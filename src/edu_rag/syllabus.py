"""Syllabus scope: units and topics read from the student's syllabus, questions matched to a topic
("Unit III · Gini index"), and coverage: which slides cover each topic."""
from collections import Counter
import json
import re

import numpy as np

from . import config, library, ollama_client, retrieve

UNITS_FILE = "units.json"
UNITS_VERSION = 3                                 # 2: whole document, course by course, with semesters
MAX_TOPIC_CHARS = 400                             # real syllabus lines run long; only garbage is this long
MAX_GROUP_CHARS = 12000                           # one LLM call; a course is far shorter than this
EXTRACT_NUM_PREDICT = 3000                        # a long unit list must not be cut off mid-JSON
HEAD_CHARS = 500                                  # top of a course's first page: its title and code
OVERVIEW_MIN_CODES = 3                            # a page listing this many course codes is an overview table, not a course
DIVIDER_MAX_CHARS = 200
SEMESTER = re.compile(r"\bsemester\s*[-:]?\s*(viii|vii|vi|iv|v|iii|ii|i|lll|ll|lv|vl|l|[1-8])\b", re.I)    # scans read I as l
ORDINAL_SEMESTER = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+semester\b", re.I)
ORDINALS = dict(zip("first second third fourth fifth sixth seventh eighth".split(), "I II III IV V VI VII VIII".split()))
ROMAN = dict(zip("12345678", "I II III IV V VI VII VIII".split()))
COURSE_START = re.compile(r"course\s*code.{0,200}?course\s*(?:title|name)|course\s*(?:title|name).{0,200}?course\s*code", re.I | re.S)
MIN_LIST_ITEMS = 3
UNIT_NUMBER = re.compile(r"\b(unit\s+)(l+|lv|vl+|il+|l+i)\b", re.I)
SCAN_L_WORD = re.compile(r"\bl[a-z]{3,}\b")
ASSIGNMENTS = re.compile(r"\bassignments?\s*:", re.I)
COURSE_TOPICS_EMBEDDED = 40
COURSE_FILE_MARGIN = 0.03                         # a file also belongs to courses this close to its best one
TO_LETTERS = str.maketrans("501", "SOI")
TO_DIGITS = str.maketrans("SOTRIL", "507111")
OVERVIEW_TABLE = re.compile(r"course\s*type", re.I)   # the programme's structure tables, which may list only two courses
COURSE_CODE = re.compile(r"\b[A-Z]{2,6}\s?\d{3}[A-Z]?\b")
CONTENT_START = re.compile(r"syllabus\s*[–—:-]|\bunit\s*no\b|\bassignments?\s*:|project\s+execution|ojt/internship\s+types", re.I)
CONTENT_END = re.compile(r"mapp?ing\s+of\s+course|learning\s+resources|reference\s+books?|text\s*books?\s*:|assessment\s+scheme", re.I)
EXTRACT_PROMPT = """Extract the units and topics of one course from this part of a syllabus.
Return JSON only: {"course": "Machine Learning", "code": "SMDS550", "units": [{"name": "Unit I: Python Fundamentals", "topics": ["Data types and type conversion", "..."]}]}
Rules:
- "course" is the course title as printed in the heading, and "code" its course code. The text comes from a scan: correct
  obvious scan errors everywhere ("lntroduction to SamplinB" is "Introduction to Sampling", "sMDS6s0" is "SMDS650").
- Include EVERY unit printed, in order. Keep the unit names as written (with their number and title).
- One topic per item the syllabus lists for that unit, copied in the syllabus's own words. Never skip, summarise or invent one.
  Keep every named term: "attribute selection measures: information gain, gain ratio, Gini index" stays one topic with all three names.
- A practical or lab course that lists assignments or numbered contents instead of units: one unit named "Assignments", one
  topic per assignment or numbered item.
- A project, internship or online course with no units: one unit named "Contents", one topic per item it lists, if any.
- Leave out hours, credits, reference books, outcomes and exam rules."""
UNIT_SPLIT = re.compile(r"(?=\b(?:unit|module|chapter)\s*[ivx\d]+\s*[:\-–])", re.I)
UNIT_HEAD = re.compile(r"^((?:unit|module|chapter)\s*[ivx\d]+)\s*[:\-–]\s*([^\n;(]*?)\s*(?:\((?:[^)]*\d[^)]*)\)|\n|$)", re.I)
GENERIC_WORDS = set("how do does did can could should would what which why when explain tell me about way ways use using "
                    "difference between work works define meaning example examples".split())
_STATE = {}                                       # topic vectors per syllabus hash


def parse_units_regex(text):
    """Fallback without the LLM: split at 'Unit I: Title (hours)' wherever it appears; topics split on ';'."""
    units = []
    for part in UNIT_SPLIT.split(text):
        m = UNIT_HEAD.match(part.strip() + "\n")
        if not m:
            continue
        name = f"{m.group(1).strip()}: {m.group(2).strip()}".strip(": ")
        body = part.strip()[m.end():] if m.end() < len(part.strip()) else ""
        body = re.split(r"\n\s*\n|reference books?|text books?", body, flags=re.I)[0]
        topics = [t.strip(" .\n") for t in re.split(r";|\n", body) if len(t.strip(" .\n")) > 2]
        if topics:
            units.append({"name": name, "topics": topics})
    return units


def split_topic(topic):
    """A syllabus line that lists several things becomes one topic each, so that each can be checked against the slides:
    'Networks: Perceptron, Hebb, SOM' -> 'Networks: Perceptron', 'Networks: Hebb', 'Networks: SOM'. Commas in brackets stay."""
    head, sep, rest = topic.partition(": ") if ": " in topic and "," not in topic.partition(": ")[0] else ("", "", topic)
    items = [i.strip(" .") for i in re.split(r"[,;]\s+(?![^()]*\))", rest) if i.strip(" .")]
    if len(items) < MIN_LIST_ITEMS:
        return [topic]
    return [f"{head}{sep}{i}" for i in items]


def repair_scan_words(text, df):
    """Scans read a capital I as l. 'Unit ll' -> 'Unit II'; 'lntelligence' -> 'Intelligence' when the course material uses that
    spelling more often. The LLM copies these errors even when told to correct them."""
    text = UNIT_NUMBER.sub(lambda m: m.group(1) + m.group(2).upper().replace("L", "I"), text)
    fix = lambda m: "I" + m.group(0)[1:] if df.get("i" + m.group(0)[1:].lower(), 0) > df.get(m.group(0).lower(), 0) else m.group(0)
    return SCAN_L_WORD.sub(fix, text).replace("Pro.iect", "Project")


def clean_units(units):
    out = []
    for u in units if isinstance(units, list) else []:
        if not isinstance(u, dict):
            continue
        lines = [str(t).strip(" .") for t in u.get("topics") or [] if 2 < len(str(t).strip()) <= MAX_TOPIC_CHARS]
        topics = [t for line in lines for t in split_topic(line)]
        if str(u.get("name", "")).strip() and topics:
            out.append({"name": str(u["name"]).strip(), "topics": topics})
    return out


def semester_name(text):
    """'F.Y. M.Sc. Semester II' or 'Second Semester' -> 'Semester II'; None when the text names no semester."""
    m = SEMESTER.search(text)
    if m:
        n = m.group(1).upper().replace("L", "I")
        return f"Semester {ROMAN.get(n, n)}"
    m = ORDINAL_SEMESTER.search(text)
    return f"Semester {ORDINALS[m.group(1).lower()]}" if m else None


def page_text(page):
    return "\n".join(b["text"] for b in page["blocks"])


def course_body(text):
    """The part of a course's pages that lists its content: after any outcomes, before books and mapping tables."""
    start = CONTENT_START.search(text)
    body = text[start.start():] if start else text
    end = CONTENT_END.search(body)
    return body[:end.start()] if end else body


def course_groups(doc):
    """One group per course: [{"semester", "code", "pages", "text"}]. A course starts on a page with both 'Course Code' and
    'Course Title'; the programme's overview tables, which list many codes, are not a course. Empty if no course is found."""
    groups, semester = [], None
    for p in doc["pages"]:
        text = page_text(p)
        codes = set(COURSE_CODE.findall(text))
        here = semester_name(text)
        overview = len(codes) >= OVERVIEW_MIN_CODES or OVERVIEW_TABLE.search(text)
        if COURSE_START.search(text) and not overview:
            semester = here or semester
            groups.append({"semester": semester, "code": min(codes) if codes else "", "pages": [p["page_no"]], "text": text})
        elif here and len(text) < DIVIDER_MAX_CHARS:    # a divider page: "Second Year / Third Semester"
            semester = here
        elif groups and not overview:
            groups[-1]["pages"].append(p["page_no"])
            groups[-1]["text"] += "\n" + text
    for g in groups:
        g["head"], g["text"] = g["text"][:HEAD_CHARS], course_body(g["text"])
    return groups


def plain_groups(doc):
    """A syllabus without course pages: pieces of at most MAX_GROUP_CHARS, each with the semester heading it falls under."""
    groups, semester = [], None
    for p in doc["pages"]:
        text = page_text(p)
        semester = semester_name(text) or semester
        if groups and groups[-1]["semester"] == semester and len(groups[-1]["text"]) + len(text) <= MAX_GROUP_CHARS:
            groups[-1]["text"] += "\n" + text
            groups[-1]["pages"].append(p["page_no"])
        else:
            groups.append({"semester": semester, "code": "", "pages": [p["page_no"]], "text": text[:MAX_GROUP_CHARS], "head": ""})
    return groups


def clean_code(code):
    """Scan errors in a course code like 'sMDSssT' -> 'SMDS557': the last three characters are digits, the rest letters."""
    code = re.sub(r"\s+", "", code).upper()
    if not re.fullmatch(r"[A-Z0-9]{5,9}", code):
        return code
    return code[:-3].translate(TO_LETTERS) + code[-3:].translate(TO_DIGITS)


def read_group(group, model=None):
    """(course name, units) of one group. LLM first (layouts vary a lot), regex as the fallback."""
    user = f"Heading of this part:\n{group['head']}\n\nContent:\n{group['text'][:MAX_GROUP_CHARS]}"
    try:
        raw, _ = ollama_client.chat([{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user}], model=model,
                                    json_mode=True, temperature=0, num_predict=EXTRACT_NUM_PREDICT)
        data = json.loads(raw)
        group["code"] = clean_code(str(data.get("code") or group["code"]))
        df = retrieve.load_state()["terms"]["df"]
        units = [{"name": repair_scan_words(u["name"], df), "topics": [repair_scan_words(t, df) for t in u["topics"]]}
                 for u in clean_units(data.get("units"))]
        return repair_scan_words(str(data.get("course") or "").strip(), df), units
    except (ValueError, AttributeError):
        return "", parse_units_regex(group["text"])


def extract_units(doc, model=None, progress=None):
    """Every unit of the syllabus: {"name", "unit", "course", "code", "semester", "pages", "topics"}. The whole document is read,
    one course at a time, so a long programme syllabus is not cut off. "name" stays unique across courses, which all
    repeat "Unit I". A course with no units or topics still gets an entry, so nothing disappears unnoticed."""
    groups = course_groups(doc)
    has_courses = bool(groups)
    groups = groups or plain_groups(doc)
    out = []
    for i, g in enumerate(groups, 1):
        course, units = read_group(g, model)
        listed = ASSIGNMENTS.search(g["text"])
        if not units and listed:                    # outcomes before the list can hide it: ask again with the list alone
            course, units = read_group(dict(g, text=g["text"][listed.start():]), model)
        course = course if has_courses else ""
        common = {"course": course, "code": g["code"], "semester": g["semester"], "pages": g["pages"]}
        for u in units:
            out.append({**common, "name": f"{course} · {u['name']}" if course else u["name"], "unit": u["name"], "topics": u["topics"]})
        if has_courses and not units:
            out.append({**common, "name": course or g["code"] or f"Course on p. {g['pages'][0]}", "unit": "", "topics": []})
        if progress:
            progress(i, len(groups))
    return out


def semesters(units):
    """Semesters in the order the syllabus gives them; empty when it names none."""
    return list(dict.fromkeys(u["semester"] for u in units if u.get("semester")))


def in_semester(units, semester):
    return [u for u in units if u.get("semester") == semester] if semester else units


def units_for(source_id, model=None, progress=None):
    """Cached in the syllabus's library folder (units.json), keyed by the file hash and how it was read."""
    folder = library.find_dir(source_id)
    doc = library.load_source(source_id)
    cache = folder / UNITS_FILE if folder else None
    if cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("hash") == doc["hash"] and data.get("version") == UNITS_VERSION:
            return data["units"]
    units = extract_units(doc, model, progress)
    if cache:
        cache.write_text(json.dumps({"hash": doc["hash"], "version": UNITS_VERSION, "units": units}, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    return units


def topic_table(units):
    """[(unit name, topic)] in syllabus order."""
    return [(u["name"], t) for u in units for t in u["topics"]]


def topic_cache_file():
    return config.INDEX_DIR / f"syllabus_topics_{re.sub(r'\W+', '_', config.EMBED_MODEL)}.npz"


def stored_topic_vectors():
    """{text: vector}, kept on disk: a long syllabus takes 30 seconds to embed, and chat tags, course matching and coverage wait for it."""
    if "topics" not in _STATE:
        path = topic_cache_file()
        topics = {}
        if path.exists():
            with np.load(path) as data:
                topics = dict(zip(data["texts"].tolist(), data["vectors"]))
        _STATE["topics"] = topics
    return _STATE["topics"]


def embedding_prefix(kind):
    return config.EMBEDDERS.get(config.EMBED_MODEL, {}).get(kind, "")


def stored_vectors(texts, prefix=""):
    """Vectors of these texts. Only texts never seen before are embedded, so changing semester or restarting costs nothing."""
    if not texts:
        return np.zeros((0, 1))
    known = stored_topic_vectors()
    missing = list(dict.fromkeys(t for t in texts if prefix + t not in known))
    if missing:
        known.update(zip([prefix + t for t in missing], ollama_client.embed(missing, config.EMBED_MODEL, prefix)))
        np.savez(topic_cache_file(), texts=np.array(list(known)), vectors=np.array(list(known.values())))
    return np.array([known[prefix + t] for t in texts])


def topic_vectors(units):
    return stored_vectors([f"{u}: {t}" for u, t in topic_table(units)], embedding_prefix("doc_prefix"))


def prepare_vectors(units):
    """At upload, so no page waits for them later."""
    topic_vectors(units)
    course_vectors(units)
    stored_vectors([t for _, t in topic_table(units)], embedding_prefix("query_prefix"))


def best_topics(query_vec, units, vectors=None, n=1):
    """[(unit, topic, cosine)] best first."""
    vectors = topic_vectors(units) if vectors is None else vectors
    if not len(vectors):
        return []
    sims = vectors @ query_vec
    table = topic_table(units)
    return [(table[i][0], table[i][1], float(sims[i])) for i in np.argsort(-sims)[:n]]


def key_words(text):
    return {w for w in index_tokens(text) if len(w) > 2 and w not in GENERIC_WORDS}


def index_tokens(text):
    from . import index
    return index.tokenize(text)


def word_match(question, units):
    """(unit, topic, share of the question's key words found in the topic), best first. Catches named terms that a short
    question embeds poorly. The topic must hold the question's rarest word: 'Gini index' otherwise matched a topic on journal
    indexing through 'index' alone, and 'Apriori algorithm' the Gram-Schmidt algorithm."""
    q = key_words(question)
    if not q:
        return None
    topics = [(u, t, key_words(t)) for u, t in topic_table(units)]
    counts = Counter(w for _, _, words in topics for w in words & q)
    rarest = min(counts.get(w, 0) for w in q)
    needed = {w for w in q if counts.get(w, 0) == rarest}
    best = max(((u, t, len(q & words) / len(q)) for u, t, words in topics if needed & words), key=lambda x: x[2], default=None)
    return best if best and best[2] > 0 else None


def scope(question, units, qvec=None):
    """The syllabus topic a question belongs to, or None when it is outside the syllabus.
    In scope when the meaning is close (cosine) or at least half of its key words name a topic."""
    if not units:
        return None
    qvec = retrieve.embed_query(question) if qvec is None else qvec
    dense = best_topics(qvec, units)
    words = word_match(question, units)
    if words and words[2] >= 0.5:
        return {"unit": words[0], "topic": words[1], "score": round(max(words[2], dense[0][2] if dense else 0), 3), "by": "words"}
    if dense and dense[0][2] >= config.SYLLABUS_MIN_SCORE:
        return {"unit": dense[0][0], "topic": dense[0][1], "score": round(dense[0][2], 3), "by": "meaning"}
    return None


def course_vectors(units):
    """{course: unit vector of its title and topics}, cached: the topics rarely change."""
    courses = {}
    for u in units:
        if u.get("course") and u["topics"]:
            courses.setdefault(u["course"], []).extend(u["topics"])
    key = json.dumps(courses, sort_keys=True)
    if _STATE.get("course_key") != key:
        prefix = embedding_prefix("doc_prefix")
        lists = {name: [name] + [f"{name}: {t}" for t in topics[:COURSE_TOPICS_EMBEDDED]] for name, topics in courses.items()}
        stored_vectors([t for texts in lists.values() for t in texts], prefix)      # one call, one save
        vectors = {}
        for name, texts in lists.items():
            v = stored_vectors(texts, prefix).mean(axis=0)
            vectors[name] = v / np.linalg.norm(v)
        _STATE.update(course_key=key, course_vectors=vectors)
    return _STATE["course_vectors"]


def file_vectors():
    """{source id: unit vector of all its passages}, once per index: a whole file against a whole course lets a big textbook
    belong only to its own subject, where a topic-by-topic search lets it win in every course."""
    st = retrieve.load_state()
    if _STATE.get("file_stamp") != st.get("stamp") or "file_vectors" not in _STATE:
        got = st["col"].get(include=["embeddings", "metadatas"])
        groups = {}
        for e, m in zip(got["embeddings"], got["metadatas"]):
            if m["role"] in config.EVIDENCE_ROLES:
                groups.setdefault(m["source_id"], []).append(e)
        vectors = {}
        for sid, rows in groups.items():
            v = np.mean(rows, axis=0)
            vectors[sid] = v / np.linalg.norm(v)
        _STATE.update(file_stamp=st.get("stamp"), file_vectors=vectors)
    return _STATE["file_vectors"]


def assign_files(files, courses, margin=None):
    """{course: [source ids]} from unit vectors: each file goes to its best course and any course nearly as close."""
    margin = COURSE_FILE_MARGIN if margin is None else margin
    out = {c: [] for c in courses}
    for sid, v in files.items():
        scores = {c: float(v @ cv) for c, cv in courses.items()}
        best = max(scores.values(), default=0)
        for c, s in scores.items():
            if s >= best - margin:
                out[c].append(sid)
    return out


def course_files(units, overrides=None):
    """Which files belong to which course: estimated from their content, unless the student has corrected it."""
    courses = course_vectors(units)
    if not courses:
        return {}
    return {**assign_files(file_vectors(), courses), **{c: f for c, f in (overrides or {}).items() if c in courses}}


def coverage(units, sources=None, progress=None, course_map=None):
    """For each topic: the best-matching slides among the student's materials, and whether the topic is covered. A topic of a
    course is searched only in that course's files: 'Control' or 'Case study' mean nothing without the subject around them."""
    table, hits = [], []
    groups = {}
    for u in units:
        groups.setdefault(u.get("course") or "", []).append(u)
    prefix = embedding_prefix("query_prefix")
    stored_vectors([t for _, t in topic_table(units)], prefix)          # one call, one save
    for course, rows in groups.items():
        part = topic_table(rows)
        files = sources
        if course and course_map is not None:
            files = [s for s in course_map.get(course, []) if not sources or s in sources]
            if not files:                           # no file of this course: nothing can cover its topics
                table += part
                hits += [[] for _ in part]
                continue
        table += part
        texts = [t for _, t in part]
        hits += retrieve.dense_search_many(texts, n=2, sources=files, qvecs=stored_vectors(texts, prefix))
    by_id = retrieve.load_state()["by_id"]
    out = []
    for (unit, topic), h in zip(table, hits):
        score = h[0][1] if h else 0.0
        slides = [by_id[cid] for cid, s in h if cid in by_id and s >= config.COVERAGE_MIN_SCORE]
        out.append({"unit": unit, "topic": topic, "score": round(score, 3), "covered": bool(slides),
                    "slides": [{"source": c["source_name"], "pages": c["page_start"], "unit": c.get("unit") or "p.",
                                "title": c["title"], "chunk_id": c["chunk_id"]} for c in slides]})
    if progress:
        progress(len(table), len(table))
    return out
