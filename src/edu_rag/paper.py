"""Question-paper generator: learn the pattern of the student's previous-year papers (sections, marks, "attempt any",
question styles), find which topics they ask about most, and write a NEW paper in the same pattern from the
student's slides, weighted towards the frequent topics. Optional answer key citing slide pages; PDF export."""
import json
import random
import re
import time
import unicodedata
from collections import Counter

from . import chunking, config, exam, export, generate, library, ollama_client, quiz, retrieve

BLUEPRINT_FILE = "blueprint.json"
MIN_CHUNK_TOKENS = 40
MIN_SLIDE_SUPPORT = 0.3                           # share of a question's content words that must be on its slide
REFILL_PASSES = 5
OVERVIEW_TITLES = {"contents", "table of contents", "agenda", "outline", "summary", "overview", "index", "references"}
MARKS_SUFFIX = re.compile(r"\s*[\(\[]\s*\d+\s*marks?\s*[\)\]]\s*$", re.I)   # the paper prints marks itself
COURSE_DECK_MIN = 2                              # past questions a deck must match to count as part of the course
LEVEL_STYLES = {                                 # the level overrides the past paper's style for a slot; marks stay
    "easy": {"compare": "explain", "program": "explain", "output": "short answer", "long answer": "explain"},
    "hard": {"definition": "apply to a small case", "short answer": "apply to a small case", "explain": "analyse and justify"},
}
PATTERN_PROMPT = """You read a university question paper and describe its PATTERN, not its content.
Return JSON only:
{"course": "...", "exam": "e.g. End Semester Examination", "duration_minutes": 180, "total_marks": 60,
 "instructions": ["..."],
 "sections": [{"name": "Section A", "instruction": "Attempt all questions", "attempt": null,
               "questions": [{"marks": 2, "style": "definition"}, ...]}]}
- One entry in "questions" per question printed in that section, in order, with its marks.
- "attempt": how many questions of the section the student must answer ("Attempt any FOUR" -> 4), null if all.
- style: one of definition, short answer, explain, compare, program, output, long answer.
- If the paper has no sections, use one section named "Questions"."""
WRITE_PROMPT = """You set a NEW university exam paper from numbered passages of the course slides.
Return JSON only: {{"questions": [{{"n": 1, "question": "..."}}, ...]}} with exactly {k} questions.
Question i must be answerable from passage [S i] and fit its slot:
{slots}
- Difficulty of the whole paper: {difficulty} The slot's marks still decide how long the answer is.
- Write like a real exam paper ("Explain ...", "Differentiate between ...", "Write a program to ...", "What is ...?").
  A 2-mark question asks for one fact; a long-answer question asks for explanation plus an example or program.
- Match the style of these questions from the student's previous papers, but never copy them:
{examples}
- Never mention the passages, the slides, their labels or the slides' example file names ("from listDemo.py"):
  a question that needs code includes the code."""


def norm(text):
    """NFKC fixes PDF ligatures ('ﬁles' -> 'files'); whitespace collapsed."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


# --- 1. the pattern ------------------------------------------------------------------------

def paper_text(doc, max_chars=9000):
    return "\n".join(b["text"] for p in doc["pages"] for b in p["blocks"])[:max_chars]


def clean_blueprint(bp):
    """Keep well-formed sections and questions; recompute total marks from what is asked."""
    sections = []
    for s in bp.get("sections") or []:
        qs = [{"marks": int(q["marks"]), "style": str(q.get("style") or "explain")}
              for q in s.get("questions") or [] if isinstance(q, dict) and str(q.get("marks", "")).isdigit() and 0 < int(q["marks"]) <= 30]
        if qs:
            att = s.get("attempt")
            att = att if isinstance(att, int) and 0 < att < len(qs) else None
            sections.append({"name": str(s.get("name") or f"Section {chr(65 + len(sections))}"),
                             "instruction": str(s.get("instruction") or ("Attempt any %d" % att if att else "Attempt all questions")),
                             "attempt": att, "questions": qs})
    return {"course": str(bp.get("course") or ""), "exam": str(bp.get("exam") or "Examination"),
            "duration_minutes": int(bp["duration_minutes"]) if str(bp.get("duration_minutes", "")).isdigit() else 180,
            "instructions": [str(i) for i in bp.get("instructions") or []][:5], "sections": sections,
            "total_marks": total_marks(sections)}


def total_marks(sections):
    """Marks a student can score: in an "attempt any N" section, the N highest-mark questions."""
    return sum(sum(sorted((q["marks"] for q in s["questions"]), reverse=True)[:s["attempt"] or len(s["questions"])])
               for s in sections)


def blueprint_from_questions(chunks):
    """Fallback without the LLM: one section per marks value, in the paper's order."""
    by_marks = {}
    for c in chunks:
        by_marks.setdefault(exam.parse_marks(c["text"]), []).append(c)
    sections = [{"name": f"Section {chr(65 + i)}", "instruction": "Attempt all questions", "attempt": None,
                 "questions": [{"marks": m, "style": "explain"} for _ in cs]} for i, (m, cs) in enumerate(sorted(by_marks.items()))]
    return {"course": "", "exam": "Examination", "duration_minutes": 180, "instructions": [], "sections": sections,
            "total_marks": total_marks(sections)}


def blueprint_for(source_id, model=None):
    """The paper's pattern, read once by the LLM and cached in its library folder (keyed by the file hash)."""
    folder, doc = library.find_dir(source_id), library.load_source(source_id)
    cache = folder / BLUEPRINT_FILE if folder else None
    if cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("hash") == doc["hash"]:
            return data["blueprint"]
    msgs = [{"role": "system", "content": PATTERN_PROMPT}, {"role": "user", "content": f"Question paper:\n{paper_text(doc)}"}]
    raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)
    try:
        bp = clean_blueprint(json.loads(raw))
    except (ValueError, TypeError, AttributeError, KeyError):
        bp = {"sections": []}
    if not bp["sections"]:
        bp = blueprint_from_questions([c for c in chunking.load_chunks() if c["source_id"] == source_id])
    bp["source"] = doc["name"]
    if cache:
        cache.write_text(json.dumps({"hash": doc["hash"], "blueprint": bp}, indent=1, ensure_ascii=False), encoding="utf-8")
    return bp


# --- 2. what past papers ask about -----------------------------------------------------------

def past_questions(pyq_sources):
    return [{"question": norm(exam.question_text(c)), "marks": exam.parse_marks(c["text"]), "paper": c["source_name"],
             "chunk_id": c["chunk_id"]}
            for c in retrieve.load_state()["chunks"] if c["role"] == "pyq" and c["source_id"] in pyq_sources]


def is_overview(chunk):
    """Contents / agenda / summary slides, or a passage titled only with its file name: not a topic."""
    title = chunking.clean_title(chunk["title"]).lower()
    return title in OVERVIEW_TITLES or title == chunk["source_name"].rsplit(".", 1)[0].lower()


def topic_frequency(pyq_sources, sources=None, units=None):
    """Each past question -> the slide that best matches it (and its syllabus unit, if there is a syllabus).
    Returns {"questions": [...with slide/topic], "topics": [(topic, count)], "decks": [(deck, count)]}."""
    from . import syllabus
    qs = past_questions(pyq_sources)
    hits = retrieve.dense_search_many([q["question"] for q in qs], n=5, sources=sources)
    by_id = retrieve.load_state()["by_id"]
    for q, h in zip(qs, hits):
        h = [(cid, s) for cid, s in h if not is_overview(by_id[cid])] if h else []   # "Contents" slides list everything
        c = by_id.get(h[0][0]) if h and h[0][1] >= config.PYQ_MIN_DENSE else None
        q.update(slide=c["chunk_id"] if c else None, deck=c["source_name"] if c else None,
                 topic=chunking.clean_title(c["title"]) if c else None, score=round(h[0][1], 3) if h else 0.0)
        if units:
            s = syllabus.scope(q["question"], units)
            q["unit"] = s["unit"] if s else None
            q["topic"] = s["topic"] if s else q["topic"]
    topics = Counter(q["topic"] for q in qs if q["topic"])
    decks = Counter(q["deck"] for q in qs if q["deck"])
    return {"questions": qs, "topics": topics.most_common(), "decks": decks.most_common(),
            "unmatched": sum(1 for q in qs if not q["topic"])}


# --- 3. a new paper ---------------------------------------------------------------------------------

def candidate_slides(freq, sources=None):
    """Slides to write questions from, with weights: the course's decks (those at least COURSE_DECK_MIN past questions
    match: one stray match from another course's deck does not count), weighted by how often each deck is asked,
    plus a bonus for the very slides past questions matched. All materials when no deck qualifies."""
    deck_w = {d: n for d, n in freq["decks"] if n >= COURSE_DECK_MIN}
    asked = Counter(q["slide"] for q in freq["questions"] if q["slide"])
    out = {}
    for c in retrieve.load_state()["chunks"]:
        if c["role"] not in config.EVIDENCE_ROLES or (sources and c["source_id"] not in sources) or c["n_tokens"] < MIN_CHUNK_TOKENS:
            continue
        if deck_w and c["source_name"] not in deck_w:
            continue
        out[c["chunk_id"]] = deck_w.get(c["source_name"], 1) + 3.0 * asked[c["chunk_id"]]
    return out


def pick_slides(weights, k, rng, avoid_titles=()):
    """k distinct slides, weighted, at most one per slide title."""
    by_id = retrieve.load_state()["by_id"]
    pool, picked, titles = dict(weights), [], set(avoid_titles)
    while pool and len(picked) < k:
        ids = list(pool)
        cid = rng.choices(ids, weights=[pool[i] for i in ids])[0]
        pool.pop(cid)
        title = chunking.clean_title(by_id[cid]["title"]).lower()
        if title not in titles:
            titles.add(title)
            picked.append(by_id[cid])
    return picked


def too_close(q, past, limit=0.7):
    """Word-overlap check so a 'new' question is not a past question reworded slightly."""
    words = set(re.findall(r"\w{3,}", q.lower()))
    for p in past:
        pw = set(re.findall(r"\w{3,}", p.lower()))
        if words and pw and len(words & pw) / len(words | pw) >= limit:
            return True
    return False


def write_section(section, slides, past, model=None, rng=None, taken=(), difficulty=quiz.DEFAULT_DIFFICULTY):
    """One LLM call per section: question i from slide i, with its marks and style."""
    rng = rng or random.Random()
    level_styles = LEVEL_STYLES.get(difficulty, {})
    slots = "\n".join(f"{i}. [S{i}] {q['marks']} marks, style: {level_styles.get(q['style'], q['style'])}"
                      for i, q in enumerate(section["questions"], 1))
    same = [p["question"] for p in past if p["marks"] in {q["marks"] for q in section["questions"]}] or [p["question"] for p in past]
    examples = "\n".join(f"  - {e}" for e in rng.sample(same, min(4, len(same))))
    msgs = [{"role": "system", "content": WRITE_PROMPT.format(k=len(slides), slots=slots, examples=examples, difficulty=quiz.difficulty_rule(difficulty))},
            {"role": "user", "content": f"Passages:\n{generate.format_passages(slides)}"}]
    raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0.5)
    try:
        items = json.loads(raw).get("questions", [])
    except (ValueError, AttributeError):
        items = []
    texts = {}
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and str(it.get("n", "")).isdigit():
            texts[int(it["n"])] = MARKS_SUFFIX.sub("", exam.clean_generated(str(it.get("question", ""))))
    taken = list(taken)
    out = []
    for i, slot in enumerate(section["questions"][:len(slides)], 1):
        text = texts.get(i, "")
        if (len(text) < 10 or too_close(text, [p["question"] for p in past]) or too_close(text, taken, 0.5)
                or not quiz.fits_level(text, difficulty)):
            continue
        c, share = best_slide(text, slides)        # the model sometimes writes question i from another passage
        if share < MIN_SLIDE_SUPPORT:
            continue                                # from no passage at all: dropped, refilled with other slides
        taken.append(text)
        out.append({"question": text, "marks": slot["marks"], "style": slot["style"], "difficulty": difficulty, "chunk_id": c["chunk_id"],
                    "source": c["source_name"], "pages": generate.page_label(c), "topic": chunking.clean_title(c["title"])})
    return out


def best_slide(question, slides):
    """(slide, share of the question's content words found in it), best slide first."""
    words = generate.content_words(question)
    if not words:
        return slides[0], 0.0
    scored = [(len(words & generate.content_words(c["text"])) / len(words), c) for c in slides]
    share, c = max(scored, key=lambda x: x[0])
    return c, share


def make(blueprint, freq, sources=None, model=None, seed=None, progress=None, difficulty=quiz.DEFAULT_DIFFICULTY):
    """A new paper in the blueprint's pattern at one level. Slots the model fills badly are retried with other slides."""
    rng = random.Random(seed)
    t0 = time.time()
    weights = candidate_slides(freq, sources)
    past = freq["questions"]
    sections, used = [], set()
    for si, s in enumerate(blueprint["sections"]):
        qs = []
        for _ in range(REFILL_PASSES):              # later passes refill slots whose question was dropped
            need = {"questions": s["questions"][len(qs):]}
            if not need["questions"]:
                break
            slides = pick_slides({k: v for k, v in weights.items() if k not in used}, len(need["questions"]), rng,
                                 avoid_titles={q["topic"].lower() for q in qs})
            used |= {c["chunk_id"] for c in slides}
            taken = [q["question"] for sec in sections for q in sec["questions"]] + [q["question"] for q in qs]
            qs += write_section(need, slides, past, model=model, rng=rng, taken=taken, difficulty=difficulty)
        sections.append(dict(s, questions=qs))
        if progress:
            progress(si + 1, len(blueprint["sections"]))
    n = 0
    for s in sections:
        for q in s["questions"]:
            n += 1
            q["n"] = n
    return {"course": blueprint.get("course", ""), "exam": blueprint.get("exam", "Examination"),
            "duration_minutes": blueprint.get("duration_minutes", 180), "instructions": blueprint.get("instructions", []),
            "sections": sections, "total_marks": total_marks(sections), "pattern_from": blueprint.get("source", ""),
            "difficulty": difficulty, "seconds": round(time.time() - t0, 1)}


# --- 4. answer key, PDFs, exam mode ---------------------------------------------------------------------

def exam_style(marks):
    """How long and how structured a full-marks answer is: roughly 25-35 words per mark, as university answers are."""
    if marks <= 2:
        return ("A 2-mark exam answer: a precise definition or the fact asked for in 2-3 sentences (40-70 words), "
                "with a one-line example if the question asks for one.")
    if marks <= 6:
        return (f"A {marks}-mark exam answer of AT LEAST {marks * 25} words (up to {marks * 35}): a one-sentence definition, "
                "then each key point as a numbered item explained in 2-3 sentences, then an example or short program from "
                "the passages with what it prints or shows.")
    return (f"A {marks}-mark long exam answer of AT LEAST {marks * 25} words (up to {marks * 35}), under these headings: "
            "**Definition**, **Explanation** (every key point numbered, each explained in 2-3 sentences), **Example** (a program "
            "or worked example from the passages, explained), **Advantages and limitations** (or **Comparison**, if the "
            "question compares things), **Conclusion** (1-2 sentences). If the question has several parts, cover each part.")


def answer_key(paper, sources=None, model=None, progress=None):
    """{question n: generate() result}: a cited answer per question, from its own slide plus the best matches."""
    by_id = retrieve.load_state()["by_id"]
    qs = [q for s in paper["sections"] for q in s["questions"]]
    key = {}
    for i, q in enumerate(qs, 1):
        extra = 3 if q["marks"] <= 2 else 5 if q["marks"] <= 6 else 7    # long answers draw on more slides
        found = retrieve.retrieve(q["question"], k=extra + 1, sources=sources)[0]
        passages = [by_id[q["chunk_id"]]] + [c for c in found if c["chunk_id"] != q["chunk_id"]][:extra]
        res = generate.generate(q["question"], passages, 1.0, style=exam_style(q["marks"]), model=model)
        key[q["n"]] = {"answer": res["answer"], "sources": res["sources"]}
        if progress:
            progress(i, len(qs))
    return key


def header_html(paper, title_suffix=""):
    course = export.esc(paper["course"] or "Course")
    return (f"<h1 style='text-align:center'>{export.esc(paper['exam'])}{title_suffix}</h1>"
            f"<h2 style='text-align:center'>{course}</h2>"
            f"<p><b>Time: {paper['duration_minutes'] // 60} h {paper['duration_minutes'] % 60:02d} min</b> &nbsp;&nbsp; "
            f"<b>Max. Marks: {paper['total_marks']}</b> &nbsp;&nbsp; "
            f"<b>Difficulty: {export.esc(paper.get('difficulty', quiz.DEFAULT_DIFFICULTY).capitalize())}</b></p>"
            + ("<p><i>" + " ".join(f"({i}) {export.esc(t)}" for i, t in enumerate(paper["instructions"], 1)) + "</i></p>"
               if paper["instructions"] else "")
            + "<p class='muted'>Generated by Edu-RAG from your slides, in the pattern of your previous-year papers.</p>")


def to_pdf(paper):
    parts = [header_html(paper)]
    for s in paper["sections"]:
        parts.append(f"<h2>{export.esc(s['name'])} <span class='muted'>({export.esc(s['instruction'])})</span></h2>")
        parts += [f"<p>{q['n']}. {export.esc(q['question'])} <b>[{q['marks']}]</b></p>" for q in s["questions"]]
    return export.html_to_pdf("".join(parts))


def key_to_pdf(paper, key):
    parts = [header_html(paper, " - Answer key")]
    for s in paper["sections"]:
        parts.append(f"<h2>{export.esc(s['name'])}</h2>")
        for q in s["questions"]:
            k = key.get(q["n"]) or key.get(str(q["n"])) or {}
            parts.append(f"<p class='q'>{q['n']}. {export.esc(q['question'])} [{q['marks']}]</p>")
            parts.append(f"<div>{export.md_inline_to_html(k.get('answer') or '')}</div>")
            if k.get("sources"):
                parts.append("<p class='src'>" + "<br/>".join(export.esc(export.source_line(x)) for x in k["sources"]) + "</p>")
    return export.html_to_pdf("".join(parts))


def to_exam(paper):
    """Exam-mode questions; each keeps its section's "attempt any N" so the score counts the best N."""
    qs = []
    for si, s in enumerate(paper["sections"]):
        for q in s["questions"]:
            qs.append(dict(q, origin="generated paper", section=si, section_name=s["name"], attempt=s["attempt"]))
    return {"questions": qs, "minutes": paper["duration_minutes"], "total_marks": paper["total_marks"],
            "difficulty": paper.get("difficulty", quiz.DEFAULT_DIFFICULTY)}
