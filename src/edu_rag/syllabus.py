"""Syllabus scope: units and topics read from the student's syllabus, questions matched to a topic
("Unit III · Gini index" or "outside your syllabus"), and coverage: which slides cover each topic."""
import json
import re

import numpy as np

from . import config, library, ollama_client, retrieve

UNITS_FILE = "units.json"
MAX_SYLLABUS_CHARS = 20000
EXTRACT_PROMPT = """Extract the units and topics from this course syllabus.
Return JSON only: {"units": [{"name": "Unit I: Python Fundamentals", "topics": ["Data types and type conversion", "..."]}]}
Rules:
- Keep the unit names as written (with their number).
- One topic per item the syllabus lists for that unit, copied in the syllabus's own words. Keep every named term:
  "attribute selection measures: information gain, gain ratio, Gini index" stays one topic with all three names.
- Leave out hours, credits, reference books, outcomes and exam rules."""
UNIT_SPLIT = re.compile(r"(?=\b(?:unit|module|chapter)\s*[ivx\d]+\s*[:\-–])", re.I)
UNIT_HEAD = re.compile(r"^((?:unit|module|chapter)\s*[ivx\d]+)\s*[:\-–]\s*([^\n;(]*?)\s*(?:\((?:[^)]*\d[^)]*)\)|\n|$)", re.I)
GENERIC_WORDS = set("how do does did can could should would what which why when explain tell me about way ways use using "
                    "difference between work works define meaning example examples".split())
_STATE = {}                                       # topic vectors per syllabus hash


def syllabus_text(doc):
    return "\n".join(b["text"] for p in doc["pages"] for b in p["blocks"])[:MAX_SYLLABUS_CHARS]


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


def clean_units(units):
    out = []
    for u in units if isinstance(units, list) else []:
        if not isinstance(u, dict):
            continue
        topics = [str(t).strip(" .") for t in u.get("topics") or [] if 2 < len(str(t).strip()) <= 120]
        if str(u.get("name", "")).strip() and topics:
            out.append({"name": str(u["name"]).strip(), "topics": topics})
    return out


def extract_units(doc, model=None):
    """Units and topics of a syllabus document. LLM first (layouts vary a lot), regex as the fallback."""
    text = syllabus_text(doc)
    msgs = [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}]
    try:
        raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)
        units = clean_units(json.loads(raw).get("units"))
    except (ValueError, AttributeError):
        units = []
    return units or parse_units_regex(text)


def units_for(source_id, model=None):
    """Cached in the syllabus's library folder (units.json), keyed by the file hash."""
    folder = library.find_dir(source_id)
    doc = library.load_source(source_id)
    cache = folder / UNITS_FILE if folder else None
    if cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("hash") == doc["hash"]:
            return data["units"]
    units = extract_units(doc, model)
    if cache:
        cache.write_text(json.dumps({"hash": doc["hash"], "units": units}, indent=1, ensure_ascii=False), encoding="utf-8")
    return units


def topic_table(units):
    """[(unit name, topic)] in syllabus order."""
    return [(u["name"], t) for u in units for t in u["topics"]]


def topic_vectors(units):
    key = json.dumps(units, sort_keys=True)
    if _STATE.get("key") != key:
        texts = [f"{u}: {t}" for u, t in topic_table(units)]
        prefix = config.EMBEDDERS.get(config.EMBED_MODEL, {}).get("doc_prefix", "")
        _STATE.update(key=key, vectors=ollama_client.embed(texts, config.EMBED_MODEL, prefix) if texts else np.zeros((0, 1)))
    return _STATE["vectors"]


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
    """(unit, topic, share of the question's key words found in the topic), best first. Catches named terms
    ('Gini index') that a short question embeds poorly."""
    q = key_words(question)
    if not q:
        return None
    best = max(((u, t, len(q & key_words(f"{t}")) / len(q)) for u, t in topic_table(units)), key=lambda x: x[2], default=None)
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


def coverage(units, sources=None, progress=None):
    """For each topic: the best-matching slides among the student's materials, and whether the topic is covered.
    One batched search; the score and the slides shown come from the same hits."""
    table = topic_table(units)
    by_id = retrieve.load_state()["by_id"]
    hits = retrieve.dense_search_many([t for _, t in table], n=2, sources=sources)
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
