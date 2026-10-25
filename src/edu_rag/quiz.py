"""Practice quizzes from the student's own slides: multiple choice (marked in code) and short answer
(marked by the LLM against the source passage). Every question keeps the slide it came from."""
import json
import random
import re

from . import chunking, config, generate, ollama_client, retrieve

MIN_CHUNK_TOKENS = 60                             # skip title-only and link slides
PASSAGES_PER_CALL = 4
GEN_PROMPT = """You write practice questions for a student from numbered passages of their course materials.
Return JSON only: {{"questions": [ ... ]}} with exactly {n} questions, each one of:
{{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0-3, "explanation": "...", "source": "S1"}}
{{"type": "short", "question": "...", "answer": "a model answer in 1-3 sentences", "source": "S2"}}
Rules:
- Use only facts from the passages; "source" is the passage the answer comes from.
- Test understanding, not trivia about slide layout. Wrong MCQ options must be plausible but clearly wrong.
- All four options must be the same kind of thing as the question asks for (if it asks for a mode, all options are modes).
- Never use "None of the above", "All of the above" or similar options. Do not prefix options with letters.
- Never mention the passages or their labels (S1, S2, ...) in a question, option or answer; the student cannot see them.
- Mix the types: about {n_mcq} mcq and {n_short} short. Vary which passage each question uses."""
MARK_PROMPT = """You mark a student's short answer against a model answer and the source passage.
Return JSON only: {"score": 0, 1 or 2, "feedback": "one or two sentences"}.
2 = states the key facts of the model answer correctly (wording may differ).
1 = states at least one key fact correctly but misses or gets another one wrong.
0 = wrong, vague, off-topic, or only repeats words from the question without a key fact.
Only credit what the student actually wrote; never attribute facts to the student that are not in their answer.
In the feedback, say what was right and what was missing."""
OPTION_PREFIX = re.compile(r"^\s*(\(?[a-dA-D][).:]|[a-dA-D]\s*-)\s+")
LABEL_MENTION = re.compile(r"\s*(,\s*)?\b(according to|as (stated|described|mentioned) in|in|from|per)\s+(passage\s+)?\[?S\d+\]?|\s*\(\[?S\d+\]?\)|\s*\[S\d+\]", re.I)


def strip_labels(text, sentence=False):
    """'... according to S1?' -> '...?': passage labels mean nothing to the student.
    sentence=True re-capitalises a question whose opening words were removed (not options: 'tell()' stays)."""
    out = re.sub(r"\s+([?.!,])", r"\1", LABEL_MENTION.sub("", text)).strip().lstrip(",;: ")
    return out[:1].upper() + out[1:] if sentence else out
BANNED_OPTIONS = re.compile(r"\b(none|all) of (the )?(above|these|them)\b", re.I)


def pick_passages(source_ids=None, topic=None, n=PASSAGES_PER_CALL * 2, seed=None):
    """Topic -> the best-matching passages; otherwise a spread of substantial slides from the chosen sources."""
    if topic:
        passages, _, _ = retrieve.retrieve(topic, k=n, sources=source_ids)
        return passages
    pool = [c for c in chunking.load_chunks() if c["role"] in config.EVIDENCE_ROLES
            and (not source_ids or c["source_id"] in source_ids) and c["n_tokens"] >= MIN_CHUNK_TOKENS]
    rng = random.Random(seed)
    by_title = {}
    for c in rng.sample(pool, len(pool)):
        by_title.setdefault((c["source_id"], c["title"]), c)          # one chunk per slide title: more variety
    picked = list(by_title.values())[:n]
    return sorted(picked, key=lambda c: (c["source_name"], c["page_start"]))


def validate(q, n_passages):
    """Well-formed question with a valid source label, else None."""
    try:
        src = int(re.fullmatch(r"\s*\[?S(\d+)\]?\s*", str(q.get("source", ""))).group(1))
    except (AttributeError, ValueError):
        return None
    if not 1 <= src <= n_passages or not str(q.get("question", "")).strip():
        return None
    if q.get("type") == "mcq":
        opts = [OPTION_PREFIX.sub("", str(o)).strip() for o in q.get("options") or []]
        idx = q.get("answer_index")
        if (len(opts) != 4 or len(set(o.lower() for o in opts)) != 4 or not isinstance(idx, int) or not 0 <= idx < 4
                or any(BANNED_OPTIONS.search(o) for o in opts)):
            return None
        return {"type": "mcq", "question": strip_labels(q["question"], True), "options": [strip_labels(o) for o in opts],
                "answer_index": idx, "explanation": strip_labels(str(q.get("explanation", "")), True), "source_n": src}
    if q.get("type") == "short" and str(q.get("answer", "")).strip():
        return {"type": "short", "question": strip_labels(q["question"], True), "answer": strip_labels(q["answer"], True),
                "source_n": src}
    return None


def shuffle_options(q, rng):
    """LLMs favour putting the right option first; shuffle so the answer position carries no hint."""
    order = list(range(4))
    rng.shuffle(order)
    q["options"] = [q["options"][i] for i in order]
    q["answer_index"] = order.index(q["answer_index"])
    return q


def make(n=6, source_ids=None, topic=None, mcq_share=0.6, model=None, seed=None, progress=None):
    """Returns a list of questions; each has its source chunk (for the citation and the slide preview)."""
    passages = pick_passages(source_ids, topic, seed=seed)
    rng = random.Random(seed)
    groups = [passages[i:i + PASSAGES_PER_CALL] for i in range(0, len(passages), PASSAGES_PER_CALL)] or [[]]
    per_call = -(-n // len(groups))
    questions, seen = [], set()
    calls = [g for g in groups if g] * 2                # second round tops up when the model returns too few
    for gi, group in enumerate(calls):
        if len(questions) >= n:
            break
        k = min(per_call, n - len(questions)) + 1           # one spare: validation drops a malformed one now and then
        n_mcq = round(k * mcq_share)
        msgs = [{"role": "system", "content": GEN_PROMPT.format(n=k, n_mcq=n_mcq, n_short=k - n_mcq)},
                {"role": "user", "content": f"Passages:\n{generate.format_passages(group)}"}]
        raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0.4)
        try:
            items = json.loads(raw).get("questions", [])
        except (ValueError, AttributeError):
            items = []
        for item in items if isinstance(items, list) else []:
            q = validate(item, len(group)) if isinstance(item, dict) else None
            key = q and re.sub(r"\W+", " ", q["question"].lower()).strip()
            if q and key not in seen:
                seen.add(key)
                c = group[q.pop("source_n") - 1]
                q.update(chunk_id=c["chunk_id"], source=c["source_name"], pages=generate.page_label(c), source_text=c["text"])
                questions.append(shuffle_options(q, rng) if q["type"] == "mcq" else q)
        if progress:
            progress(min(len(questions), n), n)
    return questions[:n]


def mark_mcq(q, chosen_index):
    return {"score": 2 if chosen_index == q["answer_index"] else 0, "max": 2, "correct": q["options"][q["answer_index"]],
            "feedback": q.get("explanation", "")}


def mark_short(q, student_answer, model=None):
    if not student_answer.strip():
        return {"score": 0, "max": 2, "feedback": "No answer given.", "correct": q["answer"]}
    msgs = [{"role": "system", "content": MARK_PROMPT},
            {"role": "user", "content": f"Passage:\n{q['source_text']}\n\nQuestion: {q['question']}\n"
                                        f"Model answer: {q['answer']}\nStudent answer: {student_answer}"}]
    raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)
    try:
        j = json.loads(raw)
        score = int(j.get("score", 0))
        score = score if score in (0, 1, 2) else 0
        return {"score": score, "max": 2, "feedback": str(j.get("feedback", "")).strip(), "correct": q["answer"]}
    except (ValueError, TypeError, AttributeError):
        return {"score": 0, "max": 2, "feedback": "Could not mark this answer automatically; compare with the model answer.",
                "correct": q["answer"]}
