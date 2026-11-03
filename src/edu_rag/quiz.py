"""Practice quizzes from the student's own slides: multiple choice (marked in code) and short answer
(marked by the LLM against the source passage). Every question keeps the slide it came from."""
import json
import random
import re

from . import chunking, config, generate, index, ollama_client, retrieve

MIN_CHUNK_TOKENS = 60                             # skip title-only and link slides
PASSAGES_PER_CALL = 4
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_DIFFICULTY = "medium"
DIFFICULTY_RULES = {
    "easy": "EASY: recall. Ask for a definition, a named term or one fact stated directly in the passage; start with What is, "
            "Define, Name, State, List or Which. Never ask for the output of code, a program, a calculation or a comparison. "
            "Wrong MCQ options are obviously different from the right answer.",
    "medium": "MEDIUM: understanding. Ask how or why something works, or to explain a concept in the student's own words; "
              "start with Explain, Describe, How or Why. Wrong MCQ options are plausible but clearly wrong.",
    "hard": "HARD: application and analysis. Never ask What is, Define, Name or List, or for a fact copied from the passage. "
            "Make the student reason, using forms like: \"Given ..., what happens when ...?\", \"Why would ... fail if ...?\", "
            "\"What does this code print, and why?\", \"Compare ... and ... and justify which suits ...\". The answer must "
            "still follow only from facts in the passage. Wrong MCQ options are common misconceptions that look almost right.",
}
# The model does not always obey the rule above, so questions of the wrong kind are dropped and asked for again.
WRONG_FOR_LEVEL = {
    "easy": re.compile(r"\b(write a|what is the output|what (will|would)|what happens|predict|compare|differentiate|calculate|"
                       r"compute|printed|print|justify|if (you|a|the)|given)\b", re.I),
    "hard": re.compile(r"^\W*(what is|what are|what does|define|name|list|state|who)\b", re.I),
}
GEN_PROMPT = """You write practice questions for a student from numbered passages of their course materials.
Difficulty of every question: {difficulty}
Return JSON only: {{"questions": [ ... ]}} with exactly {n} questions, each one of:
{{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0-3, "explanation": "...", "source": "S1"}}
{{"type": "short", "question": "...", "answer": "a model answer in 1-3 sentences", "source": "S2"}}
Rules:
- Use only facts from the passages; "source" is the passage the answer comes from.
- Test the course content, not trivia about slide layout.
- All four options must be the same kind of thing as the question asks for (if it asks for a mode, all options are modes).
- Never use "None of the above", "All of the above" or similar options. Do not prefix options with letters.
- Never mention the passages or their labels (S1, S2, ...) in a question, option or answer; the student cannot see them.
- Mix the types: about {n_mcq} mcq and {n_short} short. Vary which passage each question uses."""
CONFIRM_PROMPT = """Answer a multiple-choice question using only the passage. Work it out carefully.
Return JSON only: {"answer_index": 0-3}, or {"answer_index": null} if the passage does not settle it."""
MISSING_CODE = re.compile(r"\b(?:(?:the|this|following|provided|given|above)\s+(?:python\s+)?(?:code(?:\s+snippet)?|snippet)"
                          r"|(?:following|provided|above)\s+(?:python\s+)?(?:function|program))\b", re.I)
INLINE_CODE = re.compile(r"`([^`]*)`")
SHOWS_CODE_CHARS = 20                             # shorter inline code is a name like `tell()`, not code to read
MENTIONS_SOURCE = re.compile(r"\b(?:the passages?|provided information|given information|in example \d+)\b", re.I)
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


def difficulty_rule(level):
    """The instruction for one level; an unknown level means the default."""
    return DIFFICULTY_RULES.get(level, DIFFICULTY_RULES[DEFAULT_DIFFICULTY])


def fits_level(question, level):
    """False when the question's form belongs to another level (a 'what is' at hard, 'predict the output' at easy),
    or when it points at code it does not show: the student cannot see the slide."""
    shows_code = "```" in question or any(len(c) >= SHOWS_CODE_CHARS for c in INLINE_CODE.findall(question))
    if MENTIONS_SOURCE.search(question) or (MISSING_CODE.search(question) and not shows_code):
        return False
    wrong = WRONG_FOR_LEVEL.get(level)
    return not (wrong and wrong.search(question))


def key_confirmed(q, passage, model=None):
    """A hard question needs reasoning, and the model can reason wrongly when it writes the key. It answers again without
    seeing the key; a question whose two answers differ cannot be trusted and is dropped."""
    options = "\n".join(f"{i}. {o}" for i, o in enumerate(q["options"]))
    msgs = [{"role": "system", "content": CONFIRM_PROMPT},
            {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion: {q['question']}\nOptions:\n{options}"}]
    raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)
    try:
        return json.loads(raw).get("answer_index") == q["answer_index"]
    except (ValueError, AttributeError):
        return False


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


def json_items(raw, key, field):
    """The list under key. 3B often writes one bare item, or a bare list, instead of {key: [...]}: every quiz came back empty."""
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get(key, [data] if field in data else [])
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def fill_source(item, group):
    """3B leaves out "source" on about half its questions, so the passage sharing most of its words is used. The page shown comes from this."""
    if item.get("source"):
        return item
    words = set(index.tokenize(" ".join(str(v) for k, v in item.items() if k != "type")))
    best = max(range(len(group)), key=lambda i: len(words & set(index.tokenize(group[i]["text"]))))
    return {**item, "source": f"S{best + 1}"}


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


def make(n=6, source_ids=None, topic=None, mcq_share=0.6, model=None, seed=None, progress=None, difficulty=DEFAULT_DIFFICULTY):
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
        msgs = [{"role": "system", "content": GEN_PROMPT.format(n=k, n_mcq=n_mcq, n_short=k - n_mcq, difficulty=difficulty_rule(difficulty))},
                {"role": "user", "content": f"Passages:\n{generate.format_passages(group)}"}]
        raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0.4)
        for item in json_items(raw, "questions", "question"):
            q = validate(fill_source(item, group), len(group))
            key = q and re.sub(r"\W+", " ", q["question"].lower()).strip()
            if q and key not in seen and fits_level(q["question"], difficulty):
                seen.add(key)
                c = group[q.pop("source_n") - 1]
                if difficulty == "hard" and q["type"] == "mcq" and not key_confirmed(q, c["text"], model):
                    continue
                q.update(chunk_id=c["chunk_id"], source=c["source_name"], pages=generate.page_label(c), source_text=c["text"],
                         difficulty=difficulty)
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
