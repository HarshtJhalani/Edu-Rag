"""Exam mode: a timed mock paper. Questions come from the student's past papers (optionally only the syllabus
units they choose), topped up with exam-style questions written from their slides. Each answer is marked by the
LLM against the slide passages that answer the question, with a model answer citing those slides."""
import json
import random
import re
import unicodedata

from . import config, generate, ollama_client, quiz, retrieve, syllabus

UNIT_TOPICS_SEARCHED = 24
DIFFICULTY_MARKS = {"easy": 2, "medium": 5, "hard": 10}   # marks of a generated question at each level
MARKS_PATTERNS = [r"[\(\[]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\)\]]\s*$", r"\b(\d{1,2})\s*marks?\b", r"\b(\d{1,2})\s*M\s*$"]
SECTION_LINE = re.compile(r"^\s*(section|part)\s+[a-e1-5]\b.*$", re.I | re.M)
NUMBER_PREFIX = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?\d+\s*\(?[a-z]?\s*[.)]\s*|^\s*\(?[a-h]\)\s*", re.I)
GEN_PROMPT = """You set exam questions for a university course from numbered passages of the course slides.
Return JSON only: {{"questions": [{{"question": "...", "source": "S1"}}, ...]}} with exactly {n} questions.
Every question is worth {marks} marks. Difficulty: {difficulty}
Rules:
- Write questions the way a university exam paper does: "Explain ...", "Differentiate between ...", "Write a program ...",
  "What is ...? Give an example." 2 marks = a definition, 5 = an explanation, 10 = a long answer or comparison.
- Each question must be answerable from its passage ("source"). Vary the passages.
- Never mention the passages, "the given text", "the slides" or their labels (S1, S2, ...): the paper stands alone."""
PASSAGE_MENTION = re.compile(r"\s*,?\s*\b(?:(?:as\s+)?(?:mentioned|given|described|discussed|shown|provided)\s+)?(?:in|from)\s+(?:the\s+)?"
                             r"(?:given\s+)?(?:[\w`.-]+\s+){0,3}(?:passages?|text|slides?|materials?)\b|(?<!\bis)(?<!\bare)(?<!\bbe)(?<!\bnot)\s+(?:mentioned|provided|given)(?=\s*[?.])", re.I)
MARK_PROMPT = """You are a university examiner. Mark the student's answer to an exam question.
Inside <reference> are numbered passages from the course slides: YOUR marking scheme, never seen by the student.
Only the text inside <student_answer> is what the student wrote (closed book: citations or slide wording are never
expected). Judge each fact the question asks for: is it in <student_answer>, and is it correct?
Return JSON only: {{"score": 0-{marks}, "feedback": "...", "model_answer": "..."}}
- score: whole marks out of {marks}. An answer that gives everything the question asks for (correct facts, the
  requested example or calculation) gets full marks, even in other words or briefly. Take marks off only for missing
  or wrong content. A blank, off-topic or wrong answer gets 0. Only credit what the student actually wrote.
- feedback: 1-3 sentences to the student: what was good, what content was missing or wrong.
- model_answer: a full-marks answer as long as the marks need (about {words} words; definition, explained key points,
  an example), citing the reference passages like [S1] after each fact. Use only facts from them."""


def parse_marks(text, default=None):
    """'... (5 marks)', '... [10]', '... 5M' -> 5 / 10 / 5; the default when no marks are printed.
    A trailing section heading ('SECTION B (... 6 marks each)') belongs to the next questions, so it is ignored."""
    text = SECTION_LINE.sub("", text).strip()
    tail = text.splitlines()[-1] if text else ""
    for pat in MARKS_PATTERNS:
        m = re.search(pat, tail, re.I) or re.search(pat, text, re.I)
        if m and 1 <= int(m.group(1)) <= 30:
            return int(m.group(1))
    return default or config.EXAM_DEFAULT_MARKS


def question_text(chunk):
    """The question as printed, without the paper title the chunker puts first, the question number or the marks."""
    body = unicodedata.normalize("NFKC", chunk["text"]).split("\n\n", 1)[-1].strip()   # NFKC: PDF ligatures 'ﬁ' -> 'fi'
    body = SECTION_LINE.sub("", body).strip()       # a "SECTION B (...)" heading lands after the previous question
    body = NUMBER_PREFIX.sub("", body, count=1)
    for pat in MARKS_PATTERNS[:1]:
        body = re.sub(pat, "", body, flags=re.I).strip()
    return re.sub(r"\s+", " ", body)


def marks_level(marks):
    """A past question's level, judged by its marks: nobody labelled them, and marks are what the paper's setter gave."""
    return "easy" if marks <= 4 else "medium" if marks <= 8 else "hard"


def past_paper_questions(sources=None, units=None, chosen_units=None):
    """Questions from the included past papers; with chosen_units, only questions that fall in those units."""
    out = []
    for c in retrieve.load_state()["chunks"]:
        if c["role"] != "pyq" or (sources and c["source_id"] not in sources):
            continue
        text = question_text(c)
        if len(text) < 15:
            continue
        marks = parse_marks(c["text"])
        q = {"question": text, "marks": marks, "difficulty": marks_level(marks), "origin": "past paper", "source": c["source_name"],
             "pages": generate.page_label(c), "chunk_id": c["chunk_id"]}
        if chosen_units:
            hit = syllabus.scope(text, units)
            if not hit or hit["unit"] not in chosen_units:
                continue
            q["unit"] = hit["unit"]
        out.append(q)
    return out


def clean_generated(text):
    """'..., providing an example from the given passages.' -> '..., providing an example.'"""
    t = PASSAGE_MENTION.sub("", quiz.strip_labels(text, True))
    return re.sub(r"\s+([?.!,])", r"\1", t).strip()


def unit_passages(units, chosen_units, sources, per_topic=2, max_topics=UNIT_TOPICS_SEARCHED, seed=None, course_map=None):
    """Slide passages that best match the topics of the chosen units, searched only in each course's own files. A sample of
    topics is searched, because a semester has hundreds; a topic the slides only loosely match is skipped."""
    course_map = syllabus.course_files(units) if course_map is None else course_map
    topics = [(u.get("course"), t) for u in units if u["name"] in chosen_units for t in u["topics"]]
    picked = {}
    for course, t in random.Random(seed).sample(topics, min(max_topics, len(topics))):
        files = [s for s in course_map[course] if not sources or s in sources] if course in course_map else sources
        if course in course_map and not files:
            continue
        found, top, _ = retrieve.retrieve(t, k=per_topic, sources=files)
        if top >= config.COVERAGE_MIN_SCORE:
            for c in found:
                picked.setdefault(c["chunk_id"], c)
    return list(picked.values())


def generate_questions(n, passages, model=None, seed=None, difficulty=quiz.DEFAULT_DIFFICULTY):
    """Exam-style questions from slide passages at one level, each tied to the passage that answers it."""
    rng = random.Random(seed)
    marks = DIFFICULTY_MARKS.get(difficulty, config.EXAM_DEFAULT_MARKS)
    passages = rng.sample(passages, len(passages))
    out, seen = [], set()
    for i in range(0, len(passages), quiz.PASSAGES_PER_CALL):
        if len(out) >= n:
            break
        group = passages[i:i + quiz.PASSAGES_PER_CALL]
        k = min(n - len(out), len(group)) + 1
        msgs = [{"role": "system", "content": GEN_PROMPT.format(n=k, marks=marks, difficulty=quiz.difficulty_rule(difficulty))},
                {"role": "user", "content": f"Passages:\n{generate.format_passages(group)}"}]
        raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0.4)
        try:
            items = json.loads(raw).get("questions", [])
        except (ValueError, AttributeError):
            items = []
        for it in items if isinstance(items, list) else []:
            m = re.fullmatch(r"\s*\[?S(\d+)\]?\s*", str(it.get("source", ""))) if isinstance(it, dict) else None
            text = clean_generated(str(it.get("question", ""))) if m else ""
            if (not m or not 1 <= int(m.group(1)) <= len(group) or len(text) < 10 or text.lower() in seen
                    or not quiz.fits_level(text, difficulty)):
                continue
            seen.add(text.lower())
            c = group[int(m.group(1)) - 1]
            out.append({"question": text, "marks": marks, "difficulty": difficulty, "origin": "from your slides", "source": c["source_name"],
                        "pages": generate.page_label(c), "chunk_id": c["chunk_id"]})
    return out[:n]


def build_paper(n=5, minutes=30, sources=None, pyq_sources=None, units=None, chosen_units=None, use_past_papers=True,
                model=None, seed=None, difficulty=quiz.DEFAULT_DIFFICULTY, course_map=None):
    """{"questions", "minutes", "total_marks"}: past-paper questions of that level first (shuffled), then generated ones."""
    rng = random.Random(seed)
    qs = past_paper_questions(pyq_sources, units, chosen_units) if use_past_papers and pyq_sources else []
    qs = [q for q in qs if q["difficulty"] == difficulty]
    qs = rng.sample(qs, min(n, len(qs)))
    if len(qs) < n:
        if chosen_units and units:
            passages = unit_passages(units, chosen_units, sources, seed=seed, course_map=course_map)
        else:
            passages = quiz.pick_passages(sources, None, n=(n - len(qs)) * 2, seed=seed)
        qs += generate_questions(n - len(qs), passages, model=model, seed=seed, difficulty=difficulty)
    return {"questions": qs, "minutes": minutes, "total_marks": sum(q["marks"] for q in qs), "difficulty": difficulty}


def mark(question, answer, sources=None, model=None):
    """Marks one answer against the best-matching slide passages. Returns score, max, feedback, model answer, sources."""
    passages = retrieve.retrieve(question["question"], k=config.EXAM_MARK_K, sources=sources)[0]
    marks, blank = question["marks"], not answer.strip()
    msgs = [{"role": "system", "content": MARK_PROMPT.format(marks=marks, words=f"{marks * 25}-{marks * 35}" if marks > 2 else "40-70")},
            {"role": "user", "content": f"<reference>\n{generate.format_passages(passages)}\n</reference>\n\n"
                                        f"<question marks=\"{marks}\">{question['question']}</question>\n\n"
                                        f"<student_answer>\n{'(blank)' if blank else answer.strip()}\n</student_answer>"}]
    raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)   # also for a blank: the model answer
    try:
        j = json.loads(raw)
    except ValueError:
        j = {}
    j = j if isinstance(j, dict) else {}
    score = j.get("score") if isinstance(j.get("score"), int) and not blank else 0
    feedback = "No answer given." if blank else str(j.get("feedback", "")).strip()
    model_answer = str(j.get("model_answer", "")).strip()
    text, valid, _ = generate.parse_citations(generate.title_citations_to_labels(model_answer, passages), len(passages))
    return {"score": max(0, min(score, marks)), "max": marks, "feedback": feedback, "model_answer": text,
            "sources": [generate.source_entry(passages[n - 1], n) for n in valid]}


def paper_score(questions, marks):
    """(score, max): questions with a section "attempt" count only the best N answers of that section."""
    groups = {}
    for q, m in zip(questions, marks):
        groups.setdefault(q.get("section"), {"attempt": q.get("attempt"), "rows": []})["rows"].append((m["score"], m["max"]))
    score = best = 0
    for g in groups.values():
        n = g["attempt"] or len(g["rows"])
        score += sum(sorted((s for s, _ in g["rows"]), reverse=True)[:n])
        best += sum(sorted((mx for _, mx in g["rows"]), reverse=True)[:n])
    return score, best
