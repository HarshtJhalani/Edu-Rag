"""Grounded answers (notebook 05): [S#] citations checked in code, support check, abstain rule."""
import re
import time

from . import config, index, ollama_client, retrieve

SYSTEM_PROMPT = f"""You are a study assistant. Answer the student's question using ONLY the numbered passages from their course materials.
Rules:
- After each sentence that states a fact, cite the passage or passages it comes from, like [S1] or [S2][S3].
- Use only facts from the passages. Do not add facts from your own knowledge.
- If the passages do not contain the answer, reply with exactly {config.INSUFFICIENT} and nothing else.
- Cite with the passage label only, like [S2]. Never put a title or any other text inside square brackets.
- Copy code exactly as it appears in the passages, inside ``` fences.
- Write maths in plain text (for example b(new) = b(old) + a*t), not LaTeX.
- Do not mention page numbers or file names; the app shows sources."""

STYLES = {
    "concise": "Answer in 2 to 4 sentences.",
    "simple": "Explain for a beginner: short sentences, no jargon, and define any technical term you use.",
    "step-by-step": "Answer as a numbered list of steps.",
    "worked example": "Explain briefly, then walk through one worked example that uses code or numbers from the passages.",
}


# --- prompt -------------------------------------------------------------------

def format_passages(passages, prefix="S"):
    return "\n\n".join(f"[{prefix}{i}] ({c['source_name']}, {c['title']})\n{c['text']}" for i, c in enumerate(passages, start=1))


def build_messages(question, passages, style="concise", history=(), system=SYSTEM_PROMPT, prefix="S"):
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in list(history)[-config.MAX_HISTORY_MESSAGES:]]
    msgs.append({"role": "user", "content": f"Passages:\n{format_passages(passages, prefix)}\n\n"
                                            f"Style: {STYLES[style]}\n\nQuestion: {question}"})
    return msgs


# --- citations ----------------------------------------------------------------

def cite_pattern(prefix="S"):
    return re.compile(rf"\[\s*{prefix}\d+(?:\s*(?:,|-|–)\s*{prefix}?\d+)*\s*\]")


def cite_numbers(group):
    nums = [int(x) for x in re.findall(r"\d+", group)]
    if re.search(r"[-–]", group) and len(nums) == 2 and nums[0] <= nums[1]:
        return list(range(nums[0], nums[1] + 1))
    return nums


def title_citations_to_labels(text, passages, prefix="S"):
    """The model sometimes cites a passage by its title, e.g. [Tuples are more efficient]. Map those to [S#]."""
    titles = {re.sub(r"\W+", " ", c["title"]).strip().lower(): i for i, c in enumerate(passages, start=1)}

    def fix(m):
        key = re.sub(r"\W+", " ", m.group(1)).strip().lower()
        return f"[{prefix}{titles[key]}]" if key in titles else m.group(0)
    return re.sub(r"\[([^\]\[]{3,80})\]", fix, text)


def parse_citations(text, n_passages, prefix="S"):
    """Normalise [S1, S3] / [S1-S3] to [S1][S3]; drop numbers outside 1..n. Returns (text, valid, invalid)."""
    valid, invalid = [], []

    def fix(m):
        nums = cite_numbers(m.group(0))
        good = [x for x in nums if 1 <= x <= n_passages]
        invalid.extend(x for x in nums if not 1 <= x <= n_passages)
        valid.extend(good)
        return "".join(f"[{prefix}{x}]" for x in good)
    return cite_pattern(prefix).sub(fix, text), sorted(set(valid)), invalid


def page_label(c):
    """'p. 3', 'p. 3-5', 'slide 4', 'cell 12' (the unit comes from the source's file type)."""
    unit = c.get("unit") or "p."
    unit = unit if unit.endswith(".") else unit + " "
    unit = unit if unit.endswith(" ") else unit + " "
    return f"{unit}{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")


def source_entry(c, n, prefix="S"):
    """Pages always come from chunk metadata, never from the LLM."""
    return {"label": f"{prefix}{n}", "source": c["source_name"], "pages": page_label(c), "title": c["title"],
            "chunk_id": c["chunk_id"], "text": c["text"], "url": c.get("url", "")}


# --- support check ------------------------------------------------------------

def content_words(text):
    return {t for t in index.tokenize(text) if len(t) > 2 and not t.isdigit()}


def split_sentences(text, prefix="S"):
    """Sentences outside code blocks. Citations written after the full stop ('... plot. [S1] Next')
    belong to the sentence before them, so they are moved back onto it."""
    cite = cite_pattern(prefix)
    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", prose) if p.strip()]
    merged = []
    for p in parts:
        lead = re.match(rf"^((?:\[{prefix}\d+\]\s*)+)(.*)$", p)
        if lead and merged:
            merged[-1] += " " + lead.group(1).strip()
            p = lead.group(2).strip()
        if p:
            merged.append(p)
    return [s for s in merged if len(content_words(cite.sub("", s))) >= 3]


def support_check(answer, passages, prefix="S"):
    """Share of each sentence's content words found in the passages it cites; weak and uncited sentences flagged."""
    cite = cite_pattern(prefix)
    report = []
    for s in split_sentences(answer, prefix):
        nums = [x for g in cite.findall(s) for x in cite_numbers(g) if 1 <= x <= len(passages)]
        words = content_words(cite.sub("", s))
        if not nums:
            report.append({"sentence": s, "cited": [], "overlap": 0.0, "flag": "no citation"})
            continue
        ref = set().union(*(content_words(passages[x - 1]["text"]) for x in nums))
        overlap = len(words & ref) / len(words) if words else 1.0
        report.append({"sentence": s, "cited": nums, "overlap": round(overlap, 2),
                       "flag": "weak support" if overlap < config.SUPPORT_MIN else None})
    return report


# --- answer -------------------------------------------------------------------

MIN_ANSWER_WORDS = 8


def strip_insufficient(raw):
    """None when the model says the passages are insufficient. A real cited answer that merely ends with the
    token (the model sometimes answers, then adds it) keeps its content with the token removed."""
    if config.INSUFFICIENT not in raw:
        return raw
    rest = raw.replace(config.INSUFFICIENT, "").strip()
    has_cite = bool(cite_pattern().search(rest))
    if has_cite and len(content_words(cite_pattern().sub("", rest))) >= MIN_ANSWER_WORDS:
        return rest
    return None

def empty_result(question, style):
    return {"question": question, "style": style, "abstained": False, "abstain_reason": None, "answer": None,
            "raw": None, "sources": [], "citations_ok": False, "invalid_citations": [], "support": [],
            "llm": None, "llm_insufficient": False, "retrieved": [], "top_dense": None}


def generate(question, passages, top_dense, style="concise", history=(), model=None, on_token=None):
    """Answer from given passages. Abstain below the dense floor (no LLM call) or when the LLM says insufficient.
    on_token(piece) streams the raw answer as it is written (the app shows it live, then the checked version)."""
    out = empty_result(question, style)
    out.update(retrieved=[c["chunk_id"] for c in passages], top_dense=top_dense)
    if not passages or top_dense < config.DENSE_FLOOR:
        out.update(abstained=True, abstain_reason="off-topic (below dense floor)", answer=config.ABSTAIN_MSG)
        return out
    msgs = build_messages(question, passages, style, history)
    raw, stats = (ollama_client.chat_stream(msgs, model=model, on_token=on_token) if on_token
                  else ollama_client.chat(msgs, model=model))
    body = strip_insufficient(raw)
    out.update(raw=raw, llm=stats, llm_insufficient=body is None)
    if body is None:
        out.update(abstained=True, abstain_reason="model said insufficient", answer=config.ABSTAIN_MSG)
        return out
    text, valid, invalid = parse_citations(title_citations_to_labels(body, passages), len(passages))
    out.update(answer=text, invalid_citations=invalid, support=support_check(text, passages))
    if valid:
        out.update(citations_ok=True, sources=[source_entry(passages[n - 1], n) for n in valid])
    else:
        out["sources"] = [source_entry(c, i) for i, c in enumerate(passages[:3], start=1)]   # fallback
    return out


def answer(question, style="concise", history=(), roles=None, sources=None, model=None):
    """Fixed pipeline: retrieve top-k, then generate."""
    t0 = time.time()
    passages, top_dense, _ = retrieve.retrieve(question, roles=roles, sources=sources)
    out = generate(question, passages, top_dense, style, history, model)
    out["seconds"] = round(time.time() - t0, 2)
    return out
