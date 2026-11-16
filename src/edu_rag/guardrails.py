"""Guardrails around the answer. Each one is a plain check in code, so it adds no model call and no waiting:
  1. question      an instruction aimed at the assistant ("ignore your rules ...") is not sent to the model; an over-long message is cut
  2. passages      instructions hidden in a slide, note or web page are removed before the model reads the passage
  3. answer        how much of the answer is in the passages (measured: see support_level), and a check that the prompt was not leaked
The answer is never withheld for weak support: on the judged answers that would also have hidden correct ones, so the student is told."""
import re

from . import config, index

INJECTION = [
    ("ignore the rules", re.compile(r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|the|these|any)\b"
                                    r"[^.\n]{0,30}\b(instructions?|prompts?|rules|guidelines|context|passages)\b", re.I)),
    ("new role", re.compile(r"\byou are now\b|\bfrom now on,? you\b|\bdo anything now\b|\bjailbr(eak|oken)\b|\bdeveloper mode\b"
                            r"|\b(act|behave|respond) as (an? |the )?(unrestricted|uncensored|different|evil)\b", re.I)),
    ("reveal the prompt", re.compile(r"\b(reveal|print|show|repeat|output|tell me|what (is|are))\b[^.\n]{0,30}\b(system prompt|your (instructions|rules|prompt)|hidden prompt|initial prompt)\b", re.I)),
    ("fake system message", re.compile(r"\bnew instructions?\s*:|^\s*system\s*:|\[/?(system|inst)\]|<\|?/?(system|im_start|im_end)\|?>|###\s*(system|instruction)", re.I | re.M)),
    ("stop using the passages", re.compile(r"\bdo not (follow|obey|use|cite)\b[^.\n]{0,20}\b(rules|instructions|passages|sources)\b"
                                           r"|\b(answer|respond|reply)\b[^.\n]{0,25}\b(without|regardless of)\b[^.\n]{0,15}\b(passages|sources|materials|context)\b", re.I)),
]
PROMPT_MARKS = ("you are a study assistant", "using only the numbered passages", "reply with exactly insufficient_context")
QUESTION_REPLY = ("That message contains an instruction for the assistant, not a question about your course. "
                  "I can only answer questions from your materials. Ask about a topic from your slides or notes.")


def scan(text):
    """Names of the kinds of instruction found in a text; empty for ordinary course text."""
    return [name for name, pattern in INJECTION if pattern.search(text or "")]


def check_question(message):
    """{"message": the text to answer, "flags": [...], "reply": a fixed reply when the model must not be asked}."""
    flags, text = [], message
    if len(text) > config.MAX_QUESTION_CHARS:
        flags.append("cut to length")
        text = text[:config.MAX_QUESTION_CHARS]
    found = scan(text)
    if found:
        flags += [f"instruction: {f}" for f in found]
    return {"message": text, "flags": flags, "reply": QUESTION_REPLY if found else None}


def clean_passages(passages):
    """Passages with any instruction-like sentence removed, and what was removed. The rest of the passage is kept:
    a slide can teach about prompt injection and still be useful."""
    cleaned, removed = [], []
    for c in passages:
        if not scan(c.get("text", "")):
            cleaned.append(c)
            continue
        kept = []
        for part in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            labels = scan(part)
            if labels:
                removed.append({"source": c.get("source_name") or c.get("url"), "page": c.get("page_start"), "kinds": labels, "text": part[:200]})
            else:
                kept.append(part)
        cleaned.append({**c, "text": "\n".join(kept), "guardrail_cleaned": True})
    return cleaned, removed


def content_words(text):
    return {t for t in index.tokenize(text) if len(t) > 2 and not t.isdigit()}


def support_level(answer, passages, prefix="S"):
    """How much of the answer is in the passages the model was given: the share of each sentence's content words that occur in
    them, averaged. Measured on 155 answers judged by hand: below 0.40 caught 31 of 32 answers paired with unrelated passages and
    wrongly flagged 1 of 82 fully supported answers; below 0.58 caught 14 of 17 answers judged unsupported, wrongly flagging 11 of 82."""
    from . import generate                          # here, because generate imports this module
    body = re.sub(rf"\s*\[{prefix}\d+\]", "", answer or "")
    reference = set().union(*(content_words(c.get("text", "")) for c in passages)) if passages else set()
    shares = []
    for s in generate.split_sentences(body, prefix):
        words = content_words(s)
        if len(words) >= 3:
            shares.append(len(words & reference) / len(words))
    if not shares:
        return {"share": None, "level": "not checked"}
    share = sum(shares) / len(shares)
    level = "unsupported" if share < config.GUARD_UNSUPPORTED else "partly supported" if share < config.GUARD_PARTLY else "supported"
    return {"share": round(share, 2), "level": level}


def leaks_prompt(answer):
    low = (answer or "").lower()
    return any(mark in low for mark in PROMPT_MARKS)
