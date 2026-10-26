"""Flashcards from the student's slides, reviewed with Leitner boxes (spaced repetition).
A card: front (term or question), back (short answer), and the slide it came from. "Knew it" moves the card
up one box (seen again later); "Didn't know" sends it back to box 0 (seen again today)."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from . import config, generate, ollama_client, quiz

PROMPT = """You make study flashcards for a student from numbered passages of their course materials.
Return JSON only: {{"cards": [{{"front": "...", "back": "...", "source": "S1"}}, ...]}} with about {n} cards.
Rules:
- front: a key term, or a short question (max 15 words). back: the answer in 1-2 sentences (max 40 words).
- Prefer definitions, purposes, differences, formulas and the steps of a method. No trivia about the slides.
- Use only facts from the passages; "source" is the passage the back comes from.
- Never mention the passages or their labels (S1, S2, ...) on a card."""


def card_id(front, chunk_id):
    return hashlib.sha1(f"{chunk_id}|{front.lower().strip()}".encode()).hexdigest()[:12]


def validate(c, n_passages):
    """Well-formed card with a valid source label, else None."""
    try:
        src = int(re.fullmatch(r"\s*\[?S(\d+)\]?\s*", str(c.get("source", ""))).group(1))
    except (AttributeError, ValueError):
        return None
    front, back = quiz.strip_labels(str(c.get("front", ""))), quiz.strip_labels(str(c.get("back", "")), True)
    if not 1 <= src <= n_passages or not front or not back or front.lower() == back.lower():
        return None
    return {"front": front, "back": back, "source_n": src}


def make(n=12, source_ids=None, topic=None, model=None, seed=None, progress=None):
    """New cards (not yet saved) from a spread of slides, or from the passages that best match a topic."""
    passages = quiz.pick_passages(source_ids, topic, n=max(4, n // 2 + 2), seed=seed)
    return from_passages(passages, n, model=model, progress=progress)


def from_passages(passages, n=12, model=None, progress=None):
    """Cards from exactly these passages (e.g. one cited slide)."""
    groups =[passages[i:i + quiz.PASSAGES_PER_CALL] for i in range(0, len(passages), quiz.PASSAGES_PER_CALL)]
    per_call = -(-n // max(len(groups), 1))
    cards, seen = [], set()
    for gi, group in enumerate(groups):
        if len(cards) >= n:
            break
        msgs = [{"role": "system", "content": PROMPT.format(n=per_call + 1)},
                {"role": "user", "content": f"Passages:\n{generate.format_passages(group)}"}]
        raw, _ = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0.3)
        try:
            items = json.loads(raw).get("cards", [])
        except (ValueError, AttributeError):
            items = []
        for item in items if isinstance(items, list) else []:
            c = validate(item, len(group)) if isinstance(item, dict) else None
            key = c and re.sub(r"\W+", " ", c["front"].lower()).strip()
            if c and key not in seen:
                seen.add(key)
                p = group[c.pop("source_n") - 1]
                c.update(id=card_id(c["front"], p["chunk_id"]), chunk_id=p["chunk_id"], source_id=p["source_id"],
                         source=p["source_name"], pages=generate.page_label(p), source_text=p["text"])
                cards.append(c)
        if progress:
            progress(gi + 1, len(groups))
    return cards[:n]


# --- The saved deck ------------------------------------------------------------------------

def load(path=None):
    try:
        return json.loads(Path(path or config.FLASHCARDS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save(cards, path=None):
    p = Path(path or config.FLASHCARDS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")


def today():
    return dt.date.today().isoformat()


def add(new_cards, path=None, day=None):
    """Adds cards to the deck (box 0, due today); a card already in the deck is skipped. Returns how many were added."""
    deck = load(path)
    have = {c["id"] for c in deck}
    fresh = [c | {"box": 0, "due": day or today(), "reviews": 0, "lapses": 0} for c in new_cards if c["id"] not in have]
    save(deck + fresh, path)
    return len(fresh)


def due(deck, day=None, source_ids=None):
    """Cards due on or before the day, lowest box first (the ones the student knows least)."""
    day = day or today()
    out = [c for c in deck if c["due"] <= day and (not source_ids or c.get("source_id") in source_ids)]
    return sorted(out, key=lambda c: (c["box"], c["due"]))


def review(card, knew, day=None):
    """Leitner step; returns the updated card."""
    d = dt.date.fromisoformat(day or today())
    box = min(card["box"] + 1, len(config.LEITNER_DAYS) - 1) if knew else 0
    return card | {"box": box, "due": (d + dt.timedelta(days=config.LEITNER_DAYS[box])).isoformat(),
                   "reviews": card.get("reviews", 0) + 1, "lapses": card.get("lapses", 0) + (0 if knew else 1)}


def record(card_id_, knew, path=None, day=None):
    deck = load(path)
    deck = [review(c, knew, day) if c["id"] == card_id_ else c for c in deck]
    save(deck, path)
    return next((c for c in deck if c["id"] == card_id_), None)


def remove(card_ids, path=None):
    ids = set(card_ids)
    save([c for c in load(path) if c["id"] not in ids], path)


def stats(deck, day=None):
    day = day or today()
    top = len(config.LEITNER_DAYS) - 1
    return {"total": len(deck), "due": len(due(deck, day)), "new": sum(c.get("reviews", 0) == 0 for c in deck),
            "mastered": sum(c["box"] >= top - 1 for c in deck),
            "boxes": [sum(c["box"] == b for c in deck) for b in range(top + 1)]}
