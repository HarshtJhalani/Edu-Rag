"""The student's own activity, kept on this computer in data/study/: questions asked (with the sources that answered
them, for "most asked" in the Library) and thumbs up/down on answers (to review weak answers later)."""
import json
import re
import time
from collections import Counter
from pathlib import Path

from . import config


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                                # a half-written line after a power cut
    return out


def log_question(question, source_ids, path=None):
    """One answered question and the sources it cited."""
    if source_ids:
        _append(path or config.QUESTIONS_LOG, {"t": int(time.time()), "q": question.strip(), "sources": sorted(set(source_ids))})


def log_feedback(question, answer, rating, sources=(), path=None):
    """rating: 1 = helpful, 0 = not helpful."""
    _append(path or config.FEEDBACK_LOG, {"t": int(time.time()), "q": question, "answer": answer, "rating": rating,
                                          "sources": [f"{s.get('source')} {s.get('pages')}" for s in sources]})


def feedback(path=None):
    return _read(path or config.FEEDBACK_LOG)


def norm(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def most_asked(path=None, n=3):
    """{source_id: [(question, times asked)]}: the questions asked most often about each source, newest wording kept."""
    per_source, wording = {}, {}
    for row in _read(path or config.QUESTIONS_LOG):
        key = norm(row["q"])
        if not key:
            continue
        wording[key] = row["q"]
        for sid in row.get("sources", []):
            per_source.setdefault(sid, Counter())[key] += 1
    return {sid: [(wording[k], c) for k, c in counts.most_common(n)] for sid, counts in per_source.items()}
