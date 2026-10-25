"""Chunking (notebook 02): one slide = one chunk, same-title neighbours merged, code and tables never split."""
import difflib
import json
import re
from pathlib import Path

from . import config

BULLET_GLYPHS = re.compile(r"^[\x00-]\s*", flags=re.M)   # PDF symbol-font bullets at line start
JUNK_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f-]")


def n_tokens(text):
    return max(1, round(len(text) / 4))


def clean_title(text):
    return re.sub(r"\s+", " ", text).strip(" :-–•")


def title_key(title):
    """Titles that differ only by case or a 'contd' marker count as the same title."""
    t = title.lower()
    t = re.sub(r"\(?\s*(cont(inued|d)?\.?|\d+\s*/\s*\d+)\s*\)?$", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def clean_text(text):
    return JUNK_CHARS.sub("", BULLET_GLYPHS.sub("• ", text))


def render_block(b):
    """Text of a block as it appears in a chunk. Code is fenced so the LLM sees it as code."""
    text = clean_text(b["text"]).strip("\n")
    if b.get("is_code"):
        return f"```\n{text}\n```"
    return text.strip()


def is_atomic(b):
    return bool(b.get("is_code") or b.get("is_table"))


def clean_blocks(page, log):
    """Drop OCR faults: runaway vision output and repeated blocks. Appends what was dropped to log."""
    blocks = [b for b in page["blocks"] if b["text"].strip()]
    vision = [b for b in blocks if b.get("ocr") == "vision" and not b.get("is_heading")]
    vision_chars = sum(len(b["text"]) for b in vision)
    if vision_chars > config.VISION_MAX_CHARS:
        log.append(("runaway vision", page["page_no"], len(vision), vision_chars))
        blocks = [b for b in blocks if b not in vision]
    kept = []
    for b in blocks:
        norm = re.sub(r"\s+", " ", b["text"]).strip()
        if any(difflib.SequenceMatcher(None, norm, k).ratio() >= config.DUP_RATIO
               for k in (re.sub(r"\s+", " ", x["text"]).strip() for x in kept) if abs(len(k) - len(norm)) < 0.1 * len(norm) + 5):
            log.append(("duplicate block", page["page_no"], 1, len(b["text"])))
            continue
        kept.append(b)
    return kept


def page_title(blocks, body_font):
    """Return (title, body_blocks). Heading blocks become the title; without them,
    leading blocks in a clearly larger font do (title slides)."""
    heads = [b for b in blocks if b.get("is_heading")]
    if heads:
        return clean_title(" ".join(b["text"] for b in heads)), [b for b in blocks if not b.get("is_heading")]
    lead = []
    for b in blocks:
        if is_atomic(b) or not b["size"] or b["size"] < body_font * config.TITLE_SIZE_RATIO or len(b["text"]) > 120:
            break
        lead.append(b)
    if lead:
        return clean_title(" ".join(b["text"] for b in lead)), blocks[len(lead):]
    return "", blocks


def split_lines(lines, budget, joiner="\n", header=None):
    pieces, cur = [], []
    head = [header] if header else []
    for line in lines:
        if cur and n_tokens(joiner.join(head + cur + [line])) > budget:
            pieces.append(joiner.join(head + cur))
            cur = []
        cur.append(line)
    if cur:
        pieces.append(joiner.join(head + cur))
    return pieces


def split_block(b, budget):
    """Split one oversized block: code at line boundaries, tables by rows (header repeated), text by lines/sentences."""
    text = b["text"].strip("\n")
    if b.get("is_code"):
        return [f"```\n{p}\n```" for p in split_lines(text.split("\n"), budget - 2)]
    if b.get("is_table"):
        rows = text.split("\n")
        return split_lines(rows[1:], budget, header=rows[0])
    lines = [l for l in text.split("\n") if l.strip()]
    if all(n_tokens(l) <= budget for l in lines):
        return split_lines(lines, budget)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", " ".join(lines)):
        out += [s[i:i + budget * 4] for i in range(0, len(s), budget * 4)]
    return split_lines(out, budget, joiner=" ")


def pack(blocks, budget):
    """Fill chunks with whole rendered blocks. Returns [(body_text, has_code, has_table)]."""
    units = []
    for b in blocks:
        r = render_block(b)
        flags = (b.get("is_code", False), b.get("is_table", False))
        if n_tokens(r) <= budget:
            units.append((r, *flags))
        else:
            units += [(p, *flags) for p in split_block(b, budget)]
    chunks, cur = [], []
    for u in units:
        if cur and n_tokens("\n".join(x[0] for x in cur + [u])) > budget:
            chunks.append(cur)
            cur = []
        cur.append(u)
    if cur:
        chunks.append(cur)
    return [("\n".join(x[0] for x in c), any(x[1] for x in c), any(x[2] for x in c)) for c in chunks]


def budget_for(title):
    return config.MAX_TOKENS - n_tokens(title + "\n\n")


def _squash(s):
    return s.replace(" ", "")


def slide_pieces(doc, log):
    pieces, dropped, prev_title = [], [], ""
    for page in doc["pages"]:
        title, body_blocks = page_title(clean_blocks(page, log), doc.get("body_font_size") or 0)
        title = title or prev_title or Path(doc["name"]).stem
        prev_title = title
        body_chars = sum(len(b["text"].strip()) for b in body_blocks)
        body_key = title_key(" ".join(b["text"] for b in body_blocks))
        only_repeats_title = body_key and _squash(body_key) in _squash(title_key(doc["name"]) + title_key(title))
        if not any(is_atomic(b) for b in body_blocks) and (body_chars < config.MIN_BODY_CHARS or only_repeats_title):
            dropped.append((page["page_no"], title))
            continue
        parts = pack(body_blocks, budget_for(title))
        for body, has_code, has_table in parts:
            pieces.append({"title": title, "body": body, "page_start": page["page_no"], "page_end": page["page_no"],
                           "has_code": has_code, "has_table": has_table, "whole_slide": len(parts) == 1})
    return pieces, dropped


def merge_same_title(pieces):
    """Merge consecutive whole slides that share a title while they fit in one chunk."""
    out = []
    for p in pieces:
        last = out[-1] if out else None
        if (last and last["whole_slide"] and p["whole_slide"] and title_key(last["title"]) == title_key(p["title"])
                and n_tokens(last["body"] + "\n\n" + p["body"]) <= budget_for(last["title"])):
            last["body"] += "\n\n" + p["body"]
            last["page_end"] = p["page_end"]
            last["has_code"] |= p["has_code"]
            last["has_table"] |= p["has_table"]
        else:
            out.append(dict(p))
    return out


def section_pieces(doc, log):
    """Non-slide sources: a heading starts a new section; sections are packed to the budget."""
    sections, cur = [], None
    for page in doc["pages"]:
        for b in clean_blocks(page, log):
            if b.get("is_heading") or cur is None:
                title = clean_title(b["text"]) if b.get("is_heading") else Path(doc["name"]).stem
                cur = {"title": title, "blocks": [], "page_start": page["page_no"], "page_end": page["page_no"]}
                sections.append(cur)
                if b.get("is_heading"):
                    continue
            cur["blocks"].append(b)
            cur["page_end"] = page["page_no"]
    pieces = []
    for s in sections:
        for body, has_code, has_table in pack(s["blocks"], budget_for(s["title"])):
            pieces.append({"title": s["title"], "body": body, "page_start": s["page_start"], "page_end": s["page_end"],
                           "has_code": has_code, "has_table": has_table, "whole_slide": False})
    return pieces, []


QUESTION_START = re.compile(r"^\s*(Q\.?\s*\d+|Question\s+\d+|\d{1,2}\s*[.)]|\(?[a-h]\))\s*", re.I)


def question_pieces(doc, log):
    """Past papers: one chunk per question, on the page where it starts, so the past-papers panel can point to it.
    Text before the first question (university name, instructions) is dropped. No question markers -> one chunk per page."""
    title = Path(doc["name"]).stem
    pieces, cur = [], None
    for page in doc["pages"]:
        for b in clean_blocks(page, log):
            text = render_block(b)
            if QUESTION_START.match(b["text"]):
                cur = {"title": title, "body": text, "page_start": page["page_no"], "page_end": page["page_no"],
                       "has_code": bool(b.get("is_code")), "has_table": bool(b.get("is_table")), "whole_slide": False}
                pieces.append(cur)
            elif cur:
                cur["body"] += "\n" + text
                cur["page_end"] = page["page_no"]
    if pieces:
        return pieces, []
    return [{"title": title, "body": "\n".join(render_block(b) for b in clean_blocks(p, log)), "page_start": p["page_no"],
             "page_end": p["page_no"], "has_code": False, "has_table": False, "whole_slide": False}
            for p in doc["pages"] if any(b["text"].strip() for b in p["blocks"])], []


def chunk_doc(doc, log=None):
    """Returns (chunks, dropped_slides). log (a list) collects OCR faults that were removed."""
    log = [] if log is None else log
    is_slides = any(p["is_slide"] for p in doc["pages"])
    if doc.get("role") == "pyq":
        pieces, dropped = question_pieces(doc, log)
    else:
        pieces, dropped = slide_pieces(doc, log) if is_slides else section_pieces(doc, log)
    if is_slides and doc.get("role") != "pyq":
        pieces = merge_same_title(pieces)
    settings_hash = config.chunk_settings_hash()
    chunks = []
    for i, p in enumerate(pieces):
        text = f"{p['title']}\n\n{p['body']}"
        chunks.append({"chunk_id": f"{doc['source_id']}-{i:04d}", "source_id": doc["source_id"], "source_name": doc["name"],
                       "role": doc["role"], "page_start": p["page_start"], "page_end": p["page_end"], "title": p["title"],
                       "has_code": p["has_code"], "has_table": p["has_table"], "n_tokens": n_tokens(text), "text": text,
                       "doc_hash": doc["hash"], "chunk_settings": settings_hash})
    return chunks, dropped


def save_chunks(chunks, path=None):
    path = Path(path or config.CHUNKS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (path.parent / "settings.json").write_text(json.dumps(dict(config.chunk_settings(), hash=config.chunk_settings_hash()), indent=2))


def load_chunks(path=None):
    path = Path(path or config.CHUNKS_FILE)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
