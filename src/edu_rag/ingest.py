"""PDF / pasted-text ingestion (notebook 01): text layer first, OCR fallback, headers removed, titles and code tagged."""
import base64
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import requests

from . import config

BULLETS = ("•", "▪", "◦", "‣", "●", "○", "■", "□", "➢", "➤", "►", "✓", "✔", "–", "- ", "* ")
CODE_LINE_RE = re.compile(
    r"^\s*(>>>|\.\.\.\s|def |class |import |from \S+ import|return\b|print\(|"
    r"for .+:$|if .+:$|while .+:$|elif .+:$|else:$|try:$|except\b.*:$|with .+:$)")
MONO_HINTS = ("mono", "courier", "consol", "menlo", "inconsolata", "code")
PAGE_NUM_RE = re.compile(r"(page\s*)?#(\s*(of|/)\s*#)?")
VISION_PROMPT = (
    "Transcribe ALL text in this image exactly as written. Rules:\n"
    "- If the image is a slide, put its title on the first line starting with '# '.\n"
    "- Put any code inside ```python fences and keep its exact indentation.\n"
    "- Write tables as Markdown tables.\n"
    "- Read multi-column layouts column by column, left column first.\n"
    "- Output only the transcription, no comments."
)
_OCR = {}                                          # RapidOCR engine, loaded on first use


# --- helpers ----------------------------------------------------------------

def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    total, acc = sum(weights), 0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0] if pairs else 0


def starts_new_line(t):
    """Bullets and interactive-prompt code lines keep their own line instead of being joined."""
    t = t.lstrip()
    return t.startswith(BULLETS) or t.startswith((">>>", "...")) or bool(re.match(r"^\(?\d+[.)]\s", t))


def join_lines(lines):
    """Join wrapped lines; keep bullets/code prompts on new lines; undo line-end hyphenation."""
    out = []
    for t in lines:
        t = t.strip()
        if not t:
            continue
        if not out or starts_new_line(t) or out[-1].startswith((">>>", "...")):
            out.append(t)
        elif out[-1].endswith("-") and t[:1].islower():
            out[-1] = out[-1][:-1] + t
        else:
            out[-1] += " " + t
    return "\n".join(out)


def looks_like_code(lines):
    lines = [l for l in lines if l.strip()]
    if not lines:
        return False
    hits = sum(bool(CODE_LINE_RE.match(l)) for l in lines)
    return hits >= max(1, (len(lines) + 1) // 2) and not lines[0].lstrip().startswith(BULLETS)


# --- text layer ---------------------------------------------------------------

def _is_mono(span):
    return bool(span["flags"] & 8) or any(h in span["font"].lower() for h in MONO_HINTS)


def rebuild_code(lines, size, char_ratio=0.6):
    """PDFs often drop leading spaces in code; recover indentation from each line's x position."""
    char_w = size * char_ratio
    min_x = min(x for x, _ in lines)
    out = []
    for x, t in lines:
        lead = len(t) - len(t.lstrip(" "))
        out.append(" " * (round((x - min_x) / char_w) + lead) + t.strip())
    return "\n".join(out)


def extract_page_blocks(page):
    """Text blocks (text, size, code/bold flags, position) + rectangles of images on the page."""
    blocks, images = [], []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") == 1:
            images.append(tuple(b["bbox"]))
            continue
        if b.get("type") != 0:
            continue
        lines, sizes, weights = [], [], []
        mono = bold = total = 0
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"])
            if not text.strip():
                continue
            for s in line["spans"]:
                n = len(s["text"].strip())
                total += n
                sizes.append(s["size"])
                weights.append(max(n, 1))
                if _is_mono(s):
                    mono += n
                if s["flags"] & 16:
                    bold += n
            lines.append((line["bbox"][0], text))
        if not lines or total == 0:
            continue
        size = weighted_median(sizes, weights)
        raw = [t for _, t in lines]
        mono_font = mono / total > 0.6
        is_code = mono_font or looks_like_code(raw)
        text = rebuild_code(lines, size, 0.6 if mono_font else 0.5) if is_code else join_lines(raw)
        blocks.append({"text": text, "size": round(size, 1), "is_code": is_code, "is_table": False,
                       "bold": bold / total > 0.6, "y0": round(b["bbox"][1], 1), "y1": round(b["bbox"][3], 1)})
    return blocks, images


# --- OCR ----------------------------------------------------------------------

def page_needs_ocr(blocks):
    return sum(len(b["text"].strip()) for b in blocks) < config.MIN_CHARS_FOR_TEXT_PAGE


def ocr_result_to_blocks(result, dpi, offset=(0, 0), col_gap=1.5):
    """RapidOCR [[box, text, score], ...] -> blocks in PDF points. Wide gaps on a row become ' | ' (table columns)."""
    scale, (ox, oy) = 72 / dpi, offset
    lines = []
    for box, text, score in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append({"x0": min(xs) * scale + ox, "y0": min(ys) * scale + oy,
                      "y1": max(ys) * scale + oy, "text": text.strip(), "score": float(score)})
    lines.sort(key=lambda l: (l["y0"] + l["y1"]) / 2)
    rows = []
    for l in lines:
        c, h = (l["y0"] + l["y1"]) / 2, l["y1"] - l["y0"]
        if rows and abs(c - rows[-1]["c"]) < 0.5 * h:
            rows[-1]["items"].append(l)
        else:
            rows.append({"c": c, "items": [l]})
    row_lines = []
    for r in rows:
        items = sorted(r["items"], key=lambda l: l["x0"])
        text, cols = items[0]["text"], 1
        for prev, cur in zip(items, items[1:]):
            h = cur["y1"] - cur["y0"]
            prev_x1 = prev["x0"] + len(prev["text"]) * h * 0.5
            if cur["x0"] - prev_x1 > col_gap * h:
                text, cols = text + " | " + cur["text"], cols + 1
            else:
                text += " " + cur["text"]
        row_lines.append({"x0": items[0]["x0"], "y0": min(i["y0"] for i in items), "y1": max(i["y1"] for i in items),
                          "text": text, "score": sum(i["score"] for i in items) / len(items), "cols": cols})
    groups = []
    for l in row_lines:
        h = l["y1"] - l["y0"]
        prev = groups[-1][-1] if groups else None
        if prev and (l["y0"] - prev["y1"]) < 1.3 * h and abs(h - (prev["y1"] - prev["y0"])) < 0.4 * h:
            groups[-1].append(l)
        else:
            groups.append([l])
    blocks = []
    for g in groups:
        texts = [l["text"] for l in g]
        heights = [l["y1"] - l["y0"] for l in g]
        table = sum(l["cols"] > 1 for l in g) >= max(2, len(g) // 2)
        is_code = not table and looks_like_code(texts)
        if is_code:
            char_w = statistics.median(heights) * 0.5
            min_x = min(l["x0"] for l in g)
            text = "\n".join(" " * round((l["x0"] - min_x) / char_w) + l["text"] for l in g)
        elif table:
            text = "\n".join(texts)
        else:
            text = join_lines(texts)
        blocks.append({"text": text, "size": round(statistics.median(heights) / 1.2, 1), "is_code": is_code,
                       "is_table": table, "bold": False, "y0": round(g[0]["y0"], 1), "y1": round(g[-1]["y1"], 1),
                       "ocr_conf": round(sum(l["score"] for l in g) / len(g), 3)})
    return blocks


def vision_ocr(png_bytes, model=None):
    """Local vision model -> markdown transcription. num_predict caps runaway generation."""
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=600, json={
        "model": model or config.VISION_MODEL, "stream": False, "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": config.VISION_NUM_PREDICT, "repeat_penalty": 1.15},
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [base64.b64encode(png_bytes).decode()]}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def _flush_para(blocks, para):
    if para:
        blocks.append({"text": join_lines(para), "is_code": False, "is_table": False})
        para.clear()


def _flush_table(blocks, table):
    rows = [r for r in table if not re.fullmatch(r"\|?[\s:|-]+\|?", r.strip())]
    if rows:
        blocks.append({"text": "\n".join(" | ".join(c.strip() for c in r.strip().strip("|").split("|")) for r in rows),
                       "is_code": False, "is_table": True})
    table.clear()


def markdown_to_blocks(md, y0=0.0, y1=0.0):
    """Vision output -> blocks. '# ' first line = title, ``` fences = code, '|' lines = table."""
    blocks, para, code, table, in_code = [], [], [], [], False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                if code:
                    blocks.append({"text": "\n".join(code).rstrip(), "is_code": True, "is_table": False})
                code.clear()
            else:
                _flush_para(blocks, para)
                _flush_table(blocks, table)
            in_code = not in_code
        elif in_code:
            code.append(line)
        elif line.strip().startswith("|"):
            _flush_para(blocks, para)
            table.append(line)
        elif line.startswith("# ") and not blocks and not para:
            _flush_table(blocks, table)
            blocks.append({"text": line[2:].strip(), "is_code": False, "is_table": False, "is_title": True})
        elif not line.strip():
            _flush_para(blocks, para)
            _flush_table(blocks, table)
        else:
            _flush_table(blocks, table)
            para.append(line.lstrip("#").strip() if line.startswith("#") else line)
    if in_code and code:
        blocks.append({"text": "\n".join(code).rstrip(), "is_code": True, "is_table": False})
    _flush_para(blocks, para)
    _flush_table(blocks, table)
    step = (y1 - y0) / max(len(blocks), 1)
    for i, b in enumerate(blocks):
        b.update({"size": 0.0, "bold": False, "y0": round(y0 + i * step, 1), "y1": round(y0 + (i + 1) * step, 1), "ocr": "vision"})
    return blocks


def needs_vision(blocks):
    if not blocks:
        return True
    conf = sum(b["ocr_conf"] for b in blocks) / len(blocks)
    return conf < config.VISION_CONF_THRESHOLD or any(b["is_code"] or b["is_table"] for b in blocks)


def _ocr_engine():
    if "engine" not in _OCR:
        from rapidocr_onnxruntime import RapidOCR
        _OCR["engine"] = RapidOCR()
    return _OCR["engine"]


def ocr_page(page, dpi=None, clip=None, use_vision=None):
    """OCR the whole page or the rectangle `clip`. RapidOCR first; weak/code/table results go to the vision model."""
    import pymupdf as fitz
    use_vision = config.USE_VISION_OCR if use_vision is None else use_vision
    rect = fitz.Rect(clip) if clip else page.rect
    png = page.get_pixmap(dpi=dpi or config.OCR_DPI, clip=rect).tobytes("png")
    result, _ = _ocr_engine()(png, use_cls=False)
    blocks = ocr_result_to_blocks(result, dpi or config.OCR_DPI, offset=(rect.x0, rect.y0))
    if use_vision and needs_vision(blocks):
        try:
            vblocks = markdown_to_blocks(vision_ocr(png), rect.y0, rect.y1)
            if vblocks:
                if clip:
                    for b in vblocks:
                        b.pop("is_title", None)
                return vblocks
        except requests.RequestException:
            pass                                    # keep the RapidOCR text
    return blocks


# --- cleaning -----------------------------------------------------------------

def _norm(text):
    return re.sub(r"\d+", "#", text.lower()).strip()


def strip_headers_footers(pages, min_share=0.3, margin=0.08, max_len=120):
    """Drop short blocks in the top/bottom 8% of the page that repeat across pages, and page numbers."""
    def in_margin(b, h):
        return b["y1"] <= h * margin or b["y0"] >= h * (1 - margin)

    n = len(pages)
    counts = Counter()
    for p in pages:
        counts.update({_norm(b["text"]) for b in p["blocks"] if in_margin(b, p["height"]) and len(b["text"]) <= max_len})
    repeated = {t for t, c in counts.items() if n >= 6 and c / n >= min_share}
    removed = []
    for p in pages:
        keep = []
        for b in p["blocks"]:
            t = _norm(b["text"])
            if in_margin(b, p["height"]) and (t in repeated or PAGE_NUM_RE.fullmatch(t)):
                removed.append(b["text"])
            else:
                keep.append(b)
        p["blocks"] = keep
    return pages, removed


def merge_wrapped_titles(blocks, idxs):
    if len(idxs) < 2:
        return blocks
    first = blocks[idxs[0]]
    first["text"] = " ".join(blocks[i]["text"].strip() for i in idxs)
    first["y1"] = blocks[idxs[-1]]["y1"]
    drop = set(idxs[1:])
    return [b for i, b in enumerate(blocks) if i not in drop]


def tag_headings(pages, ratio=1.15, max_len=150):
    """Slides: the title is the largest text near the top. Documents: short, not code, larger than body text."""
    body = [(b["size"], len(b["text"])) for p in pages for b in p["blocks"] if not b["is_code"] and b["size"] > 0]
    if not body:
        return pages, 0
    body_size = weighted_median([s for s, _ in body], [w for _, w in body])
    for p in pages:
        bl = p["blocks"]
        for b in bl:
            b["is_heading"] = False
        if not bl:
            continue
        if any(b.get("is_title") for b in bl):
            for b in bl:
                b["is_heading"] = bool(b.get("is_title"))
        elif p.get("is_slide"):
            top = [i for i, b in enumerate(bl) if b["y0"] < p["height"] * 0.3
                   and not b["is_code"] and not b.get("is_table") and b["size"] > 0]
            if not top:
                continue
            biggest = max(bl[i]["size"] for i in top)
            title_idx = [i for i in top if bl[i]["size"] >= biggest * 0.9 and len(bl[i]["text"]) <= max_len]
            run = [title_idx[0]] if title_idx else []
            for i in title_idx[1:]:
                if i == run[-1] + 1:
                    run.append(i)
            if run and biggest >= body_size * 1.05:
                p["blocks"] = bl = merge_wrapped_titles(bl, run)
                bl[run[0]]["is_heading"] = True
        else:
            for b in bl:
                t = b["text"].strip()
                short = 0 < len(t) <= max_len and "\n" not in t
                bigger = b["size"] >= body_size * ratio
                bold_line = b.get("bold", False) and len(t) <= 80
                b["is_heading"] = (not b["is_code"]) and short and (bigger or bold_line)
    return pages, body_size


# --- ingest / save / load -----------------------------------------------------

def ingest_pdf(path, role="textbook", use_ocr=True, use_vision=None, progress=None, name=None):
    """progress(done_pages, total_pages) is called after each page, for the app's progress bar."""
    import pymupdf as fitz
    path = Path(path)
    t0 = time.time()
    pages, ocr_pages, image_ocr_pages = [], [], []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            blocks, images = extract_page_blocks(page)
            if use_ocr and page_needs_ocr(blocks):
                blocks = ocr_page(page, use_vision=use_vision)
                if blocks:
                    ocr_pages.append(i + 1)
            elif use_ocr and images:
                area = page.rect.width * page.rect.height
                for r in images:
                    r = fitz.Rect(r) & page.rect           # pictures can extend past the page edge
                    if not r.is_empty and r.width * r.height >= config.OCR_IMAGE_MIN_AREA * area:
                        try:
                            extra = ocr_page(page, clip=r, use_vision=use_vision)
                        except fitz.mupdf.FzErrorBase:      # a picture MuPDF cannot render must not lose the document
                            continue
                        if sum(len(b["text"]) for b in extra) >= config.MIN_OCR_CHARS:
                            for b in extra:
                                b["from_image"] = True
                            blocks.extend(extra)
                            image_ocr_pages.append(i + 1)
                blocks.sort(key=lambda b: b["y0"])
            pages.append({"page_no": i + 1, "label": page.get_label(), "height": page.rect.height,
                          "is_slide": page.rect.width > page.rect.height, "blocks": blocks})
            if progress:
                progress(i + 1, len(doc))
    pages, removed = strip_headers_footers(pages)
    pages, body_size = tag_headings(pages)
    h = file_hash(path)
    return {"source_id": h[:12], "name": name or path.name, "role": role, "hash": h, "n_pages": len(pages),
            "ocr_pages": ocr_pages,
            "vision_pages": sorted({p["page_no"] for p in pages for b in p["blocks"] if b.get("ocr") == "vision"}),
            "image_ocr_pages": sorted(set(image_ocr_pages)), "body_font_size": body_size,
            "removed_headers": sorted(set(removed))[:30], "seconds": round(time.time() - t0, 1), "pages": pages}


SUPPORTED_TYPES = [".pdf", ".pptx", ".docx", ".ipynb"]


def ingest_file(path, role="textbook", progress=None):
    """Pick the reader by file type. All readers return the same document shape; "unit" is how a page
    is cited (p. / slide / cell)."""
    from . import readers
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        doc = dict(ingest_pdf(path, role=role, progress=progress), unit="p.")
    elif suffix == ".pptx":
        doc = readers.ingest_pptx(path, role=role, progress=progress)
    elif suffix == ".docx":
        doc = readers.ingest_docx(path, role=role, progress=progress)
    elif suffix == ".ipynb":
        doc = readers.ingest_ipynb(path, role=role, progress=progress)
    else:
        raise ValueError(f"unsupported file type: {suffix} (supported: {', '.join(SUPPORTED_TYPES)})")
    doc["kind"] = suffix.lstrip(".")
    return doc


def ocr_image_bytes(data, use_vision=None):
    """OCR an embedded picture (PPTX/DOCX): RapidOCR first, the vision model for weak/code/table results.
    Formats PyMuPDF can't decode (EMF/WMF) are skipped."""
    import pymupdf as fitz
    use_vision = config.USE_VISION_OCR if use_vision is None else use_vision
    try:
        pix = fitz.Pixmap(data)
        if pix.alpha or pix.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png = pix.tobytes("png")
    except Exception:
        return []
    result, _ = _ocr_engine()(png, use_cls=False)
    blocks = ocr_result_to_blocks(result, 72)
    if use_vision and needs_vision(blocks):
        try:
            vblocks = markdown_to_blocks(vision_ocr(png))
            if vblocks:
                blocks = vblocks
        except requests.RequestException:
            pass
    for b in blocks:
        b.pop("is_title", None)
        b["from_image"] = True
    return blocks if sum(len(b["text"]) for b in blocks) >= config.MIN_OCR_CHARS else []


def ingest_text(title, text, role="note"):
    """Pasted text -> same shape as a PDF with one page; paragraphs become blocks."""
    blocks = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        heading = para.startswith("#") and "\n" not in para
        is_code = looks_like_code(para.splitlines())
        blocks.append({"text": para.lstrip("# ").strip() if heading else para, "size": 10.0, "is_code": is_code,
                       "is_table": False, "bold": False, "is_heading": heading, "y0": 0, "y1": 0})
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return {"source_id": h[:12], "name": title, "role": role, "hash": h, "n_pages": 1, "ocr_pages": [],
            "image_ocr_pages": [], "vision_pages": [], "body_font_size": 10.0, "removed_headers": [], "seconds": 0,
            "pages": [{"page_no": 1, "label": "", "height": 0, "is_slide": False, "blocks": blocks}]}


