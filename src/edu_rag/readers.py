"""Readers for PowerPoint, Word and Jupyter files. Each returns the same document shape as ingest.ingest_pdf,
so chunking, indexing and citations treat them like any other source."""
import hashlib
import json
import re
import time
from pathlib import Path

from . import config, ingest

EMU_PER_PT = 12700
MONO_FONTS = ("consolas", "courier", "mono", "menlo", "lucida console", "source code")


def _doc(path, role, unit, pages, body_size, t0, **extra):
    h = ingest.file_hash(path)
    return dict({"source_id": h[:12], "name": Path(path).name, "role": role, "hash": h, "n_pages": len(pages),
                 "ocr_pages": [], "image_ocr_pages": sorted({p["page_no"] for p in pages if any(b.get("from_image") for b in p["blocks"])}),
                 "vision_pages": sorted({p["page_no"] for p in pages if any(b.get("ocr") == "vision" for b in p["blocks"])}),
                 "body_font_size": body_size, "removed_headers": [], "seconds": round(time.time() - t0, 1), "unit": unit,
                 "pages": pages}, **extra)


def _block(text, size=0.0, code=False, table=False, heading=False, y0=0.0, bold=False):
    return {"text": text, "size": round(size, 1), "is_code": code, "is_table": table, "is_heading": heading,
            "bold": bold, "y0": round(y0, 1), "y1": round(y0, 1)}


def clean(text):
    """Office files use non-breaking and zero-width spaces that break word matching."""
    return text.replace("\xa0", " ").replace("​", "").replace("\u000b", "\n")


def _body_size(pages):
    sizes = [(b["size"], len(b["text"])) for p in pages for b in p["blocks"] if b["size"] and not b["is_heading"]]
    return ingest.weighted_median([s for s, _ in sizes], [w for _, w in sizes]) if sizes else 0.0


# --- PowerPoint ---------------------------------------------------------------

def _pptx_paragraphs(text_frame):
    """Lines of a text frame; nested bullet levels are indented, sizes and fonts collected."""
    lines, sizes, mono, total = [], [], 0, 0
    for p in text_frame.paragraphs:
        t = clean("".join(r.text for r in p.runs)).strip()
        if not t:
            continue
        lines.append(("  " * p.level) + t)
        for r in p.runs:
            n = len(r.text.strip())
            total += n
            if r.font.size:
                sizes.append(r.font.size.pt)
            if r.font.name and any(m in r.font.name.lower() for m in MONO_FONTS):
                mono += n
    return lines, (sum(sizes) / len(sizes) if sizes else 0.0), total and mono / total > 0.6


def _pptx_table(table):
    return "\n".join(" | ".join(clean(c.text).strip().replace("\n", " ") for c in row.cells) for row in table.rows)


def _pptx_shapes(shapes):
    """Flatten group shapes so their text is not lost."""
    for s in shapes:
        if s.shape_type == 6:                       # MSO_SHAPE_TYPE.GROUP
            yield from _pptx_shapes(s.shapes)
        else:
            yield s


def ingest_pptx(path, role="textbook", progress=None, use_ocr=True):
    from pptx import Presentation
    t0 = time.time()
    prs = Presentation(path)
    slide_area = prs.slide_width * prs.slide_height
    height_pt = prs.slide_height / EMU_PER_PT
    pages = []
    for i, slide in enumerate(prs.slides):
        blocks = []
        title_shape = slide.shapes.title
        for s in sorted(_pptx_shapes(slide.shapes), key=lambda s: (s.top or 0, s.left or 0)):
            y0 = (s.top or 0) / EMU_PER_PT
            if s.has_text_frame:
                lines, size, mono = _pptx_paragraphs(s.text_frame)
                if not lines:
                    continue
                is_title = title_shape is not None and s.shape_id == title_shape.shape_id
                code = not is_title and (mono or ingest.looks_like_code(lines))
                text = "\n".join(lines) if code else ingest.join_lines(lines)
                blocks.append(_block(" ".join(lines) if is_title else text, size or (32.0 if is_title else 18.0),
                                     code=code, heading=is_title, y0=y0))
            elif getattr(s, "has_table", False) and s.has_table:
                blocks.append(_block(_pptx_table(s.table), 14.0, table=True, y0=y0))
            elif use_ocr and s.shape_type == 13 and (s.width or 0) * (s.height or 0) >= config.OCR_IMAGE_MIN_AREA * slide_area:
                for b in ingest.ocr_image_bytes(s.image.blob):     # 13 = PICTURE: screenshots of code, tables
                    b["y0"] = b["y1"] = round(y0, 1)
                    b.setdefault("is_heading", False)
                    blocks.append(b)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                blocks.append(_block("Speaker notes: " + notes, 12.0, y0=height_pt))
        pages.append({"page_no": i + 1, "label": str(i + 1), "height": height_pt, "is_slide": True, "blocks": blocks})
        if progress:
            progress(i + 1, len(prs.slides))
    return _doc(path, role, "slide", pages, _body_size(pages), t0)


# --- Word -----------------------------------------------------------------------

def _docx_items(document):
    """Paragraphs and tables in document order, with the page each starts on (from Word's saved page breaks)."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    page = 1
    for el in document.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            xml = el.xml
            breaks = xml.count("lastRenderedPageBreak") + len(re.findall(r'w:br [^>]*w:type="page"', xml))
            yield page, Paragraph(el, document)
            page += breaks
        elif tag == "tbl":
            yield page, Table(el, document)


def _docx_images(paragraph):
    """Bytes of pictures inside a paragraph."""
    out = []
    for blip in paragraph._p.xpath(".//a:blip"):
        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        part = paragraph.part.related_parts.get(rid)
        if part is not None and hasattr(part, "blob") and len(part.blob) > 5000:      # skip icons and bullets
            out.append(part.blob)
    return out


def ingest_docx(path, role="textbook", progress=None, use_ocr=True):
    import docx
    t0 = time.time()
    document = docx.Document(path)
    by_page, code_buf = {}, []

    def add(page, block):
        by_page.setdefault(page, []).append(block)

    def flush_code(page):
        if code_buf:
            add(page, _block("\n".join(code_buf), 10.0, code=True))
            code_buf.clear()

    items = list(_docx_items(document))
    for n, (page, item) in enumerate(items):
        if item.__class__.__name__ == "Table":
            flush_code(page)
            add(page, _block("\n".join(" | ".join(c.text.strip().replace("\n", " ") for c in row.cells) for row in item.rows),
                             11.0, table=True))
            continue
        text = clean(item.text).rstrip()
        style = (item.style.name or "").lower() if item.style is not None else ""
        fonts = [r.font.name.lower() for r in item.runs if r.font.name]
        mono = bool(fonts) and all(any(m in f for m in MONO_FONTS) for f in fonts)
        if text.strip() and (mono or "code" in style or (ingest.looks_like_code([text]) and not style.startswith("heading"))):
            code_buf.append(text)
        else:
            flush_code(page)
            if text.strip():
                heading = style.startswith("heading") or style == "title"
                bullet = "list" in style
                size = 16.0 if heading else 11.0
                add(page, _block(("• " if bullet else "") + text.strip(), size, heading=heading,
                                 bold=any(r.bold for r in item.runs)))
        if use_ocr:
            for data in _docx_images(item):
                for b in ingest.ocr_image_bytes(data):
                    b.setdefault("is_heading", False)
                    add(page, b)
        if progress and n % 20 == 0:
            progress(n + 1, len(items))
    flush_code(items[-1][0] if items else 1)
    n_pages = max(by_page, default=1)
    pages = [{"page_no": p, "label": str(p), "height": 0, "is_slide": False, "blocks": by_page.get(p, [])}
             for p in range(1, n_pages + 1)]
    if progress:
        progress(len(items), len(items))
    return _doc(path, role, "p.", pages, 11.0, t0)


# --- Jupyter ----------------------------------------------------------------------

MAX_OUTPUT_CHARS = 600


def _markdown_blocks(src):
    blocks, para = [], []
    for line in src.splitlines() + [""]:
        m = re.match(r"^\s*#{1,6}\s+(.*)", line)
        if m or not line.strip():
            if para:
                blocks.append(_block(ingest.join_lines(para), 11.0))
                para = []
            if m:
                blocks.append(_block(m.group(1).strip(), 16.0, heading=True))
        else:
            para.append(line)
    return blocks


def _output_text(outputs):
    parts = []
    for o in outputs:
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", "")))
        elif "data" in o and "text/plain" in o["data"]:
            parts.append("".join(o["data"]["text/plain"]))
    text = "\n".join(p.strip() for p in parts if p.strip())
    return text[:MAX_OUTPUT_CHARS] + (" ..." if len(text) > MAX_OUTPUT_CHARS else "")


def ingest_ipynb(path, role="textbook", progress=None):
    import nbformat
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    pages = []
    for i, cell in enumerate(nb.cells):
        src = cell.source.strip()
        if not src:
            continue
        if cell.cell_type == "markdown":
            blocks = _markdown_blocks(src)
        elif cell.cell_type == "code":
            blocks = [_block(src, 10.0, code=True)]
            out = _output_text(cell.get("outputs", []))
            if out:
                blocks.append(_block("Output:\n" + out, 10.0, code=True))
        else:
            continue
        pages.append({"page_no": i + 1, "label": str(i + 1), "height": 0, "is_slide": False, "blocks": blocks})
        if progress:
            progress(i + 1, len(nb.cells))
    return _doc(path, role, "cell", pages, 11.0, t0)
