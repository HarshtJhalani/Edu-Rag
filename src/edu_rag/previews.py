"""Page images for citations: the cited slide/page rendered from the original file, with the cited text
highlighted where it can be found. PPTX/DOCX are rendered through a PDF copy that PowerPoint/Word export
(Windows COM, run in a hidden PowerShell); without Office the app falls back to the extracted text."""
import hashlib
import subprocess
from pathlib import Path

from . import config, library

PREVIEW_PDF = "preview.pdf"
PREVIEW_DIR = "previews"
DPI = 110
HIGHLIGHT = (1.0, 0.85, 0.2)
EXPORT_PS = {
    ".pptx": "$a = New-Object -ComObject PowerPoint.Application; "
             "$p = $a.Presentations.Open('{src}', $true, $false, $false); $p.SaveAs('{dst}', 32); $p.Close(); $a.Quit()",
    ".docx": "$a = New-Object -ComObject Word.Application; $a.Visible = $false; "
             "$d = $a.Documents.Open('{src}', $false, $true); $d.ExportAsFixedFormat('{dst}', 17); $d.Close($false); $a.Quit()",
}


def export_office_pdf(src, timeout=180):
    """PowerPoint/Word -> preview.pdf next to the file. Returns the PDF path or None (no Office, or it failed)."""
    src = Path(src)
    dst = src.parent / PREVIEW_PDF
    if dst.exists():
        return dst
    script = EXPORT_PS.get(src.suffix.lower())
    if not script:
        return None
    esc = lambda p: str(p).replace("'", "''")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        script.format(src=esc(src.resolve()), dst=esc(dst.resolve()))],
                       capture_output=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return dst if dst.exists() else None


def renderable_pdf(source_id):
    """The PDF a source's pages are drawn from: the original PDF, or the Office export."""
    f = library.original_file(source_id)
    if f is None:
        return None
    if f.suffix.lower() == ".pdf":
        return f
    pdf = f.parent / PREVIEW_PDF
    return pdf if pdf.exists() else None


def highlight_lines(chunk_text, piece=40, max_pieces=40):
    """Pieces of the cited chunk to search for on the page. A chunk line can wrap over several printed lines,
    so lines are cut at word boundaries into ~40-character pieces; the title line and very short lines are skipped."""
    body = chunk_text.split("\n\n", 1)[-1]
    out = []
    for line in (l.strip(" •–-`") for l in body.splitlines()):
        if len(line) < 12 or line.startswith("```"):
            continue
        cur = ""
        for word in line.split():
            if cur and len(cur) + 1 + len(word) > piece:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if len(cur) >= 8:
            out.append(cur)
    return out[:max_pieces]


def page_image(source_id, page_no, chunk_text=None, dpi=DPI):
    """PNG path of the page, highlighted when chunk_text is given; cached in the source's previews/ folder."""
    import pymupdf as fitz
    pdf = renderable_pdf(source_id)
    if pdf is None:
        return None
    key = hashlib.sha1((chunk_text or "").encode("utf-8")).hexdigest()[:8] if chunk_text else "plain"
    out = pdf.parent / PREVIEW_DIR / f"page{page_no:04d}-{key}.png"
    if out.exists():
        return out
    with fitz.open(pdf) as doc:
        if not 1 <= page_no <= len(doc):
            return None
        page = doc[page_no - 1]
        rects = [r for line in highlight_lines(chunk_text or "") for r in page.search_for(line)]
        pix = page.get_pixmap(dpi=dpi)
        if rects:
            shape = page.new_shape()
            for r in rects:
                shape.draw_rect(r)
            shape.finish(color=None, fill=HIGHLIGHT, fill_opacity=0.35)
            shape.commit()
            pix = page.get_pixmap(dpi=dpi)
    out.parent.mkdir(exist_ok=True)
    pix.save(out)
    return out


def prepare(source_id):
    """Called after an upload: make the Office PDF copy now, so the first preview is instant."""
    f = library.original_file(source_id)
    if f is not None and f.suffix.lower() in EXPORT_PS:
        return export_office_pdf(f)
    return None
