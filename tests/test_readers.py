"""Readers on tiny files built here, so the tests need no fixtures and no Ollama (OCR is off)."""
import nbformat
import pytest

from edu_rag import chunking, generate, readers

pptx = pytest.importorskip("pptx")
docx = pytest.importorskip("docx")


def make_pptx(path):
    from pptx.util import Inches
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])                   # title + content
    s.shapes.title.text = "Lists"
    s.placeholders[1].text_frame.text = "A list is mutable and ordered"
    s.notes_slide.notes_text_frame.text = "Mention append()"
    s2 = prs.slides.add_slide(prs.slide_layouts[5])                  # title only
    s2.shapes.title.text = "Scores"
    tbl = s2.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    for (r, c), v in {(0, 0): "name", (0, 1): "score", (1, 0): "ann", (1, 1): "90"}.items():
        tbl.cell(r, c).text = v
    prs.save(path)


def test_pptx_slides_titles_tables_notes(tmp_path):
    p = tmp_path / "deck.pptx"
    make_pptx(p)
    doc = readers.ingest_pptx(p, use_ocr=False)
    assert doc["unit"] == "slide" and doc["n_pages"] == 2
    b1, b2 = doc["pages"][0]["blocks"], doc["pages"][1]["blocks"]
    assert b1[0]["is_heading"] and b1[0]["text"] == "Lists"
    assert "mutable and ordered" in b1[1]["text"]                    # non-breaking space normalised
    assert any(b["text"] == "Speaker notes: Mention append()" for b in b1)
    assert any(b["is_table"] and b["text"] == "name | score\nann | 90" for b in b2)
    chunks, _ = chunking.chunk_doc(doc)
    assert [c["title"] for c in chunks] == ["Lists", "Scores"]


def test_docx_headings_code_and_tables(tmp_path):
    d = docx.Document()
    d.add_heading("Functions", level=1)
    d.add_paragraph("A function groups code.")
    d.add_paragraph("def add(a, b):")
    d.add_paragraph("    return a + b")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "x", "y"
    p = tmp_path / "notes.docx"
    d.save(p)
    doc = readers.ingest_docx(p, use_ocr=False)
    blocks = doc["pages"][0]["blocks"]
    assert blocks[0]["is_heading"] and blocks[0]["text"] == "Functions"
    assert any(b["is_code"] and b["text"] == "def add(a, b):\n    return a + b" for b in blocks)
    assert any(b["is_table"] and b["text"] == "x | y" for b in blocks)


def test_ipynb_cells_and_outputs(tmp_path):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Plots\nHistograms show spread."),
                                         nbformat.v4.new_code_cell("print(1 + 1)", outputs=[
                                             nbformat.v4.new_output("stream", name="stdout", text="2\n")])])
    p = tmp_path / "nb.ipynb"
    nbformat.write(nb, p)
    doc = readers.ingest_ipynb(p)
    assert doc["unit"] == "cell" and [pg["page_no"] for pg in doc["pages"]] == [1, 2]
    b = doc["pages"][1]["blocks"]
    assert b[0]["is_code"] and b[0]["text"] == "print(1 + 1)" and "Output:\n2" in b[1]["text"]


def test_citation_unit_labels():
    assert generate.page_label(dict(page_start=4, page_end=4, unit="slide")) == "slide 4"
    assert generate.page_label(dict(page_start=3, page_end=5, unit="cell")) == "cell 3-5"
    assert generate.page_label(dict(page_start=2, page_end=2, unit="p.")) == "p. 2"
    assert generate.page_label(dict(page_start=2, page_end=2)) == "p. 2"
