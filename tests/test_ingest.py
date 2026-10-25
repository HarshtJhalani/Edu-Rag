from edu_rag import ingest


def test_join_lines_keeps_bullets_and_prompts():
    assert ingest.join_lines(["• first point that", "wraps here", ">>> x = 1", ">>> y = 2"]) == \
        "• first point that wraps here\n>>> x = 1\n>>> y = 2"


def test_join_lines_undoes_hyphenation():
    assert ingest.join_lines(["classifi-", "cation model"]) == "classification model"


def test_looks_like_code():
    assert ingest.looks_like_code(["def f(x):", "    return x"])
    assert not ingest.looks_like_code(["• A list is mutable", "• A tuple is not"])


def test_vision_markdown_to_blocks():
    md = "# Python statements\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nfor i in x:\n    print(i)\n```\nSome text."
    blocks = ingest.markdown_to_blocks(md, 0, 100)
    assert blocks[0].get("is_title") and blocks[0]["text"] == "Python statements"
    assert any(b["is_table"] and b["text"] == "a | b\n1 | 2" for b in blocks)
    assert any(b["is_code"] and b["text"] == "for i in x:\n    print(i)" for b in blocks)


def test_ingest_text_shape():
    d = ingest.ingest_text("Decorators", "# Decorators\n\nA decorator extends a function.\n\ndef deco(f):\n    return f")
    b = d["pages"][0]["blocks"]
    assert [x["is_heading"] for x in b] == [True, False, False] and b[2]["is_code"]
    assert d["role"] == "note" and len(d["source_id"]) == 12


def test_header_footer_removal():
    pages = [dict(height=540, blocks=[dict(text="ACME University", y0=5, y1=15), dict(text=f"Body {i}", y0=200, y1=220),
                                      dict(text=str(i + 1), y0=520, y1=535)]) for i in range(8)]
    pages, removed = ingest.strip_headers_footers(pages)
    assert all([b["text"] for b in p["blocks"]] == [f"Body {i}"] for i, p in enumerate(pages))
