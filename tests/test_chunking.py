from edu_rag import chunking, config


def block(text, size=24.0, code=False, table=False, heading=False, **kw):
    return dict(text=text, size=size, is_code=code, is_table=table, is_heading=heading, bold=False, y0=0, y1=0, **kw)


def slide_doc(pages, name="Deck.pdf"):
    return dict(source_id="abc", name=name, role="textbook", hash="h" * 40, body_font_size=24.0,
                pages=[dict(page_no=i + 1, label="", height=540, is_slide=True, blocks=b) for i, b in enumerate(pages)])


def test_settings_hash_matches_notebook_02():
    # the embedding cache in data/embed_cache is keyed by this value
    assert config.chunk_settings_hash() == "eebcd81a8021"


def test_settings_hash_changes_with_settings():
    s = config.chunk_settings()
    assert config.chunk_settings_hash(dict(s, max_tokens=300)) != config.chunk_settings_hash(s)


def test_one_slide_one_chunk_with_title_prefix():
    doc = slide_doc([[block("Lists", 44, heading=True), block("A list is a mutable sequence of items.")]])
    chunks, _ = chunking.chunk_doc(doc)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Lists\n\n")
    assert chunks[0]["title"] == "Lists" and chunks[0]["page_start"] == chunks[0]["page_end"] == 1


def test_small_slides_with_same_title_merge():
    doc = slide_doc([[block("Tuples", 44, heading=True), block("Tuples are immutable sequences.")],
                     [block("Tuples (contd.)", 44, heading=True), block("They can be used as dictionary keys.")]])
    chunks, _ = chunking.chunk_doc(doc)
    assert len(chunks) == 1 and (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 2)


def test_code_block_never_split_across_chunks():
    code = "\n".join(f"x{i} = {i} * 2  # line" for i in range(30))          # ~200 tokens
    prose = [block("Some explanation sentence about the code here. " * 6) for _ in range(4)]
    doc = slide_doc([[block("Code", 44, heading=True), *prose, block(code, code=True)]])
    chunks, _ = chunking.chunk_doc(doc)
    assert sum(f"```\n{code}\n```" in c["text"] for c in chunks) == 1
    assert all(c["n_tokens"] <= config.MAX_TOKENS for c in chunks)


def test_oversized_code_splits_only_at_line_boundaries():
    lines = [f"value_{i} = compute_something({i}, option=True)" for i in range(80)]  # far over the limit
    doc = slide_doc([[block("Big", 44, heading=True), block("\n".join(lines), code=True)]])
    chunks, _ = chunking.chunk_doc(doc)
    assert len(chunks) > 1
    rebuilt = [l for c in chunks for l in c["text"].split("\n") if l.startswith("value_")]
    assert rebuilt == lines


def test_oversized_table_repeats_header():
    rows = ["id | name | score"] + [f"{i} | student{i} | {i * 3}" for i in range(150)]
    doc = slide_doc([[block("Table", 44, heading=True), block("\n".join(rows), table=True)]])
    chunks, _ = chunking.chunk_doc(doc)
    assert len(chunks) > 1 and all("id | name | score" in c["text"] for c in chunks)


def test_title_only_slide_dropped():
    doc = slide_doc([[block("Ch1. Machine Learning", 44, heading=True), block("Ch 1 Machine Learning", 30)]],
                    name="Ch 1 Machine Learning.pdf")
    chunks, dropped = chunking.chunk_doc(doc)
    assert chunks == [] and dropped == [(1, "Ch1. Machine Learning")]


def test_runaway_vision_output_quarantined():
    junk = [block("Structure: organ " * 30, 0, ocr="vision", from_image=True) for _ in range(30)]
    doc = slide_doc([[block("Limitations of K Means", 44, heading=True), *junk, block("Hard clusters cannot overlap.")]])
    log = []
    chunks, _ = chunking.chunk_doc(doc, log)
    assert log and log[0][0] == "runaway vision"
    assert "organ" not in chunks[0]["text"] and "Hard clusters" in chunks[0]["text"]


def test_repeated_blocks_removed():
    code = "import pandas as pd\ndf = pd.read_csv('x.csv')"
    doc = slide_doc([[block("Elbow", 44, heading=True)] + [block(code, code=True, ocr="vision") for _ in range(5)]])
    log = []
    chunks, _ = chunking.chunk_doc(doc, log)
    assert chunks[0]["text"].count("read_csv") == 1 and len(log) == 4


def test_past_paper_one_chunk_per_question_with_its_page():
    doc = dict(source_id="pyq1", name="Exam 2024.pdf", role="pyq", hash="p" * 40, body_font_size=12.0,
               pages=[dict(page_no=1, label="", height=842, is_slide=False,
                           blocks=[block("UNIVERSITY EXAM 2024", 12), block("Q1. Explain DBSCAN. (5 marks)", 12),
                                   block("Use an example.", 12), block("Q2. List vs tuple? (5 marks)", 12)]),
                      dict(page_no=2, label="", height=842, is_slide=False, blocks=[block("Q3. What is the Gini index?", 12)])])
    chunks, _ = chunking.chunk_doc(doc)
    assert [(c["page_start"], c["text"].split("\n\n", 1)[1]) for c in chunks] == [
        (1, "Q1. Explain DBSCAN. (5 marks)\nUse an example."), (1, "Q2. List vs tuple? (5 marks)"), (2, "Q3. What is the Gini index?")]


def test_pdf_bullet_glyphs_cleaned():
    assert chunking.clean_text(" Item one\n\x00 Item two") == "• Item one\n• Item two"
