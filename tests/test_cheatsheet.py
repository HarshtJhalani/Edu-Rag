from edu_rag import cheatsheet


def chunk(cid, title, page, text="x " * 40, tokens=100):
    return dict(chunk_id=cid, title=title, page_start=page, page_end=page, source_name="Deck.pdf", text=text,
                n_tokens=tokens, unit="slide")


def test_batches_keep_order_and_budget():
    cs = [chunk(f"c{i}", "T", i, tokens=1200) for i in range(5)]
    assert [[c["chunk_id"] for c in b] for b in cheatsheet.batches(cs, budget=3000)] == [["c0", "c1"], ["c2", "c3"], ["c4"]]


def test_bullets_get_pages_from_metadata_and_uncited_are_dropped():
    ps = [chunk("a", "Lists", 3, "a list is mutable and ordered"), chunk("b", "Tuples", 5, "a tuple is immutable")]
    text = "- A list is mutable and ordered [S1].\n- A tuple is immutable [S2], .\n- Something with no source.\nNot a bullet [S1]"
    bullets = cheatsheet.bullets_from(text, ps)
    assert [b["text"] for b in bullets] == ["A list is mutable and ordered.", "A tuple is immutable."]
    assert [b["refs"] for b in bullets] == [[("Deck.pdf", "slide 3")], [("Deck.pdf", "slide 5")]]


def test_sections_follow_cited_slide_titles_in_order():
    ps = [chunk("a", "Lists", 3, "a list is mutable"), chunk("b", "Tuples", 5, "a tuple is immutable"),
          chunk("c", "Lists", 4, "lists support append")]
    bullets = cheatsheet.bullets_from("- Tuples are immutable [S2]\n- Lists are mutable [S1]\n- Lists support append [S3]", ps)
    sections = cheatsheet.group_sections(bullets)
    assert [(s["title"], s["pages"], len(s["bullets"])) for s in sections] == [("Lists", "slide 3-4", 2), ("Tuples", "slide 5", 1)]
    md = cheatsheet.to_markdown({"title": "Cheat sheet: Deck", "sections": sections})
    assert "## Lists  (Deck.pdf, slide 3-4)" in md and "- Lists are mutable *(slide 3)*" in md
    assert cheatsheet.to_pdf({"title": "T", "sections": sections})[:4] == b"%PDF"
