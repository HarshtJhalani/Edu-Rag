"""Flashcards (Leitner boxes), exam question parsing and chat export. No Ollama needed."""
from edu_rag import config, exam, export, flashcards


def card(id_, box=0, due="2026-10-26", source_id="a"):
    return {"id": id_, "front": f"F{id_}", "back": "B", "box": box, "due": due, "reviews": 0, "lapses": 0, "source_id": source_id}


def test_flashcard_validate():
    assert flashcards.validate({"front": "Gini index", "back": "An impurity measure.", "source": "S2"}, 3)["source_n"] == 2
    assert flashcards.validate({"front": "x", "back": "y", "source": "S4"}, 3) is None          # label out of range
    assert flashcards.validate({"front": "Same", "back": "same", "source": "S1"}, 3) is None    # back repeats the front
    assert flashcards.validate({"front": "Term", "back": "", "source": "S1"}, 3) is None


def test_leitner_review_moves_boxes():
    c = card("1", box=2)
    up = flashcards.review(c, True, day="2026-10-26")
    assert up["box"] == 3 and up["due"] == "2026-10-30" and up["reviews"] == 1           # box 3 = 4 days
    down = flashcards.review(up, False, day="2026-10-26")
    assert down["box"] == 0 and down["due"] == "2026-10-26" and down["lapses"] == 1
    top = flashcards.review(card("2", box=len(config.LEITNER_DAYS) - 1), True, day="2026-10-26")
    assert top["box"] == len(config.LEITNER_DAYS) - 1                                    # never past the last box


def test_due_order_and_filter():
    deck = [card("a", box=2, due="2026-10-20"), card("b", box=0, due="2026-10-26"), card("c", box=0, due="2026-11-01"),
            card("d", box=1, due="2026-10-25", source_id="z")]
    assert [c["id"] for c in flashcards.due(deck, day="2026-10-26")] == ["b", "d", "a"]
    assert [c["id"] for c in flashcards.due(deck, day="2026-10-26", source_ids=["a"])] == ["b", "a"]


def test_deck_add_record_remove(tmp_path):
    path = tmp_path / "cards.json"
    assert flashcards.add([card("1"), card("2")], path=path, day="2026-10-26") == 2
    assert flashcards.add([card("2"), card("3")], path=path, day="2026-10-26") == 1     # duplicates skipped
    flashcards.record("1", True, path=path, day="2026-10-26")
    deck = {c["id"]: c for c in flashcards.load(path)}
    assert deck["1"]["box"] == 1 and deck["2"]["box"] == 0
    flashcards.remove(["2"], path=path)
    assert sorted(c["id"] for c in flashcards.load(path)) == ["1", "3"]
    assert flashcards.stats(flashcards.load(path), day="2026-10-26")["due"] == 1


def test_parse_marks():
    assert exam.parse_marks("Explain DBSCAN. (5 marks)") == 5
    assert exam.parse_marks("Differentiate between bagging and boosting. [10]") == 10
    assert exam.parse_marks("What is entropy? 2M") == 2
    assert exam.parse_marks("Define support.") == config.EXAM_DEFAULT_MARKS
    assert exam.parse_marks("Q3. Explain DBSCAN with an example") == config.EXAM_DEFAULT_MARKS


def test_question_text_strips_title_number_marks():
    c = {"text": "May 2024 paper\n\nQ3. Explain the k-means algorithm with an example. (5 marks)"}
    assert exam.question_text(c) == "Explain the k-means algorithm with an example."
    assert exam.question_text({"text": "Paper\n\n2 a) What is a decision tree? [5]"}) == "What is a decision tree?"


def test_clean_generated_drops_passage_mentions():
    assert exam.clean_generated("Explain characterization, with an example from the given passages.") == \
        "Explain characterization, with an example."
    assert exam.clean_generated("What are the functionalities mentioned? Explain one.") == "What are the functionalities? Explain one."
    assert exam.clean_generated("Explain the role of text in data mining.") == "Explain the role of text in data mining."


def test_chat_export():
    msgs = [{"role": "user", "content": "What is a tuple?"},
            {"role": "assistant", "content": "An immutable sequence [S1].",
             "result": {"answer": "An immutable sequence [S1].",
                        "sources": [{"label": "S1", "source": "Python.pdf", "pages": "p. 4", "title": "Tuples"}]},
             "deeper": {"answer": "Tuples are immutable [W1].", "sources": [{"label": "W1", "url": "https://docs.python.org"}]}}]
    md = export.chat_to_markdown(msgs)
    assert "## Q: What is a tuple?" in md and "[S1] Python.pdf, p. 4 (Tuples)" in md and "https://docs.python.org" in md
    assert export.chat_to_pdf(msgs)[:4] == b"%PDF"
    assert export.chat_title(msgs) == "What is a tuple?"
    assert export.md_inline_to_html("**a** <b>`x`") == "<b>a</b> &lt;b&gt;<code>x</code>"
