from edu_rag import generate


def passage(title, text, n=1):
    return dict(title=title, text=text, source_name="Deck.pdf", page_start=n, page_end=n, chunk_id=f"c{n}")


def test_citation_forms_normalised():
    text, valid, invalid = generate.parse_citations("Tuples are immutable [S1]. Lists are not [S2, S3].", 5)
    assert text == "Tuples are immutable [S1]. Lists are not [S2][S3]."
    assert valid == [1, 2, 3] and invalid == []


def test_citation_range_expanded():
    assert generate.parse_citations("Range [S1-S3].", 5)[1] == [1, 2, 3]


def test_out_of_range_citation_removed():
    text, valid, invalid = generate.parse_citations("Bad [S9] and good [S2].", 5)
    assert "[S9]" not in text and valid == [2] and invalid == [9]


def test_title_used_as_citation_mapped_to_label():
    ps = [passage("Tuple", "a"), passage("Tuples are more efficient", "b", 2)]
    assert generate.title_citations_to_labels("Fixed length [Tuples are more efficient].", ps) == "Fixed length [S2]."


def test_citation_after_full_stop_belongs_to_previous_sentence():
    sents = generate.split_sentences("The elbow method plots SSE against k values. [S1][S2] It looks for the bend in the curve. [S3]")
    assert sents == ["The elbow method plots SSE against k values. [S1][S2]", "It looks for the bend in the curve. [S3]"]


def test_code_ignored_by_sentence_split():
    assert generate.split_sentences("Use this code [S1].\n```\nprint('a. b. c. d')\n```") == []


def test_support_check_flags():
    ps = [passage("Tuple", "A tuple is an ordered immutable collection of values")]
    rep = generate.support_check("A tuple is an ordered immutable collection [S1]. Bananas grow quickly in tropical weather [S1]. "
                                 "Something entirely unrelated without citing anything here.", ps)
    assert [r["flag"] for r in rep] == [None, "weak support", "no citation"]


def test_page_label_from_metadata():
    assert generate.page_label(dict(page_start=3, page_end=5)) == "p. 3-5"
    assert generate.page_label(dict(page_start=7, page_end=7)) == "p. 7"


def test_history_capped_at_four_messages():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    msgs = generate.build_messages("q", [passage("T", "x")], "concise", hist)
    assert [m["content"] for m in msgs[1:-1]] == ["m6", "m7", "m8", "m9"]


def test_below_floor_abstains_without_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")
    monkeypatch.setattr(generate.ollama_client, "chat", boom)
    out = generate.generate("weather?", [passage("T", "x")], top_dense=0.2)
    assert out["abstained"] and out["answer"] == "Your materials don't cover this."


def test_llm_insufficient_abstains(monkeypatch):
    monkeypatch.setattr(generate.ollama_client, "chat", lambda *a, **k: ("INSUFFICIENT_CONTEXT", {"seconds": 0}))
    out = generate.generate("q", [passage("T", "x")], top_dense=0.9)
    assert out["abstained"] and out["abstain_reason"] == "model said insufficient"


def test_insufficient_token_handling():
    assert generate.strip_insufficient("INSUFFICIENT_CONTEXT") is None
    assert generate.strip_insufficient("INSUFFICIENT_CONTEXT [S1][S2]") is None
    assert generate.strip_insufficient("Not covered here [S1]. INSUFFICIENT_CONTEXT") is None        # too little content
    long = "Meow laadle is the title of a note whose text repeats the word gap three times [S1]."
    assert generate.strip_insufficient(long + " INSUFFICIENT_CONTEXT") == long
    assert generate.strip_insufficient("A tuple is immutable [S1].") == "A tuple is immutable [S1]."


def test_missing_citations_fall_back_to_top_chunks(monkeypatch):
    monkeypatch.setattr(generate.ollama_client, "chat", lambda *a, **k: ("An answer with no citations at all here.", {"seconds": 0}))
    ps = [passage("A", "x", 1), passage("B", "y", 2)]
    out = generate.generate("q", ps, top_dense=0.9)
    assert not out["citations_ok"] and [s["chunk_id"] for s in out["sources"]] == ["c1", "c2"]
