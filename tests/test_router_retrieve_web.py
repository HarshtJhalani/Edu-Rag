import re

import pytest

from edu_rag import index, retrieve, router, web

VOCAB = {"nominal", "attribute", "dbscan", "lambda", "functions", "tuple", "python", "gini", "index"}


def test_rrf_rewards_agreement():
    fused = retrieve.rrf([("a", 0.9), ("b", 0.8), ("c", 0.1)], [("b", 9.0), ("d", 5.0)])
    assert [cid for cid, _ in fused][:2] == ["b", "a"]


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    assert index.tokenize("What is the __init__ of read_csv?") == ["__init__", "read_csv"]


def test_where_filter_combines_roles_and_sources():
    assert retrieve.where_filter(["textbook"], ["s1"]) == {"$and": [{"role": {"$in": ["textbook"]}}, {"source_id": {"$in": ["s1"]}}]}


def test_rules_chitchat():
    for m, kind in [("hi", "greeting"), ("hello there!", "greeting"), ("thanks, that helped", "thanks"), ("ok bye", "bye")]:
        r = router.rule_route(m, VOCAB)
        assert r["type"] == "chitchat" and r["chitchat_kind"] == kind


def test_rules_definition_only_for_terms_in_slide_titles():
    assert router.rule_route("What is a nominal attribute?", VOCAB)["type"] == "definition"
    assert router.rule_route("What is the weather today?", VOCAB) is None      # goes to the LLM router
    assert router.rule_route("What is the difference between a list and a tuple?", VOCAB) is None
    assert router.rule_route("What is it used for?", VOCAB) is None


def fake_state(monkeypatch, texts):
    chunks = [dict(chunk_id=f"c{i}", text=t, title=t.split("\n")[0], role="note" if i == 0 else "textbook", source_id=f"s{i}")
              for i, t in enumerate(texts)]
    state = dict(chunks=chunks, by_id={c["chunk_id"]: c for c in chunks}, terms=retrieve.chunk_terms(chunks))
    monkeypatch.setattr(retrieve, "load_state", lambda force=False: state)


def test_rare_word_match_finds_coined_names(monkeypatch):
    fake_state(monkeypatch, ["Meow laadle\n\nGap Gap Gap"] + [f"Python tuple slide {i}" for i in range(6)])
    assert retrieve.rare_term_matches("What is Meow laadle?") == ["c0"]
    assert retrieve.rare_term_matches("What is a tuple in Python?") == []      # common words never trigger it
    assert retrieve.rare_term_matches("What is the Apriori algorithm?") == []  # unknown words don't either
    assert retrieve.rare_term_matches("What is Meow laadle?", sources=["s9"]) == []   # source filter still applies


def test_incidental_rare_word_does_not_block_match(monkeypatch):
    fake_state(monkeypatch, ["Meow laadle\n\nGap Gap Gap", "What people say about loops"] + [f"Python slide {i}" for i in range(6)])
    assert retrieve.rare_term_matches("What does my note Meow Laadle say?") == ["c0"]
    assert retrieve.rare_term_matches("What do people say?") == ["c1"]                # a single rare word is enough
    assert retrieve.rare_term_matches("Does meow say anything?") == []                 # each chunk has 1 of 2 rare words: too weak


def test_router_keeps_own_words_without_history():
    r = router.fix_route({"type": "definition", "standalone_query": "What does the note say?", "sub_queries": []},
                         "What does my note Meow Laadle say?", [], {"meow", "laadle"})
    assert r["standalone_query"] == "What does my note Meow Laadle say?"
    hist = [{"role": "user", "content": "What is a tuple?"}]
    r = router.fix_route({"type": "followup", "standalone_query": "How does a tuple differ from a list?", "sub_queries": []},
                         "How is it different from a list?", hist, set())
    assert r["standalone_query"] == "How does a tuple differ from a list?"


def test_chitchat_naming_a_source_is_a_course_question():
    vocab = {"meow", "laadle", "tuple"}
    r = router.fix_route({"type": "chitchat", "standalone_query": "meow laadle", "sub_queries": []}, "meow laadle", [], vocab)
    assert r["type"] == "definition"
    r = router.fix_route({"type": "chitchat", "standalone_query": "tell me a joke", "sub_queries": []}, "tell me a joke", [], vocab)
    assert r["type"] == "chitchat"


def test_router_json_fallback():
    assert router.parse_route("not json", "msg") == {"type": "conceptual", "standalone_query": "msg", "sub_queries": [], "parse_ok": False}
    assert router.parse_route('{"type": "bogus"}', "msg")["parse_ok"] is False
    ok = router.parse_route('{"type": "compare", "standalone_query": "a vs b", "sub_queries": ["a?", "b?", "c?", "d?"]}', "m")
    assert ok["parse_ok"] and len(ok["sub_queries"]) == 3


def test_interleave_takes_both_sides():
    a = [{"chunk_id": "a1"}, {"chunk_id": "a2"}, {"chunk_id": "x"}]
    b = [{"chunk_id": "b1"}, {"chunk_id": "x"}, {"chunk_id": "b3"}]
    assert [c["chunk_id"] for c in router.interleave([a, b], 5)] == ["a1", "b1", "a2", "x", "b3"]


def test_allowlist_and_independent_sites():
    assert web.allowlisted("https://en.wikipedia.org/wiki/X") and not web.allowlisted("https://evil-wikipedia.org.example.com")
    assert web.site_key("https://de.wikipedia.org/a") == web.site_key("https://en.wikipedia.org/b") == "wikipedia.org"


def test_rank_hits_allowlist_first_one_per_site():
    hits = [{"url": "https://blog.example.com/a"}, {"url": "https://en.wikipedia.org/wiki/A"},
            {"url": "https://en.wikipedia.org/wiki/B"}, {"url": "https://www.w3schools.com/x#frag"}]
    assert [h["url"] for h in web.rank_hits(hits)] == ["https://en.wikipedia.org/wiki/A", "https://www.w3schools.com/x",
                                                       "https://blog.example.com/a"]


def test_wikipedia_keywords():
    assert web.keywords("How do I add and remove items from a Python set?") == "add remove items python set"


def test_group_sources_one_entry_per_page():
    s = [{"label": "W1", "url": "u1"}, {"label": "W3", "url": "u1"}, {"label": "W2", "url": "u2"}]
    assert [(g["url"], g["labels"]) for g in web.group_sources(s)] == [("u1", ["W1", "W3"]), ("u2", ["W2"])]


def test_confirmed_needs_two_sites():
    ps = [dict(site="wikipedia.org", text="dropout randomly disables neurons during training"),
          dict(site="geeksforgeeks.org", text="dropout randomly disables neurons while training networks"),
          dict(site="wikipedia.org", text="dropout randomly disables neurons during training")]
    two = web.confirmed_sentences("Dropout randomly disables neurons during training [W1][W2].", ps)
    one = web.confirmed_sentences("Dropout randomly disables neurons during training [W1][W3].", ps)
    assert two[0]["confirmed"] and not one[0]["confirmed"]


def test_cache_key_depends_on_model_and_settings():
    a = index.cache_path("bge-m3", "f" * 40, "s1")
    assert a != index.cache_path("bge-m3", "f" * 40, "s2") != index.cache_path("nomic-embed-text", "f" * 40, "s1")


GIB_TEXT = """A tuple is an ordered collection which cannot be changed once it is created, while a list is mutable.
The read method reads a string from an open file; the write method writes data and the close method releases it.
Clustering groups similar objects by a distance measure such as the Euclidean or Manhattan distance. A decision tree
splits the training data on the attribute with the highest information gain or the lowest Gini index. Neural networks
learn weights for every layer by gradient descent and backpropagation, and a perceptron updates its weights after each
example. Python modules are imported with import statements; functions return values and classes define objects.
Data mining finds frequent patterns, associations and correlations in large databases, and supervised learning uses
labelled examples to build a classification or regression model that predicts the class of a new instance."""
GIB_WORDS = re.findall(r"[a-z]+", GIB_TEXT.lower())


def test_is_gibberish_flags_keyboard_mashes_only(monkeypatch):
    # A small vocabulary compresses the scores (real -1.7..-2.0, mashes -3.5..-3.7 here); config's -3.6 is set on
    # the full corpus, so this test uses a threshold for its own vocabulary.
    monkeypatch.setattr(router.config, "GIBBERISH_MAX_SCORE", -3.0)
    known = {w: 1 for w in GIB_WORDS}
    model = router.char_model(GIB_WORDS)
    for mash in ("qwertyuiasdfghjklASDFGHJK", "wr[uiw f[piue gofihjeriogue[ripogphjfvieo[rhjv", "sdfsdf fghjkl"):
        assert router.is_gibberish(mash, known, model), mash
    for real in ("what is a tuple", "how do I read a file?", "bfs vs dfs", "hi", "2+2", "explain clustering distance"):
        assert not router.is_gibberish(real, known, model), real


def test_gibberish_never_reaches_the_llm(monkeypatch):
    monkeypatch.setattr(router, "is_gibberish", lambda m: True)
    monkeypatch.setattr(router, "llm_route", lambda *a, **k: pytest.fail("router LLM called for gibberish"))
    r = router.route("qwertyuiop", history=[{"role": "user", "content": "How does read() work?"}], vocab=set())
    assert r["type"] == "chitchat" and r["chitchat_kind"] == "unclear"
