import numpy as np

from edu_rag import syllabus

UNITS = [{"name": "Unit III: Classification", "topics": ["attribute selection measures: information gain, gain ratio, Gini index",
                                                         "tree pruning"]},
         {"name": "Unit V: Neural Networks", "topics": ["multilayer networks and backpropagation"]}]
TEXT = """Course: ML
Unit I: Python Fundamentals (12 hours) Data types; strings and string methods; regular expressions.
Unit II - Data Mining (8 hours)
Knowledge discovery process; Jaccard coefficient
Reference books: Han and Kamber"""


def test_regex_fallback_units_and_topics():
    units = syllabus.parse_units_regex(TEXT)
    assert [u["name"] for u in units] == ["Unit I: Python Fundamentals", "Unit II: Data Mining"]
    assert units[0]["topics"] == ["Data types", "strings and string methods", "regular expressions"]
    assert units[1]["topics"] == ["Knowledge discovery process", "Jaccard coefficient"]


def test_clean_units_drops_malformed():
    assert syllabus.clean_units([{"name": "U1", "topics": ["a topic", ""]}, {"name": "", "topics": ["x y z"]}, "junk"]) == \
        [{"name": "U1", "topics": ["a topic"]}]


def test_named_terms_match_by_words():
    assert syllabus.word_match("What is the Gini index?", UNITS)[1].startswith("attribute selection")
    assert syllabus.word_match("Explain photosynthesis", UNITS) is None


def test_scope_words_meaning_and_outside(monkeypatch):
    far = np.array([0.0, 1.0])
    monkeypatch.setattr(syllabus, "topic_vectors", lambda units: np.array([[1.0, 0.0]] * 3))
    s = syllabus.scope("What is backpropagation?", UNITS, qvec=far)                 # words carry it despite low cosine
    assert s["by"] == "words" and s["unit"] == "Unit V: Neural Networks"
    assert syllabus.scope("Explain photosynthesis", UNITS, qvec=far) is None
    near = np.array([1.0, 0.0])
    assert syllabus.scope("How do decision trees get smaller?", UNITS, qvec=near)["by"] == "meaning"
    assert syllabus.scope("anything", [], qvec=near) is None                        # no syllabus -> no verdict
