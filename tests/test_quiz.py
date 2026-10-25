import random

from edu_rag import quiz

MCQ = {"type": "mcq", "question": "Which method reports the file position?", "options": ["a) tell()", "b) seek()", "c) open()", "d) read()"],
       "answer_index": 0, "explanation": "tell() gives the position.", "source": "S2"}


def test_valid_mcq_and_letter_prefixes_stripped():
    q = quiz.validate(MCQ, 3)
    assert q["options"] == ["tell()", "seek()", "open()", "read()"] and q["source_n"] == 2


def test_invalid_questions_dropped():
    assert quiz.validate(dict(MCQ, source="S9"), 3) is None                               # unknown passage
    assert quiz.validate(dict(MCQ, options=["x", "y", "z"]), 3) is None                   # not 4 options
    assert quiz.validate(dict(MCQ, options=["x", "y", "z", "None of the above"]), 3) is None
    assert quiz.validate(dict(MCQ, answer_index=4), 3) is None
    assert quiz.validate({"type": "short", "question": "Why?", "answer": "", "source": "S1"}, 3) is None
    assert quiz.validate({"type": "short", "question": "Why?", "answer": "Because.", "source": "[S1]"}, 3)["type"] == "short"


def test_passage_labels_removed_from_questions():
    assert quiz.strip_labels("Which is a characteristic of interval-scaled attributes according to S1?") == \
        "Which is a characteristic of interval-scaled attributes?"
    assert quiz.strip_labels("Tuples are immutable [S2].") == "Tuples are immutable."
    assert quiz.strip_labels("As described in passage S3, what does seek() do?", sentence=True) == "What does seek() do?"
    assert quiz.strip_labels("tell()") == "tell()"
    assert quiz.strip_labels("What is the Gini index (S1)?") == "What is the Gini index?"


def test_shuffle_keeps_the_right_answer():
    for seed in range(10):
        q = quiz.shuffle_options(quiz.validate(MCQ, 3), random.Random(seed))
        assert q["options"][q["answer_index"]] == "tell()"


def test_mcq_marking():
    q = quiz.validate(MCQ, 3)
    assert quiz.mark_mcq(q, 0)["score"] == 2 and quiz.mark_mcq(q, 1)["score"] == 0


def test_short_marking_handles_bad_llm_output(monkeypatch):
    q = {"question": "Why?", "answer": "Because.", "source_text": "..."}
    monkeypatch.setattr(quiz.ollama_client, "chat", lambda *a, **k: ('{"score": 7, "feedback": "x"}', {}))
    assert quiz.mark_short(q, "an answer")["score"] == 0                                   # out-of-range score
    monkeypatch.setattr(quiz.ollama_client, "chat", lambda *a, **k: ("not json", {}))
    assert quiz.mark_short(q, "an answer")["score"] == 0
    assert quiz.mark_short(q, "   ")["feedback"] == "No answer given."
