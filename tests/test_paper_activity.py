"""Question-paper generator logic and the activity log. No Ollama needed."""
from edu_rag import activity, exam, paper


def test_clean_blueprint_and_total_marks():
    raw = {"course": "Python", "duration_minutes": "180", "sections": [
        {"name": "Section A", "attempt": None, "questions": [{"marks": 2, "style": "definition"}] * 6},
        {"name": "Section B", "attempt": 4, "questions": [{"marks": 6}] * 6},
        {"name": "Section C", "attempt": 2, "questions": [{"marks": 12}] * 3},
        {"name": "Empty", "questions": [{"marks": "x"}]}]}
    bp = paper.clean_blueprint(raw)
    assert [s["name"] for s in bp["sections"]] == ["Section A", "Section B", "Section C"]   # malformed section dropped
    assert bp["total_marks"] == 12 + 24 + 24 and bp["duration_minutes"] == 180
    assert bp["sections"][1]["instruction"] == "Attempt any 4"
    odd = paper.clean_blueprint({"sections": [{"attempt": 9, "questions": [{"marks": 5}, {"marks": 5}]}]})
    assert odd["sections"][0]["attempt"] is None and odd["total_marks"] == 10                 # "attempt 9 of 2" ignored


def test_blueprint_fallback_groups_by_marks():
    chunks = [{"text": "P\n\n1. Define x. [2]"}, {"text": "P\n\n2. Explain y. [6]"}, {"text": "P\n\n3. Define z. [2]"}]
    bp = paper.blueprint_from_questions(chunks)
    assert [[q["marks"] for q in s["questions"]] for s in bp["sections"]] == [[2, 2], [6]] and bp["total_marks"] == 10


def test_paper_score_counts_best_n_in_attempt_sections():
    qs = [{"section": 0, "attempt": None}] * 2 + [{"section": 1, "attempt": 2}] * 3
    marks = [{"score": 2, "max": 2}, {"score": 1, "max": 2}, {"score": 6, "max": 6}, {"score": 0, "max": 6}, {"score": 4, "max": 6}]
    assert exam.paper_score(qs, marks) == (3 + 10, 4 + 12)
    assert exam.paper_score([{}, {}], [{"score": 1, "max": 5}, {"score": 2, "max": 5}]) == (3, 10)   # plain exam: everything


def test_section_heading_does_not_change_marks_or_text():
    text = "Paper\n\n6. What is the use of the self keyword? [2]\nSECTION B (Attempt any FOUR, 6 marks each)"
    assert exam.parse_marks(text) == 2
    assert exam.question_text({"text": text}) == "What is the use of the self keyword?"
    assert exam.question_text({"text": "P\n\n4. Use of the with statement while handling ﬁles? [2]"}).endswith("files?")


def test_new_question_checks():
    assert paper.too_close("Explain the different file opening modes in Python with examples.",
                           ["Explain the different file opening modes in Python with examples."])
    assert not paper.too_close("Write a program to read a CSV file.", ["Explain inheritance with a program."])
    assert paper.MARKS_SUFFIX.sub("", "Define a tuple. (2 marks)") == "Define a tuple."
    slides = [{"text": "Tuples are immutable sequences; indexing and slicing work on tuples."},
              {"text": "The seek() method moves the file pointer to a byte offset."}]
    best, share = paper.best_slide("Explain what seek() does to the file pointer.", slides)
    assert best is slides[1] and share >= paper.MIN_SLIDE_SUPPORT
    assert paper.best_slide("Describe neural network backpropagation.", slides)[1] < paper.MIN_SLIDE_SUPPORT


def test_activity_log_most_asked(tmp_path):
    qlog, flog = tmp_path / "q.jsonl", tmp_path / "f.jsonl"
    for q, src in [("What is a tuple?", ["a"]), ("what is a tuple", ["a"]), ("Explain seek()", ["b", "a"]), ("Hi", [])]:
        activity.log_question(q, src, path=qlog)
    with open(qlog, "a", encoding="utf-8") as f:
        f.write('{"broken": \n')                                      # half-written line after a power cut
    asked = activity.most_asked(path=qlog)
    assert asked["a"][0] == ("what is a tuple", 2) and ("Explain seek()", 1) in asked["a"] and list(asked) == ["a", "b"]
    activity.log_feedback("q", "ans", 0, [{"source": "S.pdf", "pages": "p. 3"}], path=flog)
    assert activity.feedback(path=flog)[0]["sources"] == ["S.pdf p. 3"]
