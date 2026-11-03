"""Exam mode: a timed mock paper from past papers and the slides, marked with feedback and a cited model answer."""
import time

import streamlit as st

from edu_rag import exam

from . import common


def setup_form(model):
    pyq = common.included_ids(["pyq"])
    with st.container(border=True):
        common.semester_picker("exam_semester")
        units = common.active_syllabus_units()
        all_units = common.all_syllabus_units()     # a past question is matched against all of them, or it always "fits" a semester
        whole_syllabus = len(units) == len(all_units)
        c1, c2 = st.columns(2)
        n = c1.slider("Questions", 2, 10, 4)
        minutes = c2.slider("Time (minutes)", 5, 120, 20, step=5)
        level = common.difficulty_picker("exam_difficulty",
                                         help="Easy = 2-mark recall questions, Medium = 5-mark explanations, Hard = 10-mark "
                                              "application and analysis. Past-paper questions are chosen by their marks.")
        chosen = st.multiselect("Syllabus units", [u["name"] for u in units if u["topics"]], placeholder="All units") if units else []
        chosen = chosen or ([] if whole_syllabus else [u["name"] for u in units if u["topics"]])   # a semester limits the paper
        use_pyq = st.toggle("Use my past-paper questions first", value=bool(pyq), disabled=not pyq,
                            help="Questions from your uploaded previous-year papers; the rest are written from your slides."
                            if pyq else "Upload previous-year papers in the Library to use them here.")
        if st.button("Start exam", type="primary", disabled=common.running("exam_build")):
            sources = common.evidence_filter()
            course_map = common.course_file_map()

            def keep(paper):                        # the clock starts when the paper is ready
                if paper["questions"]:
                    st.session_state.exam = {"paper": paper, "deadline": time.time() + minutes * 60, "marks": None}
                else:
                    st.session_state.exam_warning = "No questions could be set from that choice."
            common.start_job("exam_build", "Exam paper",
                             lambda progress: exam.build_paper(n=n, minutes=minutes, sources=sources, pyq_sources=pyq, units=all_units,
                                                               chosen_units=chosen, use_past_papers=use_pyq, model=model,
                                                               difficulty=level, course_map=course_map),
                             keep)
            st.rerun()
    common.job_progress("exam_build")
    if st.session_state.get("exam_warning"):
        st.warning(st.session_state.pop("exam_warning"))


@st.fragment(run_every=1)
def timer(ex):
    left = int(ex["deadline"] - time.time())
    if left <= 0 and ex["marks"] is None and not ex.get("submitted"):
        ex["time_up"] = True
        st.rerun(scope="app")
    left = max(left, 0)
    st.markdown(f"<div class='edu-timer{' low' if left < 60 else ''}'>{left // 60:02d}:{left % 60:02d}</div>",
                unsafe_allow_html=True)


def answer_sheet(ex):
    paper = ex["paper"]
    head, clock = st.columns([3, 1])
    head.markdown(f"**{len(paper['questions'])} questions · {paper['total_marks']} marks · {paper['minutes']} minutes · "
                  f"{common.difficulty_label(paper.get('difficulty'))}**  \n"
                  "<span class='small-muted'>Answer in your own words. Submit when you are done; the paper is submitted "
                  "automatically when the time runs out.</span>", unsafe_allow_html=True)
    with clock:
        timer(ex)
    section = None
    for i, q in enumerate(paper["questions"]):
        if q.get("section_name") and q["section_name"] != section:
            section = q["section_name"]
            note = f"attempt any {q['attempt']}: the best {q['attempt']} answers count" if q.get("attempt") else "attempt all"
            st.markdown(f"#### {section} <span class='small-muted'>({note})</span>", unsafe_allow_html=True)
        with st.container(border=True):
            tag = (f"past paper: {common.short_name(q['source'], 30)}, {q['pages']}" if q["origin"] == "past paper"
                   else "generated paper" if q["origin"] == "generated paper" else "from your slides")
            st.markdown(f"**Q{i + 1}.** {q['question']}  \n<span class='small-muted'>[{q['marks']} marks] · {tag}</span>",
                        unsafe_allow_html=True)
            st.text_area("Answer", key=f"exam_a{i}", height=120 if q["marks"] < 10 else 200, label_visibility="collapsed",
                         placeholder="Your answer...")
    if st.button("Submit paper", type="primary") or ex.get("time_up"):
        start_marking(ex)
        st.rerun()


def start_marking(ex):
    """Answers and time are taken now; the marking itself runs in the background."""
    qs = ex["paper"]["questions"]
    answers = [st.session_state.get(f"exam_a{i}", "") or "" for i in range(len(qs))]
    sources, model = common.evidence_filter(), st.session_state["model"]
    ex.update(submitted=True, seconds_used=int(ex["paper"]["minutes"] * 60 - max(ex["deadline"] - time.time(), 0)))

    def work(progress):
        marks = []
        for i, (q, given) in enumerate(zip(qs, answers)):
            progress(i, len(qs), f"marking answer {i + 1} of {len(qs)}")
            marks.append(exam.mark(q, given, sources=sources, model=model) | {"given": given})
        return marks

    def keep(marks, ex=ex):
        ex["marks"] = marks
    common.start_job("exam_mark", "Exam marking", work, keep)


def results(ex):
    qs, marks = ex["paper"]["questions"], ex["marks"]
    total, best = exam.paper_score(qs, marks)       # "attempt any N" sections count their best N answers
    pct = round(100 * total / best) if best else 0
    used = ex.get("seconds_used", 0)
    common.stats_row([(f"{total} / {best}", "marks"), (f"{pct}%", "score"), (f"{used // 60}:{used % 60:02d}", "time used"),
                      (sum(m["score"] == m["max"] for m in marks), "full marks")])
    if ex.get("time_up"):
        st.info("Time ran out, so the paper was submitted automatically.")
    for i, (q, m) in enumerate(zip(qs, marks)):
        share = m["score"] / m["max"] if m["max"] else 0
        icon = "✅" if share == 1 else "🟡" if share >= 0.4 else "❌"
        with st.expander(f"{icon} Q{i + 1} · {m['score']}/{m['max']} · {q['question'][:90]}", expanded=share < 1):
            st.markdown(f"**Question:** {q['question']}")
            st.markdown(f"**Your answer:** {m['given'] or '*(blank)*'}")
            if m["feedback"]:
                st.markdown(f"**Examiner:** *{m['feedback']}*")
            if m["model_answer"]:
                st.markdown("**Model answer from your slides:**")
                st.markdown(m["model_answer"])
                common.citation_chips(m["sources"], key=f"exam{i}")


def page():
    model = st.session_state["model"]
    common.page_start("Exam mode", "A timed mock paper from your past papers and slides, marked like an examiner would.")
    if not common.evidence_sources():
        st.info("Add slides or notes on the **Library** page first.")
        return
    ex = st.session_state.get("exam")
    if not ex:
        setup_form(model)
        return
    if ex["marks"] is None and ex.get("submitted"):
        if "exam_mark" in common.jobs():           # running, or finished and about to be collected
            common.job_progress("exam_mark")
        else:                                       # marking failed (see the message): let the student submit again
            ex["submitted"] = False
            st.rerun()
        return
    if ex["marks"] is None:
        answer_sheet(ex)
        if st.button("Abandon exam"):
            st.session_state.pop("exam")
            st.rerun()
        return
    results(ex)
    if st.button("New exam", type="primary"):
        for k in [k for k in st.session_state if k.startswith("exam")]:
            st.session_state.pop(k)
        st.rerun()
