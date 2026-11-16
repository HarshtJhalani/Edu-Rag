"""Question paper: a new paper in the pattern of the student's previous-year papers, written from their slides,
weighted towards the topics those papers ask most. Answer key, PDF downloads, and a timed attempt in Exam mode."""
import re
import time

import streamlit as st

from edu_rag import paper

from . import common


@st.cache_data(show_spinner=False)
def frequency(pyq_ids, evidence_ids, units_key, course=None):
    units = [u for u in common.active_syllabus_units() if not course or u.get("course") == course]
    return paper.topic_frequency(list(pyq_ids), sources=list(evidence_ids), units=units or None)


def frequency_card(freq):
    if not freq["topics"]:
        st.caption("None of the past questions matched your slides closely.")
        return
    top = freq["topics"][:12]
    st.bar_chart([{"topic": t[:45], "times asked": n} for t, n in top], x="topic", y="times asked", horizontal=True,
                 sort=False, height=max(160, 28 * len(top)), color="#70587C", x_label="", y_label="times asked")
    decks = " · ".join(f"{common.short_name(d, 32)} ({n})" for d, n in freq["decks"][:6])
    st.caption(f"By deck: {decks}" + (f" · {freq['unmatched']} question(s) matched no slide" if freq["unmatched"] else ""))
    if st.toggle(f"List all {len(freq['questions'])} past questions", key="paper_all_past"):
        for q in freq["questions"]:
            st.markdown(f"- {common.esc(q['question'][:140])} <span class='small-muted'>[{q['marks']}] · "
                        f"{common.esc(q['topic'] or 'no matching slide')} · {common.short_name(q['paper'], 30)}</span>",
                        unsafe_allow_html=True)


def show_paper(qp):
    with st.container(border=True):
        st.markdown(f"<div style='text-align:center'><b>{common.esc(qp['exam'])}</b><br/>{common.esc(qp['course'] or '')}<br/>"
                    f"<span class='small-muted'>Time: {qp['duration_minutes'] // 60} h {qp['duration_minutes'] % 60:02d} min · "
                    f"Max. marks: {qp['total_marks']} · {common.difficulty_label(qp.get('difficulty'))}</span></div>",
                    unsafe_allow_html=True)
        for s in qp["sections"]:
            st.markdown(f"**{common.esc(paper.section_heading(s))}**")
            for q in s["questions"]:
                st.markdown(f"<div style='margin:0 0 .55rem 1.2rem'>{paper.label(q)} {common.esc(q['question'])} <b>[{q['marks']}]</b></div>",
                            unsafe_allow_html=True)


def key_view(qp, model):
    """The answer key tab: a button until the key is made, then the download and every answer under its question."""
    key = st.session_state.get("qpaper_key")
    if key is None:
        st.caption("A model answer for every question, sized by its marks, written from your files. It takes a few minutes.")
        if st.button("Make answer key", type="primary", disabled=common.running("paper_key")):
            sources = common.evidence_filter()

            def keep(result, qp=qp):                # only if that paper is still the one on screen
                if st.session_state.get("qpaper") is qp:
                    st.session_state.qpaper_key = result
            common.start_job("paper_key", "Answer key",
                             lambda progress: paper.answer_key(qp, sources=sources, model=model,
                                                               progress=lambda i, n: progress(i, n, f"answer {i} of {n}")),
                             keep)
            st.rerun()
        return
    st.download_button("Download answer key (PDF)", paper.key_to_pdf(qp, key), file_name=f"{fname(qp)}_answer_key.pdf",
                       mime="application/pdf")
    st.caption(paper.KEY_NOTE)
    for s in qp["sections"]:
        st.markdown(f"**{common.esc(paper.section_heading(s))}**")
        for q in s["questions"]:
            k = key.get(q["n"]) or {}
            with st.expander(f"{paper.label(q)} {q['question'][:110]} [{q['marks']}]"):
                st.markdown(paper.key_text(k.get("answer") or "", q["question"], k.get("code_ok", True)) or "*No answer could be written from your material.*")


def fname(qp):
    return re.sub(r"[^\w\- ]+", "", f"{qp['course'] or 'Question paper'}").strip().replace(" ", "_")[:50] or "question_paper"


def page():
    model = st.session_state["model"]
    common.page_start("Question paper", "A new paper in the pattern of your previous-year papers, written from your slides "
                                              "and weighted towards the topics they ask most.")
    pyqs = [s for s in common.sources() if s["role"] == "pyq" and s["source_id"] not in common.excluded()]
    if not pyqs:
        st.info("Upload your previous-year question papers on the **Library** page (choose **Previous-year question papers**).")
        return
    qp = st.session_state.get("qpaper")
    with st.expander("Paper settings", expanded=qp is None):
        common.semester_picker("paper_semester")
        course, units, files = common.subject_and_files("paper")
        src = st.selectbox("Follow the pattern of", pyqs, format_func=lambda s: common.short_name(s["name"], 60))
        level = common.difficulty_picker("paper_difficulty",
                                         help="Changes how the questions are asked. The sections and marks stay as in your past paper.")
    if not files:
        st.info("Choose at least one file for the questions.")
        return
    with st.spinner("Reading the paper's pattern..."):
        bp = paper.blueprint_for(src["source_id"], model=model)
    with st.spinner("Matching past questions to your slides..."):
        freq = frequency(tuple(s["source_id"] for s in pyqs), tuple(files), str([u["name"] for u in units]), course)
    marks = " · ".join(f"{s['name']}: {len(s['questions'])} × {'/'.join(map(str, sorted({q['marks'] for q in s['questions']})))}"
                       for s in bp["sections"])
    st.caption(f"Pattern: {sum(len(s['questions']) for s in bp['sections'])} questions, {bp['total_marks']} marks, "
               f"{bp['duration_minutes'] // 60} h {bp['duration_minutes'] % 60:02d} min · {marks}")
    with st.expander("What your past papers ask most"):
        frequency_card(freq)
    if st.button("Generate a new paper" if qp is None else "Generate another paper", type="primary", use_container_width=True,
                 disabled=common.running("paper")):
        sources, seed, narrow = files, int(time.time()), course is None

        def keep(result):
            st.session_state.qpaper = result
            st.session_state.pop("qpaper_key", None)
        common.start_job("paper", "Question paper",
                         lambda progress: paper.make(bp, freq, sources=sources, model=model, seed=seed, difficulty=level, only_asked_decks=narrow,
                                                     course=course, progress=lambda i, n: progress(i, n, f"section {i} of {n} written")),
                         keep)
        st.rerun()
    common.job_progress("paper")                   # nothing unless the job exists
    common.job_progress("paper_key")
    if not qp:
        return
    a, b = st.columns(2)
    a.download_button("Download paper (PDF)", paper.to_pdf(qp), file_name=f"{fname(qp)}.pdf", mime="application/pdf",
                      use_container_width=True)
    if b.button("Attempt in Exam mode", use_container_width=True):
        for k in [k for k in st.session_state if k.startswith("exam_a")]:
            st.session_state.pop(k)
        p = paper.to_exam(qp)
        st.session_state.exam = {"paper": p, "deadline": time.time() + p["minutes"] * 60, "marks": None}
        st.switch_page(st.session_state.pages["exam"])
    paper_tab, key_tab = st.tabs(["Paper", "Answer key"])
    with paper_tab:
        show_paper(qp)
    with key_tab:
        key_view(qp, model)
