"""📄 Question paper: a new paper in the pattern of the student's previous-year papers, written from their slides,
weighted towards the topics those papers ask most. Answer key, PDF downloads, and a timed attempt in Exam mode."""
import re
import time

import streamlit as st

from edu_rag import paper

from . import common


@st.cache_data(show_spinner=False)
def frequency(pyq_ids, evidence_ids, units_key):
    return paper.topic_frequency(list(pyq_ids), sources=list(evidence_ids), units=common.active_syllabus_units() or None)


def pattern_card(bp):
    with st.container(border=True):
        st.markdown(f"**Pattern** <span class='small-muted'>from {common.esc(bp.get('source', ''))}</span>", unsafe_allow_html=True)
        common.stats_row([(len(bp["sections"]), "sections"), (sum(len(s["questions"]) for s in bp["sections"]), "questions"),
                          (bp["total_marks"], "marks"), (f"{bp['duration_minutes'] // 60}h {bp['duration_minutes'] % 60:02d}", "time")])
        for s in bp["sections"]:
            marks = sorted({q["marks"] for q in s["questions"]})
            st.markdown(f"- **{s['name']}**: {len(s['questions'])} questions × {'/'.join(map(str, marks))} marks · "
                        f"{s['instruction']}")


def frequency_card(freq):
    with st.container(border=True):
        st.markdown("**What your past papers ask most**")
        if not freq["topics"]:
            st.caption("None of the past questions matched your slides closely.")
            return
        top = freq["topics"][:12]
        st.bar_chart([{"topic": t[:45], "times asked": n} for t, n in top], x="topic", y="times asked", horizontal=True,
                     sort=False, height=max(160, 28 * len(top)), color="#6366F1", x_label="", y_label="times asked")
        decks = " · ".join(f"{common.short_name(d, 32)} ({n})" for d, n in freq["decks"][:6])
        st.caption(f"By deck: {decks}" + (f" · {freq['unmatched']} question(s) matched no slide" if freq["unmatched"] else ""))
        with st.expander(f"All {len(freq['questions'])} past questions"):
            for q in freq["questions"]:
                st.markdown(f"- {common.esc(q['question'][:140])} <span class='small-muted'>[{q['marks']}] · "
                            f"{common.esc(q['topic'] or 'no matching slide')} · {common.short_name(q['paper'], 30)}</span>",
                            unsafe_allow_html=True)


def show_paper(qp, show_sources):
    with st.container(border=True):
        st.markdown(f"<div style='text-align:center'><b>{common.esc(qp['exam'])}</b><br/>{common.esc(qp['course'] or '')}<br/>"
                    f"<span class='small-muted'>Time: {qp['duration_minutes'] // 60} h {qp['duration_minutes'] % 60:02d} min · "
                    f"Max. marks: {qp['total_marks']}</span></div>", unsafe_allow_html=True)
        for s in qp["sections"]:
            st.markdown(f"##### {s['name']} <span class='small-muted'>({common.esc(s['instruction'])})</span>", unsafe_allow_html=True)
            for q in s["questions"]:
                src = (f"  \n<span class='small-muted'>from {common.short_name(q['source'], 35)}, {q['pages']} · {q['topic']}</span>"
                       if show_sources else "")
                st.markdown(f"**{q['n']}.** {common.esc(q['question'])} **[{q['marks']}]**{src}", unsafe_allow_html=True)


def key_section(qp, model):
    key = st.session_state.get("qpaper_key")
    if key is None:
        if st.button("🔑 Make answer key", use_container_width=True, disabled=common.running("paper_key")):
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
    st.download_button("⬇️ Answer key PDF", paper.key_to_pdf(qp, key), file_name=f"{fname(qp)}_answer_key.pdf",
                       mime="application/pdf", use_container_width=True)


def answers_view(qp):
    key = st.session_state.get("qpaper_key")
    if not key:
        return
    st.markdown("#### 🔑 Answer key")
    for s in qp["sections"]:
        for q in s["questions"]:
            k = key.get(q["n"]) or {}
            with st.expander(f"{q['n']}. {q['question'][:100]} [{q['marks']}]"):
                st.markdown(k.get("answer") or "*No answer could be written from your slides.*")
                common.citation_chips(k.get("sources", []), key=f"qkey{q['n']}")


def fname(qp):
    return re.sub(r"[^\w\- ]+", "", f"{qp['course'] or 'Question paper'}").strip().replace(" ", "_")[:50] or "question_paper"


def page():
    model = st.session_state["model"]
    common.page_start("📄", "Question paper", "A new paper in the pattern of your previous-year papers, written from your slides "
                                              "and weighted towards the topics they ask most.")
    pyqs = [s for s in common.sources() if s["role"] == "pyq" and s["source_id"] not in common.excluded()]
    if not pyqs:
        st.info("Upload your previous-year question papers on the **📚 Library** page (choose **Previous-year question papers**).")
        return
    left, right = st.columns([1, 1])
    with left:
        src = st.selectbox("Follow the pattern of", pyqs, format_func=lambda s: common.short_name(s["name"], 60))
        with st.spinner("Reading the paper's pattern..."):
            bp = paper.blueprint_for(src["source_id"], model=model)
        pattern_card(bp)
    with right:
        with st.spinner("Matching past questions to your slides..."):
            units = common.active_syllabus_units()
            freq = frequency(tuple(s["source_id"] for s in pyqs), tuple(common.evidence_filter()),
                             str([u["name"] for u in units]))
        frequency_card(freq)

    if st.button("✨ Generate a new paper", type="primary", use_container_width=True, disabled=common.running("paper")):
        sources, seed = common.evidence_filter(), int(time.time())

        def keep(result):
            st.session_state.qpaper = result
            st.session_state.pop("qpaper_key", None)
        common.start_job("paper", "Question paper",
                         lambda progress: paper.make(bp, freq, sources=sources, model=model, seed=seed,
                                                     progress=lambda i, n: progress(i, n, f"section {i} of {n} written")),
                         keep)
        st.rerun()
    common.job_progress("paper")                   # nothing unless the job exists
    common.job_progress("paper_key")
    qp = st.session_state.get("qpaper")
    if not qp:
        return
    st.caption(f"Made in {qp['seconds']} s. Generate again for a different paper.")
    a, b, c = st.columns(3)
    a.download_button("⬇️ Paper PDF", paper.to_pdf(qp), file_name=f"{fname(qp)}.pdf", mime="application/pdf",
                      use_container_width=True)
    with b:
        key_section(qp, model)
    if c.button("⏱️ Attempt in Exam mode", use_container_width=True):
        for k in [k for k in st.session_state if k.startswith("exam_a")]:
            st.session_state.pop(k)
        p = paper.to_exam(qp)
        st.session_state.exam = {"paper": p, "deadline": time.time() + p["minutes"] * 60, "marks": None}
        st.switch_page(st.session_state.pages["exam"])
    show_paper(qp, st.toggle("Show which slide each question comes from", value=False))
    answers_view(qp)
