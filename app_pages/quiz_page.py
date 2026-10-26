"""🎯 Quiz: practice questions from the student's slides, marked with feedback and the slide each answer is on."""
import streamlit as st

from edu_rag import config, quiz

from . import common


def setup_form(srcs, model):
    with st.container(border=True):
        mode = st.radio("Questions from", ["Chosen decks", "A topic"], horizontal=True)
        if mode == "Chosen decks":
            chosen = st.multiselect("Decks", srcs, default=srcs[:1],
                                    format_func=lambda s: f"{common.KIND_ICONS.get(s.get('kind') or 'pdf', '📄')} {s['name']}")
            topic = None
        else:
            topic = st.text_input("Topic", placeholder="e.g. clustering methods")
            chosen = []
        n = st.slider("Number of questions", 3, 10, 5)
        ready = (mode == "Chosen decks" and chosen) or (mode == "A topic" and (topic or "").strip())
        if st.button("Start quiz", type="primary", disabled=not ready or common.running("quiz")):
            ids = [s["source_id"] for s in chosen] or common.evidence_filter()
            topic_ = topic.strip() if topic else None
            common.start_job("quiz", "Quiz",
                             lambda progress: quiz.make(n=n, source_ids=ids, topic=topic_, model=model,
                                                        progress=lambda i, t: progress(i, t, f"{i} of {t} questions written")),
                             lambda qs: st.session_state.update(quiz={"questions": qs, "marks": None}))
            st.rerun()
    common.job_progress("quiz")


def show_questions(qz, model):
    qs = qz["questions"]
    if not qs:
        st.warning("No questions could be made from that choice. Try another deck or a broader topic.")
        return
    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(qs):
            st.markdown(f"**Q{i + 1}.** {q['question']}")
            if q["type"] == "mcq":
                answers.append(st.radio("Answer", q["options"], index=None, key=f"q{i}", label_visibility="collapsed"))
            else:
                answers.append(st.text_area("Your answer", key=f"q{i}", height=80, label_visibility="collapsed"))
            st.divider()
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        marks = []
        with st.spinner("Marking..."):
            for q, a in zip(qs, answers):
                if q["type"] == "mcq":
                    marks.append(quiz.mark_mcq(q, q["options"].index(a) if a in q["options"] else -1) | {"given": a or "(no answer)"})
                else:
                    marks.append(quiz.mark_short(q, a or "", model=model) | {"given": a or "(no answer)"})
        qz["marks"] = marks
    if qz.get("marks"):
        show_results(qs, qz["marks"])


def show_results(qs, marks):
    total, best = sum(m["score"] for m in marks), sum(m["max"] for m in marks)
    st.subheader(f"Score: {total} / {best}")
    st.progress(total / best if best else 0.0)
    for i, (q, m) in enumerate(zip(qs, marks)):
        icon = "✅" if m["score"] == m["max"] else "🟡" if m["score"] else "❌"
        with st.expander(f"{icon} Q{i + 1}. {q['question']}", expanded=m["score"] < m["max"]):
            st.markdown(f"**Your answer:** {m['given']}  \n**{'Correct answer' if q['type'] == 'mcq' else 'Model answer'}:** {m['correct']}")
            if m.get("feedback"):
                st.markdown(f"*{m['feedback']}*")
            common.citation_chips([{"label": "Slide", "source": q["source"], "pages": q["pages"], "title": "",
                                    "chunk_id": q["chunk_id"], "text": q["source_text"]}], key=f"quiz{i}")


def page():
    model = st.session_state["model"]
    common.page_start("🎯", "Practice quiz", "Questions written from your own slides. Multiple choice is marked instantly; short answers are marked against the slide.")
    srcs = [s for s in common.sources() if s["role"] in config.EVIDENCE_ROLES and s["source_id"] not in common.excluded()]
    if not srcs:
        st.info("Add slides or notes on the **📚 Library** page first.")
        return
    qz = st.session_state.get("quiz")
    if qz and qz["questions"]:
        if st.button("↺ New quiz"):
            st.session_state.pop("quiz")
            st.rerun()
        show_questions(qz, model)
    else:
        setup_form(srcs, model)
        if qz is not None:
            show_questions(qz, model)
