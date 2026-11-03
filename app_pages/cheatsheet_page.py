"""Cheat sheet: revision notes for a deck or a topic, every bullet with its page; download as Markdown or PDF."""
import re

import streamlit as st

from edu_rag import cheatsheet, config

from . import common


def page():
    model = st.session_state["model"]
    common.page_start("Cheat sheet", "Short revision notes from your own materials. Every bullet shows the page it came from.")
    srcs = [s for s in common.sources() if s["role"] in config.EVIDENCE_ROLES]
    if not srcs:
        st.info("Add slides or notes on the **Library** page first.")
        return
    with st.container(border=True):
        mode = st.radio("Make a cheat sheet for", ["A whole deck", "A topic"], horizontal=True)
        if mode == "A whole deck":
            src = st.selectbox("Deck", srcs, format_func=lambda s: s["name"])
            topic = None
        else:
            topic = st.text_input("Topic", placeholder="e.g. decision tree attribute selection measures")
            src = None
        go = st.button("Make cheat sheet", type="primary",
                       disabled=(mode == "A topic" and not (topic or "").strip()) or common.running("cheatsheet"))
    if go:
        sid, topic_, sources = src["source_id"] if src else None, topic.strip() if topic else None, common.evidence_filter()
        common.start_job("cheatsheet", "Cheat sheet",
                         lambda progress: cheatsheet.make(source_id=sid, topic=topic_, sources=sources, model=model,
                                                          progress=lambda i, n: progress(i, n, f"part {i} of {n}")),
                         lambda sheet: st.session_state.update(cheatsheet=sheet))
        st.rerun()
    common.job_progress("cheatsheet")
    sheet = st.session_state.get("cheatsheet")
    if not sheet:
        return
    if not sheet["sections"]:
        st.warning("Nothing to summarise was found for that choice.")
        return
    n = sum(len(s["bullets"]) for s in sheet["sections"])
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.subheader(sheet["title"])
    c1.caption(f"{len(sheet['sections'])} sections · {n} points · made in {sheet['seconds']} s")
    fname = re.sub(r"[^\w\- ]+", "", sheet["title"]).strip().replace(" ", "_")[:60] or "cheat_sheet"
    c2.download_button("Markdown", cheatsheet.to_markdown(sheet), file_name=f"{fname}.md", use_container_width=True)
    c3.download_button("PDF", cheatsheet.to_pdf(sheet), file_name=f"{fname}.pdf", mime="application/pdf", use_container_width=True)
    for s in sheet["sections"]:
        with st.container(border=True):
            st.markdown(f"**{s['title']}** <span class='small-muted'>· {common.short_name(s['source'], 40)} · {s['pages']}</span>",
                        unsafe_allow_html=True)
            st.markdown("\n".join(f"- {b['text']} <span class='small-muted'>({'; '.join(p for _, p in b['refs'])})</span>"
                                  + (" ⚠️" if b["weak"] else "") for b in s["bullets"]), unsafe_allow_html=True)
    if any(b["weak"] for s in sheet["sections"] for b in s["bullets"]):
        st.caption("⚠️ = little of that point appears on the cited page; double-check it.")
