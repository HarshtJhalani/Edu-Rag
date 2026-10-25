"""💬 Ask: the chat. Streams the answer, then shows the checked version with citation chips and extras."""
import random

import streamlit as st

from edu_rag import config, generate, ollama_client, retrieve, router, syllabus, web

from . import common

ROUTE_LABELS = {"definition": "Definition", "conceptual": "Explanation", "compare": "Comparison", "followup": "Follow-up",
                "chitchat": "Small talk"}


def example_questions(n=4):
    """Starter questions from slide titles, so an empty chat shows what can be asked."""
    titles = sorted({c["title"] for c in retrieve.load_state()["chunks"]
                     if c["role"] == "textbook" and 3 <= len(c["title"]) <= 40 and not c["title"].lower().startswith(("let", "summary", "contents", "references", "assignment", "exercise", "demo", "solve", "example"))})
    rng = random.Random(st.session_state.setdefault("example_seed", random.randint(0, 10_000)))
    return [f"Explain {t.rstrip('?')}" for t in rng.sample(titles, min(n, len(titles)))]


def badges(res):
    r = res.get("route") or {}
    parts = []
    if r.get("type") and r["type"] != "chitchat":
        parts.append(f":violet-badge[{ROUTE_LABELS.get(r['type'], r['type'])} · {len(res.get('retrieved', []))} passages]")
    scope = res.get("scope")
    if scope == "outside":
        parts.append(":orange-badge[Outside your syllabus]")
    elif scope:
        parts.append(f":green-badge[{scope['unit'].split(':')[0]} · {scope['topic'][:50]}]")
    if res.get("seconds"):
        parts.append(f":gray-badge[{res['seconds']} s]")
    if parts:
        st.markdown(" ".join(parts))


def deeper_card(msg, i, style, model):
    d = msg.get("deeper")
    if not d:
        if msg["result"].get("route", {}).get("type") != "chitchat":
            if st.button("🌐 Go deeper on the web", key=f"deep_{i}", help=web.NOTICE):
                with st.spinner("Searching the web and reading pages..."):
                    msg["deeper"] = web.go_deeper(msg["result"]["route"]["standalone_query"], style, model=model)
                st.rerun()
            st.caption("Opt-in: " + web.NOTICE)
        return
    with st.container(border=True):
        st.markdown("**🌐 Beyond your materials**")
        if d["label"]:
            st.warning(d["label"], icon="🤖")
        if d["badge"]:
            st.success(f"Confirmed by 2 sources · {sum(c['confirmed'] for c in d['confirmed'])} of {len(d['confirmed'])} sentences", icon="✅")
        st.markdown(d["answer"])
        for s in web.group_sources(d["sources"]):
            st.markdown(f"[{', '.join(s['labels'])}] [{s['title']}]({s['url']})" + ("" if s["allowlisted"] else " · *not on the trusted list*"))
        if d["mode"] == "web" and d["pages"] and not d.get("added"):
            if st.button("➕ Add these pages to my sources", key=f"add_{i}"):
                with st.spinner("Adding the pages..."):
                    d["added"] = web.add_to_sources(d)["added"]
                common.refresh_indexes()
                st.rerun()
        elif d.get("added"):
            st.caption(f"Added {d['added']} page(s) to your sources.")


def past_papers_card(hits):
    if hits:
        with st.expander(f"📝 Asked in past papers ({len(hits)})"):
            for h in hits:
                st.markdown(f"**{h['source']}**, p. {h['page']} · match {h['score']:.2f}")
                st.markdown(f"> {h['text'][:500]}")


def render_assistant(msg, i, style, model):
    res = msg["result"]
    st.markdown(res["answer"] or "")
    badges(res)
    common.citation_chips(res.get("sources", []), key=f"m{i}")
    if res.get("sources") and not res.get("citations_ok"):
        st.caption("The answer had no valid citations; these are the top passages found.")
    common.support_note(res)
    past_papers_card(msg.get("past_papers"))
    deeper_card(msg, i, style, model)


def answer_live(question, history, style, model):
    """Route, retrieve and stream the answer into the page; returns the checked result."""
    status = st.empty()
    box = st.empty()
    shown = {"text": ""}

    def on_route(r):
        if r["type"] != "chitchat":
            status.caption(f"🔀 {ROUTE_LABELS.get(r['type'], r['type'])} → searching your materials" +
                           (f" for: {', '.join(r['sub_queries'])}" if r["sub_queries"] else ""))

    def on_token(piece):
        shown["text"] += piece
        text = shown["text"]
        if config.INSUFFICIENT.startswith(text.strip()[: len(config.INSUFFICIENT)]):
            return                                       # might be the "insufficient" token: don't flash it
        box.markdown(text + " ▌")

    ollama_client.unload(config.VISION_MODEL, wait=0)
    res = router.adaptive_answer(question, history, style, sources=common.evidence_filter(), model=model,
                                 on_token=on_token, on_route=on_route)
    status.empty()
    box.empty()
    units = common.active_syllabus_units()
    if units and res["route"]["type"] != "chitchat":
        res["scope"] = syllabus.scope(res["route"]["standalone_query"], units) or "outside"
    pyq = common.included_ids(["pyq"])
    hits = retrieve.past_papers(res["route"]["standalone_query"], sources=pyq) if pyq and res["route"]["type"] != "chitchat" else []
    return res, hits


def page():
    st.markdown(common.CSS, unsafe_allow_html=True)
    model, style = st.session_state["model"], common.pref("style", "concise")
    msgs = st.session_state.setdefault("messages", [])
    n_src = len([s for s in common.sources() if s["role"] in config.EVIDENCE_ROLES and s["source_id"] not in common.excluded()])

    head, clear = st.columns([5, 1])
    head.title("💬 Ask")
    if msgs and clear.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{n_src} sources in use · answer style: {style} · answers come only from your materials, with the page they came from")

    question = None
    if not msgs:
        if n_src == 0:
            st.info("Add your lecture slides or notes on the **📚 Library** page to get started.")
        else:
            st.markdown("**Try one of these:**")
            cols = st.columns(2)
            for j, q in enumerate(example_questions()):
                if cols[j % 2].button(q, key=f"ex_{j}", use_container_width=True):
                    question = q

    for i, msg in enumerate(msgs):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📘"):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg, i, style, model)

    question = st.chat_input("Ask about your materials") or question
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in msgs]
        msgs.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📘"):
            res, hits = answer_live(question, history, style, model)
            msg = {"role": "assistant", "content": res["answer"] or "", "result": res, "past_papers": hits}
            msgs.append(msg)
            render_assistant(msg, len(msgs) - 1, style, model)
