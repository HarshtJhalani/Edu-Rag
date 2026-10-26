"""💬 Ask: the chat. Streams the answer, then shows the checked version with citation chips and extras."""
import random
import re

import streamlit as st

from edu_rag import activity, config, export, flashcards, generate, ollama_client, retrieve, router, syllabus, web

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
    if res.get("route", {}).get("type") != "chitchat" and not res.get("abstained"):
        answer_tools(msg, i)
    deeper_card(msg, i, style, model)


def answer_tools(msg, i):
    """Copy the answer (with its sources) and rate it; ratings go to data/study/feedback.jsonl."""
    res = msg["result"]
    c1, c2, _ = st.columns([1, 1.4, 5])
    with c1.popover("📋", help="Copy this answer"):
        st.code(res["answer"] + "\n\n" + "\n".join(export.source_line(s) for s in res.get("sources", [])),
                language=None, wrap_lines=True)
        st.caption("Use the copy icon in the top-right corner.")
    with c2:
        rating = st.feedback("thumbs", key=f"fb_{st.session_state.get('chat_id')}_{i}")
    if rating is not None and msg.get("rated") != rating:
        msg["rated"] = rating
        activity.log_feedback(res.get("question", ""), res["answer"], rating, res.get("sources", []))
        st.toast("Thanks, feedback saved." if rating else "Noted. Try another answer style, or Go deeper.")


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
    if res.get("citations_ok") and not res.get("abstained"):
        by_id = retrieve.load_state()["by_id"]
        activity.log_question(question, [by_id[s["chunk_id"]]["source_id"] for s in res["sources"] if s.get("chunk_id") in by_id])
    pyq = common.included_ids(["pyq"])
    hits = retrieve.past_papers(res["route"]["standalone_query"], sources=pyq) if pyq and res["route"]["type"] != "chitchat" else []
    return res, hits


def new_chat_id():
    st.session_state.chat_seq = st.session_state.get("chat_seq", 0) + 1
    return f"c{st.session_state.chat_seq}"


def store_current():
    """Keep the open chat in this session's chat list (newest first)."""
    msgs, chats = st.session_state.get("messages", []), st.session_state.setdefault("chats", [])
    if not msgs:
        return
    cid = st.session_state.get("chat_id") or new_chat_id()
    st.session_state.chat_id = cid
    entry = {"id": cid, "title": export.chat_title(msgs, 40), "messages": msgs}
    chats[:] = [entry] + [c for c in chats if c["id"] != cid]


def open_chat(cid=None):
    store_current()
    chat = next((c for c in st.session_state.get("chats", []) if c["id"] == cid), None)
    st.session_state.messages = list(chat["messages"]) if chat else []
    st.session_state.chat_id = cid if chat else new_chat_id()


def history_sidebar():
    """New chat + this session's earlier chats (kept until the app is closed: lasting memory is future work)."""
    with st.session_state.get("sidebar_top") or st.sidebar:
        if st.button("✏️ New chat", use_container_width=True, disabled=not st.session_state.get("messages")):
            open_chat(None)
            st.rerun()
        store_current()
        chats = st.session_state.get("chats", [])
        if chats:
            st.caption("Recent chats")
            for c in chats[:12]:
                current = c["id"] == st.session_state.get("chat_id")
                if st.button(("▸ " if current else "") + c["title"], key=f"chat_{c['id']}", use_container_width=True,
                             type="secondary", disabled=current):
                    open_chat(c["id"])
                    st.rerun()


def export_menu(msgs):
    with st.popover("⬇️ Export", use_container_width=True):
        name = re.sub(r"[^\w\- ]+", "", export.chat_title(msgs, 40)).strip().replace(" ", "_") or "chat"
        st.download_button("Markdown (.md)", export.chat_to_markdown(msgs), file_name=f"{name}.md", use_container_width=True)
        st.download_button("PDF", export.chat_to_pdf(msgs), file_name=f"{name}.pdf", mime="application/pdf",
                           use_container_width=True)
        st.caption("Questions, answers and every cited page.")


def welcome(n_src):
    """Empty chat: what is loaded, and some questions to start with."""
    chunks = retrieve.load_state()["chunks"]
    due = flashcards.stats(flashcards.load())["due"]
    common.stats_row([(n_src, "sources in use"), (len(chunks), "passages indexed"),
                      (len({c["source_id"] for c in chunks if c["role"] == "pyq"}), "past papers"), (due, "flashcards due")])
    if n_src == 0:
        st.info("Add your lecture slides or notes on the **📚 Library** page to get started.")
        return None
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    question = None
    for j, q in enumerate(example_questions()):
        if cols[j % 2].button(f"💡 {q}", key=f"ex_{j}", use_container_width=True):
            question = q
    return question


def page():
    model, style = st.session_state["model"], common.pref("style", "concise")
    msgs = st.session_state.setdefault("messages", [])
    n_src = len(common.evidence_sources())
    history_sidebar()
    if not msgs:                                    # empty chat: centre the welcome block in the page
        st.markdown("<div class='edu-spacer'></div>", unsafe_allow_html=True)
    common.page_start("💬", "Ask your materials",
                      f"Answers come only from your {n_src} sources, with the page each fact came from · style: {style}")
    if msgs:
        _, c1, c2 = st.columns([3, 1.2, 1.1])
        with c1:
            export_menu(msgs)
        if c2.button("🗑️ Clear", use_container_width=True, help="Delete this chat"):
            st.session_state.chats = [c for c in st.session_state.get("chats", []) if c["id"] != st.session_state.get("chat_id")]
            st.session_state.messages = []
            st.session_state.pop("chat_id", None)
            st.rerun()

    question = welcome(n_src) if not msgs else None

    for i, msg in enumerate(msgs):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📘"):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg, i, style, model)

    question = st.chat_input("Ask about your materials") or question or st.session_state.pop("pending_question", None)
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
        if len(msgs) == 2:                          # first answer of a new chat: refresh the sidebar's chat list
            st.rerun()
