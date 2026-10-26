"""🃏 Flashcards: cards written from the student's slides, reviewed with spaced repetition (Leitner boxes)."""
import streamlit as st

from edu_rag import config, flashcards

from . import common

BOX_NAMES = ["New / missed", "1 day", "2 days", "4 days", "8 days", "16 days"]


def card_html(kind, text, back=False):
    return f"<div class='edu-card{' back' if back else ''}'><div class='k'>{kind}</div>{common.esc(text)}</div>"


def source_chip(c, key):
    common.citation_chips([{"label": "Slide", "source": c["source"], "pages": c["pages"], "title": "",
                            "chunk_id": c["chunk_id"], "text": c["source_text"]}], key=key)


def study_tab(deck):
    srcs = sorted({(c.get("source_id"), c["source"]) for c in deck}, key=lambda x: x[1])
    pick = st.multiselect("Only cards from", srcs, format_func=lambda s: common.short_name(s[1], 50), placeholder="All decks")
    due = flashcards.due(deck, source_ids=[s[0] for s in pick] or None)
    seen = st.session_state.setdefault("fc_seen", set())
    queue = [c for c in due if c["id"] not in seen]
    if not queue and due:                           # a round is over but missed cards are still due: next round
        seen.clear()
        queue = due
    if not queue:
        st.success("Nothing due right now. 🎉 Come back later, or make new cards.", icon="✅")
        return
    c = queue[0]
    st.caption(f"{len(queue)} due in this round · box: {BOX_NAMES[c['box']]} · {common.short_name(c['source'], 40)}, {c['pages']}")
    done = len(seen & {c["id"] for c in deck})
    st.progress(done / (done + len(queue)), text=f"{done} reviewed this round")
    shown = st.session_state.get("fc_revealed") == c["id"]
    st.markdown(card_html("Question", c["front"]), unsafe_allow_html=True)
    st.write("")
    if not shown:
        if st.button("Show answer", type="primary", use_container_width=True):
            st.session_state.fc_revealed = c["id"]
            st.rerun()
        return
    st.markdown(card_html("Answer", c["back"], back=True), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    for col, knew, label in ((a, False, "❌ Didn't know"), (b, True, "✅ Knew it")):
        if col.button(label, use_container_width=True, type="primary" if knew else "secondary", key=f"fc_{knew}"):
            flashcards.record(c["id"], knew)
            seen.add(c["id"])
            st.session_state.pop("fc_revealed", None)
            st.rerun()
    source_chip(c, f"fc_src_{c['id']}")


def make_tab(model):
    srcs = common.evidence_sources()
    if not srcs:
        st.info("Add slides or notes on the **📚 Library** page first.")
        return
    with st.container(border=True):
        mode = st.radio("Cards from", ["Chosen decks", "A topic"], horizontal=True, key="fc_mode")
        chosen = st.multiselect("Decks", srcs, default=srcs[:1], format_func=common.deck_label, key="fc_decks") \
            if mode == "Chosen decks" else []
        topic = st.text_input("Topic", placeholder="e.g. decision trees", key="fc_topic") if mode == "A topic" else None
        n = st.slider("How many cards", 4, 20, 10, key="fc_n")
        ready = chosen or (topic or "").strip()
        if st.button("Make cards", type="primary", disabled=not ready or common.running("flashcards")):
            ids, topic_ = [s["source_id"] for s in chosen] or common.evidence_filter(), topic.strip() if topic else None
            common.start_job("flashcards", "Flashcards",
                             lambda progress: flashcards.make(n=n, source_ids=ids, topic=topic_, model=model,
                                                              progress=lambda i, t: progress(i, t, f"part {i} of {t}")),
                             lambda cards: st.session_state.update(fc_new=cards))
            st.rerun()
    common.job_progress("flashcards")
    new = st.session_state.get("fc_new")
    if new is None:
        return
    if not new:
        st.warning("No cards could be made from that choice. Try another deck or topic.")
        return
    st.markdown(f"**{len(new)} new cards** · untick any you don't want")
    keep = []
    for i, c in enumerate(new):
        with st.container(border=True):
            k, body = st.columns([0.5, 9])
            if k.checkbox("keep", value=True, key=f"fc_keep_{i}", label_visibility="collapsed"):
                keep.append(c)
            body.markdown(f"**{c['front']}**  \n{c['back']}  \n<span class='small-muted'>{common.short_name(c['source'], 40)}, "
                          f"{c['pages']}</span>", unsafe_allow_html=True)
    if st.button(f"➕ Add {len(keep)} cards to my deck", type="primary", disabled=not keep):
        added = flashcards.add(keep)
        st.session_state.pop("fc_new")
        st.toast(f"Added {added} cards" + (f" ({len(keep) - added} were already in the deck)" if added < len(keep) else ""))
        st.rerun()


def deck_tab(deck):
    if not deck:
        st.info("Your deck is empty. Make cards in the **Make cards** tab.")
        return
    s = flashcards.stats(deck)
    st.bar_chart([{"box": b, "cards": n} for b, n in zip(BOX_NAMES, s["boxes"])], x="box", y="cards", x_label="", y_label="cards",
                 horizontal=True, sort=False, height=220, color="#6366F1")
    rows = [{"remove": False, "front": c["front"], "back": c["back"], "source": common.short_name(c["source"], 30),
             "page": c["pages"], "box": c["box"], "due": c["due"], "id": c["id"]} for c in deck]
    edited = st.data_editor(rows, hide_index=True, use_container_width=True, disabled=["front", "back", "source", "page", "box", "due"],
                            column_config={"id": None, "remove": st.column_config.CheckboxColumn("🗑️", width="small")},
                            key="fc_editor")
    gone = [r["id"] for r in edited if r["remove"]]
    if gone and st.button(f"Remove {len(gone)} card(s)", type="primary"):
        flashcards.remove(gone)
        st.rerun()


def page():
    model = st.session_state["model"]
    common.page_start("🃏", "Flashcards", "Cards from your own slides. Cards you know come back less often; missed ones come back today.")
    deck = flashcards.load()
    s = flashcards.stats(deck)
    common.stats_row([(s["due"], "due now"), (s["total"], "cards"), (s["new"], "never reviewed"), (s["mastered"], "well known")])
    t1, t2, t3 = st.tabs(["📖 Study", "✨ Make cards", "🗂️ Deck"])
    with t1:
        study_tab(deck)
    with t2:
        make_tab(model)
    with t3:
        deck_tab(deck)
