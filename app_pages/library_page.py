"""📚 Library: add files and notes, choose which sources answers may use, remove sources."""
import streamlit as st

from edu_rag import chunking, config, index, ingest, previews

from . import common

UPLOAD_ROLES = {"textbook": "Lecture slides / textbook / notes", "pyq": "Previous-year question papers", "syllabus": "Syllabus"}


def upload_box(model):
    with st.container(border=True):
        st.markdown("#### ➕ Add files")
        c1, c2 = st.columns([2, 3])
        role = c1.radio("These files are", list(UPLOAD_ROLES), format_func=UPLOAD_ROLES.get, key="upload_role")
        files = c2.file_uploader("PDF, PowerPoint, Word or Jupyter", type=[t.lstrip(".") for t in ingest.SUPPORTED_TYPES],
                                 accept_multiple_files=True, key=f"uploader_{st.session_state.get('upload_round', 0)}")
        c2.caption("Scanned pages and screenshots are read with OCR, which takes a little longer.")
        if st.button("Add to library", type="primary", disabled=not files):
            for f in files:
                bar = st.progress(0.0, text=f"Reading {f.name}")
                try:
                    doc, _ = index.add_file(f.name, bytes(f.getbuffer()), role, llm=model,
                                            progress=lambda i, n, f=f: bar.progress(min(i / max(n, 1), 1.0), text=f"{f.name}: {i}/{n}"))
                    unit = doc.get("unit", "p.")
                    bar.progress(1.0, text=f"✅ {f.name}: {doc['n_pages']} {'pages' if unit == 'p.' else unit + 's'}"
                                           + (f", OCR on {len(doc['ocr_pages']) + len(doc['image_ocr_pages'])}" if doc["ocr_pages"] or doc["image_ocr_pages"] else ""))
                except Exception as e:
                    bar.progress(1.0, text=f"❌ {f.name}: {e}")
            common.refresh_indexes()
            st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1     # clears the uploader
            st.rerun()


def note_box():
    with st.container(border=True):
        st.markdown("#### 🗒️ Paste a note")
        title = st.text_input("Title", key="note_title")
        text = st.text_area("Text (Markdown headings like '# Topic' become sections)", height=140, key="note_text")
        if st.button("Add note", disabled=not (title.strip() and text.strip())):
            with st.spinner("Adding note..."):
                index.add_source(ingest.ingest_text(title.strip(), text), files={"note.md": text.encode("utf-8")})
            common.refresh_indexes()
            for k in ("note_title", "note_text"):
                st.session_state.pop(k, None)
            st.rerun()


def source_rows(model):
    srcs = common.sources()
    counts = {}
    for c in chunking.load_chunks():
        counts[c["source_id"]] = counts.get(c["source_id"], 0) + 1
    excluded = common.excluded()
    for role in config.ROLES:
        group = [s for s in srcs if s["role"] == role]
        if not group:
            continue
        st.markdown(f"#### {common.ROLE_LABELS[role]} ({len(group)})")
        with st.container(border=True):
            for s in group:
                sid = s["source_id"]
                c1, c2, c3 = st.columns([0.6, 7, 0.8])
                on = c1.checkbox("use", value=sid not in excluded, key=f"use_{sid}", label_visibility="collapsed",
                                 help="Use this source in answers, quizzes and cheat sheets")
                (excluded.discard if on else excluded.add)(sid)
                unit = "p." if (s.get("kind") or "pdf") in ("pdf", "docx") else ("slides" if s.get("kind") == "pptx" else "cells" if s.get("kind") == "ipynb" else "")
                size = f"{s['n_pages']} {'pages' if unit == 'p.' else unit}" if unit else ""
                c2.markdown(f"{common.KIND_ICONS.get(s.get('kind') or 'pdf', '📄')} **{s['name']}**  \n"
                            f"<span class='small-muted'>{size}{' · ' if size else ''}{counts.get(sid, 0)} chunks"
                            f"{' · ' + s['url'] if s.get('url') else ''}</span>", unsafe_allow_html=True)
                if c3.button("🗑️", key=f"rm_{sid}", help="Remove this source and its files", use_container_width=True):
                    st.session_state.confirm_remove = sid
                if st.session_state.get("confirm_remove") == sid:
                    st.warning(f"Remove **{s['name']}** and its files from the library?")
                    a, b, _ = st.columns([1, 1, 4])
                    if a.button("Remove", key=f"yes_{sid}", type="primary"):
                        with st.spinner("Removing..."):
                            index.remove_source(sid)
                        common.refresh_indexes()
                        st.session_state.pop("confirm_remove")
                        st.rerun()
                    if b.button("Cancel", key=f"no_{sid}"):
                        st.session_state.pop("confirm_remove")
                        st.rerun()
    common.set_pref("excluded", sorted(excluded))


def page():
    st.markdown(common.CSS, unsafe_allow_html=True)
    model = st.session_state["model"]
    st.title("📚 Library")
    st.caption(f"Everything you add is kept in {config.LIBRARY_DIR}, one folder per source.")
    t1, t2 = st.tabs(["Add files", "Paste a note"])
    with t1:
        upload_box(model)
    with t2:
        note_box()
    source_rows(model)
