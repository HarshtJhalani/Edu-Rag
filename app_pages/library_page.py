"""Library: add files and notes, choose which sources answers may use, remove sources."""
import streamlit as st

from edu_rag import activity, chunking, config, index, ingest, previews, syllabus

from . import common

UPLOAD_ROLES = {"textbook": "Lecture slides / textbook / notes", "pyq": "Previous-year question papers", "syllabus": "Syllabus"}


def upload_box(model):
    with st.container(border=True):
        st.markdown("#### Add files")
        c1, c2 = st.columns([2, 3])
        role = c1.radio("These files are", list(UPLOAD_ROLES), format_func=UPLOAD_ROLES.get, key="upload_role")
        files = c2.file_uploader("PDF, PowerPoint, Word or Jupyter", type=[t.lstrip(".") for t in ingest.SUPPORTED_TYPES],
                                 accept_multiple_files=True, key=f"uploader_{st.session_state.get('upload_round', 0)}")
        c2.caption("Scanned pages and screenshots are read with OCR, which takes a little longer.")
        if st.button("Add to library", type="primary", disabled=not files or common.running("upload")):
            items = [(f.name, bytes(f.getbuffer())) for f in files]

            def keep(results):
                common.refresh_indexes()
                st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1     # clears the uploader
                st.session_state.upload_results = results
            common.start_job("upload", "Adding files", lambda progress: add_files(items, role, model, progress), keep)
            st.rerun()
        common.job_progress("upload")
        for line in st.session_state.pop("upload_results", []):
            st.markdown(line)


def add_files(items, role, model, progress):
    """In the background: the page can be left while pages are read and the search index is rebuilt."""
    results = []
    for name, data in items:
        def on_page(i, n, name=name):
            progress(i, n, f"{name}: page {i} of {n}" if i < n else f"{name}: updating the search index")
        try:
            doc, _ = index.add_file(name, data, role, llm=model, progress=on_page)
            unit = doc.get("unit", "p.")
            ocr = len(doc["ocr_pages"]) + len(doc["image_ocr_pages"])
            results.append(f"✅ {name}: {doc['n_pages']} {'pages' if unit == 'p.' else unit + 's'}" + (f", OCR on {ocr}" if ocr else ""))
            if role == "syllabus":                  # done now, so the other pages never wait for it
                units = syllabus.units_for(doc["source_id"], model=model,
                                           progress=lambda i, n: progress(i, n, f"{name}: reading course {i} of {n}"))
                progress(1, 1, f"{name}: preparing the topics for search")
                syllabus.prepare_vectors(units)     # now, or the chat and the Syllabus page would wait for it
                results.append(f"✅ {name}: {sum(len(u['topics']) for u in units)} topics in {len(units)} units")
        except Exception as e:
            results.append(f"❌ {name}: {e}")
    return results


def note_box():
    with st.container(border=True):
        st.markdown("#### Paste a note")
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
    asked = activity.most_asked()
    for role in config.ROLES:
        group = [s for s in srcs if s["role"] == role]
        if not group:
            continue
        # The label stays the same while ticking boxes, so Streamlit keeps the section open across reruns.
        removing = any(s["source_id"] == st.session_state.get("confirm_remove") for s in group)
        with st.expander(f"**{common.ROLE_LABELS[role]}** ({len(group)})", expanded=removing):
            count = st.empty()                      # filled after the checkboxes below have updated `excluded`
            for s in group:
                sid = s["source_id"]
                c1, c2, c3 = st.columns([0.6, 7, 0.8])
                on = c1.checkbox("use", value=sid not in excluded, key=f"use_{sid}", label_visibility="collapsed",
                                 help="Use this source in answers, quizzes and cheat sheets")
                (excluded.discard if on else excluded.add)(sid)
                unit = "p." if (s.get("kind") or "pdf") in ("pdf", "docx") else ("slides" if s.get("kind") == "pptx" else "cells" if s.get("kind") == "ipynb" else "")
                size = f"{s['n_pages']} {'pages' if unit == 'p.' else unit}" if unit else ""
                c2.markdown(f"**{s['name']}**  \n"
                            f"<span class='small-muted'>{size}{' · ' if size else ''}{counts.get(sid, 0)} chunks"
                            f"{' · ' + s['url'] if s.get('url') else ''}</span>", unsafe_allow_html=True)
                if asked.get(sid):
                    c2.markdown("<span class='small-muted'>Most asked: " + " · ".join(
                        f"{common.esc(q[:70])}{f' ({n}×)' if n > 1 else ''}" for q, n in asked[sid]) + "</span>",
                        unsafe_allow_html=True)
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
            count.caption(f"{sum(s['source_id'] not in excluded for s in group)} of {len(group)} in use")
    common.set_pref("excluded", sorted(excluded))


def page():
    model = st.session_state["model"]
    common.page_start("Library", f"Everything you add is kept in <code>{config.LIBRARY_DIR}</code>, one folder per source. "
                      "Untick a source to leave it out of answers, quizzes and cheat sheets.")
    srcs = common.sources()
    roles = [s["role"] for s in srcs]
    common.stats_row([(roles.count("textbook"), "slide decks & books"), (roles.count("note"), "notes"), (roles.count("web"), "web pages"),
                      (roles.count("pyq"), "past papers"), (len(chunking.load_chunks()), "passages")])
    t1, t2 = st.tabs(["Add files", "Paste a note"])
    with t1:
        upload_box(model)
    with t2:
        note_box()
    source_rows(model)
