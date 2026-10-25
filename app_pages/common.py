"""Helpers shared by the app's pages: settings, cached indexes, source filters, citation chips with slide previews."""
import json
from pathlib import Path

import streamlit as st

from edu_rag import config, generate, library, previews, retrieve, syllabus

ROLE_LABELS = {"textbook": "Slides & textbooks", "note": "Notes", "web": "From the web", "pyq": "Past papers", "syllabus": "Syllabus"}
KIND_ICONS = {"pdf": "📄", "pptx": "📊", "docx": "📝", "ipynb": "📓", "web": "🌐", "note": "🗒️"}
CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 980px;}
[data-testid="stChatMessage"] {padding: 0.6rem 0.2rem;}
.small-muted {color: rgba(128,128,128,0.95); font-size: 0.85rem;}
</style>
"""


def load_prefs():
    try:
        return json.loads(Path(config.APP_SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_prefs(prefs):
    Path(config.APP_SETTINGS_FILE).write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def pref(key, default):
    return st.session_state.setdefault("prefs", load_prefs()).get(key, default)


def set_pref(key, value):
    prefs = st.session_state.setdefault("prefs", load_prefs())
    if prefs.get(key) != value:
        prefs[key] = value
        save_prefs(prefs)


@st.cache_resource(show_spinner="Loading indexes...")
def warm_up():
    retrieve.load_state()
    return True


def refresh_indexes():
    retrieve.reset()
    warm_up.clear()
    syllabus_units.clear()


def sources():
    """Library sources, slides first, then notes, web pages, past papers, syllabus; by name within each."""
    order = {r: i for i, r in enumerate(config.ROLES)}
    return sorted(library.list_sources(), key=lambda s: (order.get(s["role"], 99), s["name"].lower()))


def excluded():
    return set(pref("excluded", []))


def included_ids(roles):
    """Source ids of the given roles that are switched on in the Library ("__none__" when nothing is: no results)."""
    ex = excluded()
    ids = [s["source_id"] for s in sources() if s["role"] in roles and s["source_id"] not in ex]
    return ids


def evidence_filter():
    return included_ids(config.EVIDENCE_ROLES) or ["__none__"]


@st.cache_data(show_spinner=False)
def syllabus_units(source_ids_and_hashes):
    """Units of the included syllabus sources (LLM-extracted once per file, cached in its folder)."""
    units = []
    for sid, _ in source_ids_and_hashes:
        units += syllabus.units_for(sid, model=st.session_state.get("model"))
    return units


def active_syllabus_units():
    ex = excluded()
    keys = tuple((s["source_id"], s["hash"]) for s in sources() if s["role"] == "syllabus" and s["source_id"] not in ex)
    return syllabus_units(keys) if keys else []


def short_name(name, n=28):
    stem = Path(name).stem if Path(name).suffix else name
    return stem if len(stem) <= n else stem[: n - 1] + "…"


def citation_chips(sources_, key):
    """One chip per cited passage; opening it shows the page image (cited text highlighted) and the passage."""
    if not sources_:
        return
    cols = st.columns(min(max(len(sources_), 3), 5))
    for i, s in enumerate(sources_):
        with cols[i % len(cols)]:
            icon = "🌐" if s.get("url") else "📄"
            with st.popover(f"{icon} {s['label']} · {s['pages']}", help=f"{s['source']} · {s['title']}", use_container_width=True):
                st.markdown(f"**{s['source']}** · {s['pages']}  \n{s['title']}")
                if s.get("url"):
                    st.markdown(f"[{s['url']}]({s['url']})")
                img = page_preview(s)
                if img:
                    st.image(str(img), use_container_width=True)
                with st.expander("Passage text", expanded=not img):
                    st.text(s["text"][:2000])


def page_preview(s):
    """The highlighted page image for a citation, if the source has a renderable file."""
    chunk = retrieve.load_state()["by_id"].get(s.get("chunk_id"))
    if not chunk:
        return None
    try:
        return previews.page_image(chunk["source_id"], chunk["page_start"], chunk["text"])
    except Exception:                               # a damaged or missing file must not break the answer
        return None


def support_note(res):
    cited = [s for s in res.get("support", []) if s["cited"]]
    weak = [s for s in cited if s["flag"] == "weak support"]
    if cited and len(weak) == len(cited):
        st.warning("Little of this answer appears in the cited passages. It may come from the model's general knowledge: "
                   "check the sources, or use Go deeper.", icon="⚠️")
    elif weak:
        st.caption("⚠️ Check against the sources: " + " | ".join(s["sentence"][:70] for s in weak))
