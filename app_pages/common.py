"""Helpers shared by the app's pages: settings, cached indexes, source filters, citation chips with slide previews."""
import html
import json
import threading
import time
from pathlib import Path

import streamlit as st

from edu_rag import config, generate, library, previews, retrieve, syllabus

ROLE_LABELS = {"textbook": "Slides & textbooks", "note": "Notes", "web": "From the web", "pyq": "Past papers", "syllabus": "Syllabus"}
KIND_ICONS = {"pdf": "📄", "pptx": "📊", "docx": "📝", "ipynb": "📓", "web": "🌐", "note": "🗒️"}
# Colours are translucent indigo/greys so the same rules read well in the light and the dark theme.
CSS = """
<style>
.block-container {padding-top: 2.4rem; max-width: 1200px;}
[data-testid="stBottomBlockContainer"] {max-width: 1200px; margin: 0 auto;}
.edu-spacer {height: 14vh;}
footer {visibility: hidden;}
.small-muted {color: rgba(128,128,128,0.95); font-size: 0.85rem;}
.edu-hero {padding: 1.05rem 1.35rem; border-radius: 16px; margin-bottom: 1.1rem;
  background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(168,85,247,0.08) 60%, rgba(99,102,241,0.02));
  border: 1px solid rgba(99,102,241,0.22);}
.edu-hero .t {font-size: 1.65rem; font-weight: 750; letter-spacing: -0.01em; line-height: 1.25;}
.edu-hero .s {opacity: 0.72; margin-top: 0.2rem; font-size: 0.95rem;}
.edu-stats {display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.2rem 0 1rem;}
.edu-stat {flex: 1 1 96px; padding: 0.7rem 0.9rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.22);
  background: rgba(128,128,128,0.05);}
.edu-stat .v {font-size: 1.35rem; font-weight: 700;}
.edu-stat .l {font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em;}
.edu-card {min-height: 190px; display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center; padding: 1.6rem; border-radius: 18px; font-size: 1.3rem; line-height: 1.45;
  border: 1px solid rgba(99,102,241,0.35); background: rgba(99,102,241,0.07);}
.edu-card .k {font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;}
.edu-card.back {background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.35); font-size: 1.1rem;}
.edu-timer {font-variant-numeric: tabular-nums; font-size: 1.7rem; font-weight: 750; text-align: right;}
.edu-timer.low {color: #EF4444;}
[data-testid="stChatMessage"] {padding: 0.7rem 0.4rem;}
[data-testid="stSidebar"] [data-testid="stButton"] button {justify-content: flex-start; text-align: left;}
</style>
"""


esc = html.escape


# --- background jobs -------------------------------------------------------------------------------
# Streamlit stops a page's script when the student opens another page, which threw away a half-written question
# paper. Long jobs therefore run in a thread: work(progress) gets a progress(i, n, text=None) callback and must not
# call Streamlit; on_done(result) runs later in a normal app run (collect_jobs), whichever page is open.

def jobs():
    return st.session_state.setdefault("jobs", {})


def start_job(name, label, work, on_done):
    job = {"label": label, "status": "running", "i": 0, "n": 0, "text": "", "result": None, "error": None,
           "on_done": on_done, "t0": time.time()}

    def progress(i, n, text=None):
        job.update(i=i, n=n, text=text or job["text"])

    def run():
        try:
            job["result"] = work(progress)
            job["status"] = "done"
        except Exception as e:                      # shown to the student by collect_jobs
            job.update(status="error", error=f"{type(e).__name__}: {e}")

    jobs()[name] = job
    threading.Thread(target=run, daemon=True, name=f"edu-rag-{name}").start()


def running(name):
    j = jobs().get(name)
    return bool(j) and j["status"] == "running"


def collect_jobs():
    """Hand finished jobs' results to their pages. Called at the start of every app run."""
    for name, j in list(jobs().items()):
        if j["status"] == "done":
            jobs().pop(name)
            j["on_done"](j["result"])
            st.toast(f"{j['label']} is ready", icon="✅")
        elif j["status"] == "error":
            jobs().pop(name)
            st.toast(f"{j['label']} failed: {j['error']}", icon="❌")


def job_text(j):
    step = f"{j['text'] or 'working'}" + (f" ({j['i']}/{j['n']})" if j["n"] and not j["text"] else "")
    return f"{j['label']}: {step} · {int(time.time() - j['t0'])} s"


@st.fragment(run_every=1)
def job_progress(name):
    """Live progress of a job on its page (nothing when there is none); the app reruns once it finishes, so the
    result appears."""
    j = jobs().get(name)
    if not j:
        return
    if j["status"] != "running":
        st.rerun(scope="app")
    st.progress(min(j["i"] / j["n"], 1.0) if j["n"] else 0.0, text=job_text(j))
    st.caption("This keeps running if you open another page.")


@st.fragment(run_every=2)
def jobs_sidebar():
    """Running jobs, shown on every page; reruns the app when one finishes so its result is collected."""
    if any(j["status"] != "running" for j in jobs().values()):
        st.rerun(scope="app")
    for j in jobs().values():
        st.caption(f"⏳ {job_text(j)}")


def page_start(icon, title, subtitle=""):
    """Every page opens with the shared style and a header card."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='edu-hero'><div class='t'>{icon} {title}</div>"
                + (f"<div class='s'>{subtitle}</div>" if subtitle else "") + "</div>", unsafe_allow_html=True)


def stats_row(items):
    """[(value, label)] as a row of small tiles."""
    st.markdown("<div class='edu-stats'>" + "".join(f"<div class='edu-stat'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
                                                     for v, l in items) + "</div>", unsafe_allow_html=True)


def evidence_sources():
    """Slides, notes and web pages that are switched on in the Library."""
    ex = excluded()
    return [s for s in sources() if s["role"] in config.EVIDENCE_ROLES and s["source_id"] not in ex]


def deck_label(s):
    return f"{KIND_ICONS.get(s.get('kind') or 'pdf', '📄')} {s['name']}"


THEME_KEYS = ["base", "primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor", "borderColor",
              "sidebar.backgroundColor", "codeBackgroundColor"]
THEMES = {
    "Light": ["light", "#6366F1", "#FFFFFF", "#F4F5FB", "#1F2433", "#E3E5EE", "#F7F8FC", "#F4F5FB"],
    "Dark": ["dark", "#818CF8", "#0F111A", "#1A1D2B", "#E6E8F0", "#2A2E40", "#141724", "#1A1D2B"],
    "System": [None] * 8,                         # Streamlit follows the OS; primary colour from config.toml
}


def apply_theme(mode):
    """Set the theme options. Streamlit sends them at the START of each run, so a change shows from the next run.
    Returns True when an option changed (the caller reruns). Options are process-wide: fine for a local single-user app."""
    from streamlit import config as st_config
    values = THEMES.get(mode, THEMES["System"])
    changed = False
    for key, value in zip(THEME_KEYS, values):
        if value is None and key == "primaryColor":
            value = "#6366F1"
        if st_config.get_option(f"theme.{key}") != value:
            st_config.set_option(f"theme.{key}", value)
            changed = True
    return changed


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
                slide_actions(s, f"{key}_{i}")


def slide_actions(s, key):
    """From a cited slide: ask about it in the chat, or turn it into flashcards."""
    chunk = retrieve.load_state()["by_id"].get(s.get("chunk_id"))
    if not chunk:
        return
    a, b = st.columns(2)
    if a.button("💬 Explain this slide", key=f"explain_{key}", use_container_width=True):
        st.session_state.pending_question = f"Explain \"{chunk['title']}\" ({short_name(chunk['source_name'], 40)}, {s['pages']})"
        if "pages" in st.session_state:
            st.switch_page(st.session_state.pages["ask"])
        st.rerun()
    if b.button("🃏 Flashcards from it", key=f"cards_{key}", use_container_width=True):
        with st.spinner("Writing cards from this slide..."):
            from edu_rag import flashcards
            cards = flashcards.from_passages([chunk], n=4, model=st.session_state.get("model"))
            added = flashcards.add(cards)
        st.toast(f"Added {added} flashcard(s) from {s['pages']}" if added else "No new cards from this slide (already in the deck?)",
                 icon="🃏")


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
