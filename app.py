"""Edu-RAG: Streamlit app. Run with start.bat (or: streamlit run app.py). Pages live in app_pages/."""
import subprocess
from pathlib import Path

import streamlit as st

from app_pages import (chat, cheatsheet_page, common, exam_page, flashcards_page, library_page, paper_page, quiz_page,
                       syllabus_page)
from edu_rag import config, generate, ollama_client, setup_check

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="Edu-RAG study assistant", page_icon=str(ASSETS / "icon.svg"), layout="wide")
config.ensure_dirs()
if common.apply_theme(common.pref("theme", "System")):
    st.rerun()                                    # first run after a server start: this run went out with the old theme


def setup_panel(status):
    """First launch: Ollama installed and running, models present. Never installs anything without a click."""
    st.title("📘 Edu-RAG setup")
    if not status["ollama_installed"]:
        st.error("Ollama is not installed. It runs the AI models locally on this computer.")
        st.code(" ".join(setup_check.WINGET_CMD))
        if st.button("Install Ollama with winget"):
            with st.spinner("Installing Ollama..."):
                out = subprocess.run(setup_check.WINGET_CMD, capture_output=True, text=True)
            st.text(out.stdout[-2000:] or out.stderr[-2000:])
        return
    if not status["ollama_running"]:
        st.warning("Ollama is installed but not running.")
        if st.button("Start Ollama"):
            setup_check.start_ollama()
            st.rerun()
        return
    vram = f"{status['vram_gb']} GB VRAM" if status["vram_gb"] else "no NVIDIA GPU found"
    st.info(f"GPU: {vram} → answer model **{status['llm']}**")
    if status["missing"]:
        st.write("These models need to be downloaded once:")
        for m in status["missing"]:
            st.write(f"- `{m}` (about {setup_check.APPROX_SIZE_GB.get(m, 1.0)} GB)")
        if st.button(f"Download {status['download_gb']} GB"):
            for m in status["missing"]:
                bar = st.progress(0.0, text=f"Downloading {m}")
                for ev in ollama_client.pull(m):
                    if ev.get("total"):
                        bar.progress(min(ev.get("completed", 0) / ev["total"], 1.0), text=f"{m}: {ev.get('status', '')}")
                bar.progress(1.0, text=f"{m}: done")
            st.session_state.pop("setup", None)
            st.rerun()


def sidebar():
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
    p = {"ask": st.Page(chat.page, title="Ask", icon="💬", url_path="ask", default=True),
         "flashcards": st.Page(flashcards_page.page, title="Flashcards", icon="🃏", url_path="flashcards"),
         "quiz": st.Page(quiz_page.page, title="Quiz", icon="🎯", url_path="quiz"),
         "exam": st.Page(exam_page.page, title="Exam mode", icon="⏱️", url_path="exam"),
         "paper": st.Page(paper_page.page, title="Question paper", icon="📄", url_path="paper"),
         "cheatsheet": st.Page(cheatsheet_page.page, title="Cheat sheet", icon="📝", url_path="cheatsheet"),
         "syllabus": st.Page(syllabus_page.page, title="Syllabus", icon="🗺️", url_path="syllabus"),
         "library": st.Page(library_page.page, title="Library", icon="📚", url_path="library")}
    st.session_state.pages = p                    # lets a page jump to another (st.switch_page)
    pages = st.navigation({"Study": [p["ask"], p["flashcards"], p["quiz"], p["exam"], p["paper"], p["cheatsheet"]],
                           "Course": [p["syllabus"], p["library"]]})
    with st.sidebar:
        st.session_state.sidebar_top = st.container()     # pages put their own sidebar items here (chat history)
        if common.jobs():
            common.jobs_sidebar()
        styles = list(generate.STYLES)
        current = common.pref("style", "concise")
        style = st.selectbox("Answer style", styles, index=styles.index(current) if current in styles else 0)
        common.set_pref("style", style)
        modes = list(common.THEMES)
        mode = st.segmented_control("Theme", modes, default=common.pref("theme", "System"), key="theme_mode",
                                    format_func={"Light": "☀️ Light", "Dark": "🌙 Dark", "System": "💻 Auto"}.get)
        if mode and mode != common.pref("theme", "System"):
            common.set_pref("theme", mode)
            common.apply_theme(mode)              # before the rerun: its first message carries the theme
            st.rerun()
        st.caption(f"🟢 {st.session_state['model']} · runs on this computer")
    return pages


if "setup" not in st.session_state or not st.session_state.setup["ready"]:
    st.session_state.setup = setup_check.check()
if not st.session_state.setup["ready"]:
    setup_panel(st.session_state.setup)
    st.stop()

st.session_state["model"] = st.session_state.setup["llm"]
common.warm_up()
common.collect_jobs()                             # results of background jobs reach their pages
sidebar().run()
