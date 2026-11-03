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


CPU_MODE_NOTE = ("Answers take longer (about 20 to 30 seconds on an 8-core processor), and scanned pages are read with RapidOCR only, "
                 "so code screenshots and tables can come out less accurately.")
DEVICE_HELP = ("On the processor, answers take from about 20 seconds to over a minute, and scanned pages are read with RapidOCR only. "
               "Files added in that mode stay that way until they are added again.")
CPU_MODEL_HELP = ("Measured on an 8-core processor. Faster: about 20 to 30 seconds per answer. More accurate: about 45 to 75 seconds. Worked examples take longer. "
                  "In the 68-question test, about 90% of answers from the more accurate model cited a correct page, against about 80% from the faster one.")


def set_device(device):
    common.set_pref("device", device)
    setup_check.use_cpu(device == setup_check.CPU)
    ollama_client.unload_all()                    # a loaded model stays where it is until it is loaded again
    st.session_state.setup = setup_check.check()
    st.rerun()


def setup_panel(status):
    """First launch: Ollama installed and running, models present. Never installs anything without a click."""
    st.title("Edu-RAG setup")
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
    if status["cpu_mode"]:
        reason = "No NVIDIA GPU found, so Edu-RAG" if status["vram_gb"] is None else "Edu-RAG"
        st.info(f"{reason} will run on the processor with the smaller answer model **{status['llm']}**. "
                f"{CPU_MODE_NOTE} At least 8 GB of RAM is recommended.")
    else:
        st.info(f"GPU: {status['vram_gb']} GB VRAM → answer model **{status['llm']}**")
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
        if setup_check.can_choose_device(status["vram_gb"]):      # the toggle is out of reach until the app is ready
            other = setup_check.GPU if status["cpu_mode"] else setup_check.CPU
            if st.button(f"Use the {'GPU' if other == setup_check.GPU else 'processor'} instead"):
                set_device(other)


def device_switch():
    """Only on a machine with a GPU: without one, or with EDU_RAG_CPU=1, there is nothing to choose."""
    status = st.session_state.setup
    if not setup_check.can_choose_device(status["vram_gb"]):
        return
    current = setup_check.CPU if status["cpu_mode"] else setup_check.GPU
    choice = st.segmented_control("Run on", [setup_check.GPU, setup_check.CPU], default=current, key="device", help=DEVICE_HELP,
                                  disabled=bool(common.jobs()))
    if choice and choice != current:
        set_device(choice)


def cpu_model_switch():
    """CPU mode only: a smaller, faster answer model or a slower one that cites the right page more often."""
    status = st.session_state.setup
    if not status["cpu_mode"]:
        return
    if not status["fits_7b"]:
        st.caption(f"More accurate answers need the larger model and {setup_check.MIN_RAM_GB_FOR_7B:.0f} GB of RAM; "
                   f"this computer has {status['ram_gb']} GB.")
        return
    choices = [setup_check.FASTER, setup_check.MORE_ACCURATE]
    current = common.pref("cpu_model", setup_check.FASTER)
    choice = st.segmented_control("Answer model", choices, default=current, key="cpu_model", disabled=bool(common.jobs()),
                                  help=CPU_MODEL_HELP)
    if choice and choice != current:
        old = st.session_state["model"]
        common.set_pref("cpu_model", choice)
        if setup_check.answer_model(status, choice) != old and old in ollama_client.loaded_models():
            ollama_client.unload(old)             # never two big models in memory at once
        st.rerun()
    if current == setup_check.MORE_ACCURATE and not status["has_7b"] and not common.running("model_download"):
        st.caption("The more accurate model is not on this computer yet. Answers stay on the faster one until it is.")
        if st.button(f"Download {setup_check.APPROX_SIZE_GB[config.LLM]} GB"):
            def keep(_):
                st.session_state.setup = setup_check.check()
            common.start_job("model_download", f"Downloading {config.LLM}",
                             lambda progress: setup_check.pull_model(config.LLM, progress), keep)
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
        device_switch()
        cpu_model_switch()
        modes = list(common.THEMES)
        mode = st.segmented_control("Theme", modes, default=common.pref("theme", "System"), key="theme_mode",
                                    format_func=lambda m: "Auto" if m == "System" else m)
        if mode and mode != common.pref("theme", "System"):
            common.set_pref("theme", mode)
            common.apply_theme(mode)              # before the rerun: its first message carries the theme
            st.rerun()
        mode = " · CPU mode: slower answers" if st.session_state.setup.get("cpu_mode") else ""
        st.caption(f"{st.session_state['model']} · runs on this computer{mode}")
    return pages


setup_check.use_cpu(common.pref("device", setup_check.GPU) == setup_check.CPU)
if "setup" not in st.session_state or not st.session_state.setup["ready"]:
    st.session_state.setup = setup_check.check()
    if st.session_state.setup["ollama_running"] and not st.session_state.setup["cpu_mode"]:
        ollama_client.unload_off_gpu()
if not st.session_state.setup["ready"]:
    setup_panel(st.session_state.setup)
    st.stop()

common.warm_up()
common.collect_jobs()                             # results of background jobs reach their pages
st.session_state["model"] = setup_check.answer_model(st.session_state.setup, common.pref("cpu_model", setup_check.FASTER))
sidebar().run()
