"""First-launch checks: Ollama installed and running, GPU memory, models present. Never installs anything by itself."""
import os
import shutil
import subprocess
from pathlib import Path

import psutil

from . import config, ollama_client

WINGET_CMD = ["winget", "install", "-e", "--id", "Ollama.Ollama"]
APPROX_SIZE_GB = {"qwen2.5:7b-instruct": 4.7, "qwen2.5:3b-instruct": 1.9, "bge-m3": 1.2,
                  "qwen3-vl:4b-instruct-q4_K_M": 3.3}
MIN_VRAM_GB_FOR_7B = 7.0
MIN_RAM_GB_FOR_7B = 12.0                            # on a CPU, 7B and the embedding model take about 6.2 GB of RAM together
FASTER, MORE_ACCURATE = "Faster", "More accurate"
GPU, CPU = "GPU", "Processor"


def ollama_path():
    found = shutil.which("ollama")
    if found:
        return found
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(default) if default.exists() else None


def start_ollama():
    """Start 'ollama serve' in the background (only called when the user clicks the button)."""
    exe = ollama_path()
    if not exe:
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    return True


def gpu_vram_gb():
    """Total VRAM of the first NVIDIA GPU, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return round(int(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def pick_llm(vram_gb):
    """The big model only with a GPU that holds it. Measured on an 8-core CPU: 3B answers in 17-32 s, 7B in 16-73 s (worked examples longer)."""
    return config.LLM if vram_gb is not None and vram_gb >= MIN_VRAM_GB_FOR_7B else config.LLM_SMALL


def use_cpu(on):
    config.CPU_ONLY = on or config.CPU_FORCED


def can_choose_device(vram_gb):
    return vram_gb is not None and not config.CPU_FORCED


def is_cpu_mode(vram_gb):
    return config.CPU_ONLY or vram_gb is None


def ram_gb():
    return round(psutil.virtual_memory().total / 1024 ** 3, 1)


def answer_model(status, choice):
    """CPU mode: the student's choice, once this machine can run it. With a GPU there is no choice to make."""
    return config.LLM if status["cpu_mode"] and choice == MORE_ACCURATE and status["fits_7b"] and status["has_7b"] else status["llm"]


def pull_model(model, progress):
    """Downloads one model; Ollama reports a failed pull as an event, so it is raised here."""
    for ev in ollama_client.pull(model):
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        if ev.get("total"):
            progress(ev.get("completed", 0), ev["total"], f"{ev.get('completed', 0) * 100 // ev['total']}%")


def check():
    """Status dict for the app's setup panel. The reranker is evaluation-only, so it is not required here."""
    status = {"ollama_installed": bool(ollama_path()), "ollama_running": ollama_client.is_running(),
              "vram_gb": gpu_vram_gb(), "missing": []}
    status["cpu_mode"] = is_cpu_mode(status["vram_gb"])
    config.CPU_MODE = status["cpu_mode"]
    status["ram_gb"] = ram_gb()
    status["fits_7b"] = status["ram_gb"] >= MIN_RAM_GB_FOR_7B
    status["has_7b"] = False
    status["llm"] = pick_llm(None if status["cpu_mode"] else status["vram_gb"])
    needed = [status["llm"], config.EMBED_MODEL]
    config.USE_VISION_OCR = not status["cpu_mode"]  # the vision model takes about 3 minutes per page on a CPU
    if config.USE_VISION_OCR:
        needed.append(config.VISION_MODEL)
    if status["ollama_running"]:
        installed = ollama_client.installed_models()
        status["missing"] = [m for m in needed if not ollama_client.has_model(m, installed)]
        status["has_7b"] = ollama_client.has_model(config.LLM, installed)
    status["download_gb"] = round(sum(APPROX_SIZE_GB.get(m, 1.0) for m in status["missing"]), 1)
    status["ready"] = status["ollama_running"] and not status["missing"]
    return status
