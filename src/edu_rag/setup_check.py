"""First-launch checks: Ollama installed and running, GPU memory, models present. Never installs anything by itself."""
import os
import shutil
import subprocess
from pathlib import Path

from . import config, ollama_client

WINGET_CMD = ["winget", "install", "-e", "--id", "Ollama.Ollama"]
APPROX_SIZE_GB = {"qwen2.5:7b-instruct": 4.7, "qwen2.5:3b-instruct": 1.9, "bge-m3": 1.2,
                  "qwen3-vl:4b-instruct-q4_K_M": 3.3}
MIN_VRAM_GB_FOR_7B = 7.0


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
    return config.LLM if vram_gb is None or vram_gb >= MIN_VRAM_GB_FOR_7B else config.LLM_SMALL


def check():
    """Status dict for the app's setup panel. The reranker is evaluation-only, so it is not required here."""
    status = {"ollama_installed": bool(ollama_path()), "ollama_running": ollama_client.is_running(),
              "vram_gb": gpu_vram_gb(), "missing": []}
    status["llm"] = pick_llm(status["vram_gb"])
    needed = [status["llm"], config.EMBED_MODEL, config.VISION_MODEL]
    if status["ollama_running"]:
        installed = ollama_client.installed_models()
        status["missing"] = [m for m in needed if not ollama_client.has_model(m, installed)]
    status["download_gb"] = round(sum(APPROX_SIZE_GB.get(m, 1.0) for m in status["missing"]), 1)
    status["ready"] = status["ollama_running"] and not status["missing"]
    return status
