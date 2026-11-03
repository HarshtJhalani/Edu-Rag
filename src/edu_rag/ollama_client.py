"""Thin functions over the Ollama HTTP API."""
import json
import time

import numpy as np
import requests

from . import config


def _url(path):
    return f"{config.OLLAMA_URL}{path}"


def cpu_options():
    """Keeps a model off the GPU when the app runs in CPU mode (Ollama falls back by itself on a machine with no GPU)."""
    return {"num_gpu": 0} if config.CPU_ONLY else {}


def is_running(timeout=3):
    try:
        return requests.get(_url("/api/version"), timeout=timeout).ok
    except requests.RequestException:
        return False


def installed_models():
    r = requests.get(_url("/api/tags"), timeout=10)
    r.raise_for_status()
    return {m["name"] for m in r.json()["models"]}


def has_model(name, installed=None):
    installed = installed if installed is not None else installed_models()
    return name in installed or f"{name}:latest" in installed


def loaded_models():
    return [m["name"] for m in requests.get(_url("/api/ps"), timeout=10).json()["models"]]


def unload(model, wait=5.0):
    """keep_alive=0 through /api/generate works for chat and embedding models alike."""
    requests.post(_url("/api/generate"), json={"model": model, "keep_alive": 0}, timeout=60)
    t0 = time.time()
    while time.time() - t0 < wait:
        if not any(m.split(":")[0] == model.split(":")[0] for m in loaded_models()):
            return True
        time.sleep(0.3)
    return False


def unload_off_gpu():
    """A model loaded on the processor stays there for later GPU requests until it times out, so it is reloaded."""
    for m in requests.get(_url("/api/ps"), timeout=10).json()["models"]:
        if not m.get("size_vram"):
            unload(m["name"])


def unload_all(wait=5.0):
    for m in loaded_models():
        unload(m, wait)


def embed(texts, model=None, prefix="", batch=None, keep_alive="10m"):
    """L2-normalised float32 matrix, one row per text."""
    model = model or config.EMBED_MODEL
    batch = batch or config.EMBED_BATCH
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(_url("/api/embed"), timeout=300,
                          json={"model": model, "input": [prefix + t for t in texts[i:i + batch]], "keep_alive": keep_alive,
                                "options": cpu_options()})
        r.raise_for_status()
        out += r.json()["embeddings"]
    v = np.asarray(out, dtype=np.float32).reshape(len(out), -1)
    return v / np.linalg.norm(v, axis=1, keepdims=True) if len(v) else v


def chat(messages, model=None, json_mode=False, temperature=None, keep_alive="10m", num_predict=None, images=None):
    """Returns (text, stats). num_ctx is always set."""
    options = {"num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE if temperature is None else temperature, **cpu_options()}
    options["num_predict"] = num_predict or (config.JSON_NUM_PREDICT if json_mode else config.ANSWER_NUM_PREDICT)
    body = {"model": model or config.LLM, "messages": messages, "stream": False, "keep_alive": keep_alive, "options": options}
    if json_mode:
        body["format"] = "json"
    t0 = time.time()
    r = requests.post(_url("/api/chat"), timeout=600, json=body)
    r.raise_for_status()
    j = r.json()
    return j["message"]["content"].strip(), {"seconds": round(time.time() - t0, 2),
                                            "prompt_tokens": j.get("prompt_eval_count", 0),
                                            "output_tokens": j.get("eval_count", 0)}


def chat_stream(messages, model=None, temperature=None, keep_alive="10m", on_token=None):
    """Like chat(), but calls on_token(piece) as text arrives. Returns (full text, stats)."""
    options = {"num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE if temperature is None else temperature,
               "num_predict": config.ANSWER_NUM_PREDICT, **cpu_options()}
    body = {"model": model or config.LLM, "messages": messages, "stream": True, "keep_alive": keep_alive, "options": options}
    t0, parts, last = time.time(), [], {}
    with requests.post(_url("/api/chat"), json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            last = json.loads(line)
            piece = last.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                if on_token:
                    on_token(piece)
    return "".join(parts).strip(), {"seconds": round(time.time() - t0, 2), "prompt_tokens": last.get("prompt_eval_count", 0),
                                    "output_tokens": last.get("eval_count", 0)}


def pull(model):
    """Yields progress dicts from /api/pull: {status, completed, total}."""
    with requests.post(_url("/api/pull"), json={"model": model, "stream": True}, stream=True, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                yield json.loads(line)
