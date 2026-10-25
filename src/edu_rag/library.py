"""The student's library on disk: data/library/<role>/<source name>/ holds the original file, source.json
(the ingested text) and previews/ (page images, made on demand). Everything under data/index is rebuilt from here."""
import json
import re
import shutil
from pathlib import Path

from . import config

SUMMARY_KEYS = ("source_id", "name", "role", "n_pages", "hash", "file", "url", "kind")


def safe_name(name, max_len=80):
    """A folder name from a source name: no extension, no characters Windows forbids."""
    stem = Path(name).stem if Path(name).suffix.lower() in (".pdf", ".pptx", ".docx", ".ipynb", ".md", ".txt") else name
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", stem)
    return re.sub(r"\s+", " ", stem).strip(" .")[:max_len] or "untitled"


def _all_source_files(root=None):
    return sorted(Path(root or config.LIBRARY_DIR).glob(f"*/*/{config.SOURCE_FILE}"))


def find_dir(source_id, root=None):
    for f in _all_source_files(root):
        if json.loads(f.read_text(encoding="utf-8"))["source_id"] == source_id:
            return f.parent
    return None


def new_dir(role, name, source_id, root=None):
    """Folder for a new source; a name already used by another source gets the id appended."""
    base = Path(root or config.LIBRARY_DIR) / role / safe_name(name)
    existing = base / config.SOURCE_FILE
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["source_id"] != source_id:
        base = base.with_name(f"{base.name} ({source_id[:6]})")
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_source(doc, files=None, root=None):
    """Write source.json plus any original files ({filename: bytes}). Returns the folder."""
    folder = find_dir(doc["source_id"], root) or new_dir(doc["role"], doc["name"], doc["source_id"], root)
    for fname, data in (files or {}).items():
        (folder / fname).write_bytes(data)
    (folder / config.SOURCE_FILE).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return folder


def load_source(source_id, root=None):
    folder = find_dir(source_id, root)
    if folder is None:
        raise KeyError(f"no source {source_id} in the library")
    return json.loads((folder / config.SOURCE_FILE).read_text(encoding="utf-8"))


def list_sources(root=None):
    """Summaries of every source (no page content), with the folder they live in."""
    out = []
    for f in _all_source_files(root):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(dict({k: d.get(k) for k in SUMMARY_KEYS}, folder=str(f.parent)))
    return out


def original_file(source_id, root=None):
    """Path of the uploaded file (PDF, PPTX, ...) for a source, or None for notes and web pages."""
    folder = find_dir(source_id, root)
    doc = json.loads((folder / config.SOURCE_FILE).read_text(encoding="utf-8")) if folder else {}
    p = folder / doc["file"] if folder and doc.get("file") else None
    return p if p and p.exists() else None


def delete_source(source_id, root=None):
    """Remove the source's folder (original file, source.json, previews). Only that folder is touched."""
    folder = find_dir(source_id, root)
    if folder and folder.parent.parent == Path(root or config.LIBRARY_DIR):
        shutil.rmtree(folder)
        return True
    return False
