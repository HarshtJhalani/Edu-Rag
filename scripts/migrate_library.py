"""One-off move from the old flat layout (data/ingested, data/pdfs, data/chunks, ...) to data/library + data/index.
Only moves files; source ids stay the same, so the embedding cache and eval results stay valid.
Run from the project root:  rp\\Scripts\\python.exe scripts\\migrate_library.py"""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from edu_rag import config, library  # noqa: E402

D = config.DATA_DIR
OLD = {"ingested": D / "ingested", "pdfs": D / "pdfs", "chunks": D / "chunks", "bm25": D / "bm25",
       "embed_cache": D / "embed_cache", "chroma": D / "chroma", "web_cache": D / "web_cache"}
config.ensure_dirs()

# 1. sources: old JSON + its PDF -> one folder per source
for f in sorted(OLD["ingested"].glob("*.json")):
    doc = json.loads(f.read_text(encoding="utf-8"))
    pdf = OLD["pdfs"] / doc["role"] / doc["name"]
    if pdf.exists():
        doc["file"], doc["kind"] = pdf.name, "pdf"
    else:
        doc.setdefault("kind", "web" if doc.get("url") else "note")
    folder = library.new_dir(doc["role"], doc["name"], doc["source_id"])
    if pdf.exists():
        shutil.move(str(pdf), folder / pdf.name)
    (folder / config.SOURCE_FILE).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    f.unlink()                                       # its content now lives in folder/source.json
    print(f"{doc['role']:9} {doc['name'][:55]:55} -> {folder.relative_to(D)}")

# 2. rebuildable index files -> data/index, page cache -> data/cache/web
moves = [(OLD["chunks"] / "chunks.jsonl", config.CHUNKS_FILE), (OLD["chunks"] / "settings.json", config.CHUNK_SETTINGS_FILE),
         (OLD["bm25"] / "bm25.pkl", config.BM25_FILE)]
for src, dst in moves:
    if src.exists():
        shutil.move(str(src), dst)
for src, dst in [(OLD["embed_cache"], config.CACHE_DIR), (OLD["chroma"], config.CHROMA_DIR), (OLD["web_cache"], config.WEB_CACHE_DIR)]:
    if src.exists():
        for item in src.iterdir():
            shutil.move(str(item), dst / item.name)

# 3. remove the old folders only if they are now empty
for p in sorted(OLD.values(), key=lambda p: -len(p.parts)):
    for sub in sorted(p.rglob("*"), key=lambda s: -len(s.parts)) if p.exists() else []:
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    if p.exists() and not any(p.iterdir()):
        p.rmdir()
    elif p.exists():
        print("left in place (not empty):", p, [x.name for x in p.iterdir()][:5])
print("done")
