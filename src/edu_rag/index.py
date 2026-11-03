"""Indexes (notebook 03): embedding cache, Chroma collection, BM25. sync() keeps all three in line with data/ingested."""
import hashlib
import os
import pickle
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import chunking, config, ingest, library, ollama_client

META_KEYS = ["chunk_id", "source_id", "source_name", "role", "page_start", "page_end",
             "title", "has_code", "has_table", "n_tokens", "doc_hash", "url", "unit"]
STOPWORDS = set("""a an the of to in on for and or is are was were be been it its this that these those with as by at from
into than then so if not no do does did can could will would should may might what which who whom how why when where
there their they them we you your our i he she his her""".split())
_STATE = {}                                       # chroma client


def tokenize(text):
    """Lowercase words and identifiers; read_csv and __init__ stay one token."""
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOPWORDS]


def model_slug(model):
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def collection_name(model=None):
    return f"chunks_{model_slug(model or config.EMBED_MODEL)}"


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_path(model, doc_hash, settings_hash):
    """Key = source file hash + embed model + chunk settings, so changed settings never reuse stale vectors."""
    return Path(config.CACHE_DIR) / model_slug(model) / f"{doc_hash[:16]}_{settings_hash}.npz"


def embed_chunks(chunks, model=None):
    """Return ({chunk_id: vector}, counts), embedding only what the cache doesn't have."""
    model = model or config.EMBED_MODEL
    prefix = config.EMBEDDERS.get(model, {}).get("doc_prefix", "")
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[(c["doc_hash"], c["chunk_settings"])].append(c)
    vectors, stats = {}, Counter()
    for (doc_hash, settings_hash), cs in by_doc.items():
        path = cache_path(model, doc_hash, settings_hash)
        cached = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            cached = {(i, h): v for i, h, v in zip(z["ids"], z["text_hashes"], z["vectors"])}
        hashes = [text_hash(c["text"]) for c in cs]
        todo = [(c, h) for c, h in zip(cs, hashes) if (c["chunk_id"], h) not in cached]
        if todo:
            new = ollama_client.embed([c["text"] for c, _ in todo], model, prefix)
            cached.update({(c["chunk_id"], h): v for (c, h), v in zip(todo, new)})
        stats["embedded"] += len(todo)
        stats["from_cache"] += len(cs) - len(todo)
        vecs = np.stack([cached[(c["chunk_id"], h)] for c, h in zip(cs, hashes)])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array([c["chunk_id"] for c in cs]), text_hashes=np.array(hashes), vectors=vecs)
        vectors.update({c["chunk_id"]: v for c, v in zip(cs, vecs)})
    return vectors, stats


def chroma_client():
    if "client" not in _STATE:
        import chromadb
        _STATE["client"] = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _STATE["client"]


def get_collection(model=None):
    """The model's collection. Rebuilt empty if it was made with other chunk settings."""
    client = chroma_client()
    name = collection_name(model)
    settings_hash = config.chunk_settings_hash()
    if name in {c.name for c in client.list_collections()}:
        col = client.get_collection(name)
        if col.metadata.get("chunk_settings") == settings_hash:
            return col
        client.delete_collection(name)
    return client.create_collection(name, metadata={"hnsw:space": "cosine", "embed_model": model or config.EMBED_MODEL,
                                                    "chunk_settings": settings_hash})


def chunk_metadata(c):
    return {k: (c.get(k) if c.get(k) is not None else "") for k in META_KEYS}


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    data = {"bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None,
            "chunk_ids": [c["chunk_id"] for c in chunks], "chunk_settings": config.chunk_settings_hash()}
    Path(config.BM25_FILE).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(config.BM25_FILE).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, config.BM25_FILE)                # all at once: index_stamp changes only when the file is complete
    return data


def index_stamp():
    """Changes whenever sync() finishes: BM25 is the last thing it writes, so chunks and Chroma are already updated."""
    try:
        return os.stat(config.BM25_FILE).st_mtime_ns
    except OSError:
        return None


def load_bm25():
    if not Path(config.BM25_FILE).exists():
        return None
    with open(config.BM25_FILE, "rb") as f:
        return pickle.load(f)


def chunk_all_docs():
    """Chunk every source in the library. Web sources keep their URL on each chunk."""
    chunks, report = [], {}
    for d in library.list_sources():
        doc = library.load_source(d["source_id"])
        log = []
        cs, dropped = chunking.chunk_doc(doc, log)
        for c in cs:
            c["url"] = doc.get("url", "")
            c["unit"] = doc.get("unit", "p.")
        chunks += cs
        report[doc["name"]] = {"chunks": len(cs), "dropped_slides": len(dropped), "ocr_faults_removed": len(log)}
    return chunks, report


def sync(model=None):
    """Bring chunks.jsonl, the Chroma collection and BM25 in line with data/library. Only new chunks are embedded."""
    model = model or config.EMBED_MODEL
    config.ensure_dirs()
    chunks, report = chunk_all_docs()
    chunking.save_chunks(chunks)
    vectors, stats = embed_chunks(chunks, model)
    col = get_collection(model)
    ids = {c["chunk_id"] for c in chunks}
    stale = set(col.get(include=[])["ids"]) - ids
    if stale:
        col.delete(ids=list(stale))
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.upsert(ids=[c["chunk_id"] for c in part], embeddings=np.stack([vectors[c["chunk_id"]] for c in part]).tolist(),
                   documents=[c["text"] for c in part], metadatas=[chunk_metadata(c) for c in part])
    build_bm25(chunks)
    return {"chunks": len(chunks), "stale_removed": len(stale), **stats, "sources": report}


def add_file(name, data, role="textbook", progress=None, llm=None):
    """Store an uploaded file in its own library folder, ingest it (OCR as needed) and update the indexes.
    The LLM is unloaded first so the vision model has the GPU to itself; the vision model is unloaded after."""
    fname = Path(name).name
    if not data:
        raise ValueError("this file is empty (0 bytes). Download or export it again, then add the new copy.")
    folder = library.new_dir(role, fname, hashlib.sha1(data).hexdigest()[:12])
    (folder / fname).write_bytes(data)
    ollama_client.unload(llm or config.LLM)
    try:
        doc = ingest.ingest_file(folder / fname, role=role, progress=progress)
    except Exception:
        shutil.rmtree(folder)                       # don't leave a half-made source behind
        raise
    ollama_client.unload(config.VISION_MODEL)
    doc["file"] = fname
    library.save_source(doc)
    from . import previews
    previews.prepare(doc["source_id"])            # PowerPoint/Word -> PDF copy for slide previews (no-op otherwise)
    return doc, sync()


def add_source(doc, files=None, model=None):
    library.save_source(doc, files)
    return sync(model)


def remove_source(source_id, model=None):
    """Remove a source's folder from the library (original file included) and from the indexes."""
    library.delete_source(source_id)
    return sync(model)
