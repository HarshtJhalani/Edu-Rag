"""Retrieval (notebooks 04/05): dense, BM25, RRF hybrid; hard role/source filters; reranker for past papers."""
from collections import defaultdict

import numpy as np

from . import chunking, config, index, ollama_client

_STATE = {}                                       # chunks, bm25, collection, reranker


def load_state(force=False):
    """Load chunks, BM25 and the collection once; call reset() after index.sync()."""
    if force or "chunks" not in _STATE:
        chunks = chunking.load_chunks()
        _STATE.update(chunks=chunks, by_id={c["chunk_id"]: c for c in chunks}, bm25=index.load_bm25(),
                      col=index.get_collection(), terms=chunk_terms(chunks))
    return _STATE


def reset():
    for k in ("chunks", "by_id", "bm25", "col", "terms"):
        _STATE.pop(k, None)


def chunk_terms(chunks):
    """{chunk_id: set of words} and document frequency per word, for the rare-word match."""
    sets = {c["chunk_id"]: set(index.tokenize(c["text"])) for c in chunks}
    df = defaultdict(int)
    for s in sets.values():
        for t in s:
            df[t] += 1
    return {"sets": sets, "df": df}


def rare_term_matches(q, roles=None, sources=None, max_df=None):
    """Chunks sharing the most rare query words (words found in 1..max_df chunks), at least two of them,
    or the only one when the query has just one. Dense embeddings miss coined names ("Meow Laadle" scored
    0.36 against its own note); exact words don't. Incidental rare words ("say") don't block a match."""
    st = load_state()
    roles, max_df = roles or config.EVIDENCE_ROLES, max_df or config.RARE_TERM_MAX_DF
    rare = {t for t in index.tokenize(q) if len(t) > 2 and 0 < st["terms"]["df"].get(t, 0) <= max_df}
    if not rare:
        return []
    counts = {cid: len(rare & words) for cid, words in st["terms"]["sets"].items()
              if st["by_id"][cid]["role"] in roles and (not sources or st["by_id"][cid]["source_id"] in sources)}
    best = max(counts.values(), default=0)
    if best < min(2, len(rare)):
        return []
    return [cid for cid, n in counts.items() if n == best]


def embed_query(q, model=None):
    model = model or config.EMBED_MODEL
    return ollama_client.embed([q], model, config.EMBEDDERS.get(model, {}).get("query_prefix", ""))[0]


def where_filter(roles, sources):
    w = {"role": {"$in": list(roles)}}
    return {"$and": [w, {"source_id": {"$in": list(sources)}}]} if sources else w


def dense_search(q, n=None, roles=None, sources=None, qvec=None):
    st = load_state()
    roles = roles or config.EVIDENCE_ROLES
    if st["col"].count() == 0:
        return []
    qvec = embed_query(q) if qvec is None else qvec
    r = st["col"].query(query_embeddings=[qvec.tolist()], n_results=n or config.CANDIDATES,
                        where=where_filter(roles, sources), include=["distances"])
    return [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]


def dense_search_many(queries, n=None, roles=None, sources=None):
    """dense_search for many queries with one embedding call and one Chroma query (syllabus coverage)."""
    st = load_state()
    if not queries or st["col"].count() == 0:
        return [[] for _ in queries]
    model = config.EMBED_MODEL
    qvecs = ollama_client.embed(list(queries), model, config.EMBEDDERS.get(model, {}).get("query_prefix", ""))
    r = st["col"].query(query_embeddings=qvecs.tolist(), n_results=n or config.CANDIDATES,
                        where=where_filter(roles or config.EVIDENCE_ROLES, sources), include=["distances"])
    return [[(i, 1 - d) for i, d in zip(ids, dists)] for ids, dists in zip(r["ids"], r["distances"])]


def bm25_search(q, n=None, roles=None, sources=None):
    st = load_state()
    bm, n, roles = st["bm25"], n or config.CANDIDATES, roles or config.EVIDENCE_ROLES
    if not bm or not bm["bm25"]:
        return []
    scores = bm["bm25"].get_scores(index.tokenize(q))
    out = []
    for i in np.argsort(-scores):
        if scores[i] <= 0:
            break
        c = st["by_id"].get(bm["chunk_ids"][i])
        if c and c["role"] in roles and (not sources or c["source_id"] in sources):
            out.append((c["chunk_id"], float(scores[i])))
            if len(out) == n:
                break
    return out


def rrf(*ranked_lists, k=None, n=None):
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank). Rank-based, no score mixing."""
    k, fused = k or config.RRF_K, defaultdict(float)
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            fused[cid] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:n or config.CANDIDATES]


def retrieve(q, k=None, roles=None, sources=None, method=None):
    """Return (top-k chunks, top score, candidate ids). The score drives the off-topic floor: it is the top dense
    cosine, raised to the floor when a rare-word match exists (an exact name match is never off-topic)."""
    st = load_state()
    method, k = method or config.RETRIEVAL_METHOD, k or config.K
    dense = dense_search(q, roles=roles, sources=sources)
    if method == "dense":
        cands = dense
    elif method == "bm25":
        cands = bm25_search(q, roles=roles, sources=sources)
    else:
        cands = rrf(dense, bm25_search(q, roles=roles, sources=sources))
    ids = [cid for cid, _ in cands]
    top = float(dense[0][1]) if dense else 0.0
    exact = rare_term_matches(q, roles, sources)[:k]
    if exact:                                       # exact name matches go first; the rest keep their order
        ids = exact + [cid for cid in ids if cid not in exact]
        top = max(top, config.DENSE_FLOOR)
    return [st["by_id"][cid] for cid in ids[:k] if cid in st["by_id"]], top, ids


def reranker():
    if "reranker" not in _STATE:
        from sentence_transformers import CrossEncoder
        _STATE["reranker"] = CrossEncoder(config.RERANKER, device="cpu")
    return _STATE["reranker"]


def rerank_scores(q, chunk_ids):
    """Sigmoid scores in 0-1 from the cross-encoder."""
    by_id = load_state()["by_id"]
    if not chunk_ids:
        return []
    return [float(s) for s in reranker().predict([(q, by_id[c]["text"]) for c in chunk_ids], batch_size=16)]


def past_papers(q, sources=None):
    """'Asked in past papers': past-paper questions on the same topic (dense cosine >= PYQ_MIN_DENSE)."""
    st = load_state()
    out = []
    for cid, score in dense_search(q, n=config.PYQ_MAX_HITS, roles=["pyq"], sources=sources):
        if score >= config.PYQ_MIN_DENSE:
            c = st["by_id"][cid]
            out.append({"source": c["source_name"], "page": c["page_start"], "score": round(score, 3),
                        "text": c["text"].split("\n\n", 1)[-1], "chunk_id": cid})
    return out
