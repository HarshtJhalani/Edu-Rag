"""Go deeper (notebook 07): opt-in web search -> fetch -> chunk/embed/retrieve -> answer with [W#] and URLs."""
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

from . import config, generate, index, ingest, library, ollama_client, retrieve

NOTICE = "Go deeper sends your question to web search (Wikipedia, DuckDuckGo) and downloads the top pages. It leaves your computer."
OFFLINE_LABEL = "AI-generated, not verified"
QUESTION_WORDS = set("how do does did i can could should would what which why when explain tell me about way ways "
                     "use using difference between".split())
WEB_SYSTEM_PROMPT = (generate.SYSTEM_PROMPT.replace("[S1]", "[W1]").replace("[S2][S3]", "[W2][W3]").replace("[S2]", "[W2]")
                     .replace("numbered passages from their course materials", "numbered passages from web pages"))
OFFLINE_PROMPT = ("You are a study assistant. The student's course materials don't cover this and the web is unavailable. "
                  "Give a short general explanation from your own knowledge. Say nothing about sources.")


# --- search -------------------------------------------------------------------

def domain_of(url):
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def allowlisted(url):
    d = domain_of(url)
    return any(d == a or d.endswith("." + a) for a in config.ALLOWLIST)


def site_key(url):
    """Independent-site key: en.wikipedia.org and de.wikipedia.org are one site."""
    d = domain_of(url)
    return next((a for a in config.ALLOWLIST if d == a or d.endswith("." + a)), d)


def keywords(query):
    """Wikipedia's search matches all words, so a full question finds odd pages."""
    return " ".join(w for w in index.tokenize(query) if w not in QUESTION_WORDS)


def wikipedia_search(query, n=3):
    r = requests.get("https://en.wikipedia.org/w/api.php", timeout=15, headers={"User-Agent": config.USER_AGENT},
                     params={"action": "query", "list": "search", "srsearch": keywords(query), "format": "json", "srlimit": n})
    r.raise_for_status()
    return [{"url": "https://en.wikipedia.org/wiki/" + s["title"].replace(" ", "_"), "title": s["title"], "engine": "wikipedia"}
            for s in r.json()["query"]["search"]]


def ddgs_search(query, n=None):
    from ddgs import DDGS
    last = None
    for backend in config.DDGS_BACKENDS:
        try:
            res = DDGS(timeout=10).text(query, max_results=n or config.N_SEARCH, backend=backend)
            if res:
                return [{"url": x["href"], "title": x.get("title", ""), "engine": f"ddgs/{backend}"} for x in res]
        except Exception as e:                      # ddgs raises its own exception types per backend
            last = e
    raise RuntimeError(f"no results from {config.DDGS_BACKENDS}: {str(last)[:60]}")


def rank_hits(hits):
    """De-duplicate, allowlisted sites first, at most one page per site."""
    seen, uniq = set(), []
    for h in hits:
        u = h["url"].split("#")[0]
        if u not in seen:
            seen.add(u)
            uniq.append(dict(h, url=u))
    picked, sites = [], set()
    for h in sorted(uniq, key=lambda h: not allowlisted(h["url"])):
        if site_key(h["url"]) not in sites:
            picked.append(h)
            sites.add(site_key(h["url"]))
    return picked


def web_search(query):
    hits, errors = [], []
    for fn in (wikipedia_search, ddgs_search):
        try:
            hits += fn(query)
        except Exception as e:
            errors.append(f"{fn.__name__}: {str(e)[:80]}")
    return rank_hits(hits), errors


# --- fetch --------------------------------------------------------------------

def fetch_page(url):
    import trafilatura
    path = Path(config.WEB_CACHE_DIR) / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")), cached=True)
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": config.USER_AGENT})
        r.raise_for_status()
        # Raw bytes: trafilatura detects the encoding. r.text falls back to ISO-8859-1 when the server sends
        # no charset, which turned "language’s" into "languageâs" (coursera.org).
        text = trafilatura.extract(r.content, include_tables=True, include_comments=False, favor_precision=True) or ""
        meta = trafilatura.extract_metadata(r.content)
        page = {"url": url, "title": meta.title if meta and meta.title else url, "text": text[:config.MAX_PAGE_CHARS], "ok": bool(text)}
    except Exception as e:
        page = {"url": url, "title": url, "text": "", "ok": False, "error": str(e)[:120]}
    if page["ok"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    return dict(page, cached=False)


def fetch_top(hits, n=None):
    pages = []
    for h in hits:
        p = fetch_page(h["url"])
        if p["ok"] and len(p["text"]) > 300:
            pages.append(p)
        if len(pages) == (n or config.N_PAGES):
            break
    return pages


# --- chunk / retrieve ---------------------------------------------------------

def chunk_page(page):
    paras = [p.strip() for p in page["text"].split("\n") if len(p.strip()) > 40]
    groups, cur = [], []
    budget = config.WEB_MAX_TOKENS - round(len(page["title"]) / 4) - 2
    for p in paras:
        if cur and round(len("\n".join(cur + [p])) / 4) > budget:
            groups.append(cur)
            cur = []
        cur.append(p[: budget * 4])
    if cur:
        groups.append(cur)
    sid = "web-" + hashlib.sha1(page["url"].encode()).hexdigest()[:10]
    return [{"chunk_id": f"{sid}-{i:03d}", "source_id": sid, "source_name": f"{domain_of(page['url'])}: {page['title']}",
             "role": "web", "url": page["url"], "domain": domain_of(page["url"]), "site": site_key(page["url"]),
             "page_start": 1, "page_end": 1, "title": page["title"], "text": f"{page['title']}\n\n" + "\n".join(g)}
            for i, g in enumerate(groups)]


def web_retrieve(query, web_chunks, k=None):
    if not web_chunks:
        return [], []
    prefix = config.EMBEDDERS.get(config.EMBED_MODEL, {}).get("doc_prefix", "")
    vecs = ollama_client.embed([c["text"] for c in web_chunks], config.EMBED_MODEL, prefix)
    sims = vecs @ retrieve.embed_query(query)
    order = np.argsort(-sims)[:k or config.WEB_K]
    return [web_chunks[i] for i in order], [float(sims[i]) for i in order]


# --- answer -------------------------------------------------------------------

def confirmed_sentences(text, passages):
    """Sentences backed (overlap >= SUPPORT_MIN) by cited passages from at least two independent sites."""
    cite = generate.cite_pattern("W")
    out = []
    for s in generate.split_sentences(text, "W"):
        nums = sorted({x for g in cite.findall(s) for x in generate.cite_numbers(g) if 1 <= x <= len(passages)})
        words = generate.content_words(cite.sub("", s))
        backing = {passages[x - 1]["site"] for x in nums
                   if words and len(words & generate.content_words(passages[x - 1]["text"])) / len(words) >= config.SUPPORT_MIN}
        out.append({"sentence": s, "sites": sorted(backing), "confirmed": len(backing) >= 2})
    return out


def group_sources(sources):
    """Several [W#] passages can come from one page; list each page once with all its labels."""
    by_url = {}
    for s in sources:
        by_url.setdefault(s["url"], dict(s, labels=[]))["labels"].append(s["label"])
    return list(by_url.values())


def offline_answer(question, style="concise", model=None):
    return ollama_client.chat([{"role": "system", "content": OFFLINE_PROMPT},
                               {"role": "user", "content": f"Style: {generate.STYLES[style]}\n\nQuestion: {question}"}], model=model)


def go_deeper(question, style="concise", offline=False, model=None):
    """Fixed pipeline, separate 'Beyond your materials' result. Only the question text leaves the machine."""
    t0 = time.time()
    res = {"section": "Beyond your materials", "notice": NOTICE, "question": question, "mode": "web", "label": None,
           "answer": None, "sources": [], "confirmed": [], "badge": False, "search_errors": [], "pages": [],
           "citations_ok": False}
    hits, errors = ([], ["offline"]) if offline else web_search(question)
    res["search_errors"] = errors
    pages = fetch_top(hits) if hits else []
    res["pages"] = [{"url": p["url"], "title": p["title"], "text": p["text"]} for p in pages]
    if not pages:
        text, stats = offline_answer(question, style, model)
        res.update(mode="offline", label=OFFLINE_LABEL, answer=text, llm=stats, seconds=round(time.time() - t0, 2))
        return res
    passages, sims = web_retrieve(question, [c for p in pages for c in chunk_page(p)])
    raw, stats = ollama_client.chat(generate.build_messages(question, passages, style, (), WEB_SYSTEM_PROMPT, "W"), model=model)
    res["llm"] = stats
    if config.INSUFFICIENT in raw:
        text, _ = offline_answer(question, style, model)
        res.update(mode="offline", label=OFFLINE_LABEL + " (the fetched pages did not cover it)", answer=text)
    else:
        text, valid, invalid = generate.parse_citations(raw, len(passages), "W")
        conf = confirmed_sentences(text, passages)
        res.update(answer=text, invalid_citations=invalid, citations_ok=bool(valid), confirmed=conf,
                   badge=any(c["confirmed"] for c in conf),
                   sources=[{"label": f"W{n}", "url": passages[n - 1]["url"], "domain": passages[n - 1]["domain"],
                             "title": passages[n - 1]["title"], "allowlisted": allowlisted(passages[n - 1]["url"]),
                             "text": passages[n - 1]["text"]} for n in valid])
    res["seconds"] = round(time.time() - t0, 2)
    return res


def page_to_doc(page):
    """A fetched page as an ingested source (role=web), so 'Add to my sources' goes through index.add_source."""
    # The heading line gives every chunk the page title; without it the chunker would use the "file name".
    doc = ingest.ingest_text(f"{domain_of(page['url'])}: {page['title']}", f"# {page['title']}\n\n{page['text']}", role="web")
    doc.update(url=page["url"], kind="web")
    return doc


def add_to_sources(res):
    """Store the pages behind a Go deeper answer as role=web sources. Returns the sync report."""
    cited = {s["url"] for s in res.get("sources", [])}
    pages = [p for p in res.get("pages", []) if not cited or p["url"] in cited]
    for p in pages:
        library.save_source(page_to_doc(p))
    report = index.sync()
    retrieve.reset()
    return dict(report, added=len(pages))
