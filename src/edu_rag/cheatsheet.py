"""Cheat sheets: short revision notes for a whole source or a topic, every bullet tied to a page.
The LLM writes bullets citing [S#]; code turns each citation into the real page from chunk metadata."""
import re
import time

from . import chunking, config, export, generate, ollama_client, retrieve

BATCH_TOKENS = 3000                               # passages per LLM call; leaves room in num_ctx for the answer
TOPIC_K = 8
PROMPT = """You write a student's revision cheat sheet from numbered passages of their course materials.
Rules:
- Write short bullet points ("- ..."): definitions, formulas, key steps, differences, and at most one tiny code example per group.
- End every bullet with the passage it comes from, like [S2].
- Use only facts from the passages. No introductions, no conclusions, no headings.
- At most {max_bullets} bullets. Plain text maths, no LaTeX."""


def batches(chunks, budget=BATCH_TOKENS):
    """Consecutive chunks grouped up to the token budget; slide order is kept."""
    out, cur, size = [], [], 0
    for c in chunks:
        if cur and size + c["n_tokens"] > budget:
            out.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += c["n_tokens"]
    if cur:
        out.append(cur)
    return out


def source_chunks(source_id):
    return [c for c in chunking.load_chunks() if c["source_id"] == source_id]


def topic_chunks(topic, sources=None, k=TOPIC_K):
    passages, _, _ = retrieve.retrieve(topic, k=k, sources=sources)
    return sorted(passages, key=lambda c: (c["source_name"], c["page_start"]))


def bullets_from(text, passages):
    """Parse '- bullet [S2]' lines; replace citations with page references from metadata. Uncited bullets are dropped."""
    cite = generate.cite_pattern()
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if not m:
            continue
        body = m.group(1).strip()
        nums = [x for g in cite.findall(body) for x in generate.cite_numbers(g) if 1 <= x <= len(passages)]
        if not nums:
            continue
        refs = sorted({(passages[n - 1]["source_name"], generate.page_label(passages[n - 1])) for n in nums})
        clean = re.sub(r"\s*[,;]?\s*([.,;:])(\s*[.,;:])*\s*$", r"\1", re.sub(r"\s+([.,;:])", r"\1", cite.sub("", body))).strip(" ;,")
        support = generate.support_check(f"{clean} [S1]", [{"text": " ".join(passages[n - 1]["text"] for n in nums)}])
        first = passages[nums[0] - 1]
        out.append({"text": clean, "refs": refs, "chunk_ids": [passages[n - 1]["chunk_id"] for n in nums],
                    "weak": bool(support) and support[0]["flag"] == "weak support",
                    "section": (first["source_name"], first["title"]), "order": (first["source_name"], first["page_start"]),
                    "first": first})
    return out


def group_sections(bullets):
    """One section per cited slide title, in slide order; the page range covers the bullets in it."""
    sections = {}
    for b in sorted(bullets, key=lambda b: b["order"]):
        s = sections.setdefault(b["section"], {"title": b["section"][1], "source": b["section"][0], "chunks": [], "bullets": []})
        s["chunks"].append(b.pop("first"))
        b.pop("order")
        s["bullets"].append(b)
    out = []
    for s in sections.values():
        lo = min(c["page_start"] for c in s["chunks"])
        hi = max(c["page_end"] for c in s["chunks"])
        s["pages"] = generate.page_label(dict(s["chunks"][0], page_start=lo, page_end=hi))
        del s["chunks"]
        out.append(s)
    return out


def make(source_id=None, topic=None, sources=None, model=None, progress=None, max_bullets=6):
    """A cheat sheet for one source (whole deck) or a topic. Returns {title, sections[{title, pages, bullets}], seconds}."""
    t0 = time.time()
    if topic:
        chunks = topic_chunks(topic, sources)
        title = f"Cheat sheet: {topic}"
    else:
        chunks = source_chunks(source_id)
        title = f"Cheat sheet: {chunks[0]['source_name']}" if chunks else "Cheat sheet"
    groups = batches(chunks)
    bullets = []
    for i, group in enumerate(groups):
        n_bullets = max(max_bullets, len({c["title"] for c in group}))      # long decks: about one bullet per slide title
        msgs = [{"role": "system", "content": PROMPT.format(max_bullets=n_bullets)},
                {"role": "user", "content": f"Passages:\n{generate.format_passages(group)}\n\nWrite the cheat sheet bullets."}]
        text, _ = ollama_client.chat(msgs, model=model, temperature=0.1)
        bullets += bullets_from(text, group)
        if progress:
            progress(i + 1, len(groups))
    return {"title": title, "sections": group_sections(bullets), "seconds": round(time.time() - t0, 1)}


def to_markdown(sheet):
    lines = [f"# {sheet['title']}", ""]
    for s in sheet["sections"]:
        lines += [f"## {s['title']}  ({s['source']}, {s['pages']})", ""]
        for b in s["bullets"]:
            refs = "; ".join(p for _, p in b["refs"])
            lines.append(f"- {b['text']} *({refs})*")
        lines.append("")
    return "\n".join(lines)


def to_pdf(sheet):
    """A printable A4 PDF. Returns bytes."""
    esc = export.esc
    parts = [f"<h1>{esc(sheet['title'])}</h1>"]
    for s in sheet["sections"]:
        parts.append(f"<h2>{esc(s['title'])} <span style='color:#888;font-size:9pt'>({esc(s['source'])}, {s['pages']})</span></h2><ul>")
        for b in s["bullets"]:
            refs = "; ".join(p for _, p in b["refs"])
            parts.append(f"<li>{esc(b['text'])} <i style='color:#888'>({refs})</i></li>")
        parts.append("</ul>")
    return export.html_to_pdf("".join(parts))
