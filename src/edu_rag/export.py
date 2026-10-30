"""Downloads: a chat as Markdown or PDF (with its citations), and the shared HTML -> PDF writer."""
import io
import re
import time

A4_MARGIN = 40
PDF_CSS = ("body{font-family:sans-serif;font-size:10pt} h1{font-size:16pt} h2{font-size:11.5pt;margin-top:10pt} "
           "li{margin-bottom:2pt} .muted{color:#888} .q{font-weight:bold;margin-top:12pt} .src{font-size:8.5pt;color:#666} "
           "pre{font-family:monospace;font-size:8.5pt;background-color:#f3f3f3;padding:4pt;margin:4pt 0}")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_to_pdf(html, css=PDF_CSS):
    """A printable A4 PDF from simple HTML (PyMuPDF Story). Returns bytes."""
    import pymupdf as fitz
    story = fitz.Story(html=html, user_css=css)
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    rect, more = fitz.paper_rect("a4"), 1
    while more:
        dev = writer.begin_page(rect)
        more, _ = story.place(rect + (A4_MARGIN, A4_MARGIN, -A4_MARGIN, -A4_MARGIN))
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def source_line(s):
    where = s.get("url") or s.get("source", "")
    return f"[{s['label']}] {where}, {s.get('pages', '')}" + (f" ({s['title']})" if s.get("title") else "")


def chat_title(messages, n=60):
    first = next((m["content"] for m in messages if m["role"] == "user"), "Chat")
    return first if len(first) <= n else first[: n - 1] + "…"


def chat_to_markdown(messages, title=None):
    """Questions, answers, each answer's sources, and any Go deeper section."""
    lines = [f"# {title or chat_title(messages)}", f"*Exported from Edu-RAG on {time.strftime('%Y-%m-%d %H:%M')}*", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
            continue
        res = m.get("result") or {}
        lines += [m.get("content") or res.get("answer") or "", ""]
        if res.get("sources"):
            lines += ["**Sources**", ""] + [f"- {source_line(s)}" for s in res["sources"]] + [""]
        d = m.get("deeper")
        if d:
            lines += ["**Beyond your materials (web)**", "", d["answer"], ""]
            lines += [f"- [{s['label']}] {s['url']}" for s in d.get("sources", [])] + [""]
    return "\n".join(lines)


FENCE = re.compile(r"```[\w+-]*[ \t]*\n?(.*?)```", re.S)


def md_inline_to_html(text):
    """```code blocks``` (kept as preformatted text), **bold**, `code` and line breaks; everything else escaped."""
    parts, last = [], 0
    for m in FENCE.finditer(text):
        parts.append(_inline(text[last:m.start()]))
        parts.append(f"<pre>{esc(m.group(1).rstrip())}</pre>")
        last = m.end()
    parts.append(_inline(text[last:]))
    return "".join(parts)


def _inline(text):
    t = esc(text.strip("\n"))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br/>")


def chat_to_pdf(messages, title=None):
    parts = [f"<h1>{esc(title or chat_title(messages))}</h1>",
             f"<p class='muted'>Exported from Edu-RAG on {time.strftime('%Y-%m-%d %H:%M')}</p>"]
    for m in messages:
        if m["role"] == "user":
            parts.append(f"<p class='q'>Q: {esc(m['content'])}</p>")
            continue
        res = m.get("result") or {}
        parts.append(f"<div>{md_inline_to_html(m.get('content') or res.get('answer') or '')}</div>")
        if res.get("sources"):
            parts.append("<p class='src'>" + "<br/>".join(esc(source_line(s)) for s in res["sources"]) + "</p>")
        d = m.get("deeper")
        if d:
            parts.append(f"<p><i>Beyond your materials (web):</i><br/>{md_inline_to_html(d['answer'])}</p>")
            parts.append("<p class='src'>" + "<br/>".join(esc(f"[{s['label']}] {s['url']}") for s in d.get("sources", [])) + "</p>")
    return html_to_pdf("".join(parts))
