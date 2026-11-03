"""Router (notebook 06), the adaptive part: classify the message, then pick the retrieval strategy."""
import json
import math
import re
import time
from collections import Counter

from . import config, generate, index, ollama_client, retrieve

ROUTER_TYPES = ["definition", "conceptual", "compare", "followup", "chitchat"]
CHITCHAT_REPLIES = {
    "greeting": "Hi! Ask me anything about your course materials.",
    "thanks": "Glad it helped. Ask another question whenever you like.",
    "bye": "Bye! Good luck with your studies.",
    "unclear": "I couldn't make sense of that. Check the spelling, or ask a question about your course materials.",
    "other": "I can only help with questions about your course materials. Try asking about a topic from your slides.",
}
GREETING = re.compile(r"^(hi+|hello|hey|good (morning|afternoon|evening)|hello there)\W*$", re.I)
THANKS = re.compile(r"^(thanks?|thank you|thx|ok(ay)?,? thanks?)\b", re.I)
BYE = re.compile(r"^(ok(ay)?\W*)?(bye|goodbye|see you|cya)\W*$", re.I)
DEFINITION = re.compile(r"^(what\s+(is|are)|what's|define|meaning of)\s+(an?\s+|the\s+)?(?P<term>[^?]+?)\??$", re.I)
REFERS_BACK = re.compile(r"\b(it|its|this|that|these|those|they|them|one|other|another)\b", re.I)
OPENS_FOLLOWUP = re.compile(r"^\s*(and|but|so|also|then|what about|how about)\b", re.I)
POINTS_BACK = re.compile(r"\b(this|these|those|that|the other|another (way|one|example)|(use|pick|choose|which) one)\b", re.I)
RELATIVE_THAT = re.compile(r"\b(a|an|the)\s+(\S+\s+){1,3}that\b", re.I)        # 'a class that inherits' points at nothing earlier
EARLY_PRONOUN = re.compile(r"^(\S+\s+){0,5}(it|its|they|them)\b", re.I)          # later on, it usually means a noun in the same sentence
ENDS_ON_PRONOUN = re.compile(r"\b(of|for|with|about|on|in|from|by)\s+(it|them)\s*\??\s*$", re.I)      # 'an example of it'
SHORT_REQUEST = re.compile(r"\b(example|examples|code|more|again|further|detail|details|elaborate|simple|simpler|simply|simplify|shorter|summari[sz]e|"
                           r"continue|next|previous|above|earlier|same|first|second|third|last|former|latter)\b", re.I)
SHORT_REQUEST_WORDS = 6
BARE_QUESTION = re.compile(r"^\s*(why|how)(\s+not|\s+so)?\s*\?*\s*$", re.I)
TYPES_OF = re.compile(r"\b(the\s+)?((different|various|main|major|common)\s+)?(types|kinds|sorts|categories|forms)\s+of\s+", re.I)
COMPARE_WORDS = re.compile(r"\b(differ|difference|vs\.?|versus|compare|compared|better than|instead of)\b", re.I)

ROUTER_PROMPT = """You route a student's message in a study assistant for their course materials.
Return JSON only: {"type": ..., "standalone_query": ..., "sub_queries": [...]}

type is one of:
- "definition": asks what a single term or concept is.
- "conceptual": asks how or why something works, steps, uses, problems, or a how-to about code.
- "compare": asks for the difference or relationship between two or more things.
- "followup": depends on the earlier conversation to make sense (uses words like it, that, this, the other, them).
- "chitchat": greetings, small talk, or anything unrelated to studying (weather, jokes, news, the assistant itself).

standalone_query: the message rewritten so it makes sense with no history (replace it/that/this with what they refer to). Keep every name, term and title from the message exactly as written. For chitchat repeat the message.
sub_queries: for "compare", one short standalone question per thing being compared. Otherwise [].
For a followup that is itself a comparison, still use type "followup" and also fill sub_queries."""


_CHARS = {}


def title_vocab(chunks=None):
    """Words of slide titles. Slide text is too broad: it has examples like 'Today it is cloudy'."""
    chunks = chunks if chunks is not None else retrieve.load_state()["chunks"]
    return {t for c in chunks for t in index.tokenize(c["title"])}


def in_materials(term, vocab):
    words = [w for w in index.tokenize(term) if len(w) > 2]
    return bool(words) and all(w in vocab for w in words)


def depends_on_chat(message):
    """Only these see the chat. With history, the LLM rewrote unrelated questions onto the last topic
    ('How is agglomerative clustering different from divisive?' -> 'How is pop different from del?') and answers cited other pages."""
    m = message.strip()
    return bool(OPENS_FOLLOWUP.search(m) or BARE_QUESTION.match(m) or EARLY_PRONOUN.search(m) or ENDS_ON_PRONOUN.search(m) or POINTS_BACK.search(RELATIVE_THAT.sub(" ", m))
                or (len(m.split()) <= SHORT_REQUEST_WORDS and SHORT_REQUEST.search(m)))


def rule_route(message, vocab):
    """Greetings/thanks/bye -> chitchat; short 'what is X' with X in slide titles -> definition; else None (LLM)."""
    m = message.strip()
    for pat, kind in ((GREETING, "greeting"), (THANKS, "thanks"), (BYE, "bye")):
        if pat.search(m):
            return {"type": "chitchat", "chitchat_kind": kind, "standalone_query": m, "sub_queries": []}
    d = DEFINITION.match(m)
    if (d and len(m.split()) <= 8 and not COMPARE_WORDS.search(m) and in_materials(d.group("term"), vocab)
            and not REFERS_BACK.search(d.group("term"))):
        return {"type": "definition", "standalone_query": m, "sub_queries": []}
    return None


def char_model(words):
    """Character-trigram counts over the corpus words: what words in these materials look like."""
    tri, bi = Counter(), Counter()
    for w in words:
        s = f"^^{w}$"
        for i in range(2, len(s)):
            tri[s[i - 2:i + 1]] += 1
            bi[s[i - 2:i]] += 1
    return tri, bi


def corpus_char_model():
    st = retrieve.load_state()
    if _CHARS.get("terms") is not st["terms"]:      # rebuilt whenever the index is reloaded
        words = [w for w in st["terms"]["df"] if re.fullmatch(r"[a-z]{2,20}", w)]
        _CHARS.update(terms=st["terms"], model=char_model(words))
    return st["terms"]["df"], _CHARS["model"]


def word_score(word, model, alphabet=28):
    """Mean log-probability of the word's letters under the trigram model (add-0.1 smoothing)."""
    tri, bi = model
    s = f"^^{word}$"
    lp = [math.log((tri[s[i - 2:i + 1]] + 0.1) / (bi[s[i - 2:i]] + 0.1 * alphabet)) for i in range(2, len(s))]
    return sum(lp) / len(lp)


def is_gibberish(message, known=None, model=None):
    """True for a keyboard mash: no word of the message occurs in the materials, and its letters look unlike them."""
    if known is None or model is None:
        known, model = corpus_char_model()
    words = [w for w in re.findall(r"[a-z]+", message.lower()) if len(w) >= config.GIBBERISH_MIN_LEN]
    if not words or any(w in known for w in words):
        return False
    score = sum(word_score(w, model) * len(w) for w in words) / sum(len(w) for w in words)
    return score < config.GIBBERISH_MAX_SCORE


def history_text(history):
    lines = [f"{m['role']}: {m['content'][:300]}" for m in list(history)[-config.MAX_HISTORY_MESSAGES:]]
    return "\n".join(lines) if lines else "(no earlier conversation)"


def parse_route(raw, message):
    """Validate the router's JSON; on anything unexpected fall back to a conceptual search on the raw message."""
    try:
        out = json.loads(raw)
        assert out.get("type") in ROUTER_TYPES
        out["standalone_query"] = str(out.get("standalone_query") or message).strip()
        out["sub_queries"] = [str(s).strip() for s in (out.get("sub_queries") or []) if str(s).strip()][:3]
        out["parse_ok"] = True
    except (ValueError, AssertionError, AttributeError, TypeError):
        out = {"type": "conceptual", "standalone_query": message, "sub_queries": [], "parse_ok": False}
    return out


def llm_route(message, history=(), model=None):
    msgs = [{"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": f"Earlier conversation:\n{history_text(history)}\n\nMessage: {message}"}]
    raw, stats = ollama_client.chat(msgs, model=model, json_mode=True, temperature=0)
    return dict(parse_route(raw, message), router_seconds=stats["seconds"])


def fix_route(r, message, history, vocab):
    """Guards on the LLM's decision:
    - no history -> nothing to resolve, so search with the student's own words (the LLM once turned
      'What does my note Meow Laadle say?' into 'What does the note say?');
    - 'chitchat' for a message whose words name one of the student's sources is a course question
      (unknown coined names look like small talk to the LLM)."""
    r = dict(r)
    if not history and r["type"] != "compare":
        r["standalone_query"] = message
    if r["type"] == "chitchat" and in_materials(message, vocab):
        r.update(type="definition", standalone_query=message, overridden="chitchat")
    return r


def route(message, history=(), vocab=None, model=None):
    vocab = vocab if vocab is not None else title_vocab()
    r = rule_route(message, vocab)
    if r:
        return dict(r, by="rule", router_seconds=0.0, parse_ok=True)
    if is_gibberish(message):
        return {"type": "chitchat", "chitchat_kind": "unclear", "standalone_query": message, "sub_queries": [],
                "by": "gibberish", "router_seconds": 0.0, "parse_ok": True}
    return dict(fix_route(llm_route(message, history, model), message, history, vocab), by="llm")


def interleave(lists, limit):
    """One from each list in turn, no duplicates: both sides of a comparison reach the LLM."""
    merged, seen = [], set()
    for i in range(max(map(len, lists), default=0)):
        for lst in lists:
            if i < len(lst) and lst[i]["chunk_id"] not in seen:
                merged.append(lst[i])
                seen.add(lst[i]["chunk_id"])
    return merged[:limit]


def singular(word):
    if len(word) < 4 or not word.endswith("s") or word.endswith(("ss", "us", "is")):
        return word
    return word[:-3] + "y" if word.endswith("ies") else word[:-2] if word.endswith(("sses", "shes", "ches", "xes")) else word[:-1]


def list_query(message):
    """What to search for 'What are the different types of loops in Python?': the loops. The words 'types of' found the 'Data Types'
    and 'Function types' slides instead. Singular forms are added because the slides say 'For loop' and word search matches exact words."""
    m = TYPES_OF.search(message)
    if not m or not message[m.end():].strip(" ?"):
        return None
    words = index.tokenize(message[m.end():])
    extra = [singular(w) for w in words if singular(w) != w]
    return " ".join([message[m.end():].strip(" ?")] + extra)


def passage_limit(model):
    return config.FASTER_MAX_PASSAGES if config.CPU_MODE and model == config.LLM_SMALL else 99


def retrieve_for_route(r, roles=None, sources=None, limit=99):
    """definition k=4, conceptual k=8, compare = per sub-query and merged, followup = standalone rewrite; never above limit."""
    if r["type"] == "compare" or (r["type"] == "followup" and len(r["sub_queries"]) >= 2):
        lists, tops = [], []
        for sq in r["sub_queries"] or [r["standalone_query"]]:
            p, top, _ = retrieve.retrieve(sq, k=config.K_PER_SUBQUERY, roles=roles, sources=sources)
            lists.append(p)
            tops.append(top)
        return interleave(lists, min(config.MAX_COMPARE_PASSAGES, limit)), max(tops, default=0.0)
    listed = list_query(r["standalone_query"])
    if listed:
        p, top, _ = retrieve.retrieve(listed, k=min(config.K_LIST, limit), roles=roles, sources=sources)
        return p, top
    p, top, _ = retrieve.retrieve(r["standalone_query"], k=min(config.K_BY_TYPE.get(r["type"], config.K), limit), roles=roles, sources=sources)
    return p, top


def adaptive_answer(message, history=(), style="concise", roles=None, sources=None, model=None, on_token=None, on_route=None):
    """on_route(route) is called once the strategy is known (the app shows it); on_token streams the answer."""
    t0 = time.time()
    history = history if depends_on_chat(message) else ()
    r = route(message, history, model=model)
    if on_route:
        on_route(r)
    if r["type"] == "chitchat":
        res = generate.empty_result(message, style)
        res.update(abstained=True, abstain_reason="chitchat", answer=CHITCHAT_REPLIES[r.get("chitchat_kind", "other")])
    else:
        passages, top = retrieve_for_route(r, roles, sources, passage_limit(model))
        # The answer step sees the standalone question, so follow-ups are answered about the right thing.
        res = generate.generate(r["standalone_query"], passages, top, style, history, model, on_token)
        res["question"] = message
    res["route"] = r
    res["seconds"] = round(time.time() - t0, 2)
    return res
