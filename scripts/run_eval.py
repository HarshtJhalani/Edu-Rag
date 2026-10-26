"""Run the 68-question evaluation through the app's own pipeline (router -> retrieval -> answer) on the CURRENT library.
    python scripts/run_eval.py <output name> [dense|bm25|hybrid]   e.g.  python scripts/run_eval.py eval_hybrid hybrid
Writes data/eval/<output name>.json with a summary and one row per question.

Metrics (answerable questions unless noted):
  hit        a retrieved passage is from the gold source and touches a gold page
  recall     share of gold pages touched by the retrieved passages
  answered   not abstained
  cites_gold the answer cites a passage from the gold source on a gold page
  abstained_una  unanswerable questions that were refused
"""
import csv
import json
import statistics
import sys
import time

from edu_rag import config, library, retrieve, router


def pages_of(c):
    return set(range(c["page_start"], c["page_end"] + 1))


def evaluate(row, sources):
    t0 = time.time()
    res = router.adaptive_answer(row["question"], sources=sources)
    by_id = retrieve.load_state()["by_id"]
    gold = {int(p) for p in row["gold_pages"].split(";") if p.strip().isdigit()}
    retrieved = [by_id[c] for c in res["retrieved"] if c in by_id]
    on_gold = [c for c in retrieved if c["source_name"] == row["gold_source"] and pages_of(c) & gold]
    cited = [by_id[s["chunk_id"]] for s in res["sources"] if res.get("citations_ok") and s.get("chunk_id") in by_id]
    touched = set().union(*(pages_of(c) for c in on_gold)) & gold if on_gold else set()
    return {"q": row["question"], "diff": row.get("difficulty"), "type": row["type"], "ans": row["answerable"] == "yes",
            "abstained": res["abstained"], "why": res["abstain_reason"], "route": res["route"]["type"],
            "hit": bool(on_gold), "recall": len(touched) / len(gold) if gold else 0.0,
            "cites_gold": any(c["source_name"] == row["gold_source"] and pages_of(c) & gold for c in cited),
            "top_dense": res.get("top_dense"), "seconds": round(time.time() - t0, 2)}


def summarise(rows):
    ans = [r for r in rows if r["ans"]]
    una = [r for r in rows if not r["ans"]]
    answered = [r for r in ans if not r["abstained"]]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None
    return {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
            "hit": mean([r["hit"] for r in ans]), "recall": mean([r["recall"] for r in ans]),
            "answered": mean([not r["abstained"] for r in ans]), "abstained_una": mean([r["abstained"] for r in una]),
            "cites_gold": mean([r["cites_gold"] for r in answered]),
            "hard_recall": mean([r["recall"] for r in ans if r["diff"] == "hard"]),
            "seconds": mean([r["seconds"] for r in rows]),
            "evidence_sources": len(library.list_sources()), "passages": len(retrieve.load_state()["chunks"])}


def main(name, method=None):
    if method:
        config.RETRIEVAL_METHOD = method            # retrieve.retrieve reads it on every call
    rows_in = list(csv.DictReader(open(config.EVAL_DIR / "questions.csv", encoding="utf-8")))
    sources = [s["source_id"] for s in library.list_sources() if s["role"] in config.EVIDENCE_ROLES]
    retrieve.load_state(force=True)
    rows = []
    for i, row in enumerate(rows_in, 1):
        rows.append(evaluate(row, sources))
        print(f"[{i}/{len(rows_in)}] hit={rows[-1]['hit']} abst={rows[-1]['abstained']} {row['question'][:60]}", flush=True)
    out = {"summary": summarise(rows) | {"method": config.RETRIEVAL_METHOD}, "rows": rows}
    (config.EVAL_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY", json.dumps(out["summary"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_run", sys.argv[2] if len(sys.argv) > 2 else None)
