"""Score the RAG system against eval/golden.jsonl.

Per item:   parse accuracy, Hit@1/3/5, MRR, Recall@k, Precision@k (filter precision for list questions), nDCG@k,
            intent / relaxation behaviour, answer checks (must / must-not contain), grounding (unsupported numbers,
            citations), optional LLM-judge faithfulness/relevance/completeness (1-5), and stage latencies.
Aggregates: per category and overall, with pass/fail gates.
Outputs:    eval/eval_report.json (read by the UI and /api/eval) and eval/eval_report.md.

    uv run python -m eval.run_eval                      # retrieval + behaviour, template answers (fast, free, no key)
    uv run python -m eval.run_eval --llm                # + real answers from OpenRouter
    uv run python -m eval.run_eval --llm --judge        # + LLM-as-judge scores
    uv run python -m eval.run_eval --category wasalt_list --limit 10
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rag.graph import resources, run
from rag.llm import LLMUnavailable, get_llm

K = 8
GATES = {"hit@5": 0.85, "mrr": 0.70, "precision@k": 0.90, "parse_accuracy": 0.90, "behaviour_pass": 0.90}
JUDGE = """You grade a real-estate assistant's answer. Given CONTEXT, QUESTION and ANSWER, return JSON only:
{"faithfulness": 1-5 (5 = every claim supported by context), "relevance": 1-5 (5 = directly answers the question),
 "completeness": 1-5 (5 = covers what the context offers for this question), "reason": "<15 words"}"""


def load_golden(path="eval/golden.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _parse_ok(expected: dict | None, got: dict) -> bool | None:
    if not expected:
        return None
    for k, v in expected.items():
        g = got.get(k)
        if isinstance(v, list):
            if not g or not set(v) <= set(g):
                return False
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if not isinstance(g, (int, float)) or abs(g - v) > max(1, 0.01 * abs(v)):
                return False
        elif g != v:
            return False
    return True


def _ndcg(ret: list[str], exp: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(ret[:k]) if r in exp)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(exp))))
    return dcg / idcg if idcg else 0.0


def score_item(it: dict, out: dict, judge: bool) -> dict:
    ret = [p["id"] for p in out.get("properties", [])][:K]
    exp = set(it["expected_ids"])
    is_set = it.get("gt_filters") is not None
    row = {"id": it["id"], "category": it["category"], "query": it["query"], "answer_mode": out.get("answer_mode"),
           "store": out.get("store"), "returned": len(ret), "expected": len(exp)}
    if exp:
        first = next((i for i, r in enumerate(ret) if r in exp), None)
        row["hit@1"] = float(first == 0)
        row["hit@3"] = float(first is not None and first < 3)
        row["hit@5"] = float(first is not None and first < 5)
        row["mrr"] = 0.0 if first is None else 1 / (first + 1)
        row["recall@k"] = len(set(ret) & exp) / min(K, len(exp))
        row["ndcg@k"] = round(_ndcg(ret, exp, K), 4)
        if is_set and ret:
            row["precision@k"] = len(set(ret) & exp) / len(ret)
    elif is_set:                                   # ground truth is empty: the system must not claim matches
        row["precision@k"] = 1.0 if not ret or out.get("relaxed") else 0.0
    row["parse_ok"] = _parse_ok(it.get("expected_parse"), (out.get("understood") or {}).get("filters", {}))
    beh = []
    if it.get("expect_intent"):
        beh.append((out.get("understood") or {}).get("intent") == it["expect_intent"] or out.get("answer_mode") == it["expect_intent"])
    if it.get("expect_relaxed") is True:
        beh.append(bool(out.get("relaxed")))
    elif isinstance(it.get("expect_relaxed"), list):
        beh.append(all(any(x in r for r in out.get("relaxed") or []) for x in it["expect_relaxed"]))
    ans = (out.get("answer") or "").lower()
    must = [m for m in it.get("answer_must_contain") or []]
    if must:
        beh_ans = all(m.lower() in ans for m in must)
        row["answer_contains_ok"] = float(beh_ans)
    if it.get("answer_must_not_contain"):
        row["answer_excludes_ok"] = float(not any(m.lower() in ans for m in it["answer_must_not_contain"]))
    if beh:
        row["behaviour_pass"] = float(all(beh))
    g = out.get("grounding") or {}
    if g.get("checked"):
        row["unsupported_numbers"] = len(g.get("unsupported_numbers") or [])
        row["has_citation"] = float(bool(g.get("has_citation")))
    t = out.get("timings") or {}
    for k in ("parse", "retrieve", "generate", "total"):
        if k in t:
            row[f"ms_{k}"] = t[k]
    row["errors"] = "; ".join(out.get("errors") or [])[:300]
    if judge and out.get("answer_mode") == "llm":
        try:
            res = run(it["query"], top_k=K, use_llm=False, debug=True)        # same retrieval -> same context
            j = get_llm().chat_json([{"role": "system", "content": JUDGE},
                                     {"role": "user", "content": f"CONTEXT:\n{res.get('context', '')[:9000]}\n\nQUESTION: {it['query']}\n\nANSWER:\n{out['answer']}"}])
            for k in ("faithfulness", "relevance", "completeness"):
                if isinstance(j.get(k), (int, float)):
                    row[f"judge_{k}"] = float(j[k])
            row["judge_reason"] = str(j.get("reason", ""))[:120]
        except (LLMUnavailable, Exception) as e:
            row["judge_error"] = str(e)[:120]
    return row


METRICS = ["hit@1", "hit@3", "hit@5", "mrr", "recall@k", "precision@k", "ndcg@k", "parse_accuracy", "behaviour_pass",
           "answer_contains_ok", "answer_excludes_ok", "has_citation", "unsupported_numbers",
           "judge_faithfulness", "judge_relevance", "judge_completeness"]


def aggregate(rows: list[dict]) -> dict:
    def agg(rs):
        out = {"n": len(rs)}
        for m in METRICS:
            src = "parse_ok" if m == "parse_accuracy" else m
            vals = [float(r[src]) for r in rs if r.get(src) is not None and r.get(src) != ""]
            if vals:
                out[m] = round(statistics.mean(vals), 4)
                out[f"{m}_n"] = len(vals)
        lat = [r["ms_total"] for r in rs if "ms_total" in r]
        if lat:
            lat.sort()
            out["latency_ms_p50"] = round(statistics.median(lat), 1)
            out["latency_ms_p95"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)
        modes = defaultdict(int)
        for r in rs:
            modes[r.get("answer_mode")] += 1
        out["answer_modes"] = dict(modes)
        return out
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {"overall": agg(rows), "by_category": {c: agg(rs) for c, rs in sorted(by_cat.items())}}


def to_markdown(rep: dict) -> str:
    o = rep["summary"]["overall"]
    L = [f"# RAG evaluation — {rep['run_at']}", "",
         f"Store: **{rep['store']}** · answers: **{rep['answer_source']}** · items: **{o['n']}** · k = {K}", "",
         "## Gates", "", "| Metric | Score | Gate | |", "|---|---|---|---|"]
    for m, th in GATES.items():
        v = o.get(m)
        L.append(f"| {m} | {v if v is not None else '—'} | ≥ {th} | {'✅' if v is not None and v >= th else '❌'} |")
    L += ["", "## By category", "", "| Category | n | Hit@1 | Hit@5 | MRR | Recall@k | Precision@k | Parse | Behaviour | Answer checks | p50 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, a in rep["summary"]["by_category"].items():
        f = lambda k: f"{a[k]:.2f}" if k in a else "—"                                     # noqa: E731
        L.append(f"| {c} | {a['n']} | {f('hit@1')} | {f('hit@5')} | {f('mrr')} | {f('recall@k')} | {f('precision@k')} | "
                 f"{f('parse_accuracy')} | {f('behaviour_pass')} | {f('answer_contains_ok')} | {a.get('latency_ms_p50', '—')} |")
    fails = [r for r in rep["results"] if r.get("hit@5") == 0.0 or r.get("behaviour_pass") == 0.0 or r.get("parse_ok") is False
             or r.get("answer_contains_ok") == 0.0 or (r.get("precision@k") is not None and r["precision@k"] < 0.75)]
    L += ["", f"## Items to look at ({len(fails)})", ""]
    for r in fails[:60]:
        why = [k for k in ("hit@5", "behaviour_pass", "answer_contains_ok") if r.get(k) == 0.0]
        if r.get("parse_ok") is False:
            why.append("parse")
        if r.get("precision@k") is not None and r["precision@k"] < 0.75:
            why.append(f"precision {r['precision@k']:.2f}")
        L.append(f"- `{r['id']}` {r['query']} — {', '.join(why)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from rag.env import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--llm", action="store_true", help="generate real answers with OpenRouter")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge scoring (needs --llm)")
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between items (free-tier rate limits)")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any gate fails")
    a = ap.parse_args(argv)
    items = load_golden(a.golden)
    if a.category:
        items = [i for i in items if i["category"] in a.category]
    if a.limit:
        items = items[:a.limit]
    resources()
    rows = []
    t0 = time.time()
    for n, it in enumerate(items, 1):
        out = run(it["query"], top_k=K, use_llm=a.llm)
        rows.append(score_item(it, out, a.judge and a.llm))
        if n % 25 == 0:
            print(f"  {n}/{len(items)} ...")
        if a.sleep:
            time.sleep(a.sleep)
    summary = aggregate(rows)
    llm = get_llm()
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "k": K, "golden": a.golden,
           "store": resources()["store"].name, "answer_source": "openrouter" if a.llm and llm.configured else "template",
           "models_used": sorted({m for m in [llm.last_model] if m}), "gates": GATES, "seconds": round(time.time() - t0, 1),
           "summary": summary, "results": rows}
    rep["gates_passed"] = {m: (summary["overall"].get(m) is not None and summary["overall"][m] >= th) for m, th in GATES.items()}
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "eval_report.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    (outd / "eval_report.md").write_text(to_markdown(rep), encoding="utf-8")
    o = summary["overall"]
    print(f"\n{o['n']} items in {rep['seconds']}s · store={rep['store']} · answers={rep['answer_source']}")
    for m in ("hit@1", "hit@5", "mrr", "recall@k", "precision@k", "ndcg@k", "parse_accuracy", "behaviour_pass",
              "answer_contains_ok", "judge_faithfulness", "latency_ms_p50", "latency_ms_p95"):
        if m in o:
            gate = GATES.get(m)
            print(f"  {m:22} {o[m]:>8}" + (f"   gate ≥ {gate} {'PASS' if o[m] >= gate else 'FAIL'}" if gate else ""))
    print(f"\nreport: {outd / 'eval_report.md'}")
    return 1 if a.strict and not all(rep["gates_passed"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
