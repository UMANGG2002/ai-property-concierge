#!/usr/bin/env python
"""Ask the RAG system from the terminal.

    uv run python -m rag.ask "3 bedroom villa in North Riyadh under 3M"
    uv run python -m rag.ask "What amenities does Marea offer?" --no-llm
    uv run python -m rag.ask "..." --json        # full API-shaped response
    uv run python -m rag.ask "..." --debug       # + context, hits, model attempts
"""
import argparse
import json

from rag.env import load_env


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    from rag.graph import run
    out = run(a.query, top_k=a.k, use_llm=not a.no_llm, debug=a.debug)
    if a.json or a.debug:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print(out["answer"], "\n")
    print(f"[{out['answer_mode']}{' · ' + out['model'] if out.get('model') else ''} · store {out.get('store')} · "
          f"{out.get('total_matches', 0)} matches · {out['timings'].get('total')} ms]")
    if out.get("filters_used"):
        print("filters:", out["filters_used"])
    if out.get("relaxed"):
        print("relaxed:", ", ".join(out["relaxed"]))
    for i, p in enumerate(out["properties"], 1):
        print(f"  {i}. {p['title']} — {p['facts']}\n     {p['url']}")
    if out.get("errors"):
        print("notes:", "; ".join(out["errors"]))


if __name__ == "__main__":
    main()
