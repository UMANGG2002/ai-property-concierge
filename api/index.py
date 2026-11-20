"""FastAPI backend (Vercel Python function at /api/*, or `uvicorn api.index:app` locally / in Docker).

Endpoints
  GET  /api/health          store + LLM + corpus status (never fails)
  POST /api/chat            {query, top_k?, use_llm?, debug?} -> answer, property cards, filters, timings
  POST /api/search          retrieval only (no LLM): property cards + filters
  GET  /api/property/{id}   full record (text, facts, images) for a property card
  GET  /api/stats           corpus report (counts by source/kind/type/country, flags)
  GET  /api/eval            latest evaluation report (summary; ?full=1 for per-item rows)
  POST /api/eval/run        run the golden-set evaluation now (header X-Eval-Token = EVAL_TOKEN)

Robustness: validated input, per-instance answer cache, simple per-IP rate limit, a global exception handler, and a
pipeline (rag.graph.run) that degrades instead of failing: Upstash -> local store, LLM -> grounded template answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.env import load_env  # noqa: E402
from rag.paths import CORPUS  # noqa: E402

load_env()

from fastapi import FastAPI, Header, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

VERSION = "1.0.0"
app = FastAPI(title="Property RAG API (DarGlobal + Wasalt)", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


class ChatIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(8, ge=1, le=20)
    use_llm: bool = True
    debug: bool = False

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("query is empty")
        return v


# ---------------------------------------------------------------- small in-process cache + rate limit
CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()
CACHE_TTL, CACHE_MAX = float(os.environ.get("CACHE_TTL_SECONDS", "3600")), 500
HITS: dict[str, deque] = defaultdict(deque)
RATE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))


def _cache_get(key: str) -> dict | None:
    item = CACHE.get(key)
    if item and time.time() - item[0] < CACHE_TTL:
        CACHE.move_to_end(key)
        return item[1]
    CACHE.pop(key, None)
    return None


def _cache_put(key: str, value: dict) -> None:
    CACHE[key] = (time.time(), value)
    while len(CACHE) > CACHE_MAX:
        CACHE.popitem(last=False)


def _rate_ok(ip: str) -> bool:
    q, now = HITS[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        return False
    q.append(now)
    return True


@app.exception_handler(Exception)
async def _any_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": f"{type(exc).__name__}",
                                                  "answer": "Sorry, something went wrong. Please try again."})


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    out = {"status": "ok", "version": VERSION}
    try:
        from rag.graph import resources
        from rag.llm import get_llm
        res = resources()
        out["store"] = res["store"].health()
        st = res["store"]
        sparse = getattr(st, "sparse", None) or getattr(st, "fallback", None) or st
        out["corpus"] = {"records": len(res["parents"]), "indexed_chunks": getattr(sparse, "N", None)}
        llm = get_llm()
        out["llm"] = {"configured": llm.configured, "models_override": bool(os.environ.get("OPENROUTER_MODELS"))}
    except Exception as e:
        out.update(status="degraded", error=f"{type(e).__name__}: {e}")
    return out


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    from rag.graph import run
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
    if not _rate_ok(ip):
        raise HTTPException(status_code=429, detail="Too many requests, please wait a minute.")
    key = hashlib.sha1(f"{body.query.lower()}|{body.top_k}|{body.use_llm}".encode()).hexdigest()
    if not body.debug and (hit := _cache_get(key)):
        return {**hit, "cached": True}
    out = run(body.query, top_k=body.top_k, use_llm=body.use_llm, debug=body.debug)
    if out.get("answer_mode") not in ("error",) and not body.debug:
        _cache_put(key, out)
    return {**out, "cached": False}


@app.post("/api/search")
def search(body: ChatIn):
    from rag.graph import run
    out = run(body.query, top_k=body.top_k, use_llm=False)
    return {k: out.get(k) for k in ("query", "properties", "total_matches", "filters_used", "relaxed", "understood",
                                    "store", "timings", "errors")}


@app.get("/api/property/{record_id:path}")
def property_detail(record_id: str):
    from rag.graph import facts_line, resources
    p = resources()["parents"].get(record_id)
    if not p:
        raise HTTPException(status_code=404, detail="unknown property id")
    return {"id": record_id, "title": p["title"], "url": p["url"], "facts": facts_line(p["metadata"]),
            "metadata": p["metadata"], "images": p.get("images", []), "summary": p.get("summary", ""),
            "full_text": p.get("full_text", ""), "extra": p.get("extra", {}), "quality_flags": p.get("quality_flags", []),
            "duplicate_of": p.get("duplicate_of"), "rich": p.get("rich") or {}}


@app.get("/api/stats")
def stats():
    path = CORPUS / "corpus_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="corpus report not built")
    rep = json.loads(path.read_text(encoding="utf-8"))
    rep.pop("validation_issues", None)
    return rep


_CHUNKS: list | None = None


def _chunks() -> list:
    global _CHUNKS
    if _CHUNKS is None:
        path = CORPUS / "chunks.jsonl"
        _CHUNKS = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    return _CHUNKS


@app.get("/api/chunks")
def chunks(record: str | None = Query(None, max_length=200)):
    """Chunking explorer: per-type statistics, or every chunk of one record (parent + children)."""
    import statistics
    rows = _chunks()
    if not rows:
        raise HTTPException(status_code=404, detail="chunks not built")
    if record:
        kids = [c for c in rows if (c.get("metadata") or {}).get("parent_id") == record or c["id"].split("#")[0] == record]
        from rag.graph import resources
        p = resources()["parents"].get(record) or {}
        return {"record": record, "title": p.get("title"), "parent_words": len((p.get("full_text") or "").split()),
                "chunks": [{"id": c["id"], "type": (c.get("metadata") or {}).get("chunk_type"), "words": len(c["text"].split()),
                            "text": c["text"], "metadata": {k: v for k, v in (c.get("metadata") or {}).items() if k not in ("content_hash",)}}
                           for c in kids]}
    by: dict = {}
    for c in rows:
        m = c.get("metadata") or {}
        by.setdefault((m.get("source"), m.get("chunk_type") or "other"), []).append(len(c["text"].split()))
    types = [{"source": s, "type": t, "count": len(v), "median_words": int(statistics.median(v)), "max_words": max(v)}
             for (s, t), v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    words = [len(c["text"].split()) for c in rows]
    return {"total": len(rows), "median_words": int(statistics.median(words)), "min_words": min(words), "max_words": max(words), "types": types}


@app.get("/api/eval")
def eval_report(full: bool = Query(False)):
    path = ROOT / "eval" / "eval_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no evaluation report yet; run `uv run python -m eval.run_eval`")
    rep = json.loads(path.read_text(encoding="utf-8"))
    if not full:
        rep.pop("results", None)
    return rep


@app.post("/api/eval/run")
def eval_run(x_eval_token: str | None = Header(None), category: str | None = None, limit: int = 0, llm: bool = False):
    token = os.environ.get("EVAL_TOKEN")
    if not token or x_eval_token != token:
        raise HTTPException(status_code=403, detail="set EVAL_TOKEN and send it as X-Eval-Token")
    from eval.run_eval import aggregate, load_golden, score_item
    from rag.graph import run
    items = load_golden(str(ROOT / "eval" / "golden.jsonl"))
    if category:
        items = [i for i in items if i["category"] == category]
    if limit:
        items = items[:limit]
    t0 = time.time()
    rows = [score_item(it, run(it["query"], top_k=8, use_llm=llm), judge=False) for it in items]
    return {"items": len(rows), "seconds": round(time.time() - t0, 1), "summary": aggregate(rows), "results": rows}


# UI: served from /public by FastAPI locally and in Docker (Vercel serves public/ itself). Mounted last so /api/* wins.
_PUBLIC = ROOT / "public"
if _PUBLIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="ui")
