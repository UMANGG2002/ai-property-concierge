"""Vector store backends behind one interface.

UpstashStore  - production: Upstash Vector hybrid index (dense + sparse) with an Upstash-hosted embedding model, so
                raw text goes in and out and no embedding model ships with the app.
LocalStore    - BM25 over the same chunks with the same filter semantics. Used for offline development/tests and as the
                automatic fallback if Upstash is unreachable, so the API keeps answering (keyword-only quality).
get_store()   - Upstash when UPSTASH_VECTOR_REST_URL/TOKEN are set, else Local; FallbackStore wraps both at runtime.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import load_chunks
from .paths import CHUNKS
from .filters import Filters, matches, to_upstash

TOKEN = re.compile(r"[a-z0-9]+|[؀-ۿ]+")
STOP = set("a an the of in on at for to and or with is are be by from as it its this that what which who how do does "
           "i me my we you your show find any some there tell about please list give all".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


class SearchError(RuntimeError):
    pass


class LocalStore:
    name = "local-bm25"

    def __init__(self, chunks: list[dict] | None = None, chunks_path: str | Path = CHUNKS):
        self.chunks = chunks if chunks is not None else load_chunks(chunks_path)
        self.docs = []
        for c in self.chunks:
            m = c["metadata"]
            title = (m.get("title") or "") + " "
            self.docs.append(tokenize(title * 2 + c["text"]))        # title counted 3x: project names matter
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.tf = [Counter(d) for d in self.docs]

    def _bm25(self, q: list[str], i: int, k1=1.4, b=0.75) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def search(self, query: str, filters: Filters | None = None, k: int = 20) -> list[dict]:
        q = tokenize(query)
        f = filters or Filters()
        hits = []
        for i, c in enumerate(self.chunks):
            if not matches(c["metadata"], f):
                continue
            s = self._bm25(q, i) if q else 0.0
            hits.append((s, i))
        hits.sort(key=lambda x: -x[0])
        return [{"id": self.chunks[i]["id"], "score": round(s, 4), "text": self.chunks[i]["text"],
                 "metadata": self.chunks[i]["metadata"]} for s, i in hits[:k]]

    def health(self) -> dict:
        return {"store": self.name, "ok": True, "chunks": self.N}


def window_vectors(chunks: list[dict]) -> list[dict]:
    """Chunks -> embedding windows. Window 1 keeps the chunk id; later windows get ~wN. Metadata carries chunk_id."""
    from .embed import MODEL_NAME, windows
    from .schema import content_hash
    out = []
    for c in chunks:
        for n, w in enumerate(windows(c["text"]), 1):
            vid = c["id"] if n == 1 else f"{c['id']}~w{n}"
            meta = {**c["metadata"], "chunk_id": c["id"], "window": n}
            meta["content_hash"] = content_hash(f"{MODEL_NAME}|{w}", c["metadata"])
            out.append({"id": vid, "text": w, "metadata": meta})
    return out


class LocalDense:
    """Brute-force cosine over precomputed window vectors (rag/index/vectors.npy). 2k x 384 floats: ~1 ms/query."""
    name = "local-dense"

    def __init__(self, vectors_path=None, ids_path=None, chunks_by_id: dict | None = None):
        import json
        import numpy as np
        from .paths import CORPUS
        self.vecs = np.load(vectors_path or CORPUS / "vectors.npy")
        meta = json.loads((ids_path or CORPUS / "vectors.json").read_text(encoding="utf-8"))
        self.ids, self.chunk_ids, self.model = meta["ids"], meta["chunk_ids"], meta["model"]
        self.chunks = chunks_by_id or {}

    def search_vector(self, vec: list[float], filters: Filters | None, k: int) -> list[dict]:
        import numpy as np
        f = filters or Filters()
        sims = self.vecs @ np.asarray(vec, dtype=np.float32)
        out = []
        for i in np.argsort(-sims):
            c = self.chunks.get(self.chunk_ids[i])
            if c is None or not matches(c["metadata"], f):
                continue
            out.append({"id": self.ids[i], "chunk_id": self.chunk_ids[i], "score": float(sims[i])})
            if len(out) >= k:
                break
        return out

    def health(self) -> dict:
        return {"store": self.name, "ok": True, "vectors": len(self.ids), "model": self.model}


class UpstashStore:
    """Upstash Vector index in one of two modes, detected from the index itself:
      hosted  - the index has an Upstash-hosted embedding model (e.g. openai/text-embedding-3-small, 1536-d):
                we upsert and query plain text; Upstash embeds both. No model ships with the app.
      vectors - custom-dimension index: we send our own vectors (self-hosted MiniLM, 384-d)."""
    name = "upstash"

    def __init__(self, url: str | None = None, token: str | None = None):
        from upstash_vector import Index
        url = url or os.environ.get("UPSTASH_VECTOR_REST_URL")
        token = token or os.environ.get("UPSTASH_VECTOR_REST_TOKEN")
        if not (url and token):
            raise SearchError("UPSTASH_VECTOR_REST_URL / UPSTASH_VECTOR_REST_TOKEN not set")
        url, token = url.strip().strip('"').strip("'"), token.strip().strip('"').strip("'")
        if not url.startswith("https://"):
            raise SearchError(f"UPSTASH_VECTOR_REST_URL must start with https:// (got {url[:30]!r})")
        self.index = Index(url=url, token=token, retries=1, retry_interval=0.5)
        try:                                     # the SDK default is a 600 s read timeout: fail fast instead
            import httpx
            self.index._client = httpx.Client(timeout=httpx.Timeout(float(os.environ.get("UPSTASH_TIMEOUT", "20")), connect=8.0))
        except Exception:
            pass
        self._mode, self.model, self.sparse_model = None, None, None

    @property
    def mode(self) -> str:
        if self._mode is None:
            forced = os.environ.get("EMBED_PROVIDER", "auto")
            if forced in ("upstash", "hosted"):
                self._mode = "hosted"
            elif forced in ("upstash-hybrid", "hosted-hybrid"):
                self._mode = "hosted-hybrid"
            elif forced in ("local", "vectors"):
                self._mode = "vectors"
            else:
                info = self.index.info()                   # raises if unreachable; caller falls back
                dense = getattr(info, "dense_index", None)
                sparse = getattr(info, "sparse_index", None)
                self.model = getattr(dense, "embedding_model", None) if dense else None
                self.sparse_model = getattr(sparse, "embedding_model", None) if sparse else None
                if self.model and self.sparse_model:
                    self._mode = "hosted-hybrid"           # Upstash runs dense + BM25 and fuses them
                elif self.model or self.sparse_model:
                    self._mode = "hosted"
                else:
                    self._mode = "vectors"
            models = "+".join(m for m in (self.model, getattr(self, "sparse_model", None)) if m)
            self.name = f"upstash-{self._mode}" + (f":{models}" if models else "")
        return self._mode

    def search_text(self, query: str, filters: Filters | None, k: int) -> list[dict]:
        flt = to_upstash(filters or Filters())
        last = None
        for attempt in range(2):
            try:
                res = self.index.query(data=query or "property", top_k=min(k, 100), include_metadata=True, filter=flt)
                return [{"id": str(r.id), "chunk_id": (r.metadata or {}).get("chunk_id") or str(r.id).split("~w")[0],
                         "score": float(r.score)} for r in res]
            except Exception as e:
                last = e
                time.sleep(0.3 * (attempt + 1))
        raise SearchError(f"upstash query failed: {type(last).__name__}: {last}")

    def sync_text(self, items: list[dict], batch: int = 100, log=print) -> dict:
        """Hosted mode: upsert chunk text (Upstash embeds it). Same incremental logic as sync()."""
        from upstash_vector.types import Data

        from .schema import content_hash
        tag = "+".join(m for m in (self.model, self.sparse_model) if m) or "hosted"
        want = {}
        for c in items:
            meta = {**c["metadata"], "chunk_id": c["id"], "window": 1}
            meta["content_hash"] = content_hash(f"{tag}|{c['text']}", c["metadata"])
            want[c["id"]] = {"id": c["id"], "text": c["text"], "metadata": meta}
        have = self.existing()
        todo = [w for wid, w in want.items() if have.get(wid) != w["metadata"]["content_hash"]]
        stale = [wid for wid in have if wid not in want]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            for attempt in range(3):
                try:
                    self.index.upsert(vectors=[Data(id=w["id"], data=w["text"], metadata=w["metadata"]) for w in part])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise SearchError(f"upsert failed at batch {i // batch}: {e}") from e
                    time.sleep(1.5 * (attempt + 1))
            log(f"  upserted {min(i + batch, len(todo))}/{len(todo)}")
        for i in range(0, len(stale), 500):
            self.index.delete(ids=stale[i:i + 500])
        return {"total": len(want), "upserted": len(todo), "unchanged": len(want) - len(todo), "deleted": len(stale)}

    def search_vector(self, vec: list[float], filters: Filters | None, k: int) -> list[dict]:
        flt = to_upstash(filters or Filters())
        last = None
        for attempt in range(2):
            try:
                res = self.index.query(vector=vec, top_k=min(k, 100), include_metadata=True, filter=flt)
                return [{"id": str(r.id), "chunk_id": (r.metadata or {}).get("chunk_id") or str(r.id).split("~w")[0],
                         "score": float(r.score)} for r in res]
            except Exception as e:
                last = e
                time.sleep(0.3 * (attempt + 1))
        raise SearchError(f"upstash query failed: {type(last).__name__}: {last}")

    def health(self) -> dict:
        try:
            info = self.index.info()
            dense = getattr(info, "dense_index", None)
            sparse = getattr(info, "sparse_index", None)
            return {"store": self.name, "ok": True, "vectors": getattr(info, "vector_count", None),
                    "dimension": getattr(info, "dimension", None), "similarity": getattr(info, "similarity_function", None),
                    "embedding_model": getattr(dense, "embedding_model", None) if dense else None,
                    "sparse_model": getattr(sparse, "embedding_model", None) if sparse else None}
        except Exception as e:
            return {"store": self.name, "ok": False, "error": f"{type(e).__name__}: {e}"}

    def existing(self) -> dict[str, str]:
        out, cursor = {}, ""
        while True:
            page = self.index.range(cursor=cursor, limit=1000, include_metadata=True)
            for v in page.vectors:
                out[str(v.id)] = (v.metadata or {}).get("content_hash", "")
            cursor = page.next_cursor
            if not cursor:
                return out

    def sync(self, windows_: list[dict], vectors: dict[str, list[float]], batch: int = 100, log=print) -> dict:
        """Upsert new/changed windows, delete vanished ones. Rerunning with no data change uploads nothing."""
        from upstash_vector.types import Vector
        have = self.existing()
        want = {w["id"]: w for w in windows_}
        todo = [w for wid, w in want.items() if have.get(wid) != w["metadata"]["content_hash"]]
        stale = [wid for wid in have if wid not in want]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            for attempt in range(3):
                try:
                    self.index.upsert(vectors=[Vector(id=w["id"], vector=vectors[w["id"]], metadata=w["metadata"])
                                               for w in part])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise SearchError(f"upsert failed at batch {i // batch}: {e}") from e
                    time.sleep(1.5 * (attempt + 1))
            log(f"  upserted {min(i + batch, len(todo))}/{len(todo)}")
        for i in range(0, len(stale), 500):
            self.index.delete(ids=stale[i:i + 500])
        return {"total": len(want), "upserted": len(todo), "unchanged": len(want) - len(todo), "deleted": len(stale)}


class HybridStore:
    """Dense (Upstash, else bundled local vectors) + sparse (BM25) fused with reciprocal-rank fusion.
    Any dense failure (no model, Upstash down) degrades to BM25 for that query and is reported in last_error."""

    def __init__(self, sparse: LocalStore, embedder, dense_backends: list, rrf_k: int = 60, w_dense: float = 1.0,
                 w_sparse: float = 1.0):
        self.sparse, self.embedder, self.dense = sparse, embedder, dense_backends
        self.rrf_k, self.w_dense, self.w_sparse = rrf_k, w_dense, w_sparse
        self.by_id = {c["id"]: c for c in sparse.chunks}
        self.name = "hybrid(" + "|".join(getattr(d, "name", "dense") for d in dense_backends) + "+bm25)"
        self.last_used, self.last_error = None, None

    def search(self, query: str, filters: Filters | None = None, k: int = 20) -> list[dict]:
        dense, used, used_mode, errs = [], None, None, []
        vec = None
        for backend in self.dense:
            try:
                mode = getattr(backend, "mode", "vectors")
                if mode in ("hosted", "hosted-hybrid"):
                    dense = backend.search_text(query, filters, k * 3)
                else:
                    if vec is None:
                        if self.embedder is None:
                            raise SearchError("no local embedder for a vector index")
                        try:
                            vec = self.embedder.embed_one(query)
                        except Exception as e:
                            errs.append(f"embedder: {str(e)[:160]}")
                            break
                    dense = backend.search_vector(vec, filters, k * 3)
                used, used_mode = backend.name, mode
                break
            except Exception as e:
                errs.append(f"{getattr(backend, 'name', 'dense')}: {str(e)[:160]}")
        self.last_error = "; ".join(errs) or None
        if used_mode == "hosted-hybrid":                  # BM25 + dense already fused inside Upstash
            self.last_used = used
            out, seen = [], set()
            for h in dense:
                cid = h["chunk_id"]
                if cid in seen or cid not in self.by_id:
                    continue
                seen.add(cid)
                c = self.by_id[cid]
                out.append({"id": cid, "score": round(h["score"] * 100, 4), "text": c["text"], "metadata": c["metadata"],
                            "dense": round(h["score"], 4)})
                if len(out) >= k:
                    break
            return out
        sparse = self.sparse.search(query, filters, k * 2)
        self.last_used = f"{used}+bm25" if used else "bm25-only"
        scores: dict[str, float] = defaultdict(float)
        best_dense: dict[str, float] = {}
        seen = set()
        rank = 0
        for h in dense:                                   # windows -> chunks (first occurrence = best window)
            cid = h["chunk_id"]
            if cid in seen or cid not in self.by_id:
                continue
            seen.add(cid)
            scores[cid] += self.w_dense / (self.rrf_k + rank + 1)
            best_dense[cid] = h["score"]
            rank += 1
        for r, h in enumerate(sparse):
            scores[h["id"]] += self.w_sparse / (self.rrf_k + r + 1)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for cid, s_ in ranked:
            c = self.by_id[cid]
            out.append({"id": cid, "score": round(s_ * 100, 4), "text": c["text"], "metadata": c["metadata"],
                        "dense": round(best_dense.get(cid, 0.0), 4)})
        return out

    def health(self) -> dict:
        emb = None if self.embedder is None else (self.embedder.status() if hasattr(self.embedder, "status") else self.embedder.health())
        return {"store": self.name, "ok": True, "embedder": emb, "dense": [d.health() for d in self.dense],
                "sparse": self.sparse.health(), "last_used": self.last_used, "last_error": self.last_error}


class FallbackStore:
    """Primary (Upstash) with automatic fallback to the local store; records which one answered."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+fallback"
        self.last_used = None
        self.last_error = None

    def search(self, query: str, filters: Filters | None = None, k: int = 20) -> list[dict]:
        try:
            hits = self.primary.search(query, filters, k)
            self.last_used, self.last_error = self.primary.name, None
            return hits
        except Exception as e:
            self.last_used, self.last_error = self.fallback.name, str(e)[:300]
            return self.fallback.search(query, filters, k)

    def health(self) -> dict:
        return {"primary": self.primary.health(), "fallback": self.fallback.health()}


def get_store(chunks_path: str | Path = CHUNKS, prefer: str | None = None):
    """RAG_STORE=local -> BM25 only. Otherwise hybrid = dense + BM25 (reciprocal-rank fusion):
      Upstash index with a hosted model (e.g. OpenAI text-embedding-3-small) -> text queries, no local model;
      Upstash custom index -> self-hosted MiniLM vectors, with bundled vectors.npy as fallback;
      no Upstash -> bundled vectors.npy if built; else BM25 only."""
    prefer = prefer or os.environ.get("RAG_STORE", "auto")
    sparse = LocalStore(chunks_path=chunks_path)
    if prefer == "local":
        return sparse
    from .paths import CORPUS
    backends: list = []
    upstash = None
    if os.environ.get("UPSTASH_VECTOR_REST_URL") and os.environ.get("UPSTASH_VECTOR_REST_TOKEN"):
        try:
            upstash = UpstashStore()
            _ = upstash.mode                                 # one info() call at startup
            backends.append(upstash)
        except Exception:
            upstash = None
    if upstash is not None and upstash.mode in ("hosted", "hosted-hybrid"):
        return HybridStore(sparse, None, backends)           # never loads the local model; BM25 local = fallback only
    if upstash is None and not (CORPUS / "vectors.npy").exists():
        return sparse                                         # no vector backend configured: BM25 only, no model needed
    try:
        from .embed import get_embedder
        embedder = get_embedder()
    except Exception:                                        # onnxruntime/tokenizers not installed (lean Vercel build)
        return HybridStore(sparse, None, backends) if backends else sparse
    if (CORPUS / "vectors.npy").exists() and (CORPUS / "vectors.json").exists():
        try:
            ld = LocalDense(chunks_by_id={c["id"]: c for c in sparse.chunks})
            if ld.model == embedder.name:                   # never mix vectors from two different models
                backends.append(ld)
        except Exception:
            pass
    if not backends:
        return sparse
    return HybridStore(sparse, embedder, backends)


def group_by_record(hits: list[dict], max_records: int = 8, chunks_per_record: int = 3) -> list[dict]:
    """Collapse chunk hits into ranked records. Record score = best chunk score + a small bonus for extra matching
    chunks (several sections of one project matching is evidence), so one listing never fills the whole result list."""
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for h in hits:
        pid = h["metadata"].get("parent_id") or h["id"]
        if pid not in groups:
            order.append(pid)
        groups[pid].append(h)
    ranked = []
    for pid in order:
        hs = sorted(groups[pid], key=lambda x: -x["score"])
        score = hs[0]["score"] + 0.1 * sum(x["score"] for x in hs[1:3])
        ranked.append({"record_id": pid, "score": round(score, 4), "chunks": hs[:chunks_per_record],
                       "metadata": hs[0]["metadata"]})
    ranked.sort(key=lambda r: -r["score"])
    return ranked[:max_records]
