"""Upstash index with a hosted embedding model (e.g. OpenAI text-embedding-3-small): text in, text queries out."""
import types

import pytest

from rag import graph, store as S
from rag.corpus import load_chunks
from rag.filters import Filters


def upstash_filter_ok(meta: dict, flt: str) -> bool:
    """Evaluate the subset of Upstash filter syntax that rag.filters.to_upstash emits (=, IN, >=, <=, CONTAINS, OR groups)."""
    import re
    if not flt:
        return True
    for clause in flt.split(" AND "):
        clause = clause.strip()
        if clause.startswith("("):
            if not any(upstash_filter_ok(meta, c) for c in clause.strip("()").split(" OR ")):
                return False
            continue
        if m := re.match(r"(\w+) CONTAINS '(.*)'$", clause):
            if m.group(2) not in (meta.get(m.group(1)) or []):
                return False
        elif m := re.match(r"(\w+) IN \((.*)\)$", clause):
            if meta.get(m.group(1)) not in re.findall(r"'([^']*)'", m.group(2)):
                return False
        elif m := re.match(r"(\w+) = '(.*)'$", clause):
            if meta.get(m.group(1)) != m.group(2):
                return False
        elif m := re.match(r"(\w+) (>=|<=) ([\d.]+)$", clause):
            v = meta.get(m.group(1))
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False
            if (m.group(2) == ">=" and not v >= float(m.group(3))) or (m.group(2) == "<=" and not v <= float(m.group(3))):
                return False
        else:
            raise AssertionError(f"unexpected clause {clause!r}")
    return True


class FakeHostedIndex:
    """Minimal stand-in for upstash_vector.Index on a hosted-model index (word-overlap scoring)."""

    def __init__(self, model="openai/text-embedding-3-small", dim=1536, sparse_model=None):
        self.model, self.dim, self.sparse_model, self.rows, self.upserts = model, dim, sparse_model, {}, 0

    def info(self):
        dense = types.SimpleNamespace(embedding_model=self.model, dimension=self.dim, similarity_function="COSINE")
        sparse = types.SimpleNamespace(embedding_model=self.sparse_model) if self.sparse_model else None
        return types.SimpleNamespace(vector_count=len(self.rows), dimension=self.dim, similarity_function="COSINE",
                                     dense_index=dense, sparse_index=sparse)

    def upsert(self, vectors):
        for d in vectors:
            assert d.data and not getattr(d, "vector", None)          # hosted mode must send text, not vectors
            self.rows[d.id] = (d.data, d.metadata)
            self.upserts += 1

    def range(self, cursor="", limit=1000, include_metadata=False):
        ids = sorted(self.rows)
        start = int(cursor or 0)
        page = [types.SimpleNamespace(id=i, metadata=self.rows[i][1]) for i in ids[start:start + limit]]
        nxt = str(start + limit) if start + limit < len(ids) else ""
        return types.SimpleNamespace(vectors=page, next_cursor=nxt)

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def query(self, data=None, top_k=10, include_metadata=False, filter=""):
        q = set(data.lower().split())
        rows = {i: v for i, v in self.rows.items() if upstash_filter_ok(v[1], filter)}
        scored = sorted(((len(q & set(t.lower().split())) / (len(q) or 1), i) for i, (t, _) in rows.items()), reverse=True)
        return [types.SimpleNamespace(id=i, score=s, metadata=self.rows[i][1]) for s, i in scored[:top_k] if s > 0]


def hosted_store(fake):
    st = S.UpstashStore.__new__(S.UpstashStore)
    st.index, st._mode, st.model, st.sparse_model, st.name = fake, None, None, None, "upstash"
    return st


def test_mode_detection_and_name():
    st = hosted_store(FakeHostedIndex())
    assert st.mode == "hosted" and st.model == "openai/text-embedding-3-small" and "text-embedding-3-small" in st.name
    st2 = hosted_store(FakeHostedIndex(model=None, dim=384))
    assert st2.mode == "vectors"


def test_sync_text_is_incremental():
    fake = FakeHostedIndex()
    st = hosted_store(fake)
    chunks = load_chunks()[:50]
    r1 = st.sync_text(chunks, log=lambda *_: None)
    assert r1["upserted"] == 50 and fake.upserts == 50
    r2 = st.sync_text(chunks, log=lambda *_: None)
    assert r2["upserted"] == 0 and r2["unchanged"] == 50 and fake.upserts == 50
    r3 = st.sync_text(chunks[:40], log=lambda *_: None)
    assert r3["deleted"] == 10 and len(fake.rows) == 40


def test_hybrid_hosted_needs_no_local_model(monkeypatch):
    fake = FakeHostedIndex()
    st = hosted_store(fake)
    sparse = S.LocalStore()
    st.sync_text(sparse.chunks, log=lambda *_: None)
    hs = S.HybridStore(sparse, None, [st])                           # embedder=None: must still work
    hits = hs.search("infinity pool fitness Les Vagues", Filters(), 10)
    assert hits and hs.last_used.startswith("upstash-hosted") and hs.last_error is None
    assert any(h["metadata"]["parent_id"] == "darglobal:les-vagues" for h in hits)


def test_get_store_hosted_skips_model(monkeypatch):
    fake = FakeHostedIndex()
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "https://example.upstash.io")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "t")
    real_init = S.UpstashStore.__init__

    def fake_init(self, url=None, token=None):
        self.index, self._mode, self.model, self.sparse_model, self.name = fake, None, None, None, "upstash"
    monkeypatch.setattr(S.UpstashStore, "__init__", fake_init)
    import rag.embed as E
    monkeypatch.setattr(E, "get_embedder", lambda: pytest.fail("hosted mode must not load the local model"))
    st = S.get_store()
    assert isinstance(st, S.HybridStore) and st.embedder is None and "upstash-hosted" in st.name
    monkeypatch.setattr(S.UpstashStore, "__init__", real_init)


def test_pipeline_and_gates_through_hosted_store(monkeypatch):
    from eval.run_eval import GATES, aggregate, load_golden, score_item
    fake = FakeHostedIndex()
    st = hosted_store(fake)
    sparse = S.LocalStore()
    st.sync_text(sparse.chunks, log=lambda *_: None)
    monkeypatch.setitem(graph.resources(), "store", S.HybridStore(sparse, None, [st]))
    out = graph.run("What amenities does Les Vagues have?", use_llm=False)
    assert out["properties"][0]["id"] == "darglobal:les-vagues" and out["store"].startswith("upstash-hosted")
    rows = [score_item(it, graph.run(it["query"], top_k=8, use_llm=False), judge=False) for it in load_golden()]
    o = aggregate(rows)["overall"]
    for m, th in GATES.items():
        assert o[m] >= th, (m, o[m])


def test_upstash_down_falls_back_to_bm25(monkeypatch):
    class Down(FakeHostedIndex):
        def query(self, **k):
            raise ConnectionError("timeout")
    st = hosted_store(Down())
    _ = st.mode
    hs = S.HybridStore(S.LocalStore(), None, [st])
    assert hs.search("villas in riyadh", None, 5) and hs.last_used == "bm25-only" and "timeout" in hs.last_error


def test_db_level_hybrid_uses_upstash_ranking_and_skips_local_bm25():
    fake = FakeHostedIndex(sparse_model="bm25")
    st = hosted_store(fake)
    assert st.mode == "hosted-hybrid" and "bm25" in st.name
    sparse = S.LocalStore()
    st.sync_text(sparse.chunks, log=lambda *_: None)
    calls = []
    orig = sparse.search
    sparse.search = lambda *a, **k: calls.append(1) or orig(*a, **k)
    hs = S.HybridStore(sparse, None, [st])
    hits = hs.search("infinity pool fitness Les Vagues", None, 10)
    assert hits and not calls and hs.last_used.startswith("upstash-hosted-hybrid")      # BM25 ran inside Upstash
    assert "darglobal:les-vagues" in [h["metadata"]["parent_id"] for h in hits[:3]]


def test_db_level_hybrid_falls_back_to_local_bm25_when_down():
    class Down(FakeHostedIndex):
        def query(self, **k):
            raise ConnectionError("timeout")
    st = hosted_store(Down(sparse_model="bm25"))
    _ = st.mode
    hs = S.HybridStore(S.LocalStore(), None, [st])
    assert hs.search("villas in riyadh", None, 5) and hs.last_used == "bm25-only"
