"""Embeddings, hybrid retrieval, question-aware highlights and the static UI."""
import json

import pytest

from rag import graph
from rag.embed import HashEmbedder, windows
from rag.store import HybridStore, LocalStore, window_vectors


def test_windows_fit_the_model_and_keep_the_title():
    text = "DarGlobal > Oman > Fairway Villas > Overview. " + " ".join(f"w{i}" for i in range(300))
    ws = windows(text)
    assert len(ws) > 1 and all(len(w.split()) <= 90 for w in ws)
    assert all(w.startswith("DarGlobal > Oman > Fairway Villas > Overview.") for w in ws)
    assert windows("short text") == ["short text"]


def test_window_vectors_ids_and_hashes():
    chunks = [{"id": "x#1", "text": "T. " + " ".join(f"w{i}" for i in range(200)), "metadata": {"parent_id": "x"}}]
    ws = window_vectors(chunks)
    assert ws[0]["id"] == "x#1" and ws[1]["id"] == "x#1~w2"
    assert all(w["metadata"]["chunk_id"] == "x#1" and w["metadata"]["content_hash"] for w in ws)
    assert len({w["metadata"]["content_hash"] for w in ws}) == len(ws)


def test_onnx_embedder_runner(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers
    from rag.embed import Embedder
    vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate("villa riyadh pool apartment".split())}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    table = np.random.default_rng(0).normal(size=(len(vocab), 384)).astype(np.float32)
    ins = [helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "s"]) for n in ("input_ids", "attention_mask")]
    out = helper.make_tensor_value_info("h", TensorProto.FLOAT, ["b", "s", 384])
    g = helper.make_graph([helper.make_node("Gather", ["t", "input_ids"], ["h"])], "g", ins, [out], [numpy_helper.from_array(table, "t")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(tmp_path / "model_optimized.onnx"))
    e = Embedder(tmp_path)
    v = np.array(e.embed(["villa riyadh", "villa riyadh pool", "apartment"]))
    assert v.shape == (3, 384) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert np.allclose(e.embed(["villa riyadh"])[0], v[0], atol=1e-5)          # padding does not change a vector


class _Dense:
    name = "fake-dense"

    def __init__(self, ids, fail=False):
        self.ids, self.fail = ids, fail

    def search_vector(self, vec, filters, k):
        if self.fail:
            raise RuntimeError("down")
        return [{"id": i, "chunk_id": i.split("~w")[0], "score": 1 - n / 10} for n, i in enumerate(self.ids)]

    def health(self):
        return {"store": self.name, "ok": not self.fail}


def test_hybrid_fuses_dense_and_sparse():
    sparse = LocalStore()
    target = next(c["id"] for c in sparse.chunks if c["metadata"]["parent_id"] == "darglobal:marea")
    hs = HybridStore(sparse, HashEmbedder(), [_Dense([f"{target}~w2", target])])
    hits = hs.search("golf course villas oman", None, 10)
    assert target in [h["id"] for h in hits] and hs.last_used == "fake-dense+bm25"
    assert [h["id"] for h in hits].count(target) == 1                          # windows collapse to one chunk


def test_hybrid_falls_back_across_dense_backends_and_to_bm25():
    sparse = LocalStore()
    hs = HybridStore(sparse, HashEmbedder(), [_Dense([], fail=True), _Dense([sparse.chunks[0]["id"]])])
    hs.search("villa", None, 5)
    assert hs.last_used == "fake-dense+bm25" and "down" in hs.last_error

    class NoModel:
        def embed_one(self, q):
            raise RuntimeError("model missing")
    hs2 = HybridStore(sparse, NoModel(), [_Dense([sparse.chunks[0]["id"]])])
    assert hs2.search("villas in riyadh", None, 5) and hs2.last_used == "bm25-only" and "model missing" in hs2.last_error


def test_vectors_from_another_model_are_ignored(tmp_path, monkeypatch):
    import numpy as np
    from rag import store
    from rag.paths import CORPUS
    if (CORPUS / "vectors.json").exists():
        pytest.skip("real vectors present")
    monkeypatch.setattr("rag.paths.CORPUS", tmp_path)
    np.save(tmp_path / "vectors.npy", np.zeros((1, 384), dtype=np.float32))
    (tmp_path / "vectors.json").write_text(json.dumps({"model": "some-other-model", "ids": ["a"], "chunk_ids": ["a"]}))
    monkeypatch.setenv("EMBED_BACKEND", "hash")
    monkeypatch.setattr("rag.embed._EMB", None)
    assert store.get_store().name == "local-bm25"


@pytest.mark.parametrize("q,topic", [("Why should I invest in Marea?", "invest"), ("What amenities does Les Vagues have?", "amenities"),
                                     ("where is Trump Tower Jeddah located?", "location"), ("show me photos of the Pagani penthouse", "gallery"),
                                     ("what is the payment plan for DG1", "payment")])
def test_topics(q, topic):
    assert topic in graph.detect_topics(q)


def test_question_aware_highlights():
    inv = graph.run("Why should I invest in Marea?", use_llm=False)["properties"][0]
    assert inv["images"] and inv["highlights"].get("why_invest", {}).get("bullets")
    amen = graph.run("What amenities does Les Vagues have?", use_llm=False)["properties"][0]
    assert amen["highlights"].get("amenities") and amen["highlights"].get("features")
    assert any(f.get("image") for f in amen["highlights"]["features"])
    one = graph.run("show me photos of the Pagani penthouse", use_llm=False)["properties"][0]
    assert one["image_count"] > 1 and one["highlights"].get("sections")
    ws = graph.run("3 bedroom apartment in East Riyadh under 1M", use_llm=False)["properties"][0]
    assert ws["source"] == "wasalt" and ws["images"] == [] and ws["highlights"]["licence"]["rega_ad_license"]


def test_ui_is_served_and_api_wins():
    from fastapi.testclient import TestClient
    from api.index import app
    c = TestClient(app)
    for path, ctype in (("/", "text/html"), ("/about.html", "text/html"), ("/chat.js", "javascript"), ("/styles.css", "text/css")):
        r = c.get(path)
        assert r.status_code == 200 and ctype in r.headers["content-type"], path
    assert c.get("/api/health").headers["content-type"].startswith("application/json")
