#!/usr/bin/env python
"""Build everything the RAG backend needs, then (if configured) sync it to Upstash Vector.

    uv run python -m rag.build                 # sources -> unified corpus -> golden set -> Upstash (if keys set)
    uv run python -m rag.build --reset         # wipe the Upstash index first
    uv run python -m rag.build --no-upload     # build files only

Inputs : data/darglobal/docs/*.json, data/darglobal/one_of_one_collections.json, data/wasalt_riyadh_sale.csv
Outputs: rag/index/{records.jsonl, chunks.jsonl, parents.json, corpus_report.json,
         vectors.npy, vectors.json}, eval/golden.jsonl
Embeddings, chosen by the Upstash index itself:
  index with a hosted model (recommended: OpenAI text-embedding-3-small, 1536, COSINE) -> chunk text is uploaded and
    Upstash embeds it; no local model is downloaded or shipped;
  custom index (Dense, 384, COSINE) or no Upstash -> self-hosted multilingual MiniLM vectors (models/, vectors.npy).
Exit code 2 if the corpus is incomplete (fewer than 74 DarGlobal or 80 Wasalt records) or has validation errors.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from rag.env import load_env
from rag.paths import CORPUS, ROOT


def find_csv(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    p = ROOT / "data" / "wasalt_riyadh_sale.csv"
    if p.exists():
        return p
    found = sorted(ROOT.glob("Wasalt*.csv")) + sorted((ROOT / "data").glob("Wasalt*.csv"))
    if not found:
        sys.exit("No Wasalt CSV found: put it at data/wasalt_riyadh_sale.csv or pass --wasalt-csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(found[0], p)
    print(f"copied {found[0].name} -> data/wasalt_riyadh_sale.csv")
    return p


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--wasalt-csv")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-embed", action="store_true", help="skip embeddings (BM25-only search)")
    a = ap.parse_args()

    from rag.corpus import build_corpus
    rep = build_corpus(a.data_dir, find_csv(a.wasalt_csv), CORPUS)
    print(f"corpus: {rep['records']} records {rep['by_source']} (expected {rep['expected']}), {rep['chunks']} chunks, "
          f"{rep['property_records']} properties; duplicates collapsed: {len(rep['duplicates_collapsed'])}")
    if not rep["ok"]:
        print("CORPUS NOT OK:", json.dumps({k: rep[k] for k in ("missing", "duplicate_chunk_ids", "orphan_chunks",
                                                                 "validation_issues")}, indent=1)[:2000])
        return 2
    from eval.build_golden import OUT, build as build_golden
    items = build_golden()
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"golden set: {len(items)} items -> {OUT}")

    import os
    import time

    from rag.corpus import load_chunks
    corpus = CORPUS
    chunks = load_chunks(corpus / "chunks.jsonl")
    have_upstash = bool(os.environ.get("UPSTASH_VECTOR_REST_URL") and os.environ.get("UPSTASH_VECTOR_REST_TOKEN"))
    store = None
    if have_upstash and not a.no_upload:
        from rag.store import UpstashStore
        try:
            store = UpstashStore()
            info = store.index.info()
        except Exception as e:
            print(f"UPSTASH UNREACHABLE: {type(e).__name__}: {e}\n  -> check UPSTASH_VECTOR_REST_URL / _TOKEN in .env")
            return 4
        dense = getattr(info, "dense_index", None)
        sparse = getattr(info, "sparse_index", None)
        model = getattr(dense, "embedding_model", None) if dense else None
        sparse_model = getattr(sparse, "embedding_model", None) if sparse else None
        kind = "HYBRID (dense + BM25 fused in Upstash)" if model and sparse_model else "dense"
        print(f"upstash index: {kind} · dimension={info.dimension} similarity={info.similarity_function} "
              f"dense_model={model or 'custom (bring your own vectors)'} sparse_model={sparse_model or '-'} "
              f"vectors={info.vector_count}")
        if model or sparse_model:                            # hosted embeddings: send text, Upstash embeds it
            if a.reset:
                store.index.reset()
                print("upstash index reset")
            print(f"uploading {len(chunks)} chunks as text; Upstash embeds them with "
                  f"{' + '.join(m for m in (model, sparse_model) if m)} ...", flush=True)
            t0 = time.time()
            res = store.sync_text(chunks)
            print(f"upstash sync: {res} in {time.time() - t0:.0f}s")
            print(f"upstash health: {store.health()}")
            return 0
    if a.no_embed:
        print("embedding skipped (--no-embed): the API will use BM25 keyword search only")
        return 0

    import numpy as np

    from rag.embed import DIM, Embedder, EmbeddingUnavailable, HashEmbedder
    from rag.store import window_vectors
    wins = window_vectors(chunks)
    emb = HashEmbedder() if os.environ.get("EMBED_BACKEND") == "hash" else Embedder(threads=os.cpu_count() or 4)
    t0 = time.time()
    print(f"embedding {len(wins)} windows with {emb.name} ({os.cpu_count()} threads; first run downloads ~220 MB)...", flush=True)

    def progress(done: int, total: int) -> None:
        if done == total or done % 256 < 32:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{total} windows · {rate:.0f}/s · ~{(total - done) / max(rate, 1e-6):.0f}s left", flush=True)
    try:
        vecs = emb.embed([w["text"] for w in wins], batch=32, progress=progress)
    except EmbeddingUnavailable as e:
        print(f"EMBEDDING FAILED: {e}\n  -> the API still works with BM25 only; fix the model download and rerun.")
        return 3
    arr = np.asarray(vecs, dtype=np.float32)
    np.save(corpus / "vectors.npy", arr)
    (corpus / "vectors.json").write_text(json.dumps({"model": emb.name, "dim": int(arr.shape[1]),
                                                     "ids": [w["id"] for w in wins],
                                                     "chunk_ids": [w["metadata"]["chunk_id"] for w in wins]}), encoding="utf-8")
    print(f"embedded {len(wins)} windows in {time.time() - t0:.1f}s -> {corpus / 'vectors.npy'} ({arr.nbytes / 1e6:.1f} MB)")
    if store is None:
        print("Upstash not used: the API will run hybrid search on the bundled vectors (rag/index/vectors.npy).")
        return 0
    if info.dimension != DIM:
        print(f"UPSTASH INDEX MISMATCH: custom index has dimension {info.dimension}, the local model produces {DIM}.\n"
              "  -> either create an index with an Upstash-hosted model (e.g. OpenAI text-embedding-3-small) or a custom "
              "Dense index with dimension 384, COSINE")
        return 5
    if a.reset:
        store.index.reset()
        print("upstash index reset")
    res = store.sync(wins, {w["id"]: v for w, v in zip(wins, vecs)})
    print(f"upstash sync: {res}")
    print(f"upstash health: {store.health()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
