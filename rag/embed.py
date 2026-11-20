"""Self-hosted multilingual embeddings (no API, no per-call cost).

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (ONNX build used by FastEmbed:
Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q). 384-dim, ~50 languages incl. Arabic, 128-token input, 0.22 GB.
It is the only multilingual model small enough for a Vercel Python function (500 MB) next to the other dependencies.

The same tiny runner (tokenizers + onnxruntime, mean pooling + L2 norm) embeds documents at ingest time and questions
at query time, so both always live in the same vector space. Long texts are embedded as overlapping ~90-word windows
(see `windows`) because the model only reads 128 tokens.

Model files are looked up in: $EMBED_MODEL_DIR, <repo>/models/<name>, /tmp/models/<name>; if missing they are
downloaded once from Hugging Face into /tmp (Vercel cold start) or <repo>/models (local).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .paths import ROOT

MODEL_NAME = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
MODEL_REPO = os.environ.get("EMBED_MODEL_REPO", "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q")
MODEL_FILE = os.environ.get("EMBED_MODEL_FILE", "model_optimized.onnx")
FILES = (MODEL_FILE, "tokenizer.json")
DIM = int(os.environ.get("EMBED_DIM", "384"))
MAX_TOKENS = 128
WINDOW_WORDS, WINDOW_OVERLAP = 90, 20


class EmbeddingUnavailable(RuntimeError):
    pass


def windows(text: str, max_words: int = WINDOW_WORDS, overlap: int = WINDOW_OVERLAP) -> list[str]:
    """Split text into overlapping windows that fit the 128-token model. The first sentence (the breadcrumb/title
    prefix every chunk starts with) is repeated in each window so every vector knows which property it is about."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    head_end = text.find(". ")
    head = text[:head_end + 1] if 0 < head_end < 160 else ""
    head_words = len(head.split())
    body = words[head_words:] if head else words
    step = max(1, max_words - head_words - overlap)
    size = max_words - head_words
    out = []
    for i in range(0, len(body), step):
        out.append((head + " " if head else "") + " ".join(body[i:i + size]))
        if i + size >= len(body):
            break
    return out


def _candidate_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("EMBED_MODEL_DIR"):
        dirs.append(Path(os.environ["EMBED_MODEL_DIR"]))
    dirs += [ROOT / "models" / MODEL_NAME, Path("/tmp/models") / MODEL_NAME]
    return dirs


def _download(target: Path, timeout: float = 120.0) -> Path:
    import httpx
    target.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = target / f
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{f}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
            if r.status_code != 200:
                raise EmbeddingUnavailable(f"download {f}: HTTP {r.status_code}")
            with open(tmp, "wb") as fh:
                for block in r.iter_bytes(1 << 20):
                    fh.write(block)
        tmp.rename(dest)
    return target


def locate_model(download: bool = True) -> Path:
    for d in _candidate_dirs():
        if all((d / f).exists() for f in FILES):
            return d
    if not download:
        raise EmbeddingUnavailable("model files not found")
    writable = ROOT / "models" / MODEL_NAME
    try:
        writable.parent.mkdir(parents=True, exist_ok=True)
        probe = writable.parent / ".write_test"
        probe.write_text("x")
        probe.unlink()
    except OSError:                                         # read-only deploy (Vercel): use /tmp
        writable = Path("/tmp/models") / MODEL_NAME
    return _download(writable)


class Embedder:
    """Thread-safe, lazily loaded ONNX sentence embedder."""

    def __init__(self, model_dir: str | Path | None = None, download: bool = True, threads: int | None = None):
        self._dir = Path(model_dir) if model_dir else None
        self._threads = threads or int(os.environ.get("EMBED_THREADS", "2"))
        self._download = download
        self._lock = threading.Lock()
        self._session = None
        self._tok = None
        self.load_ms = None
        self.name = MODEL_NAME

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            t0 = time.perf_counter()
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                raise EmbeddingUnavailable(f"missing dependency: {e}") from e
            d = self._dir or locate_model(self._download)
            tok = Tokenizer.from_file(str(d / "tokenizer.json"))
            tok.enable_truncation(max_length=MAX_TOKENS)
            pad_id = tok.token_to_id("<pad>")
            if pad_id is None:
                pad_id = tok.token_to_id("[PAD]") or 0
            tok.enable_padding(pad_id=pad_id, pad_token="<pad>")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self._threads
            self._session = ort.InferenceSession(str(d / MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"])
            self._inputs = {i.name for i in self._session.get_inputs()}
            self._tok = tok
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)

    def embed(self, texts: list[str], batch: int = 32, progress=None) -> list[list[float]]:
        """progress(done, total) is called after each batch. Texts are batched by length (less padding = faster)
        and returned in the original order."""
        import numpy as np
        try:
            self._load()
        except EmbeddingUnavailable:
            raise
        except Exception as e:
            raise EmbeddingUnavailable(f"model load failed: {type(e).__name__}: {e}") from e
        order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
        out_sorted: list[list[float]] = []
        for i in range(0, len(order), batch):
            enc = self._tok.encode_batch([texts[j] or " " for j in order[i:i + batch]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            if hidden.ndim == 3:                             # (batch, seq, dim) -> mean over real tokens
                m = mask[..., None].astype(hidden.dtype)
                pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            else:
                pooled = hidden
            pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out_sorted.extend(pooled.astype(np.float32).tolist())
            if progress:
                progress(len(out_sorted), len(texts))
        out: list[list[float]] = [[] for _ in texts]
        for pos, j in enumerate(order):
            out[j] = out_sorted[pos]
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def health(self) -> dict:
        try:
            self._load()
            return {"model": self.name, "ok": True, "dim": DIM, "load_ms": self.load_ms}
        except Exception as e:
            return {"model": self.name, "ok": False, "error": str(e)[:200]}

    def status(self) -> dict:
        """Cheap: never downloads or loads the model."""
        present = any(all((d / f).exists() for f in FILES) for d in ([self._dir] if self._dir else _candidate_dirs()))
        return {"model": self.name, "ok": True if self._session is not None or present else None, "loaded": self._session is not None,
                "files_present": present, "dim": DIM}


class HashEmbedder:
    """Deterministic stand-in for tests/offline dev only (EMBED_BACKEND=hash). Not semantically meaningful."""
    name = "hash-test"

    def embed(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        import hashlib
        import math
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in (t or "").lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def health(self) -> dict:
        return {"model": self.name, "ok": True, "dim": DIM}

    def status(self) -> dict:
        return self.health()


_EMB = None


def get_embedder():
    global _EMB
    if _EMB is None:
        _EMB = HashEmbedder() if os.environ.get("EMBED_BACKEND") == "hash" else Embedder()
    return _EMB
