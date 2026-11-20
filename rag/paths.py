"""Repo-relative paths so the code works from any working directory (CLI, tests, Vercel's /var/task)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("RAG_CORPUS_DIR") or ROOT / "rag" / "index")
CHUNKS = CORPUS / "chunks.jsonl"
PARENTS = CORPUS / "parents.json"
RECORDS = CORPUS / "records.jsonl"
