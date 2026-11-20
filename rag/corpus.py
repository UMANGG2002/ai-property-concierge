"""Build the unified corpus from both sources.

Writes rag/index/:
  records.jsonl   one line per record (154 = 74 DarGlobal + 80 Wasalt), full unified schema
  chunks.jsonl    what gets embedded (text + flat filter metadata + parent_id + chunk_type + content_hash)
  parents.json    id -> {title, url, full_text, summary, metadata}; read by the API to expand hits to whole records
  corpus_report.json  counts, validation issues, flags, chunk size health
The report fails loudly (exit code != 0 in rag/build.py) if any expected record is missing or invalid.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from .paths import CHUNKS, PARENTS, RECORDS
from .schema import PROPERTY_KINDS, filter_metadata, validate_record

EXPECTED = {"darglobal": 74, "wasalt": 80}


def build_corpus(data_dir: str | Path = "data", wasalt_csv: str | Path = "data/wasalt_riyadh_sale.csv",
                 out_dir: str | Path = "rag/index") -> dict:
    from .sources.darglobal import load_darglobal          # build-time only
    from .sources.wasalt_csv import load_wasalt_csv
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    dg_records, dg_chunks, dg_rep = load_darglobal(data_dir)
    ws_records, ws_chunks = load_wasalt_csv(wasalt_csv)
    records, chunks = dg_records + ws_records, dg_chunks + ws_chunks
    duplicates = dedupe(records, chunks)
    chunks = [c for c in chunks if c["metadata"]["parent_id"] not in duplicates]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    parents = {r["id"]: {"title": r["title"], "url": r["url"], "summary": r.get("summary") or "",
                         "full_text": r.get("full_text") or "", "images": r.get("images", [])[:12],
                         "metadata": filter_metadata(r), "extra": r.get("extra", {}),
                         "quality_flags": r.get("quality_flags", []), "duplicate_of": r.get("duplicate_of"),
                         "rich": r.get("rich") or {}} for r in records}
    (out_dir / "parents.json").write_text(json.dumps(parents, ensure_ascii=False), encoding="utf-8")

    issues = {r["id"]: validate_record(r) for r in records if validate_record(r)}
    by_source = Counter(r["source"] for r in records)
    words = [len(c["text"].split()) for c in chunks]
    ids = Counter(c["id"] for c in chunks)
    orphan = [c["id"] for c in chunks if c["metadata"].get("parent_id") not in parents]
    report = {
        "records": len(records), "chunks": len(chunks), "by_source": dict(by_source),
        "expected": EXPECTED, "missing": {s: n - by_source.get(s, 0) for s, n in EXPECTED.items() if by_source.get(s, 0) < n},
        "by_kind": dict(Counter(r["record_kind"] for r in records)),
        "property_records": sum(1 for r in records if r["record_kind"] in PROPERTY_KINDS),
        "property_types": dict(Counter(t for r in records if r["record_kind"] in PROPERTY_KINDS for t in r["property_types"])),
        "countries": dict(Counter(r["country"] for r in records if r["record_kind"] in PROPERTY_KINDS)),
        "chunk_types": dict(Counter(c["metadata"]["chunk_type"] for c in chunks)),
        "chunk_words": {"min": min(words), "median": statistics.median(words), "max": max(words),
                        "over_500": sum(w > 500 for w in words)},
        "duplicate_chunk_ids": [i for i, n in ids.items() if n > 1], "orphan_chunks": orphan,
        "validation_issues": issues,
        "flags": dict(Counter(f for r in records for f in r.get("quality_flags", []))),
        "darglobal": dg_rep,
        "duplicates_collapsed": duplicates,
        "parents_json_bytes": (out_dir / "parents.json").stat().st_size,
    }
    report["ok"] = not report["missing"] and not report["duplicate_chunk_ids"] and not orphan and not issues
    (out_dir / "corpus_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def dedupe(records: list[dict], chunks: list[dict]) -> dict[str, str]:
    """Property pages with identical content (the same project published at two URLs) keep both records, but only the
    canonical one (in the site menu, else shortest id) is indexed; the other gets `duplicate_of` + its URL as an alias."""
    import hashlib
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["record_kind"] in PROPERTY_KINDS and r.get("full_text"):
            key = hashlib.sha1(" ".join(r["full_text"].split()).encode()).hexdigest()
            groups.setdefault(key, []).append(r)
    dup: dict[str, str] = {}
    for rs in groups.values():
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: (not r.get("extra", {}).get("in_menu"), len(r["id"]), r["id"]))
        canon = rs[0]
        for other in rs[1:]:
            other["duplicate_of"] = canon["id"]
            other["quality_flags"].append(f"duplicate_of:{canon['id']}")
            canon.setdefault("extra", {}).setdefault("alias_urls", []).append(other["url"])
            dup[other["id"]] = canon["id"]
    return dup


def load_chunks(path: str | Path = CHUNKS) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_records(path: str | Path = RECORDS) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_parents(path: str | Path = PARENTS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
