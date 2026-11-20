"""Unified property schema shared by DarGlobal and Wasalt records.

A *record* is one property-level document (a development project, a one-of-one residence, a single resale listing, or an
information page). A *chunk* is what gets embedded; every chunk carries the record's filterable metadata plus
`parent_id` and `chunk_type`. Filterable fields are flat scalars or lists of strings so they map 1:1 onto Upstash
metadata filters (and onto the local fallback store).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

RECORD_KINDS = ("development_project", "one_of_one", "unit_listing", "partner", "info_page", "article")
PROPERTY_KINDS = ("development_project", "one_of_one", "unit_listing")

# canonical property types -> words that indicate them (checked in order: most specific first)
PROPERTY_TYPES: dict[str, tuple[str, ...]] = {
    "penthouse": ("penthouse",),
    "townhouse": ("townhouse", "town house", "town-house"),
    "duplex": ("duplex",),
    "mansion": ("mansion",),
    "hotel_room": ("hotel room", "hotel rooms", "rooms & suites", "rooms and suites", "suite", "suites", "hotel apartment"),
    "villa": ("villa", "villas", "فيلا"),
    "apartment": ("apartment", "apartments", "flat", "flats", "شقة", "residence", "residences", "condo",
                  "residential tower", "residential complex", "branded residential"),
    "floor": ("floor",),
    "land": ("land", "plot", "plots", "أرض", "plots"),
    "building": ("building", "عمارة"),
    "studio": ("studio",),
}

# fields every record has (None when unknown). Kept flat for filtering.
FILTER_FIELDS = (
    "source", "record_kind", "purpose", "property_type", "property_types", "usage",
    "price_amount", "price_min", "price_max", "currency", "price_per_sqm", "price_published",
    "bedrooms_min", "bedrooms_max", "bathrooms", "built_up_area_min", "built_up_area_max", "land_area_sqm",
    "country", "city", "zone", "district", "community", "parent_project",
    "status", "building_year", "furnishing", "facade", "street_width_m",
    "partner", "developer", "unit_category", "has_images",
)


def norm_property_type(text: str | None) -> str | None:
    t = f" {(text or '').lower()} "
    for canon, words in PROPERTY_TYPES.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?:e?s)?(?![a-z])", t):
                return canon
    return None


def all_property_types(text: str | None) -> list[str]:
    t = f" {(text or '').lower()} "
    out = []
    for canon, words in PROPERTY_TYPES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(w)}(?:e?s)?(?![a-z])", t) for w in words):
            out.append(canon)
    return out


def to_float(v: Any) -> float | None:
    """'112.59 SQM' -> 112.59, '12,000 /sqm' -> 12000, 7547 -> 7547.0, '5+' -> 5, garbage -> None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(v))
    return float(m.group().replace(",", "")) if m else None


def to_int(v: Any) -> int | None:
    f = to_float(v)
    return int(f) if f is not None else None


WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def bedrooms_from_text(text: str | None) -> tuple[int | None, int | None]:
    """'five-bedroom villas', '3 & 4 Bedroom', '2-5 bedrooms', '1 to 4 BR' -> (min, max)."""
    t = (text or "").lower()
    nums = []
    for m in re.finditer(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:-|&|and|to|,)?\s*"
                         r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*[- ]?(?:bed(?:room)?s?|br\b)", t):
        for g in m.groups():
            if g:
                nums.append(int(g) if g.isdigit() else WORD_NUM[g])
    nums = [n for n in nums if 0 < n <= 20]
    return (min(nums), max(nums)) if nums else (None, None)


def iso_date(v: Any) -> str | None:
    s = str(v or "")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else None


def content_hash(text: str, meta: dict) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(repr(sorted((k, str(v)) for k, v in meta.items() if k != "content_hash")).encode("utf-8"))
    return h.hexdigest()[:16]


def empty_record() -> dict:
    r = {k: None for k in FILTER_FIELDS}
    r.update({"id": None, "title": None, "url": None, "summary": "", "full_text": "", "images": [], "amenities": [],
              "extra": {}, "quality_flags": [], "fetched_at": None})
    r["property_types"] = []
    r["price_published"] = False
    r["has_images"] = False
    return r


def filter_metadata(record: dict) -> dict:
    """Flat, JSON-safe metadata for the vector store. Lists are kept only as lists of short strings."""
    out = {}
    for k in FILTER_FIELDS + ("id", "title", "url"):
        v = record.get(k)
        if v is None or v == [] or v == "":
            continue
        if isinstance(v, list):
            v = [str(x)[:80] for x in v][:30]
        elif isinstance(v, str):
            v = v[:300]
        out[k] = v
    if record.get("images"):
        out["image"] = record["images"][0]
    if record.get("amenities"):
        out["amenities"] = [str(a)[:60] for a in record["amenities"]][:30]
    return out


def validate_record(r: dict) -> list[str]:
    issues = []
    if not r.get("id"):
        issues.append("missing id")
    if r.get("record_kind") not in RECORD_KINDS:
        issues.append(f"bad record_kind {r.get('record_kind')!r}")
    if not r.get("title"):
        issues.append("missing title")
    if not r.get("url"):
        issues.append("missing url")
    if r.get("record_kind") in PROPERTY_KINDS:
        if not r.get("country"):
            issues.append("missing country")
        if not (r.get("property_type") or r.get("property_types")):
            issues.append("missing property_type")
    if r.get("price_amount") is not None and r["price_amount"] <= 0:
        issues.append("non-positive price")
    return issues
