"""Structured search filters: one model, two renderings (Upstash filter string, Python predicate).

Range semantics: records carry ranges (a project with 2-4 bedrooms, a listing with exactly 3), requests carry ranges
too ("at least 3 beds", "under 2M"). A record matches when the ranges overlap:
  request bedrooms_min  -> record.bedrooms_max >= value
  request bedrooms_max  -> record.bedrooms_min <= value
  request price_max     -> record.price_min <= value      (records with no published price never match a price filter)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Filters(BaseModel):
    sources: list[str] = Field(default_factory=list)            # darglobal / wasalt
    record_kinds: list[str] = Field(default_factory=list)
    purpose: str | None = None
    property_types: list[str] = Field(default_factory=list)
    bedrooms_min: int | None = None
    bedrooms_max: int | None = None
    price_min: float | None = None
    price_max: float | None = None
    area_min: float | None = None
    area_max: float | None = None
    country: str | None = None
    city: str | None = None
    zones: list[str] = Field(default_factory=list)
    districts: list[str] = Field(default_factory=list)
    partner: str | None = None
    parent_ids: list[str] = Field(default_factory=list)          # explicit project / listing lookup

    @field_validator("bedrooms_min", "bedrooms_max")
    @classmethod
    def _beds(cls, v):
        return None if v is None or v < 0 or v > 50 else v

    @field_validator("price_min", "price_max", "area_min", "area_max")
    @classmethod
    def _pos(cls, v):
        return None if v is None or v <= 0 else v

    def is_empty(self) -> bool:
        return not any(v not in (None, [], "") for v in self.model_dump().values())

    def active(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v not in (None, [], "")}


def _q(v: Any) -> str:
    s = str(v).replace("'", "").replace('"', "")
    return f"'{s}'"


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def to_upstash(f: Filters) -> str:
    parts: list[str] = []

    def any_of(field: str, values: list[str], op: str = "=") -> None:
        vals = [v for v in values if v]
        if not vals:
            return
        if op == "CONTAINS":
            parts.append("(" + " OR ".join(f"{field} CONTAINS {_q(v)}" for v in vals) + ")")
        elif len(vals) == 1:
            parts.append(f"{field} = {_q(vals[0])}")
        else:
            parts.append(f"{field} IN (" + ", ".join(_q(v) for v in vals) + ")")

    any_of("source", f.sources)
    any_of("record_kind", f.record_kinds)
    any_of("parent_id", f.parent_ids)
    if f.purpose:
        parts.append(f"purpose = {_q(f.purpose)}")
    any_of("property_types", f.property_types, "CONTAINS")
    if f.bedrooms_min is not None:
        parts.append(f"bedrooms_max >= {f.bedrooms_min}")
    if f.bedrooms_max is not None:
        parts.append(f"bedrooms_min <= {f.bedrooms_max}")
    if f.price_max is not None:
        parts.append(f"price_min <= {_num(f.price_max)}")
    if f.price_min is not None:
        parts.append(f"price_max >= {_num(f.price_min)}")
    if f.area_max is not None:
        parts.append(f"built_up_area_min <= {_num(f.area_max)}")
    if f.area_min is not None:
        parts.append(f"built_up_area_max >= {_num(f.area_min)}")
    for field in ("country", "city", "partner"):
        if getattr(f, field):
            parts.append(f"{field} = {_q(getattr(f, field))}")
    any_of("zone", f.zones)
    any_of("district", f.districts)
    return " AND ".join(parts)


def matches(meta: dict, f: Filters) -> bool:
    """Python twin of to_upstash() for the local store and for computing golden ground truth."""
    def num(k):
        v = meta.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    if f.sources and meta.get("source") not in f.sources:
        return False
    if f.record_kinds and meta.get("record_kind") not in f.record_kinds:
        return False
    if f.parent_ids and (meta.get("parent_id") or meta.get("id")) not in f.parent_ids:
        return False
    if f.purpose and meta.get("purpose") != f.purpose:
        return False
    if f.property_types and not set(f.property_types) & set(meta.get("property_types") or []):
        return False
    if f.bedrooms_min is not None and not (num("bedrooms_max") is not None and num("bedrooms_max") >= f.bedrooms_min):
        return False
    if f.bedrooms_max is not None and not (num("bedrooms_min") is not None and num("bedrooms_min") <= f.bedrooms_max):
        return False
    if f.price_max is not None and not (num("price_min") is not None and num("price_min") <= f.price_max):
        return False
    if f.price_min is not None and not (num("price_max") is not None and num("price_max") >= f.price_min):
        return False
    if f.area_max is not None and not (num("built_up_area_min") is not None and num("built_up_area_min") <= f.area_max):
        return False
    if f.area_min is not None and not (num("built_up_area_max") is not None and num("built_up_area_max") >= f.area_min):
        return False
    for field in ("country", "city", "partner"):
        if getattr(f, field) and meta.get(field) != getattr(f, field):
            return False
    if f.zones and meta.get("zone") not in f.zones:
        return False
    if f.districts and meta.get("district") not in f.districts:
        return False
    return True
