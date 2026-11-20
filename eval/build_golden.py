"""Build the golden evaluation set from the corpus (deterministic; seed fixed).

Ground truth is computed from the data, never typed by hand, so it is exact:
- `expected_ids` for single-answer questions (a named project, a specific listing, an FAQ's project)
- `gt_filters` for list questions: the expected set is every property record matching those filters, and filter
  precision checks that each returned property really satisfies them.
Hand-phrased questions (typos, Arabic, budget talk) still get programmatic ground truth through `gt_filters`.

    uv run python -m eval.build_golden            # -> eval/golden.jsonl (+ summary)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag.corpus import load_chunks, load_records
from rag.filters import Filters, matches
from rag.schema import PROPERTY_KINDS, filter_metadata

OUT = Path("eval/golden.jsonl")
rng = random.Random(42)


def meta(r: dict) -> dict:
    return {**filter_metadata(r), "parent_id": r["id"]}


def gt_set(records: list[dict], gt: dict) -> list[str]:
    f = Filters(**gt)
    return [r["id"] for r in records if r["record_kind"] in PROPERTY_KINDS and matches(meta(r), f)]


def plural(t: str) -> str:
    return {"villa": "villas", "apartment": "apartments", "townhouse": "townhouses", "penthouse": "penthouses"}.get(t, t + "s")


def strip_al(d: str) -> str:
    return re.sub(r"^(Al|An|Ar|As|At|Ad|Ash|Az)-", "", d)


def build() -> list[dict]:
    recs = [r for r in load_records() if not r.get("duplicate_of")]       # duplicates are answered by their canonical
    props = [r for r in recs if r["record_kind"] in PROPERTY_KINDS]
    ws = [r for r in props if r["source"] == "wasalt"]
    dg = [r for r in props if r["source"] == "darglobal"]
    items: list[dict] = []

    def add(category, query, **kw):
        item = {"id": f"{category}-{sum(1 for i in items if i['category'] == category) + 1:03d}", "category": category,
                "query": query, "expected_ids": [], "gt_filters": None, "expected_parse": None, "expect_intent": None,
                "expect_relaxed": None, "answer_must_contain": [], "answer_must_not_contain": [], "notes": ""}
        item.update(kw)
        if item["gt_filters"] is not None and not item["expected_ids"]:
            item["expected_ids"] = gt_set(recs, item["gt_filters"])
        items.append(item)

    # A. Wasalt structured list queries -----------------------------------------------------------
    zones = sorted({r["zone"] for r in ws if r["zone"]})
    for t in ("villa", "apartment"):
        for z in zones:
            gt = {"sources": ["wasalt"], "property_types": [t], "zones": [z]}
            if len(gt_set(recs, gt)) >= 2:
                add("wasalt_list", f"Show me {plural(t)} for sale in {z}", gt_filters=gt,
                    expected_parse={"property_types": [t], "zones": [z]})
    for t in ("villa", "apartment"):
        for z in zones:
            for beds in sorted({r["bedrooms_min"] for r in ws if r["property_type"] == t and r["zone"] == z and r["bedrooms_min"]}):
                gt = {"sources": ["wasalt"], "property_types": [t], "zones": [z], "bedrooms_min": beds, "bedrooms_max": beds}
                if len(gt_set(recs, gt)) >= 2:
                    add("wasalt_list", f"{beds} bedroom {t} in {z}", gt_filters=gt,
                        expected_parse={"property_types": [t], "zones": [z], "bedrooms_min": beds, "bedrooms_max": beds})
    for t in ("villa", "apartment"):
        prices = sorted(r["price_amount"] for r in ws if r["property_type"] == t and r["price_amount"])
        for q in (0.3, 0.6):
            cap = prices[int(len(prices) * q)]
            cap_m = round(cap / 1_000_000, 1) if cap >= 1_000_000 else None
            cap_v = cap_m * 1_000_000 if cap_m else round(cap / 1000) * 1000
            txt = f"{cap_m:g}M" if cap_m else f"{int(cap_v / 1000)}k"
            gt = {"property_types": [t], "price_max": cap_v, "city": "Riyadh"}
            add("wasalt_list", f"{t}s in Riyadh under {txt}", gt_filters=gt,
                expected_parse={"property_types": [t], "price_max": cap_v, "city": "Riyadh"})
    by_d = Counter(r["district"] for r in ws)
    for d, n in by_d.most_common(8):
        if n >= 2:
            add("wasalt_list", f"properties in {strip_al(d)} Riyadh", gt_filters={"districts": [d]},
                expected_parse={"districts": [d]})

    # B. Wasalt single-listing lookups ------------------------------------------------------------
    key = Counter((r["property_type"], r["district"], r["bedrooms_min"]) for r in ws)
    cands = [r for r in ws if key[(r["property_type"], r["district"], r["bedrooms_min"])] == 1 and r["price_amount"] and r["bedrooms_min"]]
    for r in rng.sample(cands, min(15, len(cands))):
        price = f"{int(r['price_amount']):,} SAR"
        add("wasalt_lookup", f"{r['bedrooms_min']} bedroom {r['property_type']} in {r['district']} listed at {price}",
            expected_ids=[r["id"]], answer_must_contain=[f"{int(r['price_amount']):,}"])

    # C. DarGlobal FAQ questions that name their project ------------------------------------------
    chunks = load_chunks()
    from rag.parser import _aliases
    titles = {r["id"]: r for r in dg}
    faq_by_proj = defaultdict(list)
    for c in chunks:
        m = c["metadata"]
        if m.get("chunk_type") == "faq" and m.get("parent_id") in titles and m.get("question"):
            faq_by_proj[m["parent_id"]].append(m["question"])
    for pid, qs in sorted(faq_by_proj.items()):
        r = titles[pid]
        aliases = _aliases(r["title"], r["url"])
        named = [q for q in qs if any(a in q.lower() for a in aliases)]
        for q in rng.sample(named, min(2, len(named))):
            add("darglobal_faq", q, expected_ids=[pid])

    # D. DarGlobal project facts --------------------------------------------------------------------
    for r in rng.sample([x for x in dg if x["record_kind"] == "development_project" and x["country"]], 20):
        short = re.split(r",| – | - | by ", r["title"])[0].strip()
        add("darglobal_fact", f"Which country is {short} in?", expected_ids=[r["id"]], answer_must_contain=[r["country"]])
    for r in rng.sample([x for x in dg if x["amenities"]], 10):
        short = re.split(r",| – | - | by ", r["title"])[0].strip()
        add("darglobal_fact", f"What amenities does {short} have?", expected_ids=[r["id"]],
            answer_must_contain=[r["amenities"][0][:25]], notes="must mention at least the first listed amenity")
    for r in rng.sample([x for x in dg if x["status"]], 8):
        short = re.split(r",| – | - | by ", r["title"])[0].strip()
        word = {"under_development": "development", "completed": "complet", "sold_out": "sold"}.get(r["status"], r["status"])
        add("darglobal_fact", f"Is {short} completed or still under construction?", expected_ids=[r["id"]],
            answer_must_contain=[word])

    # E. DarGlobal by country / brand ---------------------------------------------------------------
    for c in sorted({r["country"] for r in dg if r["country"]}):
        add("darglobal_list", f"What DarGlobal projects are there in {c}?",
            gt_filters={"sources": ["darglobal"], "country": c}, expected_parse={"country": c})
    for brand in ("Trump", "Missoni", "Lamborghini", "Pagani", "Fendi", "Aston Martin"):
        ids = [r["id"] for r in dg if brand.lower() in f"{r['title']} {r['partner'] or ''}".lower()]
        if ids:
            add("darglobal_list", f"Which properties are designed with {brand}?", expected_ids=ids,
                notes="expected = DarGlobal properties whose title or partner names the brand")

    # F. One-of-one residences ------------------------------------------------------------------------
    for r in [x for x in dg if x["record_kind"] == "one_of_one"]:
        add("one_of_one", f"Tell me about {r['title']}", expected_ids=[r["id"]], answer_must_contain=["not published"])

    # G. Behaviour: honesty, relaxation, scope, language ---------------------------------------------
    for r in rng.sample([x for x in dg if x["record_kind"] == "development_project" and not x["price_published"]], 5):
        short = re.split(r",| – | - | by ", r["title"])[0].strip()
        add("behaviour", f"What is the price of {short}?", expected_ids=[r["id"]], answer_must_contain=["not published"],
            notes="DarGlobal does not publish prices: the answer must say so, never invent one")
    add("behaviour", "villa for rent in Riyadh", expect_relaxed=["purpose"],
        gt_filters={"property_types": ["villa"], "city": "Riyadh"}, notes="no rental data: must relax purpose and say so")
    add("behaviour", "10 bedroom apartment in South Riyadh under 300k", expect_relaxed=True, notes="impossible: must relax")
    add("behaviour", "What's the weather in Paris today?", expect_intent="out_of_scope")
    add("behaviour", "write me a poem about cats", expect_intent="out_of_scope")
    add("behaviour", "فيلا للبيع في الرياض", gt_filters={"property_types": ["villa"], "city": "Riyadh"},
        expected_parse={"property_types": ["villa"]}, notes="Arabic: villa for sale in Riyadh")
    add("behaviour", "شقة للبيع في الرياض", gt_filters={"property_types": ["apartment"], "city": "Riyadh"},
        expected_parse={"property_types": ["apartment"]}, notes="Arabic: apartment for sale in Riyadh")

    # H. Hand-phrased natural questions (typos, budgets) with programmatic ground truth ----------------
    natural = [
        ("looking for a 3 bed flat in east riyadh, budget about 1 million",
         {"property_types": ["apartment"], "zones": ["East Riyadh"], "bedrooms_min": 3, "bedrooms_max": 3, "price_min": 850000, "price_max": 1150000}),
        ("villa north riyadh max 3.5m 5 bedrooms",
         {"property_types": ["villa"], "zones": ["North Riyadh"], "bedrooms_min": 5, "bedrooms_max": 5, "price_max": 3500000}),
        ("any apartmnets in west riyadh?", {"property_types": ["apartment"], "zones": ["West Riyadh"]}),
        ("big villa over 400 sqm in riyadh", {"property_types": ["villa"], "area_min": 400, "city": "Riyadh"}),
        ("2-3 bedroom apartments in Riyadh between 600k and 1m",
         {"property_types": ["apartment"], "bedrooms_min": 2, "bedrooms_max": 3, "price_min": 600000, "price_max": 1000000, "city": "Riyadh"}),
        ("luxury villas in Oman", {"sources": ["darglobal"], "property_types": ["villa"], "country": "Oman"}),
        ("penthouses in dubai", {"property_types": ["penthouse"], "city": "Dubai"}),
        ("apartments in the UAE by DarGlobal", {"sources": ["darglobal"], "property_types": ["apartment"], "country": "United Arab Emirates"}),
        ("villas in spain", {"property_types": ["villa"], "country": "Spain"}),
        ("hotel rooms to invest in", {"property_types": ["hotel_room"]}),
    ]
    for q, gt in natural:
        add("natural", q, gt_filters=gt)
    for q, ids in (("villas in AIDA", [r["id"] for r in dg if (r.get("parent_project") or "").upper() == "AIDA"
                                       and "villa" in r["property_types"]]),
                   ("cheapest apartment in East Riyadh", None)):
        if ids:
            add("natural", q, expected_ids=ids, notes="AIDA sub-projects in the villas category")
    cheap = sorted([r for r in ws if r["property_type"] == "apartment" and r["zone"] == "East Riyadh" and r["price_amount"]],
                   key=lambda r: r["price_amount"])
    add("natural", "cheapest apartment in East Riyadh", expected_ids=[cheap[0]["id"]], notes="must rank the cheapest first",
        answer_must_contain=[f"{int(cheap[0]['price_amount']):,}"])

    for it in items:
        if it["gt_filters"] is not None and not it["expected_ids"] and not it["expect_relaxed"]:
            it["notes"] = (it["notes"] + " | empty ground truth: system must say nothing matches").strip(" |")
    return items


if __name__ == "__main__":
    items = build()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = Counter(i["category"] for i in items)
    print(f"{len(items)} golden items -> {OUT}  {dict(c)}")
    print("set-type items with empty ground truth:", [i["id"] for i in items if i["gt_filters"] and not i["expected_ids"]])
