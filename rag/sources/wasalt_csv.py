"""Wasalt CSV (one row per resale listing) -> unified records + exactly one chunk per listing.

The CSV's `Property facts` JSON uses ~100 key spellings for ~35 real fields (bedrooms/Bedrooms, built_up_area as
"112.59 SQM" vs built_up_area_sqm as a number, five names for the licence expiry, ...). ALIASES folds them together.
Personal data (agent/advertiser names, phone numbers, contact persons) is dropped; broker *companies* are kept.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from ..schema import (content_hash, empty_record, filter_metadata, iso_date, norm_property_type, to_float, to_int)

ALIASES: dict[str, str] = {}
for canon, names in {
    "subtype": ["subtype", "type", "property_type"],
    "usage": ["usage", "property_usage"],
    "bedrooms": ["bedrooms"],
    "bathrooms": ["bathrooms"],
    "living_rooms": ["living_rooms", "living_room"],
    "guest_rooms": ["guest_rooms", "guest_room"],
    "built_up_area": ["built_up_area", "built_up_area_sqm", "building_area", "internal_area_sqm", "built-up_area"],
    "land_area": ["land_area", "land_area_sqm", "total_area_sqm", "area"],
    "external_area": ["external_area_sqm"],
    "price_per_sqm": ["price_per_sqm"],
    "age": ["age"],
    "building_year": ["building_year"],
    "facade": ["facade"],
    "street": ["street"],
    "street_width": ["street_width", "street_width_m"],
    "streets": ["streets"],
    "ad_source": ["ad_source"],
    "rega_ad_license": ["rega_ad_license_no", "rega_ad_license", "rega_ad_license_no."],
    "fal_license": ["fal_license_no", "fal_license", "broker_fal_license"],
    "ad_status": ["status", "rega_status"],
    "license_issue": ["rega_issue_date", "issue_date", "license_issue_date"],
    "license_expiry": ["rega_expiry_date", "expiring_on", "rega_expiring_on", "expiring_date", "expiry_date", "license_expiry_date"],
    "ref_no": ["ref_no", "reference_no", "reference_number", "rega_ref_no", "ref_no."],
    "broker": ["broker", "company", "fal_license_holder"],
    "amenities": ["amenities"],
    "fixtures": ["fixtures"],
    "utilities": ["utilities", "electricity", "water", "water_connections"],
    "plan_number": ["plan_number"],
    "land_number": ["land_number"],
    "offer_number": ["offer_number"],
    "postal_code": ["postal_code"],
    "floor": ["floor", "property_floor"],
    "floors": ["floors"],
    "unit": ["unit"],
    "project": ["project"],
    "furnishing": ["furnishing"],
    "last_updated": ["last_updated"],
    "other_obligations": ["other_obligations"],
}.items():
    for n in names:
        ALIASES[n] = canon
PII_KEYS = {"agent", "advertiser", "contact", "broker_contact", "responsible_name"}
COMPANY_MARKERS = ("شركة", "مؤسسة", "مكتب", "العقارية", "عقارات", "realty", "real estate", "company", "est.", "llc", "group")
PHONE = re.compile(r"(?:\+?966|00966|\b0)5\d[\s-]?\d{3}[\s-]?\d{4}\b|\b9665\d{8}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
STOCK_EN = re.compile(r"^(built up area|land area|price|building year|the property has|.*has electricity and water|"
                      r"(apartment|villa|floor|duplex|townhouse|building|land) is\s*:|property floor|.* is located on \d+ street)", re.I)


def _key(k: str) -> str:
    return re.sub(r"[\s\-]+", "_", k.strip().lower()).rstrip(".:")


def normalise_facts(raw: dict) -> tuple[dict, list[str]]:
    """Fold alias keys; returns (facts, unmapped_keys). First non-empty value wins, except lists which merge."""
    out: dict = {}
    unmapped = []
    for k, v in raw.items():
        kk = _key(k)
        if kk in PII_KEYS or v in (None, "", [], {}):
            continue
        canon = ALIASES.get(kk) or ALIASES.get(kk.replace("_no", ""))
        if not canon:
            unmapped.append(kk)
            out.setdefault("other", {})[kk] = v
            continue
        if canon == "utilities":
            items = v if isinstance(v, list) else [kk if v is True else f"{kk}: {v}" if kk in ("electricity", "water", "water_connections") else v]
            out.setdefault("utilities", [])
            for it in items:
                if str(it) not in out["utilities"]:
                    out["utilities"].append(str(it))
        elif canon in ("amenities", "fixtures"):
            out.setdefault(canon, [])
            for it in (v if isinstance(v, list) else [v]):
                if str(it) not in out[canon]:
                    out[canon].append(str(it))
        else:
            out.setdefault(canon, v)
    return out, unmapped


def _is_company(name: str) -> bool:
    low = (name or "").lower()
    return any(m in low for m in COMPANY_MARKERS)


def scrub(text: str) -> str:
    return EMAIL.sub("[email removed]", PHONE.sub("[phone removed]", text or ""))


def clean_description(text: str) -> str:
    lines = [ln.strip() for ln in scrub(text).splitlines()]
    keep = [ln for ln in lines if ln and not STOCK_EN.match(ln)]
    return "\n".join(keep)


def _age_years(age: str | None, year: int | None) -> int | None:
    if age and re.search(r"new", str(age), re.I):
        return 0
    n = to_int(age)
    if n is not None:
        return n
    return None


def row_to_record(row: dict, fetched_default: str | None = None) -> dict:
    facts_raw = json.loads(row.get("Property facts") or "{}")
    loc = json.loads(row.get("Location") or "{}")
    price = json.loads(row["Price"]) if (row.get("Price") or "").strip() else {}
    prov = json.loads(row.get("Provenance") or "{}")
    imgs = json.loads(row.get("Image URLs") or "[]") if (row.get("Image URLs") or "").strip() else []
    f, unmapped = normalise_facts(facts_raw)

    r = empty_record()
    lid = str(row["Listing ID"]).strip()
    r.update(id=f"wasalt:{lid}", source="wasalt", record_kind="unit_listing", title=(row.get("Title") or "").strip(),
             url=(prov.get("source_url") or row.get("Listing URL") or "").strip())
    purpose = "sale" if "/sale" in (r["url"] + (prov.get("search_url") or "")) or "sale" in (prov.get("search_url") or "") else None
    r["purpose"] = purpose or "sale"                         # file is "Riyadh Residential Sale Listings"
    ptype = norm_property_type(str(f.get("subtype") or "")) or norm_property_type(r["title"])
    r["property_type"] = ptype
    r["property_types"] = [ptype] if ptype else []
    r["usage"] = (str(f.get("usage")).lower() if f.get("usage") else "residential")
    amt = to_float(price.get("amount"))
    r["price_amount"], r["currency"] = amt, (price.get("currency") or "SAR") if amt else None
    r["price_min"] = r["price_max"] = amt
    r["price_published"] = amt is not None
    beds = to_int(f.get("bedrooms"))
    r["bedrooms_min"] = r["bedrooms_max"] = beds
    r["bathrooms"] = to_int(f.get("bathrooms"))
    bua = to_float(f.get("built_up_area"))
    land = to_float(f.get("land_area"))
    if ptype == "land" and land is None:
        land = bua
    r["built_up_area_min"] = r["built_up_area_max"] = bua
    r["land_area_sqm"] = land
    ppsqm = to_float(f.get("price_per_sqm"))
    if ppsqm is None and amt and (bua or land):
        ppsqm = round(amt / (bua or land))
    r["price_per_sqm"] = ppsqm
    r["country"] = loc.get("country") or "Saudi Arabia"
    r["city"] = loc.get("city") or "Riyadh"
    zone = loc.get("zone") or loc.get("region")
    r["zone"] = zone if zone and zone.strip().lower() != (r["city"] or "").lower() else None
    r["district"] = loc.get("district")
    yr = to_int(f.get("building_year"))
    r["building_year"] = yr if yr and 1900 < yr < 2100 else None
    r["status"] = "ready"
    r["furnishing"] = f.get("furnishing")
    r["facade"] = f.get("facade")
    r["street_width_m"] = to_float(f.get("street_width"))
    broker = f.get("broker") or prov.get("broker")
    r["developer"] = None
    r["images"] = [i for i in imgs if isinstance(i, str) and i.startswith("http")]
    r["has_images"] = bool(r["images"])
    r["amenities"] = f.get("amenities", [])
    r["fetched_at"] = prov.get("scraped_at") or fetched_default
    r["extra"] = {
        "listing_id": lid, "ref_no": f.get("ref_no") or prov.get("ref_no") or prov.get("reference_no"),
        "living_rooms": to_int(f.get("living_rooms")), "guest_rooms": to_int(f.get("guest_rooms")),
        "age": f.get("age"), "age_years": _age_years(f.get("age"), r["building_year"]),
        "street": f.get("street"), "streets": f.get("streets"), "floor": f.get("floor"), "floors": f.get("floors"),
        "fixtures": f.get("fixtures", []), "utilities": f.get("utilities", []),
        "project": f.get("project"), "unit": f.get("unit"), "external_area_sqm": to_float(f.get("external_area")),
        "plan_number": f.get("plan_number"), "land_number": f.get("land_number"), "offer_number": f.get("offer_number"),
        "rega_ad_license": f.get("rega_ad_license") or prov.get("rega_ad_license_no") or prov.get("rega_ad_license"),
        "fal_license": f.get("fal_license") or prov.get("fal_license_no") or prov.get("fal_license"),
        "ad_status": f.get("ad_status"), "license_issue": iso_date(f.get("license_issue")),
        "license_expiry": iso_date(f.get("license_expiry")), "ad_source": f.get("ad_source"),
        "broker_company": broker if broker and _is_company(str(broker)) else None,
        "search_url": prov.get("search_url"),
        "arabic_url": prov.get("arabic_source_url") or prov.get("source_url_ar") or prov.get("source_arabic_url"),
        "unmapped_keys": unmapped,
    }
    r["description_en"] = clean_description(row.get("English description") or "")
    r["description_ar"] = scrub(row.get("Arabic description") or "").strip()
    flags = []
    if not r["price_published"]:
        flags.append("price_missing")
    if not (bua or land):
        flags.append("area_missing")
    if beds is None and ptype not in ("land", "building"):
        flags.append("bedrooms_missing")
    if not r["description_en"]:
        flags.append("description_en_missing")
    if not r["images"]:
        flags.append("images_missing")
    if not r["zone"]:
        flags.append("zone_missing")
    r["quality_flags"] = flags
    r["rich"] = {"gallery": [{"url": u, "alt": r["title"]} for u in r["images"]], "amenities": r["amenities"],
                 "fixtures": r["extra"]["fixtures"], "utilities": r["extra"]["utilities"], "faqs": [], "key_features": [],
                 "why_invest": None, "location": {"text": ", ".join(x for x in (r["district"], r["zone"], r["city"]) if x),
                                                  "highlights": [], "lat": None, "lng": None},
                 "about": r["description_en"][:1200] or r["description_ar"][:1200],
                 "licence": {k: r["extra"].get(k) for k in ("rega_ad_license", "ad_status", "license_expiry", "fal_license", "broker_company")}}
    r["summary"] = card_text(r)
    r["full_text"] = listing_text(r)
    return r


def _fmt(n: float | None) -> str | None:
    return None if n is None else (f"{int(n):,}" if float(n).is_integer() else f"{n:,.2f}")


def card_text(r: dict) -> str:
    e = r["extra"]
    ptype = (r["property_type"] or "property").replace("_", " ")
    where = ", ".join(x for x in (r["district"], r["zone"], r["city"], r["country"]) if x)
    bits = [f"{ptype.title()} for {r['purpose']} in {where}."]
    bits.append(f"Price: {_fmt(r['price_amount'])} {r['currency']}" + (f" ({_fmt(r['price_per_sqm'])} SAR/sqm)." if r["price_per_sqm"] else ".")
                if r["price_published"] else "Price: not published.")
    rooms = [f"{r['bedrooms_min']} bedrooms" if r["bedrooms_min"] is not None else None,
             f"{r['bathrooms']} bathrooms" if r["bathrooms"] is not None else None,
             f"{e['living_rooms']} living rooms" if e.get("living_rooms") is not None else None,
             f"{e['guest_rooms']} guest rooms" if e.get("guest_rooms") is not None else None]
    rooms = [x for x in rooms if x]
    if rooms:
        bits.append(", ".join(rooms) + ".")
    areas = [f"built-up area {_fmt(r['built_up_area_min'])} sqm" if r["built_up_area_min"] else None,
             f"land area {_fmt(r['land_area_sqm'])} sqm" if r["land_area_sqm"] and r["land_area_sqm"] != r["built_up_area_min"] else None]
    areas = [x for x in areas if x]
    if areas:
        bits.append(("; ".join(areas)).capitalize() + ".")
    if r["facade"] or r["street_width_m"]:
        bits.append("Facade: " + ", ".join(x for x in (r["facade"], f"street width {_fmt(r['street_width_m'])} m" if r["street_width_m"] else None) if x) + ".")
    if e.get("age") or r["building_year"]:
        bits.append("Age: " + ", ".join(x for x in (str(e["age"]) if e.get("age") else None,
                                                      f"built {r['building_year']}" if r["building_year"] else None) if x) + ".")
    if r["furnishing"]:
        bits.append(f"Furnishing: {r['furnishing']}.")
    if e.get("floor"):
        bits.append(f"Floor: {e['floor']}.")
    if e.get("project"):
        bits.append(f"Project: {e['project']}" + (f", unit {e['unit']}" if e.get("unit") else "") + ".")
    if r["amenities"]:
        bits.append("Amenities: " + ", ".join(r["amenities"]) + ".")
    if e.get("fixtures"):
        bits.append("Fixtures: " + ", ".join(e["fixtures"]) + ".")
    if e.get("utilities"):
        bits.append("Utilities: " + ", ".join(e["utilities"]) + ".")
    if e.get("rega_ad_license"):
        bits.append(f"REGA ad licence {e['rega_ad_license']}" + (f", status {e['ad_status']}" if e.get("ad_status") else "")
                    + (f", valid until {e['license_expiry']}" if e.get("license_expiry") else "") + ".")
    if e.get("broker_company"):
        bits.append(f"Listed by {e['broker_company']}.")
    return " ".join(bits)


def listing_text(r: dict) -> str:
    parts = [f"{r['title']} — Wasalt listing {r['extra']['listing_id']}.", r["summary"]]
    if r["description_en"]:
        parts.append("Description: " + r["description_en"])
    if r["description_ar"]:
        parts.append("الوصف: " + r["description_ar"][:1500])
    return "\n".join(parts)


def load_wasalt_csv(path: str | Path) -> tuple[list[dict], list[dict]]:
    """-> (records, chunks). One chunk per listing; the record itself is the parent."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    records, chunks, seen = [], [], set()
    for row in rows:
        r = row_to_record(row)
        if r["id"] in seen:                                  # duplicate listing ids never overwrite silently
            r["id"] = f"{r['id']}~{len(seen)}"
            r["quality_flags"].append("duplicate_listing_id")
        seen.add(r["id"])
        records.append(r)
        meta = {**filter_metadata(r), "parent_id": r["id"], "chunk_type": "listing", "chunk_id": f"{r['id']}#listing"}
        meta["content_hash"] = content_hash(r["full_text"], meta)
        chunks.append({"id": f"{r['id']}#listing", "text": r["full_text"], "metadata": meta})
    return records, chunks
