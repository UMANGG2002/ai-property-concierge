"""DarGlobal -> unified records + chunks.

- 35 development projects: the parsed page documents (data/darglobal/docs) and section chunks (sections.py), re-tagged with the
  unified metadata; the parent text is the flattened whole-project document (sections.flatten_project).
- One-of-one residences: every /one-of-one page embeds the full site-wide collection list (oneOfOneData), so all
  collections are built from it, including ones whose own page was never fetched ("Unlisted by Pagani").
- Everything else (partners, country listing pages, about/investor/hospitality..., press, blog): ingested as
  information records so nothing on the site is dropped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .sections import build_chunks, flatten_project

from ..schema import (all_property_types, bedrooms_from_text, content_hash, empty_record, filter_metadata,
                      norm_property_type, to_float)

BASE = "https://darglobal.co.uk"
COUNTRY_ALIASES = {"uae": "United Arab Emirates", "u.a.e": "United Arab Emirates", "united arab emirates": "United Arab Emirates",
                   "ksa": "Saudi Arabia", "saudi arabia": "Saudi Arabia", "oman": "Oman", "qatar": "Qatar", "spain": "Spain",
                   "uk": "United Kingdom", "united kingdom": "United Kingdom", "england": "United Kingdom", "maldives": "Maldives"}
CITY_COUNTRY = {"dubai": "United Arab Emirates", "ras al khaimah": "United Arab Emirates", "rak": "United Arab Emirates",
                "al marjan island": "United Arab Emirates", "muscat": "Oman", "aida": "Oman", "london": "United Kingdom",
                "riyadh": "Saudi Arabia", "jeddah": "Saudi Arabia", "benahavís": "Spain", "benahavis": "Spain",
                "marbella": "Spain", "doha": "Qatar", "simaisma": "Qatar"}


def norm_country(v: str | None) -> str | None:
    if not v:
        return None
    return COUNTRY_ALIASES.get(v.strip().lower(), v.strip())


def country_in_text(*texts: str | None) -> str | None:
    blob = " ".join(t for t in texts if t).lower()
    for k, v in sorted({**COUNTRY_ALIASES, **CITY_COUNTRY}.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", blob):
            return v
    return None


def norm_status(v: str | None) -> str | None:
    s = (v or "").lower()
    if not s:
        return None
    if "sold" in s:
        return "sold_out"
    if "complete" in s or "ready" in s:
        return "completed"
    if "develop" in s or "construction" in s or "launch" in s or "off" in s:
        return "under_development"
    return s.replace(" ", "_")


def _meta_for(record: dict, chunk_type: str, chunk_id: str, extra: dict | None = None) -> dict:
    m = {**filter_metadata(record), "parent_id": record["id"], "chunk_type": chunk_type, "chunk_id": chunk_id}
    if extra:
        for k in ("question", "faq_no", "section", "image_count"):
            if extra.get(k) is not None:
                m[k] = extra[k] if not isinstance(extra[k], str) else extra[k][:300]
    return m


def _finish(record: dict, chunks: list[dict]) -> list[dict]:
    out = []
    for c in chunks:
        c["metadata"]["content_hash"] = content_hash(c["text"], c["metadata"])
        out.append(c)
    return out


# ---------------------------------------------------------------- development projects
def project_record(doc: dict) -> dict:
    f = doc.get("facts") or {}
    r = empty_record()
    r.update(id=doc["id"], source="darglobal", record_kind="development_project", title=doc.get("title"), url=doc["url"],
             developer="DarGlobal", purpose="sale")
    type_text = " ".join(str(x) for x in (f.get("property_type"), f.get("unit_type"), doc.get("title"), doc.get("tagline"),
                                          doc["url"].rsplit("/", 1)[-1].replace("-", " "),
                                          " ".join(c.get("unit_category", "") for c in doc.get("children") or [])) if x)
    types = all_property_types(type_text)
    if f.get("unit_category"):
        types = list(dict.fromkeys([norm_property_type(f["unit_category"]) or f["unit_category"]] + types))
    r["property_types"] = [t for t in types if t]
    r["property_type"] = r["property_types"][0] if r["property_types"] else None
    r["usage"] = "hospitality" if r["property_type"] == "hotel_room" else "residential"
    r["price_amount"] = to_float(f.get("price"))
    r["price_min"] = r["price_max"] = r["price_amount"]
    r["price_published"] = r["price_amount"] is not None
    r["currency"] = f.get("currency")
    bmin, bmax = f.get("bedrooms_min"), f.get("bedrooms_max")
    if bmin is None:
        bmin, bmax = bedrooms_from_text(f"{f.get('unit_type') or ''} {doc.get('tagline') or ''}")
    r["bedrooms_min"], r["bedrooms_max"] = bmin, bmax
    r["built_up_area_min"], r["built_up_area_max"] = f.get("area_sqm_min"), f.get("area_sqm_max")
    r["country"] = norm_country(f.get("country"))
    r["city"] = f.get("city")
    r["zone"] = f.get("region")
    r["community"] = f.get("community")
    if not r["country"] and r["city"]:
        r["country"] = CITY_COUNTRY.get(r["city"].lower())
    if not r["country"]:
        r["country"] = country_in_text(doc.get("title"), doc.get("tagline"), (doc.get("location") or {}).get("text"))
    r["status"] = norm_status(f.get("status"))
    partner = f.get("partner") or next((p.get("name") for p in doc.get("partners") or [] if p.get("name")), None)
    r["partner"] = None if not partner or re.match(r"(?i)(dar ?global|experience in)", partner) else partner
    parent = f.get("parent") or {}
    r["parent_project"] = parent.get("title")
    r["unit_category"] = f.get("unit_category")
    r["images"] = [g["url"] for g in doc.get("gallery") or [] if g.get("url")]
    r["has_images"] = bool(r["images"])
    r["amenities"] = [a if isinstance(a, str) else a.get("name", "") for a in doc.get("amenities") or []]
    r["fetched_at"] = (doc.get("provenance") or {}).get("fetched_at")
    r["quality_flags"] = list(doc.get("quality_flags") or [])
    r["extra"] = {"tagline": doc.get("tagline"), "unit_type": f.get("unit_type"), "property_type_raw": f.get("property_type"),
                  "lat": f.get("lat"), "lng": f.get("lng"), "area_raw": f.get("area_raw"), "partner_role": f.get("partner_role"),
                  "children": [{"title": c["title"], "unit_category": c["unit_category"], "url": c["url"]} for c in doc.get("children") or []],
                  "faq_count": len(doc.get("faq") or []), "completeness": (doc.get("completeness") or {}).get("score"),
                  "in_menu": bool(doc.get("menu_paths"))}
    if not r["country"]:
        r["quality_flags"].append("country_missing")
    flat = flatten_project(doc)
    r["full_text"] = flat["text"]
    loc = doc.get("location") or {}
    wi = doc.get("why_invest") or {}
    r["rich"] = {
        "gallery": [{"url": g["url"], "alt": g.get("alt") or ""} for g in doc.get("gallery") or [] if g.get("url")][:24],
        "key_features": [{"title": k.get("title"), "detail": k.get("distance") or "", "text": k.get("text") or "",
                          "image": k.get("image")} for k in doc.get("key_features") or [] if k.get("title")][:12],
        "amenities": r["amenities"][:40],
        "why_invest": {"title": wi.get("title"), "intro": wi.get("intro") or "", "bullets": (wi.get("bullets") or [])[:10]}
        if (wi.get("intro") or wi.get("bullets")) else None,
        "faqs": [{"q": f["q"], "a": f["a"]} for f in doc.get("faq") or []][:40],
        "location": {"text": loc.get("text") or "", "highlights": [h if isinstance(h, str) else h.get("title") for h in loc.get("highlights") or []][:8],
                     "lat": f.get("lat"), "lng": f.get("lng")},
        "about": ((doc.get("about") or {}).get("text") or "")[:1200],
    }
    return r


def project_chunks(doc: dict, r: dict) -> list[dict]:
    out = []
    for c in build_chunks(doc):
        old = c["metadata"]
        ctype = old.get("chunk_type", "other")
        out.append({"id": c["id"], "text": c["text"],
                    "metadata": _meta_for(r, ctype, c["id"], {"question": old.get("question"), "faq_no": old.get("faq_no")})})
    return _finish(r, out)


# ---------------------------------------------------------------- one-of-one residences
def _media_url(o) -> str | None:
    d = (o or {}).get("data") if isinstance(o, dict) else None
    if isinstance(d, dict):
        return (d.get("attributes") or {}).get("url")
    if isinstance(d, list) and d:
        return (d[0].get("attributes") or {}).get("url")
    return None


def one_of_one_collections(pages: list[dict]) -> dict[str, dict]:
    coll: dict[str, dict] = {}
    for pg in pages:
        pp = ((pg["content"].get("next_data") or {}).get("data") or {}).get("props", {}).get("pageProps", {})
        if "oneOfOneData" not in pp:
            continue
        items = ((pp["oneOfOneData"].get("attributes") or {}).get("one_of_one_collections") or {}).get("data") or []
        items += [{"attributes": c.get("attributes") or c} for c in pp.get("collectionsData") or []]
        for it in items:
            a = it.get("attributes") or {}
            slug = (a.get("slug") or "").strip()
            if not slug or not a.get("title"):
                continue
            prev = coll.get(slug)
            if not prev or len(a.get("content") or []) > len(prev.get("content") or []):
                coll[slug] = {**a, "_fetched_at": pg.get("fetched_at")}
    # the '/' slug is a second CMS entry for the Albert Hall residences: merge its sections into the main one
    if "/" in coll:
        alias = coll.pop("/")
        main = next((v for k, v in coll.items() if "albert" in k.lower()), None)
        if main:
            main["content"] = list(main.get("content") or []) + list(alias.get("content") or [])
            main["_alias_title"] = alias.get("title")
            main["_alias_location"] = alias.get("location")
        else:
            coll["albert-hall-mansions"] = alias
    return coll


def one_of_one_record(slug: str, a: dict, fetched_urls: set[str]) -> tuple[dict, list[dict]]:
    r = empty_record()
    url = f"{BASE}/one-of-one/{slug}".lower()
    title = re.sub(r"\s+", " ", a.get("title") or a.get("project") or slug).strip()
    r.update(id=f"darglobal:one-of-one/{slug.lower()}", source="darglobal", record_kind="one_of_one", title=title, url=url,
             developer="DarGlobal", purpose="sale", status=None)
    loc = (a.get("location") or "").replace("|", ",")
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    country = next((norm_country(p) for p in reversed(parts) if p.strip().lower() in COUNTRY_ALIASES), None)
    if not country:
        country = next((CITY_COUNTRY[p.lower()] for p in parts if p.lower() in CITY_COUNTRY), None)
    if not country:
        country = country_in_text(a.get("location"), a.get("_alias_location"), title, a.get("description"))
    r["country"] = country
    r["city"] = next((p for p in parts if norm_country(p) != country), None)
    if a.get("_alias_location") and r["city"] and "albert hall" in r["city"].lower():
        r["city"] = "London"
    if r["city"] and r["city"].upper() == "AIDA":
        r["community"], r["city"] = "AIDA", "Muscat"
    sections = [s for s in a.get("content") or [] if (s.get("description") or s.get("title"))]
    text_all = " ".join(f"{s.get('title') or ''} {s.get('description') or ''}" for s in sections) + " " + (a.get("description") or "")
    types = all_property_types(title) or all_property_types(a.get("description") or "")   # the title is the most reliable
    r["property_types"] = types
    r["property_type"] = types[0] if types else None
    r["usage"] = "residential"
    r["bedrooms_min"], r["bedrooms_max"] = bedrooms_from_text(text_all)
    partner = re.search(r"(?:by|inspired by|interiors by)\s+(Automobili Lamborghini|Fendi Casa|FENDI Casa|Pagani|Aston Martin|Trump|Missoni)", title + " " + text_all, re.I)
    r["partner"] = partner.group(1) if partner else ("Trump Organization" if "trump" in title.lower() else None)
    imgs = [u for u in [_media_url(a.get("backgroundImage"))] + [_media_url(s.get("imageSrc")) for s in sections] if u]
    r["images"] = list(dict.fromkeys(imgs))
    r["has_images"] = bool(r["images"])
    r["fetched_at"] = a.get("_fetched_at")
    r["extra"] = {"location_label": a.get("location"), "video": _media_url(a.get("videoUrl")), "sections": len(sections),
                  "alias_title": a.get("_alias_title")}
    flags = ["price_not_published"]
    if url not in fetched_urls:
        flags.append("own_page_not_fetched_data_from_collection")
    if not r["country"]:
        flags.append("country_missing")
    r["quality_flags"] = flags
    where = ", ".join(x for x in (r["community"], r["city"], r["country"]) if x)
    head = (f"{title} — a DarGlobal One of One residence" + (f" in {where}" if where else "") + ". "
            + (f"Type: {', '.join(t.replace('_', ' ') for t in types)}. " if types else "")
            + (f"{r['bedrooms_min']}" + (f"-{r['bedrooms_max']}" if r["bedrooms_max"] != r["bedrooms_min"] else "") + " bedrooms. " if r["bedrooms_min"] else "")
            + (f"Partner brand: {r['partner']}. " if r["partner"] else "") + "Price: not published (enquire with DarGlobal).")
    body = [f"## {s.get('title') or s.get('sidebarLabel') or ''}\n{(s.get('description') or '').strip()}" for s in sections]
    r["summary"] = head + " " + re.sub(r"\s+", " ", a.get("description") or "")[:500]
    r["rich"] = {"gallery": [{"url": u, "alt": title} for u in r["images"]][:24],
                 "sections": [{"title": s_.get("title") or s_.get("sidebarLabel"), "text": (s_.get("description") or "").strip()[:900],
                               "image": _media_url(s_.get("imageSrc"))} for s_ in sections][:12],
                 "amenities": [], "faqs": [], "key_features": [], "why_invest": None,
                 "location": {"text": a.get("location") or "", "highlights": [], "lat": None, "lng": None},
                 "about": re.sub(r"\s+", " ", a.get("description") or "")[:1200], "video": r["extra"].get("video")}
    r["full_text"] = f"# {title}\n{head}\n\n{a.get('description') or ''}\n\n" + "\n\n".join(body)
    chunks = [{"id": f"{r['id']}#summary", "text": r["summary"], "metadata": _meta_for(r, "summary", f"{r['id']}#summary")}]
    buf = ""
    for i, s in enumerate(sections, 1):
        piece = f"{title} — {s.get('title') or s.get('sidebarLabel') or 'Section'}. {(s.get('description') or '').strip()}"
        buf = f"{buf}\n{piece}" if buf else piece
        if len(buf.split()) >= 40 or i == len(sections):
            cid = f"{r['id']}#section-{i}"
            chunks.append({"id": cid, "text": buf, "metadata": _meta_for(r, "section", cid, {"section": s.get("title")})})
            buf = ""
    return r, _finish(r, chunks)


# ---------------------------------------------------------------- information records (partners, pages, articles)
def info_record(doc: dict) -> tuple[dict, list[dict]]:
    kind = {"partner": "partner", "article": "article", "press": "article"}.get(doc["kind"], "info_page")
    r = empty_record()
    r.update(id=doc["id"], source="darglobal", record_kind=kind, title=doc.get("title"), url=doc["url"], developer="DarGlobal")
    f = doc.get("facts") or {}
    r["country"] = norm_country(f.get("country"))
    r["city"] = f.get("city")
    if kind == "partner":
        r["partner"] = doc.get("title")
    r["fetched_at"] = (doc.get("provenance") or {}).get("fetched_at")
    r["quality_flags"] = list(doc.get("quality_flags") or [])
    raw = build_chunks(doc)
    chunks = []
    for c in raw:
        old = c["metadata"]
        chunks.append({"id": c["id"], "text": c["text"], "metadata": _meta_for(r, old.get("chunk_type", "other"), c["id"])})
    r["full_text"] = f"# {r['title']}\n" + "\n\n".join(c["text"] for c in raw)
    r["rich"] = {"faqs": [{"q": f["q"], "a": f["a"]} for f in doc.get("faq") or []][:30], "gallery": [], "amenities": [],
                 "key_features": [], "why_invest": None, "location": None, "about": ((doc.get("about") or {}).get("text") or "")[:1200]}
    r["summary"] = (raw[0]["text"][:400] if raw else r["title"]) or ""
    return r, _finish(r, chunks)


# ---------------------------------------------------------------- loader
def load_darglobal(data_dir: str | Path) -> tuple[list[dict], list[dict], dict]:
    data_dir = Path(data_dir)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "darglobal" / "docs").glob("*.json"))]
    collections = json.loads((data_dir / "darglobal" / "one_of_one_collections.json").read_text(encoding="utf-8"))
    records, chunks = [], []
    report = {"docs": len(docs), "projects": 0, "one_of_one": 0, "info": 0, "skipped_duplicates": []}
    one_urls = set()
    for d in docs:
        if "/one-of-one/" in d["url"].lower():
            one_urls.add(d["url"].lower())
    for d in docs:
        if d["kind"] == "project":
            r = project_record(d)
            records.append(r)
            chunks += project_chunks(d, r)
            report["projects"] += 1
        elif d["url"].lower() in one_urls:
            continue                                          # built from the collection data below (richer)
        else:
            r, ch = info_record(d)
            records.append(r)
            chunks += ch
            report["info"] += 1
    for slug, a in collections.items():
        r, ch = one_of_one_record(slug, a, one_urls)
        records.append(r)
        chunks += ch
        report["one_of_one"] += 1
    report["one_of_one_pages_seen"] = len(one_urls)
    return records, chunks, report
