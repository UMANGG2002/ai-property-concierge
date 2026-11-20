"""DarGlobal section chunking and parent flattening (works on the parsed documents in data/darglobal/docs).

build_chunks(doc) -> section chunks (summary, facts, overview, key features, amenities, why invest, location, gallery,
one per FAQ, property cards, related projects...), each prefixed with a breadcrumb so it reads on its own.
flatten_project(doc) -> the whole project as one parent text (cards, drawer, "tell me about" answers).
"""
from __future__ import annotations

import re
from collections import Counter

def words(s: str) -> int:
    return len(s.split())





MAX_WORDS = 320


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s]


def _split(text: str, limit: int = MAX_WORDS) -> list[str]:
    """Split on paragraph, then sentence boundaries; never mid-sentence."""
    if words(text) <= limit:
        return [text]
    out, cur = [], ""
    for para in text.split("\n"):
        for unit in ([para] if words(para) <= limit else _sentences(para)):
            if cur and words(cur) + words(unit) > limit:
                out.append(cur.strip())
                cur = ""
            cur += ("\n" if cur and para else " ") + unit if cur else unit
    if cur.strip():
        out.append(cur.strip())
    return out


def _prefix(doc: dict, *tail: str) -> str:
    site = {"darglobal": "DarGlobal", "wasalt": "Wasalt"}.get(doc["site"], doc["site"])
    path = [p for p in doc["provenance"]["nav_path"] if p]
    if doc["kind"] == "partner":
        path = ["Collaborations"]
    elif doc["kind"] in ("article", "press"):
        path = ["Newsroom", "News" if doc["kind"] == "press" else "Blog"]
    elif not path:
        path = ["Real Estate developments"] if doc["kind"] == "project" else []
    if doc["kind"] == "project":
        f = doc["facts"]
        path = [p for p in path if p != doc["title"]]
        if f.get("country") and f["country"] not in path:
            path = path[:2] + [f["country"]] if len(path) >= 2 else path + [f["country"]]
    return " > ".join([site] + path + [doc["title"] or ""] + [t for t in tail if t])


_SQM = re.compile(r"\s*(sq\.?\s*m|sqm)\.?\s*$", re.I)


def _meta(doc: dict, ctype: str, extra: dict | None = None) -> dict:
    f = doc.get("facts") or {}
    m = {
        "doc_id": doc["id"], "chunk_type": ctype, "kind": doc["kind"], "source": doc["site"], "language": doc["language"],
        "name": doc["title"], "url": doc["url"], "menu_path": doc["provenance"]["nav_path"],
        "country": f.get("country"), "city": f.get("city"), "region": f.get("region"),
        "property_type": f.get("property_type"), "status": f.get("status"),
        "unit_category": f.get("unit_category"), "parent": (f.get("parent") or {}).get("title"),
        "bedrooms_min": f.get("bedrooms_min"), "bedrooms_max": f.get("bedrooms_max"),
        "area_sqm_min": f.get("area_sqm_min"), "area_sqm_max": f.get("area_sqm_max"),
        "price": f.get("price"), "currency": f.get("currency"),
        "partners": [p["name"] for p in doc.get("partners") or [] if p.get("name")] or ([doc["title"]] if doc["kind"] == "partner" else []),
        "page_updated": ((doc.get("dates") or {}).get("updated") or "")[:10] or None,
        "fetched_at": doc["provenance"]["fetched_at"][:10], "snapshot": doc["provenance"]["snapshot"],
        "quality_flags": doc["quality_flags"],
    }
    if extra:
        m.update(extra)
    out = {k: v for k, v in m.items() if v not in (None, [], "")}
    if doc["kind"] == "project" and f.get("price") is None:
        out["price"] = None                      # explicit: price not published
    return out


def _imgs(items: list[dict], n: int = 4) -> list[dict]:
    return [{"url": i["url"], "caption": i.get("alt") or i.get("caption") or ""} for i in items[:n] if i.get("url")]


def _adder(doc: dict, ch: list):
    """add(ctype, title, text, extra): splits long text, builds unique ids, prefixes the breadcrumb."""
    used: set[str] = set()

    def add(ctype, title, text, extra=None):
        for i, part in enumerate(_split(text)):
            cid = f"{doc['id']}#faq-{extra['faq_no']}" if ctype == "faq" else f"{doc['id']}#{ctype}"
            if i:
                cid += f"-{i + 1}"
            base, n = cid, 1
            while cid in used:
                n += 1
                cid = f"{base}~{n}"
            used.add(cid)
            ch.append({"id": cid, "text": f"{_prefix(doc, title)}. {part}", "metadata": _meta(doc, ctype, extra)})
    return add


def chunk_project(doc: dict) -> list[dict]:
    f, ch = doc["facts"], []
    add = _adder(doc, ch)
    where = ", ".join(x for x in (f.get("city"), f.get("region"), f.get("country")) if x)
    bits = [f"{doc['title']}" + (f", {doc['tagline']}" if doc.get("tagline") and doc["tagline"].lower() not in doc["title"].lower() else "") + "."]
    if where:
        bits.append(f"Location: {where}.")
    for lab, key in (("Property type", "property_type"), ("Status", "status"), ("Unit type", "unit_type")):
        if f.get(key):
            bits.append(f"{lab}: {f[key]}.")
    if f.get("parent"):
        bits.append(f"Part of {f['parent']['title']}" + (f" ({f['unit_category']})" if f.get("unit_category") else "") + ".")
    if f.get("area_raw"):
        area = _SQM.sub("", f["area_raw"]).strip()
        bits.append(f"Area: {area} sqm.")
    if f.get("area_text_ranges"):
        bits.append("The page text also states " + "; ".join(f"{int(a)}–{int(b)} sqm" for a, b in f["area_text_ranges"]) + ".")
    if f.get("partner"):
        bits.append(f"Partner: {f['partner']}" + (f" ({f['partner_role']})" if f.get("partner_role") else "") + ".")
    bits.append(f"Price: {f['price_raw']}." if f.get("price_raw") else "Price: not published on this page.")
    add("facts", "Key facts", " ".join(bits))
    if doc["about"]["text"]:
        add("overview", "Overview", (doc["about"]["title"] + ". " if doc["about"]["title"] else "") + doc["about"]["text"],
            {"images": _imgs(doc["gallery"], 3)})
    if doc["key_features"]:
        lines = [f"{k['title']}" + (f" - {k['distance']}" if k["distance"] else "") + (f". {k['text']}" if k["text"] else "") for k in doc["key_features"]]
        head = " ".join(x for x in (doc.get("key_features_title"), doc.get("key_features_text")) if x)
        add("key_features", "Key features", (head + "\n" if head else "") + "\n".join(lines),
            {"images": [{"url": k["image"], "caption": k["title"]} for k in doc["key_features"] if k["image"]][:6]})
    if doc["amenities"]:
        add("amenities", "Amenities", ((doc["amenities_text"] + "\n") if doc["amenities_text"] else "") + "Amenities: " + "; ".join(doc["amenities"]) + ".")
    wi = doc["why_invest"]
    if wi["bullets"] or wi["intro"]:
        add("why_invest", "Why invest",
            f"Developer's investment claims ({wi['claimed_by']} states):\n" + (wi["intro"] + "\n" if wi["intro"] else "") + "\n".join(f"- {b}" for b in wi["bullets"]),
            {"claimed_by": wi["claimed_by"]})
    lo = doc["location"]
    if lo["text"] or lo["highlights"] or lo.get("lat"):
        txt = lo["text"] + ("\n" if lo["text"] else (f"{doc['title']} is located in {where}.\n" if where else ""))
        if lo["highlights"]:
            txt += "Key location highlights (time or distance): " + "; ".join(lo["highlights"]) + ". "
        if lo.get("lat"):
            txt += f"Coordinates: {lo['lat']}, {lo['lng']}."
        add("location", "Location", txt.strip(), {"lat": lo.get("lat"), "lng": lo.get("lng")})
    core = [g for g in doc["gallery"] if g["role"] == "gallery"]
    if core:
        caps = [g["alt"] for g in core if g["alt"]]
        add("gallery", "Photo gallery", f"Photo gallery with {len(core)} images" + (": " + "; ".join(dict.fromkeys(caps)) + "." if caps else "."),
            {"images": _imgs(core, 15), "image_count": len(core), "other_image_count": len(doc["gallery"]) - len(core)})
    pend: list[dict] = []

    def flush_cards():
        if pend:
            txt = "\n".join(" ".join(x for x in (c["category"], c["title"], c["subtitle"], c["text"]) if x) for c in pend)
            add("property_card", pend[0]["title"] if len(pend) == 1 else "Property types", txt,
                {"card_urls": [c["url"] for c in pend if c["url"]],
                 "images": _imgs([{"url": c["image"], "alt": c["title"]} for c in pend if c["image"]], 4)})
            pend.clear()
    for c in doc.get("cards") or []:
        if not (c["title"] or c["text"]):
            continue
        pend.append(c)
        if words(" ".join(x["text"] for x in pend)) >= 30:       # small cards share a chunk; big ones stand alone
            flush_cards()
    flush_cards()
    for i, q in enumerate(doc["faq"], 1):
        add("faq", "FAQ", f"Q: {q['q']} A: {q['a']}", {"faq_no": i, "question": q["q"]})
    return ch


def chunk_partner(doc: dict) -> list[dict]:
    ch = []
    add = _adder(doc, ch)
    projs = doc["related"]["projects"]
    head = f"{doc['title']} is a DarGlobal collaboration partner."
    add("overview", "Overview", " ".join(x for x in (head, doc["about"]["title"], doc["about"]["text"]) if x))
    if projs:
        add("related_projects", "Related projects",
            f"Projects with {doc['title']}: " + "; ".join(f"{p['title']}" + (f" ({p['location']})" if p["location"] else "") for p in projs) + ".",
            {"related_project_urls": [p["url"] for p in projs if p["url"]], "images": _imgs([{"url": p["image"], "alt": p["title"]} for p in projs if p["image"]], 4)})
    if doc["articles"]:
        add("articles", "Articles and news", "Related articles: " + "; ".join(f"{a['title']}" + (f" ({str(a['date'])[:10]})" if a["date"] else "") for a in doc["articles"]) + ".")
    for i, q in enumerate(doc["faq"], 1):
        add("faq", "FAQ", f"Q: {q['q']} A: {q['a']}", {"faq_no": i, "question": q["q"]})
    return ch


def chunk_article(doc: dict) -> list[dict]:
    body, ch = doc["body"], []
    paras = [p for p in body.split("\n") if p.strip()]
    groups, cur = [], []
    for p in paras:
        if cur and words(" ".join(cur)) + words(p) > 260:
            groups.append(cur)
            cur = []
        cur.append(p)
    if cur:
        groups.append(cur)
    if len(groups) > 1 and words(" ".join(groups[-1])) < 60:
        groups[-2] += groups.pop()
    for i, g in enumerate(groups, 1):
        ch.append({"id": f"{doc['id']}#body-{i}", "text": f"{_prefix(doc, f'Part {i}')}. " + " ".join(g),
                   "metadata": _meta(doc, "article_body", {"part": i, "parts": len(groups), "published": (doc['dates'].get('date') or '')[:10] or None})})
    return ch


def chunk_page(doc: dict) -> list[dict]:
    """One chunk per section; sections under 40 words are merged with the following ones."""
    merged, buf = [], None
    for s in doc["sections"]:
        buf = dict(s) if buf is None else {"name": buf["name"], "topic": buf["topic"], "title": f"{buf['title']} / {s['title']}",
                                          "text": buf["text"] + "\n" + s["text"]}
        if words(buf["text"]) >= 40:
            merged.append(buf)
            buf = None
    if buf:
        if merged and words(buf["text"]) < 40:
            merged[-1] = {**merged[-1], "title": f"{merged[-1]['title']} / {buf['title']}", "text": merged[-1]["text"] + "\n" + buf["text"]}
        else:
            merged.append(buf)
    ch, ids = [], Counter()
    for s in merged:
        for i, part in enumerate(_split(s["text"])):
            cid = f"{doc['id']}#{s['name']}" + (f"-{i + 1}" if i else "")
            ids[cid] += 1
            ch.append({"id": cid if ids[cid] == 1 else f"{cid}~{ids[cid]}",
                       "text": f"{_prefix(doc, s['title'])}. {part}", "metadata": _meta(doc, s["topic"], {"section": s["name"]})})
    return ch


def build_chunks(doc: dict) -> list[dict]:
    fn = {"project": chunk_project, "partner": chunk_partner, "article": chunk_article, "press": chunk_article}.get(doc["kind"], chunk_page)
    ch = fn(doc)
    if doc.get("children"):
        by_cat: dict[str, list[str]] = {}
        for c in doc["children"]:
            by_cat.setdefault(c["unit_category"], [])
            if c["title"].title() not in by_cat[c["unit_category"]]:
                by_cat[c["unit_category"]].append(c["title"].title())
        txt = (f"{doc['title']} includes {len(doc['children'])} projects. "
               + " ".join(f"{k.capitalize()}: {', '.join(v)}." for k, v in by_cat.items()))
        ch.append({"id": f"{doc['id']}#sub_projects", "text": f"{_prefix(doc, 'Sub-projects')}. {txt}",
                   "metadata": _meta(doc, "sub_projects", {"children": [{"title": c["title"], "unit_category": c["unit_category"], "url": c["url"]} for c in doc["children"]]})})
    if doc["kind"] == "project":
        ch.insert(0, {"id": f"{doc['id']}#summary", "text": summary_card(doc), "metadata": _meta(doc, "summary")})
        for c in ch:
            c["metadata"]["parent_id"] = doc["id"]        # -> data/flat/<site>/<id>.json
    return ch


# ---------------------------------------------------------------- parent text




def _where(f: dict) -> str:
    return ", ".join(x for x in (f.get("city"), f.get("region"), f.get("country")) if x)


def _area(f: dict) -> str | None:
    if not f.get("area_raw"):
        return None
    return re.sub(r"\s*(sq\.?\s*m|sqm)\.?\s*$", "", f["area_raw"], flags=re.I).strip() + " sqm"


def flatten_project(doc: dict) -> dict:
    f, L = doc["facts"], []
    L.append(f"# {doc['title']}" + (f" - {doc['tagline']}" if doc.get("tagline") else ""))
    facts = [("Country", f.get("country")), ("Location", _where(f) or None), ("Community", f.get("community")), ("Part of", (f.get("parent") or {}).get("title")),
             ("Category", f.get("unit_category")),
             ("Property type", f.get("property_type")), ("Status", f.get("status")), ("Unit type", f.get("unit_type")),
             ("Area", _area(f)), ("Price", f.get("price_raw") or "not published"), ("Developer partner", f.get("partner"))]
    L.append("## Project details\n" + "\n".join(f"- {k}: {v}" for k, v in facts if v))
    if (doc.get("about") or {}).get("text"):
        L.append("## About\n" + (f"{doc['about']['title']}\n" if doc["about"].get("title") else "") + doc["about"]["text"])
    if doc.get("key_features"):
        L.append("## Key features\n" + "\n".join(
            f"- {k['title']}" + (f" ({k['duration']})" if k.get("duration") else "") + (f": {k['text']}" if k.get("text") else "")
            for k in doc["key_features"]))
    if doc.get("amenities"):
        L.append("## Amenities\n" + "\n".join(f"- {a}" for a in doc["amenities"]))
    wi = doc.get("why_invest") or {}
    if wi.get("bullets") or wi.get("intro"):
        L.append(f"## {wi.get('title') or 'Why invest'}\n" + (wi["intro"] + "\n" if wi.get("intro") else "")
                 + "\n".join(f"- {b}" for b in wi.get("bullets", [])))
    lo = doc.get("location") or {}
    if lo.get("text") or lo.get("highlights"):
        L.append("## Location\n" + (lo["text"] + "\n" if lo.get("text") else "")
                 + "\n".join(f"- {h['title']}" + (f" ({h['distance']})" if h.get("distance") else "") if isinstance(h, dict) else f"- {h}"
                             for h in lo.get("highlights", [])))
    if doc.get("children"):
        L.append("## Sub-projects\n" + "\n".join(f"- {c['title'].title()} ({c['unit_category']}): {c['url']}" for c in doc["children"]))
    cards = [c for c in doc.get("cards") or [] if c.get("title") or c.get("text")]
    if cards and not doc.get("key_features") and not doc.get("children"):
        L.append("## Highlights\n" + "\n".join(f"- {c['title']}: {c['text']}" if c.get("text") else f"- {c['title']}" for c in cards))
    if doc.get("faq"):
        L.append("## FAQ\n" + "\n\n".join(f"Q: {q['q']}\nA: {q['a']}" for q in doc["faq"]))
    elif (doc.get("seo") or {}).get("description") and not (doc.get("about") or {}).get("text"):
        L.append("## Summary\n" + doc["seo"]["description"])
    text = "\n\n".join(L)

    images = []
    seen = set()

    def img(url, alt, role):
        if url and url not in seen:
            seen.add(url)
            images.append({"url": url, "alt": alt or "", "role": role})
    for g in doc.get("gallery") or []:
        img(g["url"], g.get("alt"), g.get("role", "gallery"))
    for k in doc.get("key_features") or []:
        for im in k.get("images") or []:
            img(im["url"] if isinstance(im, dict) else im, k["title"], "key_feature")
    for c in doc.get("cards") or []:
        img(c.get("image"), c.get("title"), "card")
    for p in doc.get("progress_images") or []:
        img(p["url"] if isinstance(p, dict) else p, "", "progress")
    meta = {"doc_id": doc["id"], "kind": "project", "name": doc["title"], "url": doc["url"], "country": f.get("country"),
            "city": f.get("city"), "property_type": f.get("property_type"), "status": f.get("status"),
            "menu_path": doc["provenance"]["nav_path"], "quality_flags": doc["quality_flags"],
            "completeness": doc["completeness"]["score"], "fetched_at": doc["provenance"]["fetched_at"][:10]}
    return {"id": doc["id"], "text": text, "words": words(text), "metadata": {k: v for k, v in meta.items() if v not in (None, "", [])},
            "images": images, "image_count": len(images)}


def summary_card(doc: dict) -> str:
    """~100-200 words: what it is, where, key facts, one line each on amenities / why-invest. For listing + comparison queries."""
    f = doc["facts"]
    s = [f"{doc['title']} is a {f.get('property_type') or 'property development'}" + (f" in {_where(f)}" if _where(f) else "")
         + (f", {f['status'].lower()}" if f.get("status") else "") + "."]
    if _area(f):
        s.append(f"Area: {_area(f)}.")
    if f.get("community"):
        s.append(f"Part of {f['community']}.")
    about = (doc.get("about") or {}).get("text", "")
    if about:
        sents, out = re.split(r"(?<=[.!?])\s+|\n+", " ".join(about.split("\n"))), ""
        for t in sents:
            if out and len(out) + len(t) > 350:
                break
            out = f"{out} {t}".strip()
        s.append(out if out.endswith((".", "!", "?")) else out + ".")
    if doc.get("amenities"):
        s.append("Amenities include " + ", ".join(doc["amenities"][:6]) + ".")
    b = (doc.get("why_invest") or {}).get("bullets") or []
    if b:
        s.append("Why invest: " + "; ".join(b[:3]) + ".")
    s.append("Price: " + (f.get("price_raw") or "not published") + ".")
    return " ".join(s)
