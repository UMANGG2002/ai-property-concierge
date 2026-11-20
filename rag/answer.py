"""Question-first answering: pick the passage that actually answers the question, before any property is suggested.

For a question about a named development we score every answerable "unit" of its parent record (FAQ pairs, why-invest
bullets, location, amenities, about/summary) against the question and lead with the best one. Property listings and
related developments come second.
"""
from __future__ import annotations

import re

STOP = set("""a an the of in on at to for and or is are was were be been do does did can could would should will what which who whom
whose when where why how this that these those it its with from by about me my i you your tell give show please there any
some more also than then so as if into over under around near development developments project projects property properties long much many good need want like get got know let thinking looking expect interested kind wondering really actually just buyer buyers buy purchase""".split())

# query word -> related words that may appear in the answer (kept deliberately small and domain specific)
SYN = {
    "move": {"handover", "complet", "delivery", "ready", "scheduled"},
    "ready": {"complet", "handover", "delivery", "scheduled"},
    "overseas": {"foreign", "international", "freehold", "ownership", "nationalities", "nationals"},
    "buyer": {"ownership", "freehold", "purchase"},
    "drive": {"minutes", "away", "distance", "airport"},
    "big": {"sqm", "size", "area", "range", "ranging"},
    "grow": {"growth", "capital", "increase", "appreciation"},
    "value": {"price", "cost", "pricing", "worth", "aed", "usd", "sar", "starting", "from", "capital"},
    "foreign": {"freehold", "ownership", "nationalities", "nationals", "expat"},
    "foreigner": {"foreign", "freehold", "ownership", "nationalities", "nationals", "expat", "international"},
    "abroad": {"freehold", "ownership", "nationalities", "nationals", "foreign"},
    "own": {"freehold", "ownership"},
    "residency": {"visa", "golden", "residence", "lifetime"},
    "rental": {"yield", "income", "rent", "return", "roi", "occupancy"},
    "children": {"kids", "play", "family", "school", "playground"},
    "family": {"kids", "play", "school", "children", "playground"},
    "swim": {"pool", "swimming"},
    "walk": {"minutes", "beach", "promenade", "trail"},
    "complet": {"handover", "delivery", "deliver", "ready", "construction", "finish", "expected", "scheduled", "open"},
    "handover": {"complet", "delivery", "construction", "scheduled"},
    "price": {"pricing", "cost", "value", "starting", "aed", "usd", "sar", "from"},
    "cost": {"price", "pricing", "value", "starting"},
    "pay": {"payment", "instalment", "installment", "plan", "deposit", "down"},
    "payment": {"plan", "instalment", "installment", "deposit", "down", "pay"},
    "golf": {"course", "club", "fairway", "links"},
    "club": {"golf", "membership", "member", "clubhouse"},
    "beach": {"beachfront", "shore", "sea", "waterfront", "sand", "coast"},
    "pool": {"swimming", "infinity", "lagoon"},
    "spa": {"wellness", "sauna", "massage"},
    "gym": {"fitness", "training"},
    "kid": {"children", "play", "family"},
    "visa": {"residency", "golden", "residence"},
    "tax": {"taxation", "income", "property"},
    "invest": {"investment", "return", "yield", "roi", "capital", "growth", "appreciation"},
    "return": {"yield", "roi", "invest", "income", "growth", "capital", "appreciation", "occupancy", "rental"},
    "locat": {"situated", "island", "minutes", "away", "district", "area", "address"},
    "airport": {"minutes", "away", "drive", "distance"},
    "bedroom": {"bed", "beds", "br", "unit", "units", "sizes", "range"},
    "size": {"sqm", "sqft", "area", "range", "ranging", "square"},
    "unit": {"sizes", "types", "bedroom", "residences", "apartments", "villas"},
    "design": {"designed", "interiors", "interior", "architect", "partner", "brand", "collaboration"},
    "brand": {"partner", "designed", "interiors", "collaboration"},
    "partner": {"brand", "collaboration", "designed", "interiors"},
    "secur": {"24/7", "guard", "gated", "safety"},
    "furnish": {"furnished", "fixtures", "interiors", "turnkey"},
    "rent": {"rental", "yield", "income", "lease"},
}

PRICE_Q = re.compile(r"\b(price|prices|pricing|cost|costs|how much (does|do|is|would|will)|value of|worth|starting from|starts? at|budget)\b", re.I)


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def toks(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9/]+", text.lower()) if w not in STOP and len(w) > 1]


def _prefix_in(stem: str, bag: set[str]) -> bool:
    return stem in bag or (len(stem) >= 6 and any(b.startswith(stem) for b in bag))


def units(parent: dict) -> list[dict]:
    """Answerable passages of one parent record."""
    rich = parent.get("rich") or {}
    title = parent.get("title") or ""
    out = []
    for f in rich.get("faqs") or []:
        out.append({"kind": "faq", "q": f["q"], "text": f["a"]})
    wi = rich.get("why_invest") or {}
    for b in wi.get("bullets") or []:
        out.append({"kind": "why_invest", "q": "why invest", "text": b.rstrip(" .;")})
    if wi.get("intro"):
        out.append({"kind": "why_invest", "q": "why invest market", "text": wi["intro"]})
    amen = rich.get("amenities") or (parent.get("metadata") or {}).get("amenities") or []
    if amen:
        out.append({"kind": "amenities", "q": "amenities facilities features",
                    "text": "Amenities include " + ", ".join(a.rstrip('.') for a in amen[:14]) + "."})
    loc = (rich.get("location") or {}).get("text")
    if loc:
        out.append({"kind": "location", "q": "where location situated", "text": loc})
    kf = [f"{k['title']} ({k['detail']})" if isinstance(k, dict) and k.get("detail") else (k.get("title") if isinstance(k, dict) else str(k))
          for k in rich.get("key_features") or []]
    if kf:
        out.append({"kind": "key_features", "q": "key features", "text": "; ".join(x for x in kf[:8] if x)})
    hl = [f"{h['title']} {h['detail']}" for h in (rich.get("location") or {}).get("highlights") or [] if isinstance(h, dict) and h.get("title")]
    if hl and loc:
        out[-1 if not kf else -2]["text"] = loc + " Nearby: " + ", ".join(hl[:8]) + "."
    about = rich.get("about") or parent.get("summary")
    if about:
        out.append({"kind": "about", "q": "about overview what is", "text": about})
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 2]


def _hit(w: str, bag: set[str]) -> float:
    """1.0 for the word itself, 0.55 for a domain synonym, else 0."""
    if _prefix_in(w, bag):
        return 1.0
    return 0.55 if {_stem(x) for x in SYN.get(w, ())} & bag else 0.0


def _score(qt: list[str], unit: dict) -> float:
    """0..1: how well one passage answers the question. FAQs: question + answer overlap; prose: its single best sentence."""
    if not qt:
        return 0.0
    if unit["kind"] == "faq":
        body, qbag = set(toks(unit["text"])), set(toks(unit["q"]))
        s = sum(2.0 * _hit(w, qbag) + _hit(w, body) for w in qt) / (3.0 * len(qt))
        return min(1.0, s * 1.15)
    best, bestsent = 0.0, ""
    for sent in _sentences(unit["text"]) or [unit["text"]]:
        bag = set(toks(sent))
        sc = sum(_hit(w, bag) for w in qt) / len(qt)
        if sc > best:
            best, bestsent = sc, sent
    unit["focus_sentence"] = bestsent
    return best * (0.6 if unit["kind"] == "about" else 1.0)


def best_units(query: str, parent: dict, name_words: set[str] | None = None, n: int = 2, min_score: float = 0.34) -> list[tuple[float, dict]]:
    drop = {_stem(w) for w in (name_words or set())}
    qt = [w for w in toks(query) if w not in drop]
    us = units(parent)
    present: set[str] = set()                                       # only words the property's own text can speak to count
    for u in us:
        present |= set(toks(u["text"])) | set(toks(u["q"]))
    qt = [w for w in qt if _hit(w, present)]
    scored = sorted(((_score(qt, u), u) for u in us), key=lambda x: -x[0])
    out = []
    for s, u in scored[:n]:
        if s >= min_score:
            u["focus"] = focus(u, qt)
            out.append((s, u))
    return out


def focus(unit: dict, qt: list[str], max_words: int = 60) -> str:
    """FAQ answers whole; prose as its best sentence plus the next one when it still fits."""
    if unit["kind"] in ("faq", "amenities", "why_invest", "key_features"):
        return unit["text"]
    sents = _sentences(unit["text"])
    best = unit.get("focus_sentence") or (sents[0] if sents else unit["text"])
    out = [best]
    if best in sents:
        k = sents.index(best)
        if k + 1 < len(sents) and len((best + " " + sents[k + 1]).split()) <= max_words:
            out.append(sents[k + 1])
    return " ".join(out)


def name_words(parent: dict) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (parent.get("title") or "").lower()))


def snippet(text: str, query: str, words: int = 28) -> str:
    """The sentence of `text` that overlaps the question most, capped to `words` words."""
    qt = {w for w in toks(query)}
    exp = {_stem(x) for w in qt for x in SYN.get(w, ())}
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    best = max(sents, key=lambda x: (sum(1 for t in set(toks(x)) if _prefix_in(t, qt)) * 2 + len(exp & set(toks(x)))), default=text)
    ws = best.split()
    return " ".join(ws[:words]) + (" …" if len(ws) > words else "")


AMEN_WORDS = {_stem(w) for w in """golf pool swimming spa gym fitness sauna steam jacuzzi beach cinema theatre kid child children play playground nursery
school tennis padel squash yoga pilates concierge security guard restaurant cafe lounge clubhouse marina yacht helipad garden park trail
hiking cycling elevator lift parking valet wellness massage retail shopping clinic mosque kindergarten chess gaming""".split()}
