"""Question-first answering: pick the passage that actually answers the question, before any property is suggested.

For a question about a named development we score every answerable "unit" of its parent record (FAQ pairs, why-invest
bullets, location, amenities, about/summary) against the question and lead with the best one. Property listings and
related developments come second.
"""
from __future__ import annotations

import re

STOP = set("""a an the of in on at to for and or is are was were be been do does did can could would should will what which who whom
whose when where why how this that these those it its with from by about me my i you your tell give show please there any
some more also than then so as if into over under around near development developments project projects property properties""".split())

# query word -> related words that may appear in the answer (kept deliberately small and domain specific)
SYN = {
    "complet": {"handover", "delivery", "deliver", "ready", "construction", "finish", "expected", "scheduled", "open"},
    "handover": {"complet", "delivery", "construction", "scheduled"},
    "value": {"price", "cost", "pricing", "worth", "aed", "usd", "sar", "starting", "from"},
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
    "return": {"yield", "roi", "invest", "income"},
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

PRICE_Q = re.compile(r"\b(price|prices|pricing|cost|costs|how much|value|worth|starting from|starts? at|budget)\b", re.I)


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def toks(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9/]+", text.lower()) if w not in STOP and len(w) > 1]


def _prefix_in(stem: str, bag: set[str]) -> bool:
    return stem in bag or any(len(stem) >= 5 and b.startswith(stem[:5]) and len(b) >= 5 for b in bag)


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


def _score(qt: list[str], unit: dict) -> float:
    if not qt:
        return 0.0
    body = set(toks(unit["text"]))
    qbag = set(toks(unit["q"]))
    s = 0.0
    for w in qt:
        exp = {_stem(x) for x in SYN.get(w, ())}
        if _prefix_in(w, qbag):
            s += 2.0
        elif exp & qbag:
            s += 1.4
        if _prefix_in(w, body):
            s += 1.0
        elif exp & body:
            s += 0.5
    if unit["kind"] == "faq":
        s *= 1.15
    elif unit["kind"] == "about":
        s *= 0.6
    return s / (len(qt) * 3.0)          # 0..~1


def best_units(query: str, parent: dict, name_words: set[str] | None = None, n: int = 2, min_score: float = 0.22) -> list[tuple[float, dict]]:
    drop = {_stem(w) for w in (name_words or set())}
    qt = [w for w in toks(query) if w not in drop]
    scored = sorted(((_score(qt, u), u) for u in units(parent)), key=lambda x: -x[0])
    return [(s, u) for s, u in scored[:n] if s >= min_score]


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
