"""Query understanding: natural-language question -> Filters + project mentions.

Rule-based parsing always runs (deterministic, instant, never fails). An LLM parser can optionally fill gaps
(RAG_LLM_PARSER=1); its output is validated against the corpus vocabulary and never overrides numbers the rules found.
Vocabulary (countries, cities, zones, districts, partners, project names) is built from the corpus, so new data
extends the parser without code changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .filters import Filters
from .paths import PARENTS

PARSE_TYPES = {  # stricter than schema.PROPERTY_TYPES: only words a user types to ask for a type
    "penthouse": r"penthouses?", "townhouse": r"town ?houses?", "duplex": r"duplex(?:es)?", "mansion": r"mansions?",
    "hotel_room": r"hotel (?:rooms?|apartments?|units?)|rooms? (?:&|and) suites", "villa": r"villas?|فيلا|فلل",
    "apartment": r"apartments?|flats?|شقة|شقق|condos?", "floor": r"(?:residential )?floors? (?:unit|for sale)|\bfloor unit",
    "land": r"lands?|plots?|أرض|اراضي", "building": r"buildings?(?! year)|عمارة", "studio": r"studios?",
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
MONEY = r"(\d+(?:[.,]\d+)*)\s*(k|thousand|m|mn|mil|million|b|billion)?\s*(?:sar|riyals?|sr|aed|usd|\$|eur|€|ريال)?"
ZONE_WORDS = ("north", "south", "east", "west", "central", "center")


def _money(num: str, unit: str | None) -> float:
    n = float(num.replace(",", "")) if num.count(",") and len(num.split(",")[-1]) == 3 else float(num.replace(",", "."))
    unit = (unit or "").lower()
    if unit in ("k", "thousand"):
        n *= 1_000
    elif unit in ("m", "mn", "mil", "million"):
        n *= 1_000_000
    elif unit in ("b", "billion"):
        n *= 1_000_000_000
    elif n < 1000:                               # "budget 2" / "under 2.5" in a property context = millions
        n *= 1_000_000
    return n


def _key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"^(al|an|ar|as|at|ad|ash|az|adh|ath|as)[-\s]", "", s)
    s = re.sub(r"[^a-z0-9؀-ۿ]+", "", s)
    return s


@dataclass
class Vocab:
    countries: dict[str, str] = field(default_factory=dict)
    cities: dict[str, str] = field(default_factory=dict)
    zones: dict[str, str] = field(default_factory=dict)
    districts: dict[str, str] = field(default_factory=dict)
    partners: dict[str, str] = field(default_factory=dict)
    projects: dict[str, str] = field(default_factory=dict)        # alias -> record id
    brands: dict[str, list[str]] = field(default_factory=dict)    # brand word -> record ids (soft focus)
    children: dict[str, list[str]] = field(default_factory=dict)  # parent title -> child record ids

    @classmethod
    def from_parents(cls, parents: dict) -> "Vocab":
        v = cls()
        v.countries.update({"uae": "United Arab Emirates", "emirates": "United Arab Emirates", "ksa": "Saudi Arabia",
                            "saudi": "Saudi Arabia", "uk": "United Kingdom", "england": "United Kingdom",
                            "السعودية": "Saudi Arabia", "الإمارات": "United Arab Emirates", "عمان": "Oman", "قطر": "Qatar"})
        arabic_cities = {"الرياض": "Riyadh", "جدة": "Jeddah", "دبي": "Dubai", "مسقط": "Muscat", "الدوحة": "Doha"}
        by_title = {}
        for rid, p in parents.items():
            if p.get("duplicate_of"):
                continue                                     # answered by the canonical record
            m = p["metadata"]
            if m.get("country"):
                v.countries[m["country"].lower()] = m["country"]
            if m.get("city") and len(m["city"]) > 2:
                v.cities[m["city"].lower()] = m["city"]
            if m.get("zone") and m["zone"].lower() != (m.get("city") or "").lower() and " " in m["zone"]:
                v.zones[m["zone"].lower()] = m["zone"]
            if m.get("district"):
                v.districts[_key(m["district"])] = m["district"]
            kind = m.get("record_kind")
            if kind in ("development_project", "one_of_one", "partner"):
                for b in BRANDS:
                    blob = f"{p.get('title') or ''} {m.get('partner') or ''}".lower()
                    if b in blob:
                        v.brands.setdefault(b, []).append(rid)
            if kind in ("development_project", "one_of_one") and p.get("title"):
                by_title[p["title"]] = rid
                for alias in _aliases(p["title"], p.get("url", "")):
                    v.projects.setdefault(alias, rid)
            if m.get("parent_project"):
                v.children.setdefault(m["parent_project"].lower(), []).append(rid)
        for ar, en in arabic_cities.items():
            if en.lower() in v.cities:
                v.cities[ar] = v.cities[en.lower()]
        return v


BRANDS = ("trump", "missoni", "lamborghini", "pagani", "fendi", "aston martin", "elie saab", "mouawad", "w hotels",
          "marriott", "paramount", "nickelodeon", "insignia")


def _aliases(title: str, url: str) -> list[str]:
    t = title.lower().strip()
    out = {t}
    for sep in (",", " – ", " - ", " by ", " interiors by ", " design inspired", "|", "(", ":"):
        head = t.split(sep)[0].strip()
        if len(head) >= 4:
            out.add(head)
    slug = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").lower()
    if len(slug) >= 4 and not slug.isdigit():
        out.add(slug)
    generic = {"the", "villas", "villa", "apartments", "residences", "home", "projects", "rooms & suites"}
    return [a for a in out if a not in generic and (len(a) >= 4 or (len(a) >= 3 and any(ch.isdigit() for ch in a)))]


TYPO_VOCAB = ["apartment", "apartments", "villa", "villas", "townhouse", "townhouses", "penthouse", "penthouses",
              "duplex", "mansion", "mansions", "building", "buildings", "bedroom", "bedrooms", "property", "properties",
              "riyadh", "jeddah", "dubai", "north", "south", "east", "west", "central", "under", "between", "million"]


def fix_typos(q: str, extra: list[str] | None = None) -> str:
    """Correct near-miss words ('apartmnets', 'vila', 'riyad') against a small domain vocabulary."""
    import difflib
    vocab = TYPO_VOCAB + (extra or [])
    known = set(vocab)
    out = []
    for w in re.findall(r"\S+", q):
        core = re.sub(r"[^a-z]", "", w)
        if len(core) >= 4 and core not in known:
            m = difflib.get_close_matches(core, vocab, n=1, cutoff=0.82 if len(core) >= 7 else 0.85)
            if m:
                w = w.replace(core, m[0])
        out.append(w)
    return " ".join(out)


@dataclass
class Parsed:
    filters: Filters
    project_ids: list[str] = field(default_factory=list)
    primary_ids: list[str] = field(default_factory=list)       # projects named directly (ranked above their children)
    project_names: list[str] = field(default_factory=list)
    intent: str = "search"                      # search | project_info | out_of_scope
    notes: list[str] = field(default_factory=list)
    llm_used: bool = False

    def to_dict(self) -> dict:
        return {"filters": self.filters.active(), "project_ids": self.project_ids, "primary_ids": self.primary_ids,
                "project_names": self.project_names,
                "intent": self.intent, "notes": self.notes, "llm_used": self.llm_used}


DOMAIN = re.compile(r"propert|villa|apartment|flat|house|home|penthouse|duplex|townhouse|mansion|land|plot|building|"
                    r"bed|bath|sqm|m2|price|sar|riyal|budget|million|sale|buy|rent|invest|project|develop|amenit|"
                    r"location|district|community|darglobal|dar global|wasalt|listing|residence|hotel|golf|beach|"
                    r"payment|handover|brochure|faq|partner|brand|luxury|riyadh|jeddah|dubai|oman|qatar|spain|london|"
                    r"club|course|member|complet|handover|delivery|expected|value|worth|ready|status|launch|unit|tower|marina|island|resort|"
                    r"sea view|interior|design|furnish|security|spa|pool|gym|cinema|visa|tax|yield|roi|return|gated|view|floor|size|"
                    r"عقار|شقة|فيلا|أرض|سعر|للبيع|غرف", re.I)


def parse_rules(query: str, vocab: Vocab) -> Parsed:
    q = " " + fix_typos(re.sub(r"\s+", " ", re.sub(r"\bgold (club|course|villa|villas|resort)", r"golf \1", query.lower()))) + " "
    f = Filters()
    notes = []

    # --- property types
    for canon, pat in PARSE_TYPES.items():
        if re.search(rf"(?<![a-z]){pat}(?![a-z])", q):
            f.property_types.append(canon)
    if "townhouse" in f.property_types and "villa" in f.property_types and re.search(r"villa townhouse|townhouse villa", q):
        f.property_types = ["townhouse"]

    # --- bedrooms
    bed = r"(?:bed(?:room)?s?|br|bhk|غرف)"
    num = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    toint = lambda s: int(s) if s.isdigit() else NUM_WORDS[s]                       # noqa: E731
    if m := re.search(rf"{num}\s*(?:-|to|or|–)\s*{num}\s*[- ]?{bed}", q):
        f.bedrooms_min, f.bedrooms_max = sorted((toint(m.group(1)), toint(m.group(2))))
    elif m := re.search(rf"(?:at least|minimum|min\.?|more than|over|\+)\s*{num}\s*[- ]?{bed}|{num}\s*\+\s*[- ]?{bed}|{num}\s*[- ]?{bed}\s*(?:or more|\+|and above|plus)", q):
        n = toint(next(g for g in m.groups() if g))
        f.bedrooms_min = n + (1 if "more than" in m.group(0) or "over" in m.group(0) else 0)
    elif m := re.search(rf"(?:up to|at most|max(?:imum)?|less than|under|fewer than)\s*{num}\s*[- ]?{bed}", q):
        n = toint(m.group(1))
        f.bedrooms_max = n - (1 if "less than" in m.group(0) or "fewer" in m.group(0) or "under" in m.group(0) else 0)
    elif m := re.search(rf"{num}\s*[- ]?{bed}", q):
        f.bedrooms_min = f.bedrooms_max = toint(m.group(1))

    # --- price
    qp = re.sub(rf"\d+\s*[- ]?{bed}", " ", q)                                         # don't read "4 bedrooms" as money
    qp = re.sub(r"\d+(?:\.\d+)?\s*(?:sqm|sq\.? ?m|m2|m²|square met(?:er|re)s?)", " ", qp)
    if m := re.search(rf"(?:between|from)\s*{MONEY}\s*(?:and|to|-|–)\s*{MONEY}", qp):
        a, b = _money(m.group(1), m.group(2) or m.group(4)), _money(m.group(3), m.group(4))
        f.price_min, f.price_max = min(a, b), max(a, b)
    else:
        if m := re.search(rf"(?:under|below|less than|max(?:imum)?|up to|within|budget(?: of| is)?|cheaper than|not more than|<)\s*{MONEY}", qp):
            f.price_max = _money(m.group(1), m.group(2))
        if m := re.search(rf"(?:over|above|more than|at least|min(?:imum)?|starting (?:from|at)|>)\s*{MONEY}", qp):
            v = _money(m.group(1), m.group(2))
            if v >= 10_000:
                f.price_min = v
        if f.price_min is None and f.price_max is None:
            if m := re.search(rf"(?:around|about|approximately|approx\.?|~|near)\s*{MONEY}", qp):
                v = _money(m.group(1), m.group(2))
                if v >= 10_000:
                    f.price_min, f.price_max = v * 0.85, v * 1.15
                    notes.append("price 'around' interpreted as ±15%")
            elif m := re.search(r"(\d+(?:[.,]\d+)?)\s*(m|million|k|thousand)\b|(\d{1,3}(?:,\d{3}){2,})\s*(?:sar|riyals?)?", qp):
                v = _money(m.group(1), m.group(2)) if m.group(1) else _money(m.group(3), None)
                if v >= 50_000:
                    f.price_max = v * 1.05
                    notes.append("bare price interpreted as a budget ceiling (+5%)")
    if f.price_min and f.price_max and f.price_min > f.price_max:
        f.price_min, f.price_max = f.price_max, f.price_min

    # --- area
    area = r"(\d+(?:\.\d+)?)\s*(?:sqm|sq\.? ?m|m2|m²|square met(?:er|re)s?)"
    if m := re.search(rf"(?:over|above|more than|at least|min(?:imum)?|bigger than|larger than)\s*{area}", q):
        f.area_min = float(m.group(1))
    if m := re.search(rf"(?:under|below|less than|max(?:imum)?|up to|smaller than)\s*{area}", q):
        f.area_max = float(m.group(1))
    if f.area_min is None and f.area_max is None and (m := re.search(area, q)):
        v = float(m.group(1))
        f.area_min, f.area_max = v * 0.85, v * 1.15
        notes.append("area interpreted as ±15%")

    # --- purpose / source
    if re.search(r"\b(for rent|to rent|rental|rent|lease|إيجار)\b", q):
        f.purpose = "rent"
    elif re.search(r"\b(for sale|buy|to buy|purchase|للبيع)\b", q):
        f.purpose = "sale"
    if re.search(r"\bwasalt\b|وصلت", q):
        f.sources.append("wasalt")
    if re.search(r"dar ?global", q) and not f.sources:
        f.sources.append("darglobal")

    # --- project mentions first: their words must not also become hard location filters
    pids, pnames, primary = [], [], []
    for alias in sorted(vocab.projects, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
            rid = vocab.projects[alias]
            if rid not in pids and not any(alias in n for n in pnames):
                pids.append(rid)
                primary.append(rid)
                pnames.append(alias)
                pids += [c for c in vocab.children.get(alias, []) if c not in pids]
    in_project = lambda word: any(word in n for n in pnames)                            # noqa: E731
    brand_hits = [b for b in BRANDS if re.search(rf"(?<![a-z]){re.escape(b)}(?![a-z])", q)]
    for b in brand_hits:
        if not any(b in n for n in pnames):
            pids += [r for r in vocab.brands.get(b, []) if r not in pids]
            notes.append(f"brand '{b}' -> focus on its projects")

    # --- location (longest names first so "north riyadh" beats "riyadh")
    for k in sorted(vocab.zones, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", q):
            f.zones.append(vocab.zones[k])
    if not f.zones:
        for w in ZONE_WORDS:
            if re.search(rf"\b{w}(?:ern)?\s+(?:part of\s+)?(?:riyadh)\b|\briyadh\s+{w}", q):
                z = vocab.zones.get(f"{'central' if w == 'center' else w} riyadh")
                if z:
                    f.zones.append(z)
    qk = " " + " ".join(_key(w) for w in re.findall(r"[a-z؀-ۿ\-']+", q)) + " "
    qkeys = set(qk.split())
    joined = re.sub(r"[^a-z0-9؀-ۿ]", "", q)
    for k, d in vocab.districts.items():
        if len(k) >= 4 and (k in qkeys or (len(k) >= 7 and k in joined)):
            f.districts.append(d)
    for k in sorted(vocab.countries, key=len, reverse=True):
        if re.search(rf"(?<![a-z؀-ۿ]){re.escape(k)}(?![a-z؀-ۿ])", q) and not in_project(k):
            f.country = vocab.countries[k]
            break
    for k in sorted(vocab.cities, key=len, reverse=True):
        if re.search(rf"(?<![a-z؀-ۿ]){re.escape(k)}(?![a-z؀-ۿ])", q) and vocab.countries.get(k) is None \
                and not in_project(k):
            if f.zones and k in {z.lower().split()[-1] for z in f.zones}:
                break                                        # "north riyadh" already says riyadh
            f.city = vocab.cities[k]
            break
    if f.districts:
        f.city = f.city or None
    p = Parsed(filters=f, project_ids=pids, primary_ids=primary, project_names=pnames, notes=notes)
    if pids and f.is_empty():
        p.intent = "project_info"
    elif f.is_empty() and not pids and not DOMAIN.search(q):
        p.intent = "out_of_scope"
    return p


_VOCAB: Vocab | None = None


def get_vocab(parents_path: str | Path = PARENTS) -> Vocab:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab.from_parents(json.loads(Path(parents_path).read_text(encoding="utf-8")))
    return _VOCAB


def parse(query: str, vocab: Vocab | None = None, llm=None) -> Parsed:
    vocab = vocab or get_vocab()
    p = parse_rules(query, vocab)
    if llm is not None:
        try:
            p = _merge_llm(p, llm(query, vocab), vocab)
        except Exception as e:                                 # LLM parsing is a bonus, never a failure point
            p.notes.append(f"llm parser skipped: {type(e).__name__}")
    return p


def _merge_llm(p: Parsed, data: dict, vocab: Vocab) -> Parsed:
    f = p.filters
    allowed_types = set(PARSE_TYPES)
    if not f.property_types:
        f.property_types = [t for t in data.get("property_types") or [] if t in allowed_types]
    for k in ("bedrooms_min", "bedrooms_max", "price_min", "price_max", "area_min", "area_max"):
        if getattr(f, k) is None and isinstance(data.get(k), (int, float)) and data[k] > 0:
            setattr(f, k, data[k])
    if not f.country and (c := vocab.countries.get(str(data.get("country") or "").lower())):
        f.country = c
    if not f.zones:
        f.zones = [vocab.zones[z.lower()] for z in data.get("zones") or [] if z and z.lower() in vocab.zones]
    if not f.districts:
        f.districts = [vocab.districts[_key(d)] for d in data.get("districts") or [] if d and _key(d) in vocab.districts]
    if not f.purpose and data.get("purpose") in ("sale", "rent"):
        f.purpose = data["purpose"]
    p.llm_used = True
    return p
