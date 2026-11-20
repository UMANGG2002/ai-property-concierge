"""Generate varied, answerable questions from the parent records — one aspect at a time — with ground truth from the data.

Every question has a known source (a FAQ pair, an amenity list, a price flag, a location, a why-invest bullet …), so
`expect_terms` / `expect_any` are computed, never hand-typed. The evaluator then checks that the *lead* of the answer
(the text before any "Related:" suggestions) contains them — i.e. the question is answered first, properties second.

    uv run python -m eval.build_questions     # -> eval/questions.jsonl + public/prompts.json (UI suggestions)
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from rag.paths import PARENTS

rng = random.Random(7)
OUT = Path("eval/questions.jsonl")
UI = Path("public/prompts.json")

AMENITY_KW = {                       # keyword -> (question words, answer must contain)
    "golf": "golf", "beach": "beach", "pool": "pool", "spa": "spa", "fitness": "fitness", "cinema": "cinema",
    "kids": "kids", "sauna": "sauna", "yoga": "yoga", "security": "security", "marina": "marina", "padel": "padel",
    "tennis": "tennis", "concierge": "concierge", "helipad": "helipad", "restaurant": "restaurant", "gym": "gym",
}
PHRASE = {"golf": "golf course", "beach": "private beach", "pool": "swimming pool", "spa": "spa", "fitness": "fitness center", "kids": "kids play area",
          "security": "24/7 security", "gym": "gym", "cinema": "cinema", "sauna": "sauna", "yoga": "yoga lawn"}
NOART = {"security", "tennis", "padel", "yoga", "concierge"}


def art(kw: str) -> str:
    return "" if kw in NOART else "a "


Q_TEMPLATES = {
    "location": ["Where is {t} located?", "Which area is {t} in?", "What is the location of {t}?"],
    "price": ["What is the price of {t}?", "How much does {t} cost?", "What is the value of {t}?", "What are the starting prices at {t}?"],
    "amenities": ["What amenities does {t} offer?", "What facilities are available at {t}?", "What lifestyle features come with {t}?"],
    "invest": ["Why should I invest in {t}?", "{t} why to invest", "What makes {t} a good investment?"],
    "units": ["What types of units are available at {t}?", "Which property types does {t} have?"],
    "partner": ["Who is the brand partner of {t}?", "Who designed the interiors of {t}?"],
}


def short(title: str) -> str:
    return re.split(r",| – | - ", title)[0].strip()


def key_terms(answer: str, question: str, n: int = 2) -> list[str]:
    """Distinctive facts of an FAQ answer: numbers/years first, then capitalised words not already in the question."""
    qlow = question.lower()
    nums = [m for m in re.findall(r"\b\d[\d,.]*\b", answer) if len(m) >= 2]
    caps = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", answer) if w.lower() not in qlow and w not in ("This", "The", "That", "With", "DarGlobal")]
    out: list[str] = []
    for t in nums + caps:
        if t.lower() not in [o.lower() for o in out]:
            out.append(t)
        if len(out) >= n:
            break
    return out


def build() -> list[dict]:
    P = json.loads(Path(PARENTS).read_text(encoding="utf-8"))
    items: list[dict] = []
    amen_index: dict[str, list[str]] = defaultdict(list)

    def add(cat, q, pid, **kw):
        items.append({"id": f"{cat}-{len(items) + 1:03d}", "category": cat, "query": q, "expected_ids": [pid] if pid else kw.pop("expected_ids", []),
                      "expect_terms": kw.get("terms", []), "expect_any": kw.get("any", []), "answer_first": kw.get("first", True)})

    for pid, p in P.items():
        m, rich, kind = p.get("metadata") or {}, p.get("rich") or {}, (p.get("metadata") or {}).get("record_kind")
        if kind not in ("development_project", "one_of_one") or p.get("duplicate_of"):
            continue
        title, t = p["title"], short(p["title"])
        place = [x for x in (m.get("district"), m.get("community"), m.get("city"), m.get("country")) if x]
        # 1. real FAQs (ground truth = the FAQ answer)
        for f in (rich.get("faqs") or [])[:3]:
            terms = key_terms(f["a"], f["q"])
            if terms and (t.lower() in f["q"].lower() or max(re.findall(r"[A-Za-z]+", t), key=len, default="~~").lower() in f["q"].lower()):          # the question must name the property, as a user's would
                add("faq", f["q"], pid, terms=terms)
        # 2. location
        if place:
            add("location", rng.choice(Q_TEMPLATES["location"]).format(t=t), pid, any=place)
        # 3. price (never invented)
        if m.get("price_published") is False:
            add("price", rng.choice(Q_TEMPLATES["price"]).format(t=t), pid, any=["not published"])
        elif m.get("price_amount"):
            add("price", rng.choice(Q_TEMPLATES["price"]).format(t=t), pid, any=[f"{m['price_amount']:,.0f}", f"{m['price_amount'] / 1e6:g}"])
        # 4. amenities, generic + one specific amenity
        amen = rich.get("amenities") or []
        if amen:
            add("amenities", rng.choice(Q_TEMPLATES["amenities"]).format(t=t), pid, terms=[amen[0].rstrip(".")])
            text = " ".join(amen).lower()
            for kw, ans in AMENITY_KW.items():
                if kw in text:
                    amen_index[kw].append(pid)
            have = [kw for kw in AMENITY_KW if kw in text]
            if have:
                kw = rng.choice(have)
                add("amenity_specific", rng.choice([f"Does {t} have {art(kw)}{PHRASE.get(kw, kw)}?", f"Is there {art(kw) or ''}{PHRASE.get(kw, kw)} at {t}?", f"{t} {PHRASE.get(kw, kw)}"]), pid, any=[AMENITY_KW[kw]])
        # 5. why invest
        bl = (rich.get("why_invest") or {}).get("bullets") or []
        if bl:
            head = re.split(r"[:–-]", bl[0])[0].strip()
            add("invest", rng.choice(Q_TEMPLATES["invest"]).format(t=t), pid, terms=[head[:30]])
        # 6. units
        pt = (m.get("property_types") or [])
        if pt:
            add("units", rng.choice(Q_TEMPLATES["units"]).format(t=t), pid, any=[x.replace("_", " ") for x in pt] + [x.replace("_", " ").split()[0] for x in pt])
        # 7. partner / brand
        if m.get("partner"):
            add("partner", rng.choice(Q_TEMPLATES["partner"]).format(t=t), pid, any=[m["partner"].split()[0]])
    # 8. cross-property amenity questions: the answer is a list of developments (hit@5 on any development having it)
    for kw, pids in amen_index.items():
        if len(pids) >= 2:
            ph = PHRASE.get(kw, kw)
            q = rng.choice([f"Which developments have {art(kw)}{ph}?", f"Where can I find {art(kw)}{ph}?", f"{ph} properties"])
            add("amenity_search", q, None, expected_ids=pids, first=False)
    # 9. Wasalt listings: location + size + budget phrased the way people talk
    wl = [(k, v) for k, v in P.items() if k.startswith("wasalt:") and (v["metadata"].get("district")) and v["metadata"].get("price_amount")]
    for pid, p in rng.sample(wl, min(25, len(wl))):
        m = p["metadata"]
        beds, typ, d, price = m.get("bedrooms_min"), (m.get("property_type") or "property"), m["district"], m["price_amount"]
        cap = round(price * 1.1 / 1e5) * 1e5
        q = rng.choice([f"{typ} in {d}, Riyadh under {cap / 1e6:g}M", f"{int(beds)} bedroom {typ} for sale in {d}" if beds else f"{typ} for sale in {d}",
                        f"how much is a {typ} in {d}?"])
        add("wasalt", q, pid, first=False)
    return items


def ui_prompts(items: list[dict], per: int = 6) -> dict:
    """Aspect-balanced suggestions for the landing page: short, natural, and each about a different property."""
    order = ["faq", "invest", "location", "price", "amenities", "amenity_specific", "units", "partner", "amenity_search", "wasalt"]
    labels = {"invest": "Investment", "location": "Location", "price": "Price & value", "amenities": "Amenities", "amenity_specific": "Lifestyle",
              "units": "Units & sizes", "partner": "Brand & design", "faq": "Good to know", "amenity_search": "Across developments", "wasalt": "Riyadh homes"}
    used: set[str] = set()
    out = {}
    for cat in order:
        pool = [i for i in items if i["category"] == cat and len(i["query"]) <= (90 if cat == "faq" else 58)]
        rng.shuffle(pool)
        pick = []
        for it in pool:
            key = it["expected_ids"][0] if it["expected_ids"] and cat not in ("amenity_search",) else it["query"]
            if key in used:
                continue
            used.add(key)
            pick.append(it["query"])
            if len(pick) >= per:
                break
        if pick:
            out[labels[cat]] = pick
    return out


def main() -> int:
    items = build()
    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    UI.write_text(json.dumps(ui_prompts(items), ensure_ascii=False, indent=1), encoding="utf-8")
    cats = defaultdict(int)
    for i in items:
        cats[i["category"]] += 1
    print(f"{len(items)} questions -> {OUT}\n" + "\n".join(f"  {c:18} {n}" for c, n in sorted(cats.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
