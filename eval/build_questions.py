"""Generate visitor-style questions from the parent records, with ground truth taken from the data.

The questions are what a real-estate website visitor would ask — an investor asking about residency and returns, a family asking
about kids and schools, a golfer, someone with a fixed budget, someone who needs three bedrooms — and every question is generated
*only when the data can answer it* (a why-invest bullet mentioning a visa, a location paragraph naming the airport, an amenity list
containing a golf course, a bedroom range, a FAQ …). `expect_terms` / `expect_any` come from that source text, never typed by hand.

The evaluator checks that the *lead* of the answer (text before any "Related:" suggestions) contains them: question first.

    uv run python -m eval.build_questions     # -> eval/questions.jsonl + public/prompts.json (landing-page suggestions)
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from rag.paths import PARENTS

rng = random.Random(11)
OUT = Path("eval/questions.jsonl")
UI = Path("public/prompts.json")

# persona -> (tab label on the landing page)
TABS = {
    "faq": "Straight from the site", "residency": "Investing & residency", "ownership": "Investing & residency", "tax": "Investing & residency",
    "returns": "Investing & residency", "family": "Family & lifestyle", "golf": "Family & lifestyle", "beach": "Family & lifestyle",
    "wellness": "Family & lifestyle", "timeline": "Timeline & budget", "budget": "Timeline & budget", "layout": "Layout & size",
    "sizes": "Layout & size", "airport": "Getting there", "brand": "Brand & design", "compare": "Compare developments", "riyadh": "Riyadh homes",
}
TAB_ORDER = ["Straight from the site", "Investing & residency", "Family & lifestyle", "Timeline & budget", "Layout & size", "Getting there",
             "Brand & design", "Compare developments", "Riyadh homes"]

T = {                                    # several natural phrasings per intent
    "residency": ["Does buying at {t} get me a residency visa?", "I live abroad — can I get residency if I invest in {t}?", "Is there a residency visa for buyers at {t}?"],
    "ownership": ["I'm not a local — can I own a home at {t} outright?", "Can foreigners buy at {t}?", "Is ownership at {t} freehold for overseas buyers?"],
    "tax": ["Will I pay income tax or property tax if I own at {t}?", "What are the tax benefits of buying at {t}?"],
    "returns": ["What returns can I expect from investing in {t}?", "Is {t} a good investment for rental income?", "How much could property values grow around {t}?"],
    "family": ["Is {t} a good place to raise young kids?", "We have children — what is there for the family at {t}?"],
    "golf": ["I play golf — can I use a golf course if I buy at {t}?", "Is there a golf course at {t}?"],
    "beach": ["Can I get to the beach easily from {t}?", "Is {t} close to the beach?"],
    "wellness": ["I like to keep fit — does {t} have a gym or pool?", "Is there a spa or fitness centre at {t}?"],
    "airport": ["How long is the drive from {t} to the airport?", "How far is {t} from the airport?"],
    "brand": ["Is {t} actually designed by {p}?", "Who is behind the interiors at {t}?"],
    "budget": ["I'm planning my budget — what are the prices at {t}?", "What does a home at {t} cost?", "What is the value of {t}?"],
    "layout_yes": ["I need {n} bedrooms for my family — does {t} have that?", "Does {t} have {n}-bedroom homes?"],
    "layout_no": ["Can I get a {n}-bedroom home at {t}?"],
    "sizes": ["What sizes of homes does {t} have?", "How big are the residences at {t}?"],
}
CROSS = {          # keyword in an amenity -> how a visitor asks for it across developments
    "golf": ["I play golf — which developments have a golf course?", "Which developments let me live next to a golf course?"],
    "beach": ["Which developments have private beach access?", "I want a home on the beach — which developments offer it?"],
    "kids": ["We have young children — which developments have a kids play area?"],
    "spa": ["Which developments have a spa or wellness centre?"],
    "cinema": ["Which developments have a private cinema?"],
    "pool": ["Which developments have a swimming pool?"],
    "tennis": ["Which developments have tennis courts?"],
    "padel": ["Which developments have padel courts?"],
    "yoga": ["Which developments have a yoga lawn?"],
    "concierge": ["Which developments offer a concierge service?"],
    "sauna": ["Which developments have a sauna?"],
}


def short(title: str) -> str:
    return re.split(r",| – | - ", title)[0].strip()


def key_terms(answer: str, question: str, n: int = 2) -> list[str]:
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


def pick(seq):
    return rng.choice(seq)


def build() -> list[dict]:
    P = json.loads(Path(PARENTS).read_text(encoding="utf-8"))
    items: list[dict] = []
    amen_index: dict[str, list[str]] = defaultdict(list)

    def add(intent, q, pid, terms=(), any_=(), first=True, ids=None):
        items.append({"id": f"{intent}-{len(items) + 1:03d}", "category": intent, "query": q, "expected_ids": ids if ids is not None else [pid],
                      "expect_terms": list(terms), "expect_any": list(any_), "answer_first": first})

    for pid, p in P.items():
        m, rich = p.get("metadata") or {}, p.get("rich") or {}
        if m.get("record_kind") not in ("development_project", "one_of_one") or p.get("duplicate_of"):
            continue
        t, title = short(p["title"]), p["title"]
        bullets = (rich.get("why_invest") or {}).get("bullets") or []
        amen = rich.get("amenities") or []
        atext = " ".join(amen).lower()
        loc = (rich.get("location") or {}).get("text") or ""
        faqs = rich.get("faqs") or []

        for f in faqs[:3]:                                                       # real questions from the developer's own FAQ
            terms = key_terms(f["a"], f["q"])
            if terms and (t.lower() in f["q"].lower() or max(re.findall(r"[A-Za-z]+", t), key=len, default="~~").lower() in f["q"].lower()):
                add("faq", f["q"], pid, terms=terms)

        def bullet(rx):
            return next((b for b in bullets if re.search(rx, b, re.I)), None)

        if bullet(r"visa|residen"):
            add("residency", pick(T["residency"]).format(t=t), pid, any_=["visa", "residen"])
        ob = bullet(r"freehold|foreign|ownership")
        if ob:                                                                   # ask about what this property's bullet actually says
            if re.search(r"freehold", ob, re.I):
                add("ownership", pick(["Is ownership at {t} freehold?", "Can I own a home at {t} outright, as freehold?"]).format(t=t), pid, any_=["freehold"])
            else:
                add("ownership", pick(["Can foreigners buy at {t}?", "I'm not a local — can I buy at {t}?"]).format(t=t), pid, any_=["foreign", "international", "ownership"])
        if bullet(r"\btax"):
            add("tax", pick(T["tax"]).format(t=t), pid, any_=["tax"])
        for rx, qs, kws in ((r"yield|rental|occupancy", ["What rental yield can I expect at {t}?", "Is {t} good for rental income?"], ["yield", "rental", "occupancy"]),
                            (r"capital|growth|appreciation", ["How much could property values grow around {t}?", "Is property in the {t} area growing in value?"], ["capital", "growth", "appreciation"]),
                            (r"\broi\b|return", ["What returns can I expect from investing in {t}?"], ["roi", "return"])):
            rb = bullet(rx)
            if rb:
                nums = re.findall(r"\d[\d.,]*\s?%", rb)
                add("returns", pick(qs).format(t=t), pid, any_=kws + [n.replace(" ", "") for n in nums[:1]])
                break
        am = next((a for a in amen if re.search(r"kid|child|play|school|nursery", a, re.I)), None)
        if am:
            add("family", pick(T["family"]).format(t=t), pid, any_=[w for w in re.findall(r"[A-Za-z]{3,}", am)][:3])
        if "golf" in atext or "golf" in loc.lower():
            add("golf", pick(T["golf"]).format(t=t), pid, any_=["golf"])
        if "beach" in atext or "beach" in loc.lower():
            add("beach", pick(T["beach"]).format(t=t), pid, any_=["beach"])
        if re.search(r"gym|fitness|pool|spa\b|sauna", atext):
            add("wellness", pick(T["wellness"]).format(t=t), pid, any_=["gym", "fitness", "pool", "spa", "sauna"])
        sent = next((s for s in re.split(r"(?<=[.!?])\s+", loc) if re.search(r"airport", s, re.I) and re.search(r"\d+[- ]?(min|minute)", s, re.I)), None)
        if sent:
            add("airport", pick(T["airport"]).format(t=t), pid, any_=[re.search(r"(\d+)[- ]?(?:min|minute)", sent, re.I).group(1)])
        if m.get("partner"):
            add("brand", pick(T["brand"]).format(t=t, p=m["partner"]), pid, any_=[m["partner"].split()[0]])
        # timeline: a FAQ about completion/handover wins; otherwise the recorded status
        tf = next((f for f in faqs if re.search(r"complet|handover|ready", f["q"], re.I) and re.search(r"20\d\d", f["a"])), None)
        if tf:
            add("timeline", pick(["When can I expect to move into {t}?", "When will {t} be ready?"]).format(t=t), pid, any_=[re.search(r"20\d\d", tf["a"]).group(0)])
        elif m.get("status") == "sold_out":
            add("timeline", pick(["Can I still buy at {t}?", "Is {t} still available?"]).format(t=t), pid, any_=["sold out"])
        elif m.get("status") == "completed":
            add("timeline", f"Is {t} ready to move into?", pid, any_=["completed", "complete"])
        # budget
        if m.get("price_published") is False:
            add("budget", pick(T["budget"]).format(t=t), pid, any_=["not published"])
        elif m.get("price_amount"):
            add("budget", pick(T["budget"]).format(t=t), pid, any_=[f"{m['price_amount']:,.0f}", f"{m['price_amount'] / 1e6:g}"])
        # layout
        lo, hi = m.get("bedrooms_min"), m.get("bedrooms_max") or m.get("bedrooms_min")
        if lo is not None:
            lo, hi = int(lo), int(hi)
            add("layout", pick(T["layout_yes"]).format(t=t, n=rng.randint(lo, hi)), pid, any_=["Yes"])
            if hi + 2 <= 9:
                add("layout", pick(T["layout_no"]).format(t=t, n=hi + 2), pid, any_=["No —"])
        if m.get("built_up_area_min"):
            add("sizes", pick(T["sizes"]).format(t=t), pid, any_=[f"{m['built_up_area_min']:,.0f}"])
        for kw in CROSS:
            if re.search(rf"\b{kw}", atext):
                amen_index[kw].append(pid)

    for kw, pids in amen_index.items():                                          # one question per amenity, across developments
        if len(pids) >= 2:
            add("compare", pick(CROSS[kw]), None, ids=pids, first=False)

    wl = [(k, v) for k, v in P.items() if k.startswith("wasalt:") and v["metadata"].get("district") and v["metadata"].get("price_amount")]
    for pid, p in rng.sample(wl, min(30, len(wl))):                              # Riyadh: how people talk about budget + place + size
        m = p["metadata"]
        beds, typ, d, z, price = m.get("bedrooms_min"), (m.get("property_type") or "property"), m["district"], m.get("zone"), m["price_amount"]
        cap = f"{round(price * 1.1 / 1e5) / 10:g}"
        qs = [f"Looking for a {int(beds)}-bedroom {typ} in {d}, Riyadh, budget up to {cap}M SAR" if beds else f"Looking for a {typ} in {d}, Riyadh, budget up to {cap}M SAR",
              f"What can I get for {cap}M SAR in {d}?", f"Any {typ} for sale in {d} under {cap}M SAR?"]
        add("riyadh", pick(qs), pid, first=False)
    return items


def ui_prompts(items: list[dict], per: int = 6) -> dict:
    """Landing-page suggestions grouped by visitor intent: natural, answerable from the data, varied in wording and in property."""
    uses: dict[str, int] = defaultdict(int)                                      # a property appears at most twice across the whole page
    out: dict[str, list[str]] = {}
    for tab in TAB_ORDER:
        cats = [c for c, v in TABS.items() if v == tab]
        pool = [i for i in items if i["category"] in cats and len(i["query"]) <= 92]
        rng.shuffle(pool)
        sel, sigs, per_cat, here = [], set(), defaultdict(int), set()
        for it in pool:
            pid = (it["expected_ids"] or [it["query"]])[0] if it["category"] != "compare" else it["query"]
            sig = " ".join(it["query"].lower().split()[:3])                        # same opening = same template
            if pid in here or uses[pid] >= 2 or sig in sigs or per_cat[it["category"]] >= 2:
                continue
            here.add(pid); sigs.add(sig); uses[pid] += 1; per_cat[it["category"]] += 1
            sel.append(it["query"])
            if len(sel) >= per:
                break
        if sel:
            out[tab] = sel
    return out


def main() -> int:
    items = build()
    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    UI.write_text(json.dumps(ui_prompts(items), ensure_ascii=False, indent=1), encoding="utf-8")
    cats = defaultdict(int)
    for i in items:
        cats[i["category"]] += 1
    print(f"{len(items)} questions -> {OUT}\n" + "\n".join(f"  {c:12} {n}" for c, n in sorted(cats.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
