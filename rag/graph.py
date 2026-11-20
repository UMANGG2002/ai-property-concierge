"""LangGraph RAG pipeline.

    parse -> retrieve (+ progressive filter relaxation) -> build_context -> generate -> verify -> END
       \\-> out_of_scope -> END

Every node is defensive: retrieval falls back from Upstash to the local BM25 store, generation falls back across free
OpenRouter models and finally to a deterministic answer built from the retrieved records, and verification checks that
numbers in the answer exist in the context. The pipeline always returns a well-formed result with timings.
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .corpus import load_parents
from .filters import Filters
from .llm import LLMUnavailable, get_llm
from .parser import Parsed, get_vocab, parse
from .store import get_store, group_by_record

MAX_RECORDS = 8           # returned to the UI
CONTEXT_RECORDS = 5       # passed to the LLM
CONTEXT_WORDS = 3500

SYSTEM = """You are a real-estate assistant for two sources: DarGlobal (luxury developments and one-of-one residences in
the UAE, Oman, Saudi Arabia, Qatar, Spain and the UK) and Wasalt (resale listings in Riyadh, Saudi Arabia).
Rules:
- Use ONLY the numbered context. Never invent prices, sizes, bedroom counts, dates, or amenities.
- Cite every property or fact with its context number, like [1] or [2][3].
- If a price is not published, say "price not published" instead of guessing.
- If the context does not answer the question, say so plainly and suggest how to rephrase.
- If filters were relaxed (see NOTE), say that no exact match was found and that these are the closest options.
- Answer in the user's language. Start with a direct one-sentence answer, then up to 5 bullet points
  (name, location, type, bedrooms, size, price, one distinctive detail) with citations. Keep it under 220 words."""


class RAGState(TypedDict, total=False):
    query: str
    top_k: int
    use_llm: bool
    parsed: dict
    filters_used: dict
    relaxed: list[str]
    hits: list[dict]
    records: list[dict]
    total_matches: int
    context: str
    answer: str
    answer_mode: str           # llm | template | out_of_scope | no_results
    model: str | None
    llm_attempts: list[dict]
    grounding: dict
    store: str
    store_error: str | None
    timings: dict
    errors: list[str]


# ---------------------------------------------------------------- shared resources (loaded once per process)
_RES: dict[str, Any] = {}


def resources() -> dict:
    if not _RES:
        _RES["parents"] = load_parents()
        _RES["store"] = get_store()
        _RES["vocab"] = get_vocab()
    return _RES


def _timed(name: str):
    def deco(fn):
        def wrapper(state: RAGState) -> RAGState:
            t0 = time.perf_counter()
            try:
                out = fn(state)
            except Exception as e:                                  # a node must never crash the request
                out = {"errors": state.get("errors", []) + [f"{name}: {type(e).__name__}: {e}"]}
            timings = dict(state.get("timings") or {})
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)
            out["timings"] = timings
            return out
        return wrapper
    return deco


# ---------------------------------------------------------------- nodes
SORT_ASC = re.compile(r"cheapest|lowest price|most affordable|least expensive|budget option|smallest", re.I)
SORT_DESC = re.compile(r"most expensive|highest price|priciest|largest|biggest|most luxurious", re.I)


TOPICS = {
    "invest": r"invest|\broi\b|returns?\b|yield|worth (it|buying)|why (buy|invest)|residen(cy|ce visa)|visa|tax|capital gain|appreciat|rental income|profit",
    "amenities": r"amenit|facilit|pool|gym|\bspa\b|club ?house|features?\b|lifestyle|kids|play ?area|what does .* (have|offer)|offer\?",
    "location": r"\bwhere\b|locat|nearby|\bnear\b|distance|minutes|airport|beach|school|\bmall\b|\bmap\b",
    "gallery": r"photo|picture|image|gallery|look like|\bpics?\b|show me",
    "payment": r"payment|instal?lment|down ?payment|handover|\bprice|\bcost|how much|budget",
    "faq": r"\?\s*$|^(how|when|can|is|are|does|do|what|which|who|why)\b",
}


def detect_topics(q: str) -> list[str]:
    ql = q.lower().strip()
    return [t for t, pat in TOPICS.items() if re.search(pat, ql)]


@_timed("parse")
def node_parse(state: RAGState) -> RAGState:
    p = parse(state["query"], resources()["vocab"])
    d = p.to_dict()
    d["topics"] = detect_topics(state["query"])
    d["sort"] = "price_asc" if SORT_ASC.search(state["query"]) else "price_desc" if SORT_DESC.search(state["query"]) else None
    if re.search(r"largest|biggest|smallest", state["query"], re.I):
        d["sort"] = "area_desc" if re.search(r"largest|biggest", state["query"], re.I) else "area_asc"
    return {"parsed": d}


def _relaxation_steps(f: Filters) -> list[tuple[str, dict]]:
    steps = []
    if f.districts:
        steps.append(("district", {"districts": []}))
    if f.bedrooms_min is not None and f.bedrooms_max is not None and f.bedrooms_min == f.bedrooms_max:
        steps.append(("exact bedrooms -> at least", {"bedrooms_max": None}))
    if f.area_min is not None or f.area_max is not None:
        steps.append(("area", {"area_min": None, "area_max": None}))
    if f.price_max is not None:
        steps.append(("budget +25%", {"price_max": f.price_max * 1.25}))
    if f.price_min is not None or f.price_max is not None:
        steps.append(("price", {"price_min": None, "price_max": None}))
    if f.bedrooms_min is not None or f.bedrooms_max is not None:
        steps.append(("bedrooms", {"bedrooms_min": None, "bedrooms_max": None}))
    if f.zones:
        steps.append(("zone", {"zones": []}))
    if f.purpose:
        steps.append(("purpose", {"purpose": None}))
    if f.property_types:
        steps.append(("property type", {"property_types": []}))
    if f.city:
        steps.append(("city", {"city": None}))
    if f.country:
        steps.append(("country", {"country": None}))
    if f.sources:
        steps.append(("source", {"sources": []}))
    return steps


@_timed("retrieve")
def node_retrieve(state: RAGState) -> RAGState:
    res, parsed = resources(), state["parsed"]
    store = res["store"]
    f = Filters(**parsed["filters"])
    q, k = state["query"], max(40, state.get("top_k", MAX_RECORDS) * 6)
    relaxed: list[str] = []
    hits: list[dict] = []
    pids = parsed.get("project_ids") or []
    if pids:
        hits = store.search(q, f.model_copy(update={"parent_ids": pids}), k)
        if parsed.get("intent") == "project_info" or not hits:
            general = store.search(q, f, 15)
            seen = {h["id"] for h in hits}
            hits += [h for h in general if h["id"] not in seen]
    else:
        hits = store.search(q, f, k)
    if not hits and not f.is_empty():
        cur = f
        for label, change in _relaxation_steps(f):
            cur = cur.model_copy(update=change)
            relaxed.append(label)
            hits = store.search(q, cur, k)
            if hits:
                f = cur
                break
        if not hits:
            hits = store.search(q, Filters(), k)
            relaxed.append("all filters")
            f = Filters()
    records = group_by_record(hits, max_records=max(state.get("top_k", MAX_RECORDS), MAX_RECORDS) * 3)
    primary = set(parsed.get("primary_ids") or [])
    if primary:                                  # the project the user named outranks its sub-projects / neighbours
        top = max((r["score"] for r in records), default=0.0)
        for r in records:
            if r["record_id"] in primary:
                r["score"] = round(r["score"] + top + 1.0, 4)
        records.sort(key=lambda r: -r["score"])
    total = len(records)
    sort = parsed.get("sort")
    if sort:
        key, rev = {"price_asc": ("price_amount", False), "price_desc": ("price_amount", True),
                    "area_asc": ("built_up_area_min", False), "area_desc": ("built_up_area_max", True)}[sort]
        with_v = [r for r in records if isinstance(r["metadata"].get(key), (int, float))]
        without = [r for r in records if r not in with_v]
        records = sorted(with_v, key=lambda r: r["metadata"][key], reverse=rev) + without
    records = records[:state.get("top_k", MAX_RECORDS)]
    return {"hits": hits[:60], "records": records, "relaxed": relaxed, "filters_used": f.active(), "total_matches": total,
            "store": getattr(store, "last_used", None) or store.name, "store_error": getattr(store, "last_error", None)}


def _fmt_money(v, cur) -> str:
    if not isinstance(v, (int, float)):
        return "price not published"
    return f"{int(v):,} {cur or 'SAR'}"


def facts_line(m: dict) -> str:
    where = ", ".join(x for x in (m.get("district"), m.get("community"), m.get("zone"), m.get("city"), m.get("country")) if x)
    beds = m.get("bedrooms_min")
    beds_s = None if beds is None else (f"{beds}" if m.get("bedrooms_max") in (None, beds) else f"{beds}-{m['bedrooms_max']}") + " bedrooms"
    a1, a2 = m.get("built_up_area_min"), m.get("built_up_area_max")
    area_s = None if a1 is None else (f"{a1:g}" if a2 in (None, a1) else f"{a1:g}-{a2:g}") + " sqm built-up"
    if m.get("land_area_sqm") and m.get("land_area_sqm") != a1:
        area_s = ", ".join(x for x in (area_s, f"{m['land_area_sqm']:g} sqm land") if x)
    price = _fmt_money(m.get("price_amount"), m.get("currency")) if m.get("price_published") else "price not published"
    kind = {"unit_listing": "Wasalt resale listing", "development_project": "DarGlobal development",
            "one_of_one": "DarGlobal One of One residence", "partner": "DarGlobal brand partner",
            "info_page": "DarGlobal page", "article": "DarGlobal article"}.get(m.get("record_kind"), m.get("record_kind"))
    types = ", ".join(t.replace("_", " ") for t in m.get("property_types") or []) or None
    status = (m.get("status") or "").replace("_", " ") or None
    bits = [kind, types, where, beds_s, area_s, price if m.get("record_kind") in ("unit_listing", "development_project", "one_of_one") else None,
            f"status: {status}" if status else None, f"partner: {m['partner']}" if m.get("partner") else None]
    return " | ".join(b for b in bits if b)


@_timed("build_context")
def node_build_context(state: RAGState) -> RAGState:
    parents = resources()["parents"]
    parts, used = [], 0
    project_mode = state["parsed"].get("intent") == "project_info"
    for i, r in enumerate(state.get("records", [])[:CONTEXT_RECORDS], 1):
        p = parents.get(r["record_id"], {})
        m = p.get("metadata") or r["metadata"]
        block = [f"[{i}] {p.get('title') or m.get('title')} — {p.get('url') or m.get('url')}", "Facts: " + facts_line(m)]
        if project_mode and i <= 2 and p.get("full_text"):
            block.append(" ".join(p["full_text"].split()[:1200]))
        else:
            for c in r["chunks"]:
                block.append(" ".join(c["text"].split()[:300]))
        text = "\n".join(block)
        words = len(text.split())
        if used + words > CONTEXT_WORDS and parts:
            break
        parts.append(text)
        used += words
    return {"context": "\n\n".join(parts)}


def describe(m: dict, title: str | None = None, compact: bool = False) -> str:
    """Facts as a readable sentence (keeps every fact facts_line carries: kind, type, place, beds, area, price, status, partner)."""
    where = ", ".join(dict.fromkeys(x for x in (m.get("district"), m.get("community"), m.get("zone"), m.get("city"), m.get("country")) if x))
    b1, b2 = m.get("bedrooms_min"), m.get("bedrooms_max")
    beds = None if b1 is None else (f"{b1}-bedroom" if b2 in (None, b1) else f"{b1}–{b2} bedroom")
    tl = [t.replace("_", " ") for t in (m.get("property_types") or [])[:2]]
    if m.get("record_kind") == "development_project":
        tl = [t if t.endswith("s") else t + "s" for t in tl]
    types = " and ".join(tl) or "residence"
    a1, a2 = m.get("built_up_area_min"), m.get("built_up_area_max")
    area = None if a1 is None else (f"{a1:,.0f}" if a2 in (None, a1) else f"{a1:,.0f}–{a2:,.0f}") + " sqm built-up"
    if m.get("land_area_sqm") and m.get("land_area_sqm") != a1:
        area = ", ".join(x for x in (area, f"{m['land_area_sqm']:,.0f} sqm of land") if x)
    is_prop = m.get("record_kind") in ("unit_listing", "development_project", "one_of_one")
    price = (_fmt_money(m.get("price_amount"), m.get("currency")) if m.get("price_published") else None) if is_prop else None
    status = (m.get("status") or "").replace("_", " ") or None
    unit = " ".join(x for x in (beds, types) if x)
    if compact and not is_prop:
        return {"partner": "DarGlobal brand partner", "info_page": "DarGlobal page", "article": "DarGlobal article"}.get(m.get("record_kind"), "DarGlobal") + (f" in {where}" if where else "")
    if compact:
        bits = [unit + (f" in {where}" if where else ""), area, price or ("price not published" if is_prop else None)]
        return ", ".join(x for x in bits if x)
    kind = {"unit_listing": "a resale", "development_project": "a DarGlobal development of", "one_of_one": "a DarGlobal One of One",
            "partner": "a DarGlobal brand partner", "info_page": "a DarGlobal page", "article": "a DarGlobal article"}.get(m.get("record_kind"), "a")
    if not is_prop:
        s = f"{title or m.get('title')} is {kind}" + (f" ({where})" if where else "") + "."
        return s + (f" Partner: {m['partner']}." if m.get("partner") else "")
    s = f"{title or m.get('title')} is {kind} {unit}" + (f" in {where}" if where else "") + (f", with {area}" if area else "") + "."
    tail = []
    if status:
        tail.append(f"It is {status}" + (f", in partnership with {m['partner']}" if m.get("partner") else ""))
    elif m.get("partner"):
        tail.append(f"It is created in partnership with {m['partner']}")
    tail.append(f"the asking price is {price}" if price else "pricing is not published")
    return s + " " + "; ".join(tail)[0].upper() + "; ".join(tail)[1:] + "."


def _clean_chunk(text: str) -> str:
    text = re.sub(r"^(?:[^.>\n]+>\s*)+[^.\n]*\.\s*", "", text)            # drop the "DarGlobal > ... > FAQ." breadcrumb
    qa = re.match(r"\s*Q:\s*(.+?)\s+A:\s*(.+)", text, re.S)
    if qa:
        return f"**{qa.group(1).strip()}**\n{qa.group(2).strip()}"
    return text.strip()


def template_answer(state: RAGState) -> str:
    recs = state.get("records") or []
    if not recs:
        return ("I couldn't find a property matching that in the DarGlobal or Wasalt collections. "
                "Try a different location, property type or budget.")
    parents = resources()["parents"]
    parsed = state.get("parsed") or {}
    primary = set(parsed.get("primary_ids") or [])
    topics = parsed.get("topics") or []
    if primary and recs[0]["record_id"] in primary:                  # question about one named project
        p = parents.get(recs[0]["record_id"], {})
        m = p.get("metadata") or recs[0]["metadata"]
        rich = p.get("rich") or {}
        lines = [describe(m, f"**{p.get('title')}**") + " [1]"]
        wi = rich.get("why_invest")
        if wi and wi.get("bullets") and "invest" in topics:
            lines += ["#### Why invest"] + [f"- {b.rstrip('.;')}" for b in wi["bullets"][:5]]
        amen = rich.get("amenities") or m.get("amenities")
        if amen:
            shown = [a.rstrip(".") for a in amen[:8]]
            more = f", and {len(amen) - 8} more" if len(amen) > 8 else ""
            lines.append(f"Amenities include {', '.join(shown[:-1])}{' and ' if len(shown) > 1 else ''}{shown[-1]}{more}. [1]")
        best = next((c["text"] for c in recs[0]["chunks"] if c["metadata"].get("chunk_type") not in ("facts", "summary", "amenities", "why_invest")), None)
        best = _clean_chunk(best) if best else None
        if best:
            words = best.split(" ")
            lines.append(" ".join(words[:90]) + (" …" if len(words) > 90 else "") + " [1]")
        elif p.get("summary"):
            lines.append(" ".join(p["summary"].split()[:90]) + " [1]")
        others = recs[1:4]
        if others:
            lines.append("You may also consider " + ", ".join(f"{parents.get(r['record_id'], {}).get('title') or r['metadata'].get('title')} [{i}]"
                                                              for i, r in enumerate(others, 2)) + ".")
        return "\n".join(lines)
    total = state.get("total_matches", len(recs))
    if state.get("relaxed"):
        human = {"exact bedrooms -> at least": "allowing more bedrooms", "budget +25%": "stretching the budget by 25%", "all filters": "dropping the filters",
                 "price": "removing the price limit", "bedrooms": "removing the bedroom count", "area": "removing the size limit"}
        why = ", ".join(human.get(x, f"widening the {x}") for x in state["relaxed"])
        intro = f"Nothing matched every detail, so I looked a little wider ({why}). The closest options:"
    else:
        intro = f"I found {total} matching {'property' if total == 1 else 'properties'}. The strongest matches:"
    lines = [intro]
    for i, r in enumerate(recs[:5], 1):
        p = parents.get(r["record_id"], {})
        m = p.get("metadata") or r["metadata"]
        lines.append(f"- **{p.get('title') or m.get('title')}**: {describe(m, compact=True)} [{i}]")
    return "\n".join(lines)


@_timed("generate")
def node_generate(state: RAGState) -> RAGState:
    if not state.get("records"):
        return {"answer": template_answer(state), "answer_mode": "no_results"}
    llm = get_llm()
    if not state.get("use_llm", True) or not llm.configured:
        return {"answer": template_answer(state), "answer_mode": "template"}
    note = f"NOTE: no exact match; relaxed filters: {', '.join(state['relaxed'])}.\n" if state.get("relaxed") else ""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{note}Context:\n{state['context']}\n\nQuestion: {state['query']}"}]
    try:
        answer = llm.chat(msgs, max_tokens=700)
        return {"answer": answer, "answer_mode": "llm", "model": llm.last_model, "llm_attempts": llm.last_attempts}
    except LLMUnavailable as e:
        return {"answer": template_answer(state), "answer_mode": "template", "llm_attempts": llm.last_attempts,
                "errors": state.get("errors", []) + [f"llm: {str(e)[:200]}"]}


NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:million|m\b|k\b)|\d{4,}(?:\.\d+)?|\d+\.\d+")


def _norm_numbers(text: str) -> set[float]:
    out = set()
    for tok in NUM.findall(text.lower()):
        t = tok.replace(",", "").strip()
        mult = 1_000_000 if re.search(r"million|m$", t) else 1_000 if t.endswith("k") else 1
        try:
            out.add(round(float(re.sub(r"[^\d.]", "", t)) * mult, 2))
        except ValueError:
            pass
    return out


@_timed("verify")
def node_verify(state: RAGState) -> RAGState:
    answer = state.get("answer") or ""
    if state.get("answer_mode") != "llm":
        return {"grounding": {"checked": False, "reason": state.get("answer_mode")}}
    ctx_nums = _norm_numbers(state.get("context", "")) | {float(i) for i in range(0, 32)}
    ans_nums = _norm_numbers(answer)
    unsupported = sorted(n for n in ans_nums if n not in ctx_nums and not any(abs(n - c) <= 0.01 * max(c, 1) for c in ctx_nums))
    cites = sorted({int(c) for c in re.findall(r"\[(\d+)\]", answer)})
    n_ctx = min(len(state.get("records") or []), CONTEXT_RECORDS)
    bad_cites = [c for c in cites if c < 1 or c > n_ctx]
    g = {"checked": True, "numbers_in_answer": len(ans_nums), "unsupported_numbers": unsupported,
         "citations": cites, "invalid_citations": bad_cites, "has_citation": bool(cites)}
    out: RAGState = {"grounding": g}
    if len(unsupported) > 2 or bad_cites:
        out["answer"] = template_answer(state)
        out["answer_mode"] = "template"
        g["replaced"] = "answer had unsupported numbers or citations; replaced with grounded template"
    elif not cites and state.get("records"):
        out["answer"] = answer + "\n\nSources: " + ", ".join(f"[{i}]" for i in range(1, n_ctx + 1))
    return out


@_timed("out_of_scope")
def node_out_of_scope(state: RAGState) -> RAGState:
    return {"answer": "I can only help with properties from DarGlobal developments and Wasalt listings in Riyadh — for "
                      "example \"3-bedroom villa in North Riyadh under 2M\" or \"What amenities does Marea offer?\".",
            "answer_mode": "out_of_scope", "records": [], "relaxed": [], "total_matches": 0}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("parse", node_parse)
    g.add_node("retrieve", node_retrieve)
    g.add_node("build_context", node_build_context)
    g.add_node("generate", node_generate)
    g.add_node("verify", node_verify)
    g.add_node("out_of_scope", node_out_of_scope)
    g.set_entry_point("parse")
    g.add_conditional_edges("parse", lambda s: "out_of_scope" if (s.get("parsed") or {}).get("intent") == "out_of_scope" else "retrieve",
                            {"out_of_scope": "out_of_scope", "retrieve": "retrieve"})
    g.add_edge("retrieve", "build_context")
    g.add_edge("build_context", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    g.add_edge("out_of_scope", END)
    return g.compile()


_GRAPH = None


FAQ_TOPIC = {"invest": r"invest|return|yield|rent|visa|residen|tax|roi|profit|freehold|own",
             "payment": r"pay|price|cost|instal|handover|deposit|plan|mortgage|financ",
             "location": r"where|locat|near|distance|airport|beach|minutes|access"}


def _faq_from_chunk(text: str) -> dict | None:
    m = re.search(r"Q:\s*(.+?)\s+A:\s*(.+)$", text, re.S)
    return {"q": m.group(1).strip(), "a": m.group(2).strip()} if m else None


def highlights(r: dict, rich: dict, topics: list[str]) -> dict:
    """Question-aware extras for a property card: matched FAQs always; investment, amenities, location, payment
    and gallery sections only when the question is about them."""
    h: dict[str, Any] = {}
    faqs = []
    for c in r.get("chunks", []):
        if c["metadata"].get("chunk_type") == "faq" and (f := _faq_from_chunk(c["text"])) and f["q"] not in [x["q"] for x in faqs]:
            faqs.append(f)
    for t in ("invest", "payment", "location"):
        if t in topics:
            for f in rich.get("faqs") or []:
                if re.search(FAQ_TOPIC[t], f["q"], re.I) and f["q"] not in [x["q"] for x in faqs]:
                    faqs.append(f)
    if faqs:
        h["faqs"] = faqs[:5]
    if "invest" in topics and rich.get("why_invest"):
        h["why_invest"] = rich["why_invest"]
    if "amenities" in topics or "gallery" in topics:
        if rich.get("amenities"):
            h["amenities"] = rich["amenities"][:24]
        kf = [k for k in rich.get("key_features") or [] if k.get("image")]
        if kf:
            h["features"] = kf[:8]
        if rich.get("fixtures"):
            h["fixtures"] = rich["fixtures"][:12]
    if "location" in topics:
        loc = rich.get("location") or {}
        if loc.get("text") or loc.get("highlights"):
            h["location"] = loc
        kf = [k for k in rich.get("key_features") or [] if k.get("detail")]
        if kf and "features" not in h:
            h["features"] = kf[:8]
    if rich.get("sections") and ({"amenities", "gallery"} & set(topics) or not topics or "faq" in topics):
        h["sections"] = [x for x in rich["sections"] if x.get("text")][:6]
    if rich.get("licence") and any(rich["licence"].values()):
        h["licence"] = rich["licence"]
    return h


def card(r: dict, topics: list[str] | None = None, detailed: bool = True) -> dict:
    p = resources()["parents"].get(r["record_id"], {})
    m = p.get("metadata") or r["metadata"]
    rich = p.get("rich") or {}
    gallery = [g["url"] for g in rich.get("gallery") or [] if g.get("url")] or list(p.get("images") or [])
    out = {"id": r["record_id"], "title": p.get("title") or m.get("title"), "url": p.get("url") or m.get("url"),
           "source": m.get("source"), "record_kind": m.get("record_kind"), "property_types": m.get("property_types") or [],
           "price": m.get("price_amount"), "currency": m.get("currency"), "price_published": m.get("price_published", False),
           "price_per_sqm": m.get("price_per_sqm"),
           "bedrooms_min": m.get("bedrooms_min"), "bedrooms_max": m.get("bedrooms_max"),
           "area_min": m.get("built_up_area_min"), "area_max": m.get("built_up_area_max"), "land_area": m.get("land_area_sqm"),
           "country": m.get("country"), "city": m.get("city"), "zone": m.get("zone"), "district": m.get("district"),
           "community": m.get("community"), "status": m.get("status"), "partner": m.get("partner"),
           "image": gallery[0] if gallery else None, "images": gallery[:12], "image_count": len(gallery),
           "facts": facts_line(m), "summary": " ".join((rich.get("about") or p.get("summary") or "").split()[:60]),
           "score": r["score"], "matched_sections": [c["metadata"].get("chunk_type") for c in r["chunks"]],
           "quality_flags": p.get("quality_flags", [])}
    if detailed:
        out["highlights"] = highlights(r, rich, topics or [])
    return out


def run(query: str, top_k: int = MAX_RECORDS, use_llm: bool = True, debug: bool = False) -> dict:
    """Never raises: any unexpected failure returns a valid error-shaped result."""
    global _GRAPH
    t0 = time.perf_counter()
    query = (query or "").strip()[:500]
    if not query:
        return {"answer": "Please ask a question about a property.", "answer_mode": "empty", "properties": [], "timings": {}}
    try:
        if _GRAPH is None:
            _GRAPH = build_graph()
        s = _GRAPH.invoke({"query": query, "top_k": max(1, min(int(top_k), 20)), "use_llm": use_llm, "errors": [], "timings": {}})
    except Exception as e:
        return {"answer": "Sorry, something went wrong while searching. Please try again.", "answer_mode": "error",
                "properties": [], "errors": [f"{type(e).__name__}: {e}"], "timings": {}}
    timings = dict(s.get("timings") or {})
    timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
    out = {"query": query, "answer": s.get("answer", ""), "answer_mode": s.get("answer_mode"),
           "properties": [card(r, (s.get("parsed") or {}).get("topics"), detailed=i < 4)
                          for i, r in enumerate(s.get("records") or [])], "total_matches": s.get("total_matches", 0),
           "filters_used": s.get("filters_used", {}), "relaxed": s.get("relaxed", []),
           "understood": {k: v for k, v in (s.get("parsed") or {}).items() if k in ("filters", "project_names", "intent", "notes", "sort", "topics")},
           "model": s.get("model"), "store": s.get("store"), "grounding": s.get("grounding", {}),
           "timings": timings, "errors": s.get("errors", [])}
    if s.get("store_error"):
        out["errors"].append(f"vector store fallback: {s['store_error']}")
    if debug:
        out["context"] = s.get("context", "")
        out["llm_attempts"] = s.get("llm_attempts", [])
        out["hits"] = [{"id": h["id"], "score": h["score"], "chunk_type": h["metadata"].get("chunk_type")} for h in s.get("hits", [])[:30]]
    return out
