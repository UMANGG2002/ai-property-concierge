"""RAG backend tests: data completeness, normalisation, filters, parser, retrieval, failure handling, API, eval gates."""
import json

import httpx
import pytest

from rag import graph
from rag.corpus import load_chunks, load_parents, load_records
from rag.filters import Filters, matches, to_upstash
from rag.llm import LLMUnavailable, OpenRouterLLM
from rag.parser import parse
from rag.schema import bedrooms_from_text, iso_date, norm_property_type, to_float
from rag.sources.wasalt_csv import normalise_facts, scrub
from rag.store import FallbackStore, LocalStore, group_by_record


# ---------------------------------------------------------------- data completeness
def test_corpus_has_every_record():
    recs = load_records()
    by = {}
    for r in recs:
        by[r["source"]] = by.get(r["source"], 0) + 1
    assert by == {"darglobal": 74, "wasalt": 80}
    props = [r for r in recs if r["record_kind"] in ("development_project", "one_of_one", "unit_listing")]
    assert all(r["country"] and (r["property_type"] or r["property_types"]) for r in props)
    assert len({r["id"] for r in recs}) == len(recs)


def test_every_chunk_has_a_parent_and_unique_id():
    chunks, parents = load_chunks(), load_parents()
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert all(c["metadata"]["parent_id"] in parents for c in chunks)
    wasalt = [c for c in chunks if c["metadata"]["source"] == "wasalt"]
    assert len(wasalt) == 80 and all(c["metadata"]["chunk_type"] == "listing" for c in wasalt)   # one chunk per listing


# ---------------------------------------------------------------- normalisation
def test_schema_helpers():
    assert norm_property_type("Villa 252 SQM") == "villa" and norm_property_type("Rooms & Suites") == "hotel_room"
    assert bedrooms_from_text("five-bedroom villas") == (5, 5) and bedrooms_from_text("1 to 4 BR") == (1, 4)
    assert to_float("12,000 /sqm") == 12000 and to_float("112.59 SQM") == 112.59 and to_float(None) is None
    assert iso_date("30/09/2027") == "2027-09-30" and iso_date("2026-06-16") == "2026-06-16"


def test_wasalt_alias_folding_and_pii():
    f, unmapped = normalise_facts({"Bedrooms": 3, "built_up_area_sqm": 120, "Built-up area": "125 SQM", "agent": "Ali",
                                   "rega_expiry_date": "30/09/2027", "electricity": "Connected", "water": "Connected"})
    assert f["bedrooms"] == 3 and f["built_up_area"] == 120 and "agent" not in str(f) and not unmapped
    assert f["license_expiry"] == "30/09/2027" and len(f["utilities"]) == 2
    assert "0501234567" not in scrub("call 0501234567 or +966 50 123 4567")


# ---------------------------------------------------------------- filters (Upstash string and Python twin agree)
def test_filter_render_and_match():
    f = Filters(property_types=["villa"], zones=["North Riyadh"], bedrooms_min=4, price_max=3_000_000)
    s = to_upstash(f)
    assert "property_types CONTAINS 'villa'" in s and "bedrooms_max >= 4" in s and "price_min <= 3000000" in s
    ok = {"property_types": ["villa"], "zone": "North Riyadh", "bedrooms_min": 5, "bedrooms_max": 5, "price_min": 2.5e6}
    assert matches(ok, f)
    assert not matches({**ok, "price_min": 3.5e6}, f)
    assert not matches({k: v for k, v in ok.items() if k != "price_min"}, f)       # no price never matches a price filter
    assert to_upstash(Filters(zones=["x'; DROP"])) == "zone = 'x; DROP'"           # quotes stripped


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("q,expect", [
    ("4 bedroom villa in North Riyadh under 3M", {"property_types": ["villa"], "bedrooms_min": 4, "price_max": 3e6, "zones": ["North Riyadh"]}),
    ("apartments in narjis between 1m and 1.5 million", {"price_min": 1e6, "price_max": 1.5e6, "districts": ["An-Narjis"]}),
    ("any apartmnets in west riyadh?", {"property_types": ["apartment"], "zones": ["West Riyadh"]}),
    ("شقة للبيع في الرياض 3 غرف", {"property_types": ["apartment"], "bedrooms_min": 3, "city": "Riyadh"}),
    ("villa for rent in Riyadh", {"purpose": "rent"}),
])
def test_parser(q, expect):
    got = parse(q).filters.active()
    for k, v in expect.items():
        assert (set(v) <= set(got.get(k) or []) if isinstance(v, list) else got.get(k) == v), (q, k, got)


def test_parser_projects_and_scope():
    p = parse("What amenities does Marea offer?")
    assert p.intent == "project_info" and p.primary_ids == ["darglobal:marea"]
    assert "darglobal:landing-page/aida" in parse("villas in AIDA").primary_ids
    assert parse("what's the weather in Paris").intent == "out_of_scope"


# ---------------------------------------------------------------- retrieval + pipeline behaviour
def test_group_by_record_dedupes():
    hits = [{"id": f"a#{i}", "score": 5 - i, "metadata": {"parent_id": "a"}} for i in range(4)]
    hits += [{"id": "b#1", "score": 4.5, "metadata": {"parent_id": "b"}}]
    ranked = group_by_record(hits)
    assert [r["record_id"] for r in ranked] == ["a", "b"] and len(ranked[0]["chunks"]) == 3


def test_relaxation_when_nothing_matches():
    out = graph.run("10 bedroom apartment in South Riyadh under 300k", use_llm=False)
    assert out["relaxed"] and out["properties"] and "Nothing matched every detail" in out["answer"]


def test_named_project_ranks_first():
    out = graph.run("What amenities does Fairway Villas have?", use_llm=False)
    assert out["properties"][0]["id"] == "darglobal:fairway-villas-aida-oman" and "Amenities include" in out["answer"]


def test_price_never_invented_for_darglobal():
    out = graph.run("What is the price of Marea?", use_llm=False)
    assert "not published" in out["answer"].lower()


def test_empty_and_huge_queries_do_not_fail():
    assert graph.run("")["answer_mode"] == "empty"
    assert graph.run("villa " * 400, use_llm=False)["answer"]


class _Broken:
    name = "broken"

    def search(self, *a, **k):
        raise RuntimeError("upstash down")

    def health(self):
        return {"ok": False}


def test_store_fallback(monkeypatch):
    fb = FallbackStore(_Broken(), LocalStore())
    monkeypatch.setitem(graph.resources(), "store", fb)
    out = graph.run("villas in East Riyadh", use_llm=False)
    assert out["properties"] and out["store"] == "local-bm25" and any("fallback" in e for e in out["errors"])


def test_total_store_failure_still_answers(monkeypatch):
    monkeypatch.setitem(graph.resources(), "store", _Broken())
    out = graph.run("villas in East Riyadh", use_llm=False)
    assert out["answer"] and out["properties"] == [] and any("retrieve" in e for e in out["errors"])


class _FakeLLM:
    configured = True
    last_model = "fake/model:free"
    last_attempts = []

    def __init__(self, text=None, fail=False):
        self.text, self.fail = text, fail

    def chat(self, *a, **k):
        if self.fail:
            raise LLMUnavailable("all models failed")
        return self.text


def test_llm_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(graph, "get_llm", lambda: _FakeLLM(fail=True))
    out = graph.run("villas in East Riyadh")
    assert out["answer_mode"] == "template" and out["properties"] and any(e.startswith("llm") for e in out["errors"])


def test_hallucinated_numbers_are_replaced(monkeypatch):
    fake = "The villa costs 9,999,999 SAR with 777.7 sqm and 123,456 SAR service fees [1]."
    monkeypatch.setattr(graph, "get_llm", lambda: _FakeLLM(text=fake))
    out = graph.run("villas in East Riyadh")
    assert out["answer_mode"] == "template" and "9,999,999" not in out["answer"]
    assert out["grounding"]["unsupported_numbers"]


def test_grounded_llm_answer_is_kept(monkeypatch):
    first = graph.run("cheapest apartment in East Riyadh", use_llm=False)["properties"][0]
    fake = f"The cheapest is {first['title']} at {int(first['price']):,} SAR [1]."
    monkeypatch.setattr(graph, "get_llm", lambda: _FakeLLM(text=fake))
    out = graph.run("cheapest apartment in East Riyadh")
    assert out["answer_mode"] == "llm" and out["grounding"]["unsupported_numbers"] == [] and out["grounding"]["has_citation"]


# ---------------------------------------------------------------- OpenRouter client
def test_openrouter_rotation_and_free_only():
    seen = []

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "a/llama-3.3-70b-instruct:free", "context_length": 128000, "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "b/gemma-3-27b-it:free", "context_length": 96000, "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "c/paid", "context_length": 128000, "pricing": {"prompt": "0.1", "completion": "0.1"}}]})
        model = json.loads(req.content)["model"]
        seen.append(model)
        return httpx.Response(429) if "llama" in model else httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenRouterLLM(api_key="k", transport=httpx.MockTransport(handler))
    assert llm.models() == ["a/llama-3.3-70b-instruct:free", "b/gemma-3-27b-it:free"]
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok" and llm.last_model.startswith("b/")
    with pytest.raises(LLMUnavailable):
        OpenRouterLLM(api_key=None).chat([])


# ---------------------------------------------------------------- API
def test_api():
    from fastapi.testclient import TestClient
    from api.index import app
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    r = c.post("/api/chat", json={"query": "villas in North Riyadh", "use_llm": False}).json()
    assert r["properties"] and {"answer", "filters_used", "timings", "total_matches"} <= set(r)
    assert c.post("/api/chat", json={"query": ""}).status_code == 422
    assert c.post("/api/chat", json={"query": "x" * 501}).status_code == 422
    assert c.get("/api/property/darglobal:marea").json()["title"].startswith("Marea")
    assert c.get("/api/property/nope").status_code == 404
    assert c.post("/api/eval/run").status_code == 403


# ---------------------------------------------------------------- golden-set regression gates (retrieval, no LLM)
def test_golden_gates():
    from eval.run_eval import GATES, aggregate, load_golden, score_item
    items = load_golden() + load_golden("eval/questions.jsonl")
    assert len(items) >= 150
    rows = [score_item(it, graph.run(it["query"], top_k=8, use_llm=False), judge=False) for it in items]
    o = aggregate(rows)["overall"]
    for m, th in GATES.items():
        assert o[m] >= th, (m, o[m])


# ---------------------------------------------------------------- question first, properties second
def test_question_first_answers():
    lead = lambda q: graph.run(q, use_llm=False)["answer"].split("\n\n")[0]                        # noqa: E731
    assert "December 2028" in lead("When is The Astera expected to be completed?")
    assert "not published" in lead("What is the value of The Astera development?")
    assert "Robust tourism" in lead("Marea, Interiors by Missoni why to invest")
    assert lead("Does Les Vagues have a swimming pool?").startswith("Yes")
    assert "don't see" in lead("Does Les Vagues have a spa?")                                    # honest no, not a guess
    out = graph.run("gold club", use_llm=False)                                                       # typo for "golf club"; never rejected
    assert out["answer_mode"] != "out_of_scope" and "golf" in out["answer"].lower()


def test_suggestions_follow_the_answer():
    ans = graph.run("When is The Astera expected to be completed?", use_llm=False)["answer"]
    assert ans.index("December 2028") < ans.index("Related:")


def test_visitor_questions_get_the_specific_answer_first():
    lead = lambda q: graph.run(q, use_llm=False)["answer"].split("\n\n")[0]                        # noqa: E731
    assert "25 minutes" in lead("How far is Trump Cliff Villas from the airport?")
    assert lead("Does Coastal Investment Villas have 3-bedroom homes?").startswith("Yes")
    assert lead("Can I get a 5-bedroom home at Coastal Investment Villas?").startswith("No")        # a named project is never swapped for another
    assert "not published" in lead("What is the price of Neptune?")                                  # an investment threshold ("SAR 4M+") is not a price
    assert "2028" in lead("When can I expect to move into The Astera?")
