# Property RAG — design

## Data
| Source | Records | What a record is |
|---|---|---|
| DarGlobal | 74 | 35 development projects, 7 One of One residences, 11 brand partners, 19 information pages, 2 articles |
| Wasalt | 80 | one resale listing each (Riyadh, residential, for sale) |

All 154 records share one schema (`rag/schema.py`). Filterable fields are flat:
`source, record_kind, purpose, property_type(s), usage, price_amount/min/max, currency, price_per_sqm, price_published,
bedrooms_min/max, bathrooms, built_up_area_min/max, land_area_sqm, country, city, zone, district, community,
parent_project, unit_category, status, building_year, furnishing, facade, street_width_m, partner, developer, has_images`.
Source-specific detail (REGA/FAL licences, deed data, gallery, FAQ count, children...) lives in `extra`.

Normalisation highlights: Wasalt's ~100 key spellings fold into ~35 fields; "112.59 SQM"/"20 m"/"12,000 /sqm" become
numbers; dates become ISO; agent/advertiser names and phone numbers are removed. DarGlobal property types come from the
CMS type, unit types, title and URL; the One of One collection list embedded in every /one-of-one page yields all seven
residences (one never had its own page fetched). Two byte-identical duplicate pages are collapsed onto their canonical
record (both records kept, one indexed).

## Chunks (905)
* Wasalt: **one chunk per listing** — a facts card + English description + Arabic description.
* DarGlobal projects: section chunks (summary, facts, overview, key features, amenities, why-invest, location, gallery,
  property cards, sub-projects, one per FAQ); the whole flattened project is the *parent* text.
* One of One: summary + merged content sections. Partners/pages/articles: their existing section chunks.
Every chunk carries the record's filter metadata + `parent_id` + `chunk_type` + `content_hash` (incremental sync).

## Embeddings and vector store
* **Recommended:** Upstash **Hybrid** index — dense `openai/text-embedding-3-small` (1536-d, COSINE, 8,191-token input,
  so one vector per chunk: 905) + sparse **BM25**, both hosted by Upstash. Chunk text is upserted with its metadata;
  a question is sent as text with the metadata filter and Upstash fuses dense + BM25 results. The app's bundled BM25 is
  used only when Upstash is unreachable.
* **Detected automatically** from `index.info()`: hosted dense + sparse -> DB-level hybrid; hosted dense only -> app
  fuses Upstash dense with its own BM25 (RRF, k = 60); custom index -> self-hosted multilingual MiniLM (384-d, ONNX,
  ~90-word windows, 2,237 vectors, bundled `vectors.npy` fallback). Vectors from different models are never mixed.

## Retrieval and answer (LangGraph, `rag/graph.py`)
`parse → retrieve → build_context → generate → verify` (+ `out_of_scope`).
1. **parse** (`rag/parser.py`) — rules (always) turn the question into filters: type, bedrooms, price (under/between/
   around/"2M"), area, purpose, source, country/city/zone/district (vocabulary built from the corpus, typo-tolerant,
   Arabic city/type words), named projects (ranked first) and brands (soft focus). Optional LLM gap-filling.
2. **retrieve** — Upstash hybrid search (OpenAI 3-small + BM25, fused in Upstash) with the metadata filter; if nothing matches, filters are relaxed
   one at a time (district → exact beds → area → budget +25% → price → beds → zone → purpose → type → city → country)
   and the answer says so. Hits are grouped per record so one listing never fills the list.
3. **build_context** — top 5 records: facts line + best chunks (or the parent text for "tell me about X").
4. **generate** — OpenRouter free models, discovered at runtime and rotated on errors; total time budget.
5. **verify** — every number in the answer must exist in the context and citations must point at real sources;
   otherwise the answer is replaced by a grounded template built from the records.

## Failure handling
| Failure | Behaviour |
|---|---|
| Upstash unreachable / errors / timeout (20 s) | bundled BM25 with the same filters (custom-vector setups also use bundled vectors), reported in `errors` |
| No OpenRouter key / all models fail / timeout | grounded template answer from the retrieved records |
| LLM invents numbers or citations | answer replaced with the template, `grounding.replaced` explains why |
| Any node raises | error recorded, pipeline continues; `run()` never raises |
| Bad input | 422 (empty, > 500 chars); rate limit 429; global handler returns JSON |

## Evaluation (`eval/`)
`eval/golden.jsonl` (165 items) is generated from the data, so ground truth is exact: Wasalt list searches with
`gt_filters`, single-listing lookups, DarGlobal FAQ questions that name their project, project facts (country,
amenities, status), country/brand lists, One of One, behaviour (unpublished prices, rentals, impossible requests,
off-topic, Arabic) and hand-phrased natural questions. Metrics: Hit@1/3/5, MRR, Recall@k, Precision@k (filter
precision), nDCG@k, parser accuracy, behaviour pass, answer must-contain checks, grounding (unsupported numbers,
citations), optional LLM-judge faithfulness/relevance/completeness, latency p50/p95 per stage. Gates fail CI-style with
`--strict`. Reports: `data/reports/eval_report.{md,json}`, `eval_results.csv`, served at `GET /api/eval`.

## UI (`public/`)
Static HTML/CSS/JS, no build. `index.html`: chat, example questions, filters understood, relaxation notes, answer
mode/model/search/latency, property cards (photo carousel + lightbox, facts, source link), and per-question details:
related FAQs always; why-invest for investment questions; amenities + key-feature photos for amenity/photo questions;
location for "where" questions; illustrated sections for One of One residences; REGA/FAL licence data for Wasalt.
`about.html`: architecture, integration table, diagrams, live status (`/api/health`), coverage (`/api/stats`) and the
benchmark (`/api/eval`).
