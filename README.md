# Concierge: a grounded property assistant for DarGlobal and Wasalt

Ask about 74 DarGlobal developments and residences and 80 Wasalt homes for sale in Riyadh, in English or Arabic.
Answers cite their sources and come with photo galleries, prices and a full detail panel for each residence.
The **Architecture** page in the app tells the whole build story: data collection, the attribute spec, chunking, models and an honest benchmark.

**Stack:** FastAPI · LangGraph · Upstash Vector (hybrid: OpenAI `text-embedding-3-small` + BM25) · OpenRouter free LLMs · static HTML/CSS/JS (no build step)

---

## Repository layout

```
api/index.py        FastAPI app and static UI host (/api/chat, /api/search, /api/property/{id}, /api/chunks, /api/stats, /api/eval)
rag/                the RAG system
  schema.py  parser.py  filters.py  graph.py  llm.py  store.py  embed.py  env.py  paths.py
  sources/          darglobal.py + sections.py (parent–child chunks), wasalt_csv.py (one chunk per listing)
  corpus.py         builds rag/index/ and validates the expected 74 + 80 records
  build.py          CLI: build the corpus and upload it to Upstash   (python -m rag.build)
  ask.py            CLI: ask a question from the terminal            (python -m rag.ask "...")
eval/               golden set (165 questions), scorer (python -m eval.run_eval), latest eval_report.json
public/             the UI: index.html (chat), about.html (the build story and benchmark)
rag/index/          the built index files the app serves (committed)
docs/               architecture diagrams and the attribute spec
data/               local only, git-ignored: raw sources for rebuilding (see below)
tests/              offline tests
```

## Run locally

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/), the only package manager used: dependencies live in `pyproject.toml`, pinned in `uv.lock`.

```bash
uv sync                          # creates .venv from uv.lock (app + dev tools)
cp .env.example .env             # add your Upstash and OpenRouter keys (both optional locally)
uv run uvicorn api.index:app --reload --port 8000
```

Open http://localhost:8000. With no keys, the app uses the bundled BM25 index and grounded summary answers.

```bash
uv run python -m rag.ask "4 bedroom villa in North Riyadh under 3M"
uv run pytest -q
uv run python -m eval.run_eval        # retrieval benchmark → eval/eval_report.json
uv run python -m eval.run_eval --llm --judge   # also scores LLM answers (needs OPENROUTER_API_KEY)
```

## Run with Docker

```bash
cp .env.example .env        # add your keys (optional: without them the app uses BM25 and grounded summaries)
docker compose up --build   # http://localhost:8000
```

`docker-compose.yml` builds the image from `Dockerfile` and passes `UPSTASH_VECTOR_REST_URL`, `UPSTASH_VECTOR_REST_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODELS` and `EVAL_TOKEN` from your `.env`. The same image runs on any host that runs containers (a VM, Fly.io, Render, Railway, Cloud Run): build it, set those variables and expose port 8000. `PORT` is honoured when the platform sets it.

## Set up Upstash Vector (once)

1. In the [Upstash console](https://console.upstash.com/vector), create an index:
   **Type** Hybrid · **Dense model** `OpenAI/text-embedding-3-small` (1536, COSINE) · **Sparse model** `BM25`.
2. Copy `UPSTASH_VECTOR_REST_URL` and `UPSTASH_VECTOR_REST_TOKEN` into `.env`.
3. Upload the 905 chunks. Upstash embeds the text itself, so nothing is embedded on your machine:

   ```bash
   uv run python -m rag.build
   ```

   The script detects the index type, uploads each chunk's text and metadata, and prints the vector count. Re-running only updates what changed.
4. Check it: `uv run python -c "from rag.env import load_env; load_env(); from rag.store import UpstashStore; print(UpstashStore().health())"` should report about 905 vectors.

`rag/build.py` always rebuilds `rag/index/` from the sources first (73 parsed DarGlobal documents + the CSV) and checks it reproduces 74 + 80 records. Add `--no-upload` to rebuild without touching Upstash.

## Deploy to Vercel

The repository is ready to deploy as is. `public/` is served as static files, and `api/index.py` runs as one Python function with the corpus bundled. `vercel.json` is already configured.

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Concierge RAG assistant"
   git branch -M main
   git remote add origin https://github.com/<you>/ai-chat-bot.git
   git push -u origin main
   ```

2. **Import into Vercel.** Go to vercel.com → **Add New… → Project**, choose the repository and keep these settings:
   - Framework preset: **Other**
   - Root directory: `./`
   - Build command: *(leave empty)* · Output directory: *(leave empty; `public/` is picked up)*
   - Install command: *(leave default; Vercel's Python runtime installs from `pyproject.toml` + `uv.lock` with uv)*

3. **Add environment variables** (Project → Settings → Environment Variables, for Production and Preview):

   | Name | Value |
   |---|---|
   | `UPSTASH_VECTOR_REST_URL` | from the Upstash console |
   | `UPSTASH_VECTOR_REST_TOKEN` | from the Upstash console |
   | `OPENROUTER_API_KEY` | from openrouter.ai/keys |
   | `OPENROUTER_MODELS` | *optional*: pin free models, comma-separated, best first |
   | `EVAL_TOKEN` | *optional*: enables `POST /api/eval/run` |

4. **Deploy.** Vercel builds in about a minute. Then check:
   - `https://<your-app>.vercel.app/api/health`: `status: ok`, the store is Upstash hybrid, and the LLM is configured
   - `https://<your-app>.vercel.app/`: the chat
   - `https://<your-app>.vercel.app/about.html`: the build story and benchmark

Later pushes to `main` redeploy automatically. If you change the data or chunking, run `rag/build.py` locally, commit `rag/index/`, and push.

**Notes**
- Dependencies are managed only by uv. If your Vercel project doesn't pick up `pyproject.toml`, generate a lock-equivalent file for it with `uv export --no-dev --no-hashes --format requirements-txt -o requirements.txt` and commit it.
- The function has a 60-second limit (`maxDuration`). LLM calls stay within a 40-second budget, and if no model answers in time the user still gets the grounded summary.
- The app's dependencies are FastAPI, LangGraph, Pydantic, upstash-vector and httpx, about 60 MB installed. No embedding model is shipped, because Upstash embeds queries itself.
- If Upstash is unreachable, search falls back to the bundled BM25 index automatically. If OpenRouter is rate-limited, answers fall back to grounded summaries.

## Data sources and rebuilding

`data/` is git-ignored: it holds the raw sources and stays on your machine. To rebuild the index, put these there:
- `data/darglobal/docs/*.json`: 73 DarGlobal pages parsed from each page's CMS payload (collected with Playwright and Chromium, honouring `robots.txt`).
- `data/darglobal/one_of_one_collections.json`: the 7 One of One residences.
- `data/wasalt_riyadh_sale.csv`: 80 Wasalt listings extracted with ChatGPT and the TinyFish agent (Wasalt's bot protection blocks automated browsers). Agents' names, phones and emails are removed at build time.

Then run `uv run python -m rag.build`. The committed `rag/index/` already contains the result, so running or deploying the app needs none of `data/`.
