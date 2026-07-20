# RAG

A Retrieval-Augmented Generation (RAG) pipeline: it chunks `.txt` documents, embeds them with Gemini, stores the vectors in ChromaDB, and answers questions by retrieving relevant chunks and generating a grounded, cited answer with Gemini.

## How it works

1. `main.py` reads every `.txt` file in `data/`, skipping files that are already indexed
2. Each file is split into sentence-aware, overlapping chunks (`rag/chunker.py`)
3. Chunks are embedded in batches via the Gemini embedding model, using the `RETRIEVAL_DOCUMENT` task type (`rag/embedder.py`)
4. Chunks + embeddings + source metadata are upserted into a local ChromaDB collection with deterministic IDs, so re-running never duplicates data (`rag/vector_store.py`)
5. You ask a question in the terminal; it's embedded with the `RETRIEVAL_QUERY` task type, the top matching chunks are retrieved, and passed to Gemini to generate a cited answer that's grounded in the sources — the model is instructed to say so if the answer isn't in the retrieved context (`rag/generator.py`)

Via the API, step 5 runs on a separate worker pool instead of inline in the request — see [Async `/ask` & scaling](#async-ask--scaling-rabbitmq).

## Requirements

- Python 3.10+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))
- Redis and RabbitMQ (provided automatically by Docker Compose; see below for running them yourself)

## Setup

```bash
git clone https://github.com/leman-cap13/RAG.git
cd RAG

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### CLI

Add your `.txt` documents to the `data/` folder, then run:

```bash
python main.py
```

On first run it will chunk, embed, and index all files in `data/`. You'll then be prompted to ask questions:

```
question (output: q): what does this document say about X?
```

Type `q` to quit.

### API

Start Redis and RabbitMQ (or point `REDIS_URL`/`RABBITMQ_URL` at existing instances), then run the API and at least one worker:

```bash
uvicorn api:app --reload
python worker.py       # in a separate terminal; run multiple copies to parallelize /ask
```

A chat UI ("Dory") is served at `http://127.0.0.1:8000/`. Interactive API docs are at `http://127.0.0.1:8000/docs`. Endpoints:

| Method | Path                 | Description                                   |
| ------ | -------------------- | ---------------------------------------------- |
| GET    | `/`                    | chat UI (`frontend/`, served as static files)  |
| GET    | `/health`             | health check                                   |
| POST   | `/index`              | chunk, embed, and index new files in `data/`   |
| POST   | `/ask`                | `{"question": "...", "top_k": 4, "session_id": null}` → cited answer + `session_id` |
| GET    | `/sources`            | list indexed source filenames                 |
| DELETE | `/sources/{filename}` | remove a source's chunks from the vector store |

## Conversation memory

`/ask` accepts an optional `session_id`. Omit it (or send `null`) on the first message — the response comes back with a freshly minted `session_id`; send that same value on follow-up questions and the assistant will recall the conversation so far (e.g. "orada hansı proqramlar var?" resolves to whichever university was discussed earlier).

History is stored server-side in Redis (`chat_session:{session_id}`, capped at `max_history_turns` exchanges, `session_ttl` seconds TTL — both in `config.py`), not on the client, so it survives page reloads as long as the frontend keeps the `session_id` (it does, in `localStorage`). Because a cached answer isn't scoped to any one conversation, the semantic cache in `rag/cache.py` is only consulted/written on session-less first turns — once a session has history, every question goes through the full pipeline so context is always respected.

### Docker

Build and run the API in a container (requires a `.env` file with `GEMINI_API_KEY` set, as above):

```bash
docker compose up --build
```

This starts the API, one `worker` instance, Redis, and RabbitMQ, and mounts `data/` and `chroma_db/` as volumes so indexed documents and vectors persist across container restarts. The API is available at `http://127.0.0.1:8000/docs`. Scale the worker pool to handle more concurrent `/ask` traffic:

```bash
docker compose up --build --scale worker=3
```

Without Compose:

```bash
docker build -t rag .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/chroma_db:/app/chroma_db rag
```

## Caching

`/ask` responses are cached in Redis, keyed by the (question, top_k) pair, with a TTL (`cache_ttl`, default 1 hour). Repeated questions are served instantly without calling Gemini again. The cache is cleared automatically whenever `/index` indexes new files or a source is deleted via `DELETE /sources/{filename}`, so stale answers don't stick around after the data changes.

With Docker Compose, a `redis` service is started alongside `rag` automatically. Running locally without Compose, point `REDIS_URL` (in `.env`) at a running Redis instance — it defaults to `redis://localhost:6379/0`.

## Async `/ask` & scaling (RabbitMQ)

On a cache miss, `/ask` doesn't run the embed/retrieve/generate pipeline inline — it publishes an RPC-style request to RabbitMQ (`ask_queue`) and awaits the reply, so the API process stays responsive under load instead of tying up a request per in-flight question. The actual pipeline work happens in `worker.py`, a separate consumer process:

1. `api.py` publishes `{"question", "top_k"}` with a `correlation_id` and a per-process reply queue (`rabbit.py`)
2. Any number of `worker.py` instances pull one job at a time (`prefetch_count=1`) from `ask_queue`, run embed → retrieve → generate, cache the result, and publish the response back to the caller's reply queue
3. `/ask` waits up to `rabbitmq_rpc_timeout` seconds (default 45) for the reply, returning `504` on timeout or `502` if the worker hit an error

Because workers are stateless and pull from a shared queue, throughput scales with worker count, not with threads in the API process — run more `worker.py` instances (or `docker compose up --scale worker=N`) to handle more concurrent questions in parallel. The real ceiling is the Gemini API's own rate limits, which no amount of workers can bypass.

### Backpressure

Before publishing, `/ask` checks how many jobs are already waiting in `ask_queue` (`rabbitmq_max_queue_depth`, default 300). If the backlog is already at capacity, the request is rejected immediately with `503` instead of being queued behind hundreds of others and silently timing out later — a fast, clear "try again" beats a slow, silent failure. Tune `rabbitmq_max_queue_depth` relative to your worker count and expected traffic.

RabbitMQ's management UI is available at `http://127.0.0.1:15672` (`guest`/`guest`) when running via Docker Compose.

**Note:** `worker.py` and `api.py` both read/write the same local ChromaDB path. This is fine for the read-heavy `/ask` path across multiple worker processes, but running `/index` concurrently with heavy `/ask` traffic is not guarded against — ChromaDB's local persistent client isn't designed for high-concurrency multi-process writes.

## Logging

The app emits one line per event to stdout (`logging_config.py`): request start/finish (method, path, status code, duration), cache hits/misses, vector store queries, Gemini embedding/generation calls, and full stack traces on failures. Every request gets a `request_id` (also returned as the `X-Request-ID` response header) that's attached to all log lines produced while handling it, so you can trace a single request across the pipeline.

Two formats are available via `LOG_FORMAT` in `.env`:
- `text` (default) — human-readable, colorized in a terminal: `12:06:56.013 INFO  worker    job_completed  top_k=3`. Use this for local development and `docker compose logs`.
- `json` — structured JSON, one object per line. Switch to this in real production deployments so a log aggregator (ELK, Loki, CloudWatch) can parse and index fields.

Set the log level with `LOG_LEVEL` in `.env` (default `INFO`).

## Testing

Integration tests (`tests/`) spin up the real FastAPI app and a real `worker.py` consumer in-process against a real Redis and RabbitMQ, with the Gemini client mocked out (no API key/cost needed). They cover the full `/ask` path: cache miss → RabbitMQ RPC → worker pipeline → cache write, plus `/index`, queue-full backpressure (`503`), and validation errors.

```bash
pip install -r requirements-dev.txt
docker compose up -d redis rabbitmq   # or point REDIS_URL/RABBITMQ_URL at running instances
python -m pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, a Docker build, and this test suite against fresh Redis/RabbitMQ service containers on every push and PR to `main`.

## Configuration

Settings live in `config.py` (backed by `.env`, see `Settings` for defaults): embedding/generation model names, chunk size/overlap, batch size, top-k, ChromaDB path, data directory, Redis cache URL/TTL, log level, and RabbitMQ URL/queue name/RPC timeout.

## Project structure

```
.
├── data/               # source .txt documents to index
├── frontend/            # chat UI ("Dory"), served as static files by api.py
│   ├── index.html
│   ├── style.css        # blue-toned theme, animated starfield background
│   └── app.js           # chat logic + starfield canvas animation
├── rag/
│   ├── chunker.py      # sentence-aware chunking with overlap
│   ├── embedder.py     # batched embeddings via Gemini (document/query task types)
│   ├── ingest.py        # shared indexing logic used by CLI and API
│   ├── vector_store.py # ChromaDB storage/query/list/delete with metadata + dedup
│   ├── cache.py         # Redis semantic cache for /ask responses (first-turn only)
│   ├── session.py       # Redis-backed per-session conversation history
│   └── generator.py    # generates cited, grounded answers via Gemini
├── config.py            # centralized settings (pydantic-settings, reads .env)
├── logging_config.py    # structured JSON logging setup
├── rabbit.py              # async RabbitMQ RPC client used by api.py's /ask
├── worker.py             # RabbitMQ consumer: runs the embed/retrieve/generate pipeline
├── api.py                # FastAPI app: /ask, /index, /sources endpoints + frontend/ static mount
├── main.py               # CLI: ingestion + Q&A loop
├── tests/                # integration tests (see Testing section below)
└── requirements.txt
```
