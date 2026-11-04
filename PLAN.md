# Company Snapshot Assistant

A small multi-agent project: given a stock ticker, fetch live data via MCP
tools and produce a short, source-cited summary using Grok + LangGraph.

Goal: learn real multi-agent state-passing, real MCP tool-calling, and basic
tracing/eval — in under ~15-20 hours of build time. Deliberately simplified
(see "Explicitly out of scope" below).

## What it does

Input: one or more stock tickers (e.g. "NVDA", or "AAPL, NVDA")
Output: for each ticker, a quote, recent headlines, and a ~150-word
summary with every factual claim tagged to its source, e.g.
"(source: quote)" or "(source: news)"

Another app gets that payload from the HTTP API in stage 7. This repo
does not build that app. A test page served by the same container is
how we check the contract.

## Architecture

### MCP server (`mcp_server.py`)
Exposes three tools:
1. `get_quote(ticker: str) -> dict`
   - Uses `yfinance`, no API key needed
   - Returns: current price, market cap, trailing P/E, 52-week high/low
   - Invalid ticker → return an error field in the dict, don't raise

2. `get_news(ticker: str, limit: int = 5) -> list[dict]`
   - Uses `feedparser` against Yahoo Finance's RSS feed for the ticker
   - Each item: headline, link, published date
3. `search_filings(ticker: str, query: str) -> list[dict]`
   - Uses local `all-MiniLM-L6-v2` embeddings and a persisted Chroma index
   - Returns relevant SEC filing chunks filtered to the ticker

### LangGraph workflow (`graph.py`)
Two nodes, linear, no conditional edges or loops:

`START → researcher → writer → END`

- **researcher**: calls all three MCP tools over stdio, writes structured findings
  into shared state (not prose — just the data)
- **writer**: reads findings from state, formats quote figures, then asks
  Grok (`ChatXAI`, `grok-4.7`) for a ~150-word summary with inline source
  tags and no invented numbers

Shared state (`SnapshotState`):
- `ticker: str`
- `quote_data: dict`
- `news_data: list[dict]`
- `filing_excerpts: list[dict]`
- `summary: str`

Quote figures in graph state stay raw numbers. The writer prompt sees
formatted copies (`$230.48`, `$5.57T`, `29.14`) so it can copy them.

### Model
- Researcher: no LLM; direct MCP tool calls
- Writer: Grok 4.7 (`grok-4.7`) via `langchain-xai`

### HTTP API (stage 7)
`graph.py` stays the library. A FastAPI app runs the same graph once
per ticker and streams each finished snapshot to the caller. The
service runs in a local container. No auth. The port is bound to
localhost. Keys come from `.env` at run time and are not copied into
the image.

### Tracing
LangSmith via environment variables only, no custom instrumentation:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=CLAUDESTOCKPROFILE

### Eval
No formal framework. Manually run 5 tickers through the full pipeline and
check, per ticker:
- [ ] All three tools were called
- [ ] Every number in the summary has a source tag
- [ ] Nothing in the summary is unsupported by the tool data

## Explicitly out of scope (v1)

- No document/filings search tool (no embeddings, no vector DB)
- No Critic/feedback loop between agents
- No formal eval harness — manual checklist only
- No product UI for the consuming app. The only page in this repo is
  the stage 7 test page.
- No auth, no public deploy, no database, no job queue

Natural next steps once Stage 8 works, in order of effort: add a Critic loop
back in → build a real eval set → automate filing updates.

## Build stages

Build and verify each stage before moving to the next.

### 1. MCP server standalone — done

File: `mcp_server.py`

Two tools, no agents, no LLM.

- `get_quote(ticker)` via yfinance: price, market cap, trailing P/E,
  52-week high/low
- `get_news(ticker, limit=5)` via Yahoo Finance RSS + feedparser:
  headline, link, published date
- Invalid ticker or fetch failure returns an `error` field; tools do not
  raise

Done looks like: a real ticker returns quote/news; a fake ticker returns
an error dict.

### 2. Grok ↔ MCP smoke test — done

File: `test_tool_call.py`

Starts `mcp_server.py` over stdio, offers both tools to Grok
(`grok-4.7` via `xai_sdk.AsyncClient`), runs a tool loop, prints each
call and the final answer. No LangGraph.

`GRPC_ENABLE_FORK_SUPPORT=0` is set before importing the xAI SDK so
MCP's fork+exec does not log gRPC fork warnings.

Run: `./venv/bin/python test_tool_call.py`

Done looks like: both `get_quote` and `get_news` are called, and the
answer uses the returned figures and headlines.

### 3. LangGraph skeleton — done

File: `graph.py`

Linear graph `START → researcher → writer → END` with shared
`SnapshotState`. Hardcoded ticker `NVDA`. Prints raw JSON state after
each node, then the rendered summary.

- Researcher writes `quote_data` and `news_data`
- Writer writes `summary`

Run: `./venv/bin/python graph.py`

Done looks like: after researcher, state has quote+news and no summary;
after writer, the same findings plus a summary.

### 4. Prompt-tune the writer — done

Same file: `graph.py` (writer prompt + `writer_findings`)

Writer rules: 140–160 words; one `(source: quote)` or `(source: news)`
tag per claim; copy formatted quote figures exactly; paraphrase a
headline only if it names the ticker or is clearly about that stock.

Quote formatting happens in code before the LLM call (`format_price`,
`format_cap`, `format_ratio`). Raw tool JSON stays in graph state.

NVDA check: 158 words, 4 quote tags, 4 news tags; figures `$230.48`,
`$5.57T`, `29.14`; unrelated RSS items omitted.

Run: `./venv/bin/python graph.py`

### 5. LangSmith tracing — done

Env vars only, no custom instrumentation:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=...`
- `LANGCHAIN_PROJECT=CLAUDESTOCKPROFILE`

`.env` already has these set. `graph.py` loads `.env` before LangChain
imports so tracing is actually enabled, then flushes traces on exit.

Run: `./venv/bin/python graph.py`

Check: open https://smith.langchain.com → project **CLAUDESTOCKPROFILE**
→ Tracing → latest run named `company-snapshot`. The tree should show
`researcher`, `writer`, and a nested `ChatXAI` LLM call.

Verified: a `company-snapshot` root run with spans `researcher` (chain),
`writer` (chain), and `ChatXAI` (llm).

### 6. Manual eval pass — done

Same file: `graph.py`. With no arguments it still runs hardcoded `NVDA`.
Arguments run that same graph once per ticker in one MCP session.

Run: `./venv/bin/python graph.py AAPL NVDA MSFT GOOGL AMZN`

Checked AAPL, NVDA, MSFT, GOOGL, and AMZN. GOOGL's feed had no headline
that names Alphabet, so it stood in for the thin-news case. No invalid
ticker in this pass.

Per ticker:

- [x] Both tools were called. Researcher state has `quote_data` and
  `news_data` before `summary` exists.
- [x] Every numeral sits in a sentence with `(source: quote)` or
  `(source: news)`.
- [x] Quote figures match the formatters exactly, and news lines
  paraphrase only headlines that name the company. Unrelated RSS items
  are omitted. GOOGL says the feed does not name GOOGL.

| Ticker | Words | Quote tags | News tags | Figures copied |
| --- | --- | --- | --- | --- |
| AAPL | 145 | 4 | 5 | `$336.64`, `$4.91T`, `38.61`, `$345.34`, `$243.42` |
| NVDA | 160 | 6 | 5 | `$229.28`, `$5.54T`, `28.99`, `$243.37`, `$164.27` |
| MSFT | 154 | 7 | 6 | `$535.07`, `$3.97T`, `29.81`, `$553.72`, `$349.20` |
| GOOGL | 159 | 9 | 3 | `$351.66`, `$4.30T`, `17.64`, `$408.61`, `$235.84` |
| AMZN | 150 | 6 | 3 | `$262.43`, `$2.83T`, `21.11`, `$287.20`, `$196.00` |

No writer-prompt or formatter change. The writer sometimes repeats a
price to say it sits inside the 52-week high and low. Those sentences
use only the formatted quote figures.

### 7. Containerized snapshot API — done

Host the graph as a local HTTP API so another app can render the
snapshot. One container serves the API and a test page. No `run.py`.
`python graph.py` stays as the CLI used in stages 3–6.

Files: `api.py`, `static/index.html`, `requirements.txt`,
`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `README.md`

`requirements.txt` pins what the image installs, including the packages
already in `./venv` plus `fastapi` and `uvicorn`. `.dockerignore`
excludes `venv/`, `.env`, and `__pycache__/`.

#### API

`POST /snapshots` with `{"tickers": ["AAPL", "NVDA"]}`.

Response is `text/event-stream`. Tickers run sequentially in one MCP
session, in request order. Each ticker is one SSE event, sent when
that ticker's graph run finishes. Do not buffer the stream.

Normalize symbols with strip and upper. Drop blanks. Keep order.
Reject an empty list with 400. Cap the list at 10.

Events:

- `snapshot` — one finished ticker
- `error` — that ticker's run raised; continue with the next ticker
- `done` — stream finished, with a count of snapshot events

A tool payload that already has an `error` field is still a
`snapshot`. The writer already covers a failed quote. Do not invent
a price.

`snapshot` data:

- `ticker`
- `quote`: raw `quote_data` (numbers, or `error`)
- `quote_display`: formatter output (`$336.64`, `$4.91T`, `38.61`),
  omitted when `quote.error` is set
- `news`: `headline`, `link`, `published` for each item
- `filings`: retrieved `text`, `source_file`, and `chunk_id`
- `summary`: the writer prose

`GET /health` returns a small ok payload. `GET /` serves the test page.

CORS allows `http://localhost` and `http://127.0.0.1` origins so a
local browser app can call the API. No wildcard origin.

#### Test page

One HTML page, no frontend build. A text field accepts one or more
tickers, separated by commas or whitespace. Submit starts
`POST /snapshots` and reads the stream.

Render each ticker when its event arrives. Do not wait for the rest
of the list. Each card shows:

- ticker
- formatted price, market cap, trailing P/E, 52-week high and low
- the quote error, when present, and no price
- headlines as links, with the published time
- retrieved filing excerpts in a collapsed evidence section
- the summary, with the source tags left visible
- a pending state for tickers that have not arrived

#### Container

`docker-compose.yml` builds the image and publishes the port on
localhost only:

`127.0.0.1:8000:8000`

Pass `.env` with `env_file`. Do not `COPY` `.env` into the image.
The process needs `XAI_API_KEY`. LangSmith vars stay optional and
keep tracing on when they are present.

Run: `docker compose up --build`

Open: http://127.0.0.1:8000

README covers the `.env` keys, that command, and a `curl` example of
the event stream. It can point at `graph.py` and `test_tool_call.py`
for the earlier stages.

Done looks like: the page accepts AAPL and NVDA, the AAPL card appears
before NVDA finishes, both cards show quote figures, linked headlines,
and the sourced summary, and a bad symbol in the same request renders
its error without dropping the valid tickers.

Verified with AAPL, NOPE999, and NVDA in one request. All three cards
started pending. AAPL filled first, while NOPE999 and NVDA were still
waiting. NOPE999 then showed `Invalid ticker: NOPE999` and no price.
NVDA filled last. The stream ended with 3 snapshots.

| Ticker | Result |
| --- | --- |
| AAPL | `$336.64`, `$4.91T`, `38.61`, `$345.34`, `$243.42`; five linked headlines; quote and news tags |
| NOPE999 | quote error, no figures, empty news, summary says the quote failed |
| NVDA | `$229.28`, `$5.54T`, `28.99`, `$243.37`, `$164.27`; linked headlines; quote and news tags |

## Environment

- Python 3.10+ (venv at `./venv`; container image uses 3.12)
- Dependencies: `mcp`, `yfinance`, `feedparser`, `langgraph`,
  `xai-sdk`, `langchain-xai`, `python-dotenv`, and for stage 7
  `fastapi`, `uvicorn`; stage 8 adds `chromadb` and
  `sentence-transformers`
- API keys in `.env` (gitignored): `XAI_API_KEY`, `LANGCHAIN_API_KEY`
- Stage 7 runs through Docker Compose, bound to `127.0.0.1:8000`

## Stage 8: RAG — filings search (post-v1 addition) — done

Adds a third tool so the Researcher can pull grounded excerpts from actual
company filings, not just live price/news. This is the one part of the
project that uses real retrieval-augmented generation (chunk → embed →
store → similarity search → inject into context).

### Scope and constraints

- Corpus is static and manual: 3-5 tickers, 1-2 filings each (10-K or
  earnings transcript), saved as plain .txt files in `filings/`.
  No automated scraping/fetching of filings — that's a future stage, not this one.
- Embeddings run locally via `sentence-transformers`, no API key, no cost.
- Vector store is ChromaDB, persisted to a local folder (`./chroma_db`).
  No hosted/cloud vector DB for this stage.
- Indexing is a one-time offline step (`build_index.py`), not something
  that runs on every query.

### New dependencies

pip install chromadb sentence-transformers


### New files

**`filings/`** — folder of manually downloaded .txt filings, one or more
per ticker, sourced from SEC EDGAR.

**`build_index.py`** — run once, not part of the live app:
1. Read each .txt file in `filings/`
2. Chunk into ~500-word pieces with some overlap (simple sliding window)
3. Tag each chunk with metadata: `{ticker, source_file, chunk_id}`
4. Embed each chunk using `sentence-transformers` (`all-MiniLM-L6-v2`)
5. Store vector + text + metadata in ChromaDB at `./chroma_db`
6. Print a count of chunks indexed per ticker when done, as a sanity check

### New MCP tool (added to `mcp_server.py`)

`search_filings(ticker: str, query: str) -> list[dict]`
- Embeds the query with the same model used in `build_index.py`
- Runs similarity search against ChromaDB, filtered to that ticker
- Returns top 2-3 matching chunks: `{text, source_file, chunk_id}`
- No matching ticker in the index → return an empty list, don't raise

### Changes to existing components

- **Researcher node**: now calls three tools instead of two
  (`get_quote`, `get_news`, `search_filings`)
- **Writer node**: add a third citation category — "(source: filing)" —
  alongside quote/news
- **Shared state**: add `filing_excerpts: list[dict]` field

### Build order

1. Manually gather 3-5 filings as .txt files into `filings/`
2. Write and run `build_index.py`; verify chunk count via Chroma's
   `.count()` before moving on
3. Add `search_filings` to `mcp_server.py`; test standalone with a few
   queries per ticker — confirm returned chunks are actually relevant
   before touching the agents
4. Add the tool to the Researcher node's tool list; re-run a
   hardcoded-ticker test and confirm filing excerpts show up in state
5. Update the Writer prompt for the third citation type
6. Re-run the 5-ticker manual eval checklist, now including filing
   citations

### Eval additions for this stage

- [x] `search_filings` returns relevant (not just any) chunks for a
      sample of test queries
- [x] Writer summary includes at least one filing-sourced claim when
      relevant filing content exists for that ticker
- [x] No filing claim appears in the summary without a "(source: filing)" tag

### Implementation and verification

- Corpus: one official SEC 2025 10-K each for AAPL, NVDA, MSFT, GOOGL,
  and AMZN in `filings/`, with the EDGAR source URL in each file.
- `build_index.py` produced 557 overlapping 500-word chunks:
  AAPL 76, NVDA 129, MSFT 117, GOOGL 130, and AMZN 105.
- `test_filings.py` returned three relevant excerpts for every indexed
  ticker and an empty list for `NOPE999`.
- The five-ticker graph eval produced filing-tagged summaries for every
  ticker. Filing-tag counts were AAPL 2, NVDA 3, MSFT 4, GOOGL 4, and
  AMZN 3; no unsupported filing claim was observed.
- The local embedding model is loaded without network access at query time.
- The Docker image builds the static index during image creation. A clean
  Compose build and container run passed; its AAPL SSE event contained three
  filing excerpts and four filing-tagged claims, followed by `done`.

### Explicitly out of scope (still, even at Stage 8)

- No automated filing fetching/updating
- No chunking-strategy tuning beyond a basic fixed-size sliding window
- No hosted vector DB
- No re-ranking or hybrid search — plain similarity search only