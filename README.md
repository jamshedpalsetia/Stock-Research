# Company Snapshot Assistant

A ticker goes in. A live quote, recent headlines, relevant SEC filing
excerpts, and a short source-tagged summary come out. `graph.py` is the
LangGraph pipeline. This container serves that pipeline over HTTP, plus a
test page.

## Filing index

`filings/` contains one SEC 10-K text file for each supported evaluation
ticker. Rebuild the local Chroma index after changing that corpus:

```bash
./venv/bin/python build_index.py
```

The command chunks each filing into overlapping 500-word windows, embeds
them locally with `all-MiniLM-L6-v2`, and writes `chroma_db/`. The Docker
image performs this build automatically. `search_filings` returns up to
three matching chunks for a ticker; tickers absent from the index return an
empty list.

## Keys

Put these in `.env` in this directory. Compose reads that file at run time.
The image does not contain it.

- `XAI_API_KEY` is required.
- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` are
  optional. When they are set, LangSmith tracing stays on. The project name
  used in earlier stages is `CLAUDESTOCKPROFILE`.

## Run the API

```bash
docker compose up --build
```

Open http://127.0.0.1:8000

The port is published on localhost only. Inside the container the process
listens on `0.0.0.0:8000` so that publish can reach it.

The test page takes one or more tickers, separated by commas or spaces, and
draws each card when that ticker's event arrives. Each card includes a
collapsed **Filing evidence** section showing the retrieved SEC excerpts,
source filename, and chunk number.

## Event stream

`POST /snapshots` with `{"tickers": ["AAPL", "NVDA"]}`. The response is
`text/event-stream`. Tickers run in order, in one MCP session. An empty list
or more than 10 tickers returns 400. Symbols are stripped and uppercased.

```bash
curl -N -X POST http://127.0.0.1:8000/snapshots \
  -H 'Content-Type: application/json' \
  -d '{"tickers":["AAPL","NVDA"]}'
```

`-N` turns off curl's buffer so each event prints when it is sent.

```text
event: snapshot
data: {"ticker":"AAPL","quote":{...},"quote_display":{"current_price":"$336.64","market_cap":"$4.91T","trailing_pe":"38.61",...},"news":[{"headline":"...","link":"...","published":"..."}],"filings":[{"text":"...","source_file":"AAPL_2025_10-K.txt","chunk_id":1}],"summary":"... (source: filing) ..."}

event: done
data: {"count":1}
```

`quote` is the raw tool payload. `quote_display` is omitted when `quote` has
an `error` field. That ticker is still a `snapshot`. A run that raises is an
`error` event, and the stream continues with the next ticker.

`GET /health` returns `{"ok": true}`.

## Earlier stages

```bash
./venv/bin/python test_tool_call.py
./venv/bin/python test_filings.py
./venv/bin/python graph.py
./venv/bin/python graph.py AAPL NVDA MSFT GOOGL AMZN
```

`test_tool_call.py` is the Grok tool-call smoke test. `graph.py` is the
researcher → writer CLI.
