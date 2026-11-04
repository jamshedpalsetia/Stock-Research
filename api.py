"""Local HTTP API for company snapshots.

graph.py stays the library. This process runs that graph once per ticker,
in one MCP session, and streams each finished snapshot as a server-sent event.
"""

from __future__ import annotations

import os

# MCP stdio fork+execs mcp_server.py. Disable gRPC's atfork handlers before
# LangChain imports so a live gRPC thread does not skip them and warn.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import asyncio
import json
import logging
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

from graph import (
    MODEL,
    REASONING_EFFORT,
    ROOT,
    SERVER,
    ChatXAI,
    build_graph,
    format_cap,
    format_price,
    format_ratio,
    wait_for_all_tracers,
)

log = logging.getLogger("snapshot")

INDEX = Path(__file__).resolve().parent / "static" / "index.html"
MAX_TICKERS = 10
_LOCAL_ORIGIN = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")
_SECRET_NAMES = ("XAI_API_KEY", "LANGCHAIN_API_KEY")


class SnapshotRequest(BaseModel):
    tickers: list[str]


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Strip, uppercase, and drop blanks. Keep the caller's order."""
    return [item.strip().upper() for item in tickers if item and item.strip()]


def public_error(exc: BaseException) -> str:
    """Return an error string with API key values removed."""
    message = str(exc) or exc.__class__.__name__
    for name in _SECRET_NAMES:
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, name)
    return message


def quote_display(quote: dict[str, Any]) -> dict[str, str]:
    """Format quote figures the same way the writer sees them."""
    display: dict[str, str] = {}
    for key, formatted in (
        ("current_price", format_price(quote.get("current_price"))),
        ("market_cap", format_cap(quote.get("market_cap"))),
        ("trailing_pe", format_ratio(quote.get("trailing_pe"))),
        ("fifty_two_week_high", format_price(quote.get("fifty_two_week_high"))),
        ("fifty_two_week_low", format_price(quote.get("fifty_two_week_low"))),
    ):
        if formatted is not None:
            display[key] = formatted
    return display


def snapshot_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Build one snapshot event. Omit quote_display when the quote failed."""
    quote = state.get("quote_data")
    if not isinstance(quote, dict):
        quote = {"error": f"Unexpected quote payload: {quote!r}"}

    news: list[dict[str, str]] = []
    for item in state.get("news_data") or []:
        if not isinstance(item, dict):
            news.append(
                {
                    "headline": "",
                    "link": "",
                    "published": "",
                    "error": f"Unexpected news item: {item!r}",
                }
            )
            continue
        row = {
            "headline": str(item.get("headline") or ""),
            "link": str(item.get("link") or ""),
            "published": str(item.get("published") or ""),
        }
        if item.get("error"):
            row["error"] = str(item["error"])
        news.append(row)

    payload: dict[str, Any] = {
        "ticker": state.get("ticker") or "",
        "quote": quote,
        "news": news,
        "filings": [
            {
                "text": str(item.get("text") or ""),
                "source_file": str(item.get("source_file") or ""),
                "chunk_id": item.get("chunk_id"),
            }
            for item in state.get("filing_excerpts") or []
            if isinstance(item, dict)
        ],
        "summary": state.get("summary") or "",
    }
    if not quote.get("error"):
        payload["quote_display"] = quote_display(quote)
    return payload


def sse(event: str, data: dict[str, Any]) -> str:
    """Encode one SSE event. JSON stays on a single data line."""
    body = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def make_llm() -> ChatXAI:
    """Writer model. Same settings as the graph.py CLI."""
    return ChatXAI(
        model=MODEL,
        temperature=0,
        timeout=600,
        extra_body={"reasoning_effort": REASONING_EFFORT},
    )


async def run_ticker(compiled, ticker: str) -> dict[str, Any]:
    """Run one ticker through the graph and return the merged state."""
    state: dict[str, Any] = {"ticker": ticker}
    async for update in compiled.astream(
        {"ticker": ticker},
        stream_mode="updates",
        config={"run_name": "company-snapshot", "metadata": {"ticker": ticker}},
    ):
        for payload in update.values():
            if isinstance(payload, dict):
                state.update(payload)
    return state


async def iter_snapshot_events(tickers: list[str]) -> AsyncIterator[str]:
    """Run tickers sequentially in one MCP session and yield SSE frames."""
    # Send a comment immediately so the response is not buffered until the
    # first ticker's model call returns.
    yield ": open\n\n"
    count = 0
    try:
        server = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER)],
            cwd=ROOT,
        )
        async with Client(server) as mcp_client:
            compiled = build_graph(mcp_client, make_llm())
            for ticker in tickers:
                log.info("snapshot start %s", ticker)
                try:
                    state = await run_ticker(compiled, ticker)
                except Exception as exc:
                    log.exception("snapshot failed %s", ticker)
                    yield sse("error", {"ticker": ticker, "error": public_error(exc)})
                    continue
                count += 1
                log.info("snapshot ready %s", ticker)
                yield sse("snapshot", snapshot_payload(state))
            try:
                await asyncio.to_thread(wait_for_all_tracers)
            except Exception:
                log.exception("trace flush failed")
    except Exception as exc:
        log.exception("snapshot session failed")
        yield sse("error", {"error": public_error(exc)})
    yield sse("done", {"count": count})


class LocalhostCORS:
    """Allow only localhost origins, without buffering the SSE body.

    Starlette's CORSMiddleware reads the response through BaseHTTPMiddleware,
    which holds a stream until it ends. These snapshots must leave as each
    ticker finishes.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        origin = ""
        requested_headers = b"Content-Type, Accept"
        for key, value in scope.get("headers") or []:
            if key == b"origin":
                origin = value.decode("latin-1")
            elif key == b"access-control-request-headers" and b"\n" not in value and b"\r" not in value:
                requested_headers = value

        allowed = bool(origin and _LOCAL_ORIGIN.fullmatch(origin))
        if scope["method"] == "OPTIONS" and allowed:
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", origin.encode("latin-1")),
                        (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                        (b"access-control-allow-headers", requested_headers),
                        (b"access-control-max-age", b"600"),
                        (b"vary", b"Origin"),
                        (b"content-length", b"0"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message):
            if allowed and message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"access-control-allow-origin", origin.encode("latin-1")))
                headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)


app = FastAPI(title="Company Snapshot", version="1.0")
app.add_middleware(LocalhostCORS)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        INDEX,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/snapshots")
async def snapshots(body: SnapshotRequest) -> StreamingResponse:
    tickers = normalize_tickers(body.tickers)
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker.")
    if len(tickers) > MAX_TICKERS:
        raise HTTPException(status_code=400, detail="At most 10 tickers.")
    if not os.environ.get("XAI_API_KEY"):
        raise HTTPException(status_code=503, detail="XAI_API_KEY is missing.")
    return StreamingResponse(
        iter_snapshot_events(tickers),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
