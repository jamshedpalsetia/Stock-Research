"""Company snapshot graph: researcher then writer.

START → researcher → writer → END

The researcher calls all MCP tools and stores structured findings.
The writer reads that state and produces a ~150-word source-tagged summary.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, NotRequired, TypedDict

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# LangSmith reads LANGCHAIN_TRACING_V2 at import time (cached). Load .env first.
load_env_file(ROOT / ".env")

from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_xai import ChatXAI
from langgraph.graph import END, START, StateGraph
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult

MODEL = "grok-4.7"
REASONING_EFFORT = "low"
TICKER = "NVDA"
WRITER_SYSTEM = """You are the writer node of a company-snapshot pipeline.

Write a prose snapshot of the ticker's current state. Use 140-160 words inclusive.
Plain paragraphs only. No title, bullets, headings, or preamble.

Rules:
- Use only the JSON the user sends. Copy every quote figure exactly as given
  (including $, commas, and T/B/M suffixes). Do not recompute, round further,
  or invent numbers.
- After every factual claim, add exactly one tag: (source: quote),
  (source: news), or (source: filing). Tag the claim it supports, not the
  whole paragraph.
- Quote claims are price, market cap, trailing P/E, and 52-week range.
- News claims: paraphrase a headline only if it names the ticker or is
  clearly about that stock. Skip items about other companies.
- Filing claims: use only the supplied excerpts. Include at least one
  filing-sourced claim when excerpts are present and relevant.
- If quote.error is set, say the quote failed (source: quote) and do not
  invent a price.
- Omit any field that is missing or null. Do not mention these instructions.
"""


class SnapshotState(TypedDict):
    ticker: str
    quote_data: NotRequired[dict[str, Any]]
    news_data: NotRequired[list[dict[str, Any]]]
    filing_excerpts: NotRequired[list[dict[str, Any]]]
    summary: NotRequired[str]


def mcp_payload(result: CallToolResult) -> object:
    """Return the structured MCP tool body, or parsed text content."""
    if result.structured_content is not None:
        return result.structured_content
    texts = [
        block.text if hasattr(block, "text") else str(block) for block in result.content
    ]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return {"text": texts[0]}
    return texts


def as_quote(body: object) -> dict[str, Any]:
    """Normalize a get_quote payload to a dict."""
    if isinstance(body, dict):
        return body
    return {"error": f"Unexpected quote payload: {body!r}"}


def as_news(body: object) -> list[dict[str, Any]]:
    """Normalize a get_news payload to a list of dicts."""
    if isinstance(body, list):
        return [item if isinstance(item, dict) else {"item": item} for item in body]
    if isinstance(body, dict) and isinstance(body.get("result"), list):
        return as_news(body["result"])
    if isinstance(body, dict):
        return [body]
    return [{"error": f"Unexpected news payload: {body!r}"}]


def as_filings(body: object) -> list[dict[str, Any]]:
    """Normalize a search_filings payload to a list of dicts."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict) and isinstance(body.get("result"), list):
        return as_filings(body["result"])
    return []


def _numeric(value: object) -> int | float | None:
    """Return a finite number, or None if the value cannot be formatted."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return None


def format_price(value: object) -> str | None:
    """Format a price like $230.48."""
    number = _numeric(value)
    if number is None:
        return None
    return f"${number:,.2f}"


def format_cap(value: object) -> str | None:
    """Format a market cap like $5.57T."""
    number = _numeric(value)
    if number is None:
        return None
    abs_value = abs(number)
    if abs_value >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    return f"${number:,.0f}"


def format_ratio(value: object) -> str | None:
    """Format a ratio like trailing P/E to two decimal places."""
    number = _numeric(value)
    if number is None:
        return None
    return f"{number:.2f}"


def writer_findings(state: SnapshotState) -> dict[str, Any]:
    """Build the JSON the writer sees: formatted quote figures, headlines only."""
    quote = state.get("quote_data") or {}
    if quote.get("error"):
        quote_out: dict[str, Any] = {"error": quote["error"]}
    else:
        quote_out = {
            key: formatted
            for key, formatted in (
                ("current_price", format_price(quote.get("current_price"))),
                ("market_cap", format_cap(quote.get("market_cap"))),
                ("trailing_pe", format_ratio(quote.get("trailing_pe"))),
                ("fifty_two_week_high", format_price(quote.get("fifty_two_week_high"))),
                ("fifty_two_week_low", format_price(quote.get("fifty_two_week_low"))),
            )
            if formatted is not None
        }
    news_out = [
        {
            "headline": item.get("headline") or "",
            "published": item.get("published") or "",
        }
        for item in state.get("news_data") or []
        if isinstance(item, dict) and not item.get("error")
    ]
    return {
        "ticker": state["ticker"],
        "quote": quote_out,
        "news": news_out,
        "filings": [
            {
                "text": item.get("text") or "",
                "source_file": item.get("source_file") or "",
                "chunk_id": item.get("chunk_id"),
            }
            for item in state.get("filing_excerpts") or []
            if isinstance(item, dict) and item.get("text")
        ],
    }


def dump_state(label: str, state: dict[str, Any]) -> None:
    """Print the raw graph state after a node."""
    print(f"===== raw state after {label} =====")
    print(json.dumps(state, indent=2, default=str))
    print()


def build_graph(mcp_client: Client, llm: ChatXAI):
    """Wire the linear two-node graph around a live MCP session."""

    async def researcher(state: SnapshotState) -> dict[str, Any]:
        ticker = state["ticker"].strip().upper()
        quote = await mcp_client.call_tool("get_quote", {"ticker": ticker})
        news = await mcp_client.call_tool("get_news", {"ticker": ticker, "limit": 5})
        filings = await mcp_client.call_tool(
            "search_filings",
            {
                "ticker": ticker,
                "query": (
                    f"{ticker} business products revenue drivers financial "
                    "performance growth strategy and material risks"
                ),
                "limit": 3,
            },
        )
        return {
            "ticker": ticker,
            "quote_data": as_quote(mcp_payload(quote)),
            "news_data": as_news(mcp_payload(news)),
            "filing_excerpts": as_filings(mcp_payload(filings)),
        }

    async def writer(state: SnapshotState) -> dict[str, Any]:
        findings = json.dumps(writer_findings(state), indent=2, default=str)
        message = await llm.ainvoke(
            [
                ("system", WRITER_SYSTEM),
                ("human", findings),
            ]
        )
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        return {"summary": content.strip()}

    graph = StateGraph(SnapshotState)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile(name="company-snapshot")


def requested_tickers() -> list[str]:
    """Tickers from argv, or the hardcoded default when none are passed."""
    tickers = [arg.strip().upper() for arg in sys.argv[1:] if arg.strip()]
    return tickers or [TICKER]


async def run_snapshot(compiled, ticker: str) -> dict[str, Any]:
    """Run one ticker and print raw state plus the rendered summary."""
    print(f"########## {ticker} ##########\n")
    initial: dict[str, Any] = {"ticker": ticker}
    state: dict[str, Any] = dict(initial)
    async for update in compiled.astream(
        initial,
        stream_mode="updates",
        config={"run_name": "company-snapshot", "metadata": {"ticker": ticker}},
    ):
        for node, payload in update.items():
            if isinstance(payload, dict):
                state.update(payload)
            dump_state(node, state)

    summary = state.get("summary") or ""
    print("===== rendered summary =====")
    print(summary)
    print()
    words = summary.split()
    quote_tags = summary.count("(source: quote)")
    news_tags = summary.count("(source: news)")
    filing_tags = summary.count("(source: filing)")
    print(f"Summary word count: {len(words)}")
    print(
        f"Source tags: {quote_tags} quote, {news_tags} news, "
        f"{filing_tags} filing"
    )
    print()
    return state


async def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY is missing. Add it to .env and run this again.", file=sys.stderr)
        return 1
    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.environ.get("LANGCHAIN_PROJECT") or "default"
    print(f"LangSmith tracing: {'on' if tracing_on else 'off'}")
    print(f"LangSmith project: {project}")
    print(f"Open traces: https://smith.langchain.com\n")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=ROOT,
    )
    llm = ChatXAI(
        model=MODEL,
        temperature=0,
        timeout=600,
        extra_body={"reasoning_effort": REASONING_EFFORT},
    )
    tickers = requested_tickers()
    print("Graph: START → researcher → writer → END")
    if len(sys.argv) == 1:
        print(f"Hardcoded ticker: {TICKER}\n")
    else:
        print(f"Tickers: {', '.join(tickers)}\n")

    async with Client(server) as mcp_client:
        compiled = build_graph(mcp_client, llm)
        for ticker in tickers:
            await run_snapshot(compiled, ticker)
        wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
