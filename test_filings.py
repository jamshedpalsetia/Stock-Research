"""Standalone smoke test for the Stage 8 filing-search MCP tool."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from graph import as_filings, mcp_payload

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"
QUERIES = {
    "AAPL": "How do products and services contribute to net sales?",
    "NVDA": "What drives data center demand and what supply risks exist?",
    "MSFT": "How are cloud and Azure affecting revenue growth?",
    "GOOGL": "What drives advertising revenue and what are the AI risks?",
    "AMZN": "How are AWS sales and operating income performing?",
}


async def main() -> int:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=ROOT,
    )
    async with Client(server) as client:
        for ticker, query in QUERIES.items():
            result = await client.call_tool(
                "search_filings",
                {"ticker": ticker, "query": query, "limit": 3},
            )
            excerpts = as_filings(mcp_payload(result))
            if not excerpts:
                raise AssertionError(f"No filing excerpts returned for {ticker}")
            preview = " ".join(str(excerpts[0]["text"]).split())[:240]
            print(f"{ticker}: {len(excerpts)} excerpts")
            print(f"  {preview}…")

        missing = await client.call_tool(
            "search_filings",
            {"ticker": "NOPE999", "query": "revenue and risks"},
        )
        missing_payload = as_filings(mcp_payload(missing))
        if missing_payload:
            raise AssertionError(
                f"Expected no excerpts for unknown ticker: "
                f"{json.dumps(missing_payload)[:200]}"
            )
        print("NOPE999: 0 excerpts (expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
