"""MCP server for live stock data and a local SEC filing index."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import feedparser
import yfinance as yf
from mcp.server import MCPServer

mcp = MCPServer(
    "company-snapshot",
    instructions=(
        "Fetch a live stock quote, recent Yahoo Finance headlines, and relevant "
        "excerpts from a local SEC filing index."
    ),
    dependencies=["yfinance", "feedparser", "chromadb", "sentence-transformers"],
)

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "company_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Yahoo's headline RSS. The ticker is interpolated after it is normalized.
_NEWS_FEED = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={ticker}&region=US&lang=en-US"
)
_REQUEST_TIMEOUT_SECONDS = 15


def _number(value: object) -> int | float | None:
    """Turn a yfinance value into a plain JSON number, or None if it is missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


@mcp.tool()
def get_quote(ticker: str) -> dict[str, Any]:
    """Return the current quote for a stock ticker.

    Includes current price, market cap, trailing P/E, and 52-week high and low.
    An unrecognized ticker returns a dict with an error field and does not raise.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        return {"error": "Ticker is required."}

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # yfinance raises for some missing or blocked symbols
        return {"ticker": symbol, "error": f"Could not fetch quote: {exc}"}

    price = info.get("currentPrice")
    if price is None:
        price = info.get("regularMarketPrice")

    # A symbol Yahoo does not recognize comes back as a nearly empty dict,
    # often only {"trailingPegRatio": None}, with no price and no quote type.
    if price is None and not info.get("quoteType"):
        return {"ticker": symbol, "error": f"Invalid ticker: {symbol}"}

    return {
        "ticker": info.get("symbol") or symbol,
        "current_price": _number(price),
        "market_cap": _number(info.get("marketCap")),
        "trailing_pe": _number(info.get("trailingPE")),
        "fifty_two_week_high": _number(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _number(info.get("fiftyTwoWeekLow")),
    }


@mcp.tool()
def get_news(ticker: str, limit: int = 5) -> list[dict[str, str]]:
    """Return recent Yahoo Finance headlines for a stock ticker.

    Each item has headline, link, and published date. limit defaults to 5.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        return [{"error": "Ticker is required."}]
    if limit < 1:
        return [{"error": "limit must be at least 1."}]

    url = _NEWS_FEED.format(ticker=quote(symbol, safe=".-"))
    request = Request(url, headers={"User-Agent": "company-snapshot/1.0"})
    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        return [{"error": f"Could not fetch news: {exc}"}]

    feed = feedparser.parse(payload)
    items: list[dict[str, str]] = []
    for entry in feed.entries[:limit]:
        items.append(
            {
                "headline": entry.get("title") or "",
                "link": entry.get("link") or "",
                "published": entry.get("published") or entry.get("updated") or "",
            }
        )
    return items


@lru_cache(maxsize=1)
def _filings_collection():
    """Open the local index lazily so quote/news calls stay lightweight."""
    if not CHROMA_DIR.is_dir():
        return None

    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        local_files_only=True,
    )
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except Exception:
        return None


@mcp.tool()
def search_filings(
    ticker: str, query: str, limit: int = 3
) -> list[dict[str, Any]]:
    """Return relevant chunks from locally indexed SEC filings.

    Results are filtered to the normalized ticker and contain text, source_file,
    and chunk_id. A missing ticker/index or blank query returns an empty list.
    """
    symbol = ticker.strip().upper()
    search_text = query.strip()
    if not symbol or not search_text or limit < 1:
        return []

    collection = _filings_collection()
    if collection is None:
        return []

    try:
        result = collection.query(
            query_texts=[search_text],
            n_results=min(limit, 3),
            where={"ticker": symbol},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    excerpts: list[dict[str, Any]] = []
    for text, metadata in zip(documents, metadatas):
        if not isinstance(text, str) or not isinstance(metadata, dict):
            continue
        excerpts.append(
            {
                "text": text,
                "source_file": str(metadata.get("source_file") or ""),
                "chunk_id": metadata.get("chunk_id"),
            }
        )
    return excerpts


if __name__ == "__main__":
    mcp.run()
