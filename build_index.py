"""Build the persistent Chroma index for the static SEC filing corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).resolve().parent
FILINGS_DIR = ROOT / "filings"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "company_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 500
CHUNK_OVERLAP = 75
BATCH_SIZE = 100


def chunks(text: str) -> list[str]:
    """Split text into fixed-size word windows with overlap."""
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [
        " ".join(words[start : start + CHUNK_WORDS])
        for start in range(0, len(words), step)
        if words[start : start + CHUNK_WORDS]
    ]


def ticker_from_path(path: Path) -> str:
    """Read the ticker prefix from names such as AAPL_2025_10-K.txt."""
    ticker = path.stem.split("_", 1)[0].strip().upper()
    if not ticker:
        raise ValueError(f"Could not infer ticker from {path.name}")
    return ticker


def main() -> int:
    files = sorted(FILINGS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt filings found in {FILINGS_DIR}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    records: list[tuple[str, str, dict[str, str | int]]] = []
    counts: Counter[str] = Counter()
    for path in files:
        ticker = ticker_from_path(path)
        for chunk_id, text in enumerate(chunks(path.read_text(encoding="utf-8"))):
            records.append(
                (
                    f"{ticker}:{path.name}:{chunk_id}",
                    text,
                    {
                        "ticker": ticker,
                        "source_file": path.name,
                        "chunk_id": chunk_id,
                    },
                )
            )
            counts[ticker] += 1

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        collection.add(
            ids=[record[0] for record in batch],
            documents=[record[1] for record in batch],
            metadatas=[record[2] for record in batch],
        )

    for ticker in sorted(counts):
        print(f"{ticker}: {counts[ticker]} chunks")
    print(f"Total: {collection.count()} chunks in {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
