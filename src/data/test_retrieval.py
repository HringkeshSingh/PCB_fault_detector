"""
Sanity-check script: query ChromaDB with sample defect descriptions.

This is NOT the Phase 3 RAG retrieval pipeline — it only confirms embeddings
and storage work correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.ingest_cases import COLLECTION_NAME, DEFAULT_CHROMA_DIR, EMBEDDING_MODEL
from src.vision.constants import PROJECT_ROOT

SAMPLE_QUERIES: list[dict[str, str]] = [
    {
        "label": "open circuit / etching",
        "query": (
            "Open defect on a signal trace caused by under-etching or broken "
            "copper after thermal stress"
        ),
        "expected_class": "open",
    },
    {
        "label": "short / contamination",
        "query": (
            "Electrical short between adjacent pads from ionic contamination "
            "or solder bridging"
        ),
        "expected_class": "short",
    },
    {
        "label": "pin-hole / plating void",
        "query": (
            "Pin hole void in copper plane from incomplete electroless seed "
            "or air bubble during plating"
        ),
        "expected_class": "pin-hole",
    },
]


def query_cases(
    query_text: str,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
    n_results: int = 3,
) -> list[dict[str, Any]]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    chroma_dir = chroma_dir.resolve()
    if not chroma_dir.is_dir():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {chroma_dir}. Run ingest_cases.py first."
        )

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)

    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    hits: list[dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append(
            {
                "case_id": meta.get("case_id"),
                "defect_class": meta.get("defect_class"),
                "root_cause": meta.get("root_cause"),
                "distance": results["distances"][0][i],
            }
        )
    return hits


def run_sanity_checks(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    n_results: int = 3,
) -> bool:
    """Run sample queries and print whether top hit matches expected defect class."""
    all_passed = True
    print(f"ChromaDB sanity check — collection '{COLLECTION_NAME}'\n")

    for sample in SAMPLE_QUERIES:
        print(f"Query: {sample['label']}")
        print(f"  Text: {sample['query'][:80]}...")
        hits = query_cases(sample["query"], chroma_dir=chroma_dir, n_results=n_results)

        for rank, hit in enumerate(hits, start=1):
            print(
                f"  #{rank} {hit['case_id']} [{hit['defect_class']}] "
                f"dist={hit['distance']:.4f}"
            )
            print(f"       {hit['root_cause'][:90]}...")

        top_class = hits[0]["defect_class"] if hits else None
        ok = top_class == sample["expected_class"]
        status = "PASS" if ok else "FAIL"
        print(f"  Expected top class '{sample['expected_class']}' -> {status}\n")
        all_passed = all_passed and ok

    if all_passed:
        print("All sanity checks passed.")
    else:
        print("Some checks failed — review embeddings or case coverage.")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ChromaDB case retrieval")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--query", type=str, default=None, help="Custom query text")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.query:
        hits = query_cases(args.query, chroma_dir=args.chroma_dir, n_results=args.top_k)
        for hit in hits:
            print(hit)
    else:
        ok = run_sanity_checks(chroma_dir=args.chroma_dir, n_results=args.top_k)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
