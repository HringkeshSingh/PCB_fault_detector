"""Embed historical cases and ingest into ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.data.schemas import HistoricalCaseCollection, HistoricalDefectCase
from src.vision.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_CASES_JSON = PROJECT_ROOT / "data" / "cases" / "historical_cases.json"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pcb_defect_cases"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_cases(json_path: Path) -> HistoricalCaseCollection:
    json_path = json_path.resolve()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Cases file not found: {json_path}. Run generate_cases.py first."
        )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return HistoricalCaseCollection.model_validate(data)


def case_to_metadata(case: HistoricalDefectCase) -> dict[str, Any]:
    """Chroma metadata must be str/int/float/bool — flatten dates to ISO strings."""
    return {
        "case_id": case.case_id,
        "defect_class": case.defect_class,
        "component_type": case.component_type,
        "root_cause": case.root_cause,
        "corrective_action": case.corrective_action,
        "severity": case.severity,
        "date_recorded": case.date_recorded.isoformat(),
        "outcome_notes": case.outcome_notes,
    }


def embedding_text(collection: HistoricalCaseCollection, case: HistoricalDefectCase) -> str:
    return collection.to_embedding_text(case)


def ingest_cases(
    cases_json: Path = DEFAULT_CASES_JSON,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
    reset: bool = True,
) -> int:
    """
    Load cases, embed combined text fields, store in ChromaDB.

    Returns the number of cases ingested.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    collection_data = load_cases(cases_json)
    cases = collection_data.cases
    if not cases:
        raise ValueError("No cases found in JSON file")

    texts = [embedding_text(collection_data, c) for c in cases]
    ids = [c.case_id for c in cases]
    metadatas = [case_to_metadata(c) for c in cases]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    chroma_dir = chroma_dir.resolve()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception as exc:
            # Collection not existing on first run is expected — anything else is re-raised.
            exc_str = str(exc).lower()
            if "does not exist" in exc_str or "not found" in exc_str or "no collection" in exc_str:
                logger.debug("Collection '%s' did not exist; nothing to delete", collection_name)
            else:
                logger.error("Unexpected error deleting collection '%s': %s", collection_name, exc)
                raise

    chroma_collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    chroma_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    # Verify ingested count matches what we sent.
    stored_count = chroma_collection.count()
    if stored_count != len(cases):
        raise RuntimeError(
            f"Post-ingest count mismatch: ingested {len(cases)} cases "
            f"but collection reports {stored_count}. Check for duplicate IDs."
        )

    print(
        f"Ingested {len(cases)} cases into '{collection_name}' at {chroma_dir} "
        f"(model: {model_name})"
    )
    return len(cases)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest historical cases into ChromaDB")
    parser.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete existing collection before ingest",
    )
    args = parser.parse_args()

    ingest_cases(
        cases_json=args.cases_json,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model_name=args.model,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
