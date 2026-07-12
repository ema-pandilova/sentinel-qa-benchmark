"""
Populate reference_doc_ids in benchmark.jsonl by matching each item's
source PDF to its chunk IDs in the Chroma database.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import load_db


def main(
    benchmark_path: str = "data/benchmark.jsonl",
    output_path: str | None = None,
):
    output_path = output_path or benchmark_path

    # Build source -> chunk_ids index from Chroma
    db = load_db()
    all_data = db.get(include=["metadatas"])

    source_to_chunk_ids: dict[str, list[str]] = defaultdict(list)
    for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        source = meta.get("source") or meta.get("doc_id")
        if source:
            source_to_chunk_ids[source].append(chunk_id)

    print(f"Indexed {len(source_to_chunk_ids)} unique sources from Chroma "
          f"({sum(len(v) for v in source_to_chunk_ids.values())} total chunks)")

    # Read benchmark, populate reference_doc_ids
    in_path = Path(benchmark_path)
    items = []
    matched = 0
    unmatched = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            source = item.get("source", "")

            chunk_ids = source_to_chunk_ids.get(source, [])
            if chunk_ids:
                item["reference_doc_ids"] = chunk_ids
                matched += 1
            else:
                item["reference_doc_ids"] = []
                unmatched.append(source)

            items.append(item)

    # Write back
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Matched {matched}/{len(items)} items to Chroma chunks")
    if unmatched:
        print(f"WARNING: {len(unmatched)} items had no matching chunks:")
        for src in unmatched[:10]:
            print(f"  - {src}")


if __name__ == "__main__":
    main()
