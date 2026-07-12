"""
Benchmark runner supporting multi-model, k-ablation, and graph-ablation experiments.

Usage:
  # Run main multi-model experiment (3 generators × 4 modes × k={1,3,5})
  python -m bench.run_benchmark --experiment multi-model

  # Run graph ablation (GPT-4o × rag_graph × k=3 × 9 graph configs)
  python -m bench.run_benchmark --experiment graph-ablation

  # Dry run (2 items only, for verification)
  python -m bench.run_benchmark --experiment multi-model --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

from bench.schema import BenchmarkItem
from rag.pipelines import (
    answer_zeroshot,
    answer_rag_classic,
    answer_rag_dspy,
    answer_graph_rag,
)

MODES = ["zeroshot", "rag_classic", "rag_dspy", "rag_graph"]

K_VALUES = [1, 3, 5]

GENERATOR_MODELS = [
    {"provider": "openai", "model_name": "gpt-4o"},
    {"provider": "anthropic", "model_name": "claude-sonnet-4-20250514"},
    {"provider": "deepseek", "model_name": "deepseek-chat"},
]

# Default graph config for multi-model experiment
DEFAULT_GRAPH_PATH = "rag/graph/graph_t0.8.pkl"
DEFAULT_MAX_NODES = 10

# Graph ablation: threshold × max_nodes
GRAPH_THRESHOLDS = [0.7, 0.8, 0.9]
GRAPH_MAX_NODES = [5, 10, 20]


def load_benchmark(path: Path) -> List[BenchmarkItem]:
    items: List[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if "doc_id" in data:
                if "reference_doc_ids" not in data or data["reference_doc_ids"] is None:
                    data["reference_doc_ids"] = [] if data["doc_id"] is None else [data["doc_id"]]
                del data["doc_id"]

            known_keys = {f.name for f in BenchmarkItem.__dataclass_fields__.values()}
            extra_keys = set(data.keys()) - known_keys
            extra = {k: data.pop(k) for k in extra_keys}

            item = BenchmarkItem(**data, extra=extra)
            items.append(item)
    return items


def _write_record(f_out, item: BenchmarkItem, result) -> None:
    bench_dict = asdict(item)
    # Flatten extra fields to top level for downstream compatibility
    extra = bench_dict.pop("extra", None) or {}
    bench_dict.update(extra)
    record: Dict[str, Any] = {
        "benchmark_item": bench_dict,
        "answer_result": result.to_dict(),
    }
    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_item_mode(
    item: BenchmarkItem,
    mode: str,
    k: int,
    provider: str,
    model_name: str,
    graph_path: str = DEFAULT_GRAPH_PATH,
    max_nodes: int = DEFAULT_MAX_NODES,
):
    if mode == "zeroshot":
        return answer_zeroshot(item.id, item.question, provider=provider, model_name=model_name)
    elif mode == "rag_classic":
        return answer_rag_classic(item.id, item.question, k=k, provider=provider, model_name=model_name)
    elif mode == "rag_dspy":
        return answer_rag_dspy(item.id, item.question, k=k)
    elif mode == "rag_graph":
        return answer_graph_rag(
            item.id, item.question, k=k,
            provider=provider, model_name=model_name,
            graph_path=graph_path, max_nodes=max_nodes,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _load_done_keys(path: Path) -> set:
    """Load already-completed (item_id, mode) pairs from an existing results file."""
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                item_id = rec["benchmark_item"]["id"]
                mode = rec["answer_result"]["mode"]
                done.add((item_id, mode))
            except Exception:
                continue
    return done


def run_multi_model(
    items: List[BenchmarkItem],
    output_path: str = "data/benchmark_results.jsonl",
    dry_run: bool = False,
):
    """Multi-model experiment: 3 generators × 4 modes × k={1,3,5}."""
    if dry_run:
        items = items[:2]

    out_file = Path(output_path)

    # Resume support: skip already-completed combos
    done_keys = _load_done_keys(out_file)
    if done_keys:
        print(f"  Resuming: {len(done_keys)} results already exist, skipping them")

    total = 0
    errors = 0
    skipped = 0

    with out_file.open("a", encoding="utf-8") as f_out:
        for gen_cfg in GENERATOR_MODELS:
            provider = gen_cfg["provider"]
            model_name = gen_cfg["model_name"]

            for item in items:
                for mode in MODES:
                    # DSPy only supports OpenAI
                    if mode == "rag_dspy" and provider != "openai":
                        continue

                    k_list = [1] if mode == "zeroshot" else K_VALUES
                    for k in k_list:
                        # Build expected mode string to check if already done
                        if mode == "zeroshot":
                            expected_mode = f"zeroshot:{provider}"
                        elif mode == "rag_dspy":
                            expected_mode = f"rag_dspy:openai:k={k}"
                        elif mode == "rag_graph":
                            expected_mode = f"rag_graph:{provider}:k={k}:mn={DEFAULT_MAX_NODES}"
                        else:
                            expected_mode = f"{mode}:{provider}:k={k}"

                        if (item.id, expected_mode) in done_keys:
                            skipped += 1
                            continue

                        try:
                            result = _run_item_mode(
                                item, mode, k=k,
                                provider=provider, model_name=model_name,
                            )
                            _write_record(f_out, item, result)
                            total += 1
                            label = mode if mode == "zeroshot" else f"{mode}:k={k}"
                            print(f"  [{total}] {label} | {provider}/{model_name} | {item.id[:40]}")
                        except Exception as e:
                            errors += 1
                            print(f"  [ERROR] {mode}:k={k} | {provider}/{model_name} | {item.id[:40]} | {e}")

    print(f"\n[OK] Multi-model experiment: {total} new + {skipped} skipped = {total + skipped} total ({errors} errors)")


def run_graph_ablation(
    items: List[BenchmarkItem],
    output_path: str = "data/benchmark_results_graph_ablation.jsonl",
    dry_run: bool = False,
):
    """Graph ablation: GPT-4o × rag_graph × k=3 × 9 graph configs."""
    if dry_run:
        items = items[:2]

    out_file = Path(output_path)
    done_keys = _load_done_keys(out_file)
    if done_keys:
        print(f"  Resuming: {len(done_keys)} results already exist, skipping them")

    total = 0
    errors = 0
    skipped = 0
    k = 3  # Fixed k for graph ablation

    with out_file.open("a", encoding="utf-8") as f_out:
        for threshold in GRAPH_THRESHOLDS:
            graph_path = f"rag/graph/graph_t{threshold}.pkl"

            for max_nodes in GRAPH_MAX_NODES:
                for item in items:
                    expected_mode = f"rag_graph:openai:k={k}:mn={max_nodes}"
                    if (item.id, expected_mode) in done_keys:
                        skipped += 1
                        continue

                    try:
                        result = answer_graph_rag(
                            item.id, item.question, k=k,
                            provider="openai", model_name="gpt-4o",
                            graph_path=graph_path, max_nodes=max_nodes,
                        )
                        _write_record(f_out, item, result)
                        total += 1
                        print(f"  [{total}] graph t={threshold} mn={max_nodes} | {item.id[:40]}")
                    except Exception as e:
                        errors += 1
                        print(f"  [ERROR] graph t={threshold} mn={max_nodes} | {item.id[:40]} | {e}")

    print(f"\n[OK] Graph ablation: {total} new + {skipped} skipped = {total + skipped} total ({errors} errors)")


def build_all_graphs():
    """Pre-build graphs for all thresholds."""
    from rag.graph.build_graph import build_graph

    for threshold in GRAPH_THRESHOLDS:
        graph_path = f"rag/graph/graph_t{threshold}.pkl"
        print(f"\nBuilding graph: threshold={threshold} -> {graph_path}")
        build_graph(similarity_threshold=threshold, graph_path=graph_path)


def main():
    ap = argparse.ArgumentParser(description="Run RAG benchmark experiments")
    ap.add_argument(
        "--experiment",
        choices=["multi-model", "graph-ablation", "build-graphs"],
        required=True,
        help="Which experiment to run",
    )
    ap.add_argument(
        "--benchmark-path",
        default="data/benchmark.jsonl",
        help="Path to benchmark JSONL",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output path (defaults depend on experiment)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on first 2 items only for verification",
    )
    args = ap.parse_args()

    if args.experiment == "build-graphs":
        build_all_graphs()
        return

    items = load_benchmark(Path(args.benchmark_path))
    print(f"Loaded {len(items)} benchmark items")

    if args.experiment == "multi-model":
        output = args.output or "data/benchmark_results.jsonl"
        run_multi_model(items, output_path=output, dry_run=args.dry_run)

    elif args.experiment == "graph-ablation":
        output = args.output or "data/benchmark_results_graph_ablation.jsonl"
        run_graph_ablation(items, output_path=output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
