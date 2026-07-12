"""
Qualitative error analysis: identify cases where GraphRAG beats classic RAG
and vice versa. Outputs structured case studies for the paper.

Usage:
  python -m bench.error_analysis
  python -m bench.error_analysis --judgements data/benchmark_judgements.jsonl --results data/benchmark_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_base_mode(mode: str) -> str:
    """Extract base pipeline name (e.g., 'rag_classic' from 'rag_classic:openai:k=3')."""
    return mode.split(":")[0]


def load_judgements_by_item(path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Returns: {item_id: {base_mode: [judgement_records]}}
    """
    items: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bench = rec.get("benchmark_item", {})
            ans = rec.get("answer_result", {})

            item_id = bench.get("id") or ans.get("question_id", "")
            mode = ans.get("mode", "")
            base_mode = _extract_base_mode(mode)

            items[item_id][base_mode].append(rec)

    return dict(items)


def load_results_by_item(path: str) -> Dict[str, Dict[str, Dict]]:
    """
    Returns: {item_id: {mode_string: answer_result}}
    """
    items: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bench = rec.get("benchmark_item", {})
            ans = rec.get("answer_result", {})

            item_id = bench.get("id") or ans.get("question_id", "")
            mode = ans.get("mode", "")
            items[item_id][mode] = ans

    return dict(items)


def compute_deltas(
    judgements: Dict[str, Dict[str, List[Dict]]],
    metric_key: str = "factual_correctness_ratio",
    mode_a: str = "rag_graph",
    mode_b: str = "rag_classic",
) -> List[Dict[str, Any]]:
    """
    For each item, compute the mean score delta: mode_a - mode_b.
    Returns sorted list (highest delta first = mode_a wins most).
    """
    deltas = []

    for item_id, modes in judgements.items():
        recs_a = modes.get(mode_a, [])
        recs_b = modes.get(mode_b, [])

        if not recs_a or not recs_b:
            continue

        def _mean_score(recs: List[Dict], key: str) -> Optional[float]:
            scores_factual = [r.get("scores_factual", {}) for r in recs]
            scores_ground = [r.get("scores_ground", {}) for r in recs]
            vals = []
            for sf, sg in zip(scores_factual, scores_ground):
                v = sf.get(key) if sf.get(key) is not None else sg.get(key)
                if v is not None:
                    vals.append(float(v))
            return sum(vals) / len(vals) if vals else None

        score_a = _mean_score(recs_a, metric_key)
        score_b = _mean_score(recs_b, metric_key)

        if score_a is None or score_b is None:
            continue

        bench = recs_a[0].get("benchmark_item", {})
        deltas.append({
            "item_id": item_id,
            "question": bench.get("question", ""),
            "difficulty": bench.get("difficulty", ""),
            "language": bench.get("language", ""),
            f"{mode_a}_{metric_key}": round(score_a, 3),
            f"{mode_b}_{metric_key}": round(score_b, 3),
            "delta": round(score_a - score_b, 3),
        })

    deltas.sort(key=lambda x: x["delta"], reverse=True)
    return deltas


def build_case_studies(
    deltas: List[Dict],
    results: Dict[str, Dict[str, Dict]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Build detailed case study records for top wins and losses."""
    cases = []

    # Top N where GraphRAG wins
    graph_wins = deltas[:top_n]
    # Top N where classic RAG wins
    classic_wins = deltas[-top_n:][::-1] if deltas else []

    for entry in graph_wins + classic_wins:
        item_id = entry["item_id"]
        item_results = results.get(item_id, {})

        # Find the actual answers for each mode
        graph_answer = ""
        classic_answer = ""
        graph_docs = []
        classic_docs = []

        for mode_str, ans in item_results.items():
            base = _extract_base_mode(mode_str)
            if base == "rag_graph" and not graph_answer:
                graph_answer = ans.get("answer", "")
                graph_docs = ans.get("retrieved_docs", [])
            elif base == "rag_classic" and not classic_answer:
                classic_answer = ans.get("answer", "")
                classic_docs = ans.get("retrieved_docs", [])

        case = {
            **entry,
            "graph_answer": graph_answer[:500],
            "classic_answer": classic_answer[:500],
            "graph_retrieved_sources": [d.get("source", "") for d in graph_docs[:5]],
            "classic_retrieved_sources": [d.get("source", "") for d in classic_docs[:5]],
            "winner": "graph_rag" if entry["delta"] > 0 else "rag_classic",
        }
        cases.append(case)

    return cases


def main():
    ap = argparse.ArgumentParser(description="Qualitative error analysis: GraphRAG vs classic RAG")
    ap.add_argument("--judgements", default="data/benchmark_judgements.jsonl")
    ap.add_argument("--results", default="data/benchmark_results.jsonl")
    ap.add_argument("--outdir", default="data/analysis")
    ap.add_argument("--top-n", type=int, default=5, help="Number of top wins/losses to surface")
    ap.add_argument("--metric", default="factual_correctness_ratio",
                    choices=["factual_correctness_ratio", "grounding_ratio", "hallucination_rate"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    judgements = load_judgements_by_item(args.judgements)
    print(f"Loaded judgements for {len(judgements)} items")

    deltas = compute_deltas(judgements, metric_key=args.metric)
    print(f"Computed deltas for {len(deltas)} items on {args.metric}")

    if not deltas:
        print("No comparable items found between rag_graph and rag_classic.")
        return

    # Print summary
    print(f"\n=== Top {args.top_n} items where GraphRAG wins ===")
    for d in deltas[:args.top_n]:
        print(f"  {d['item_id'][:50]}  delta={d['delta']:+.3f}  "
              f"graph={d[f'rag_graph_{args.metric}']:.3f}  "
              f"classic={d[f'rag_classic_{args.metric}']:.3f}")

    print(f"\n=== Top {args.top_n} items where Classic RAG wins ===")
    for d in deltas[-args.top_n:]:
        print(f"  {d['item_id'][:50]}  delta={d['delta']:+.3f}  "
              f"graph={d[f'rag_graph_{args.metric}']:.3f}  "
              f"classic={d[f'rag_classic_{args.metric}']:.3f}")

    # Build case studies with full answers
    results_path = Path(args.results)
    if results_path.exists():
        results = load_results_by_item(args.results)
        cases = build_case_studies(deltas, results, top_n=args.top_n)

        out_path = outdir / "error_analysis.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Wrote {len(cases)} case studies to {out_path}")
    else:
        print(f"\n[WARN] Results file not found: {results_path} — skipping case study generation")

    # Also save full deltas
    deltas_path = outdir / "graph_vs_classic_deltas.json"
    with deltas_path.open("w", encoding="utf-8") as f:
        json.dump(deltas, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote all deltas to {deltas_path}")


if __name__ == "__main__":
    main()
