import json
from pathlib import Path
from typing import List, Dict, Any
import statistics


def precision_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in gold_ids)
    return hits / len(top_k)


def hit_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    if not retrieved_ids or not gold_ids:
        return 0.0
    return 1.0 if any(g in retrieved_ids[:k] for g in gold_ids) else 0.0


def compute_retrieval_metrics(
    results_path: str = "data/benchmark_results.jsonl",
    k_values: List[int] | None = None,
) -> Dict[str, Any]:
    if k_values is None:
        k_values = [1, 3, 5]

    per_mode_stats: Dict[str, Dict[str, List[float]]] = {}

    in_file = Path(results_path)
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            bench = record["benchmark_item"]
            ans = record["answer_result"]

            mode = ans["mode"]
            gold_ids: List[str] = bench.get("reference_doc_ids") or []
            retrieved_docs: List[Dict[str, Any]] = ans.get("retrieved_docs") or []

            retrieved_ids = [d.get("chunk_id") for d in retrieved_docs if d.get("chunk_id")]

            if mode not in per_mode_stats:
                per_mode_stats[mode] = {
                    f"P@{k}": [] for k in k_values
                } | {
                    f"Hit@{k}": [] for k in k_values
                }

            for k in k_values:
                p = precision_at_k(retrieved_ids, gold_ids, k)
                h = hit_at_k(retrieved_ids, gold_ids, k)
                per_mode_stats[mode][f"P@{k}"].append(p)
                per_mode_stats[mode][f"Hit@{k}"].append(h)

    summary: Dict[str, Any] = {}
    for mode, metrics in per_mode_stats.items():
        summary[mode] = {}
        for metric_name, vals in metrics.items():
            if vals:
                summary[mode][metric_name] = {
                    "mean": statistics.mean(vals),
                    "stdev": statistics.pstdev(vals),
                    "n": len(vals),
                }
            else:
                summary[mode][metric_name] = {"mean": 0.0, "stdev": 0.0, "n": 0}

    return summary


def _percentile(vals: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100)."""
    sorted_vals = sorted(vals)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def compute_latency_stats(results_path: str = "data/benchmark_results.jsonl") -> Dict[str, Any]:
    per_mode_latencies: Dict[str, List[float]] = {}

    in_file = Path(results_path)
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ans = record["answer_result"]
            mode = ans["mode"]
            latency = ans["latency_sec"]
            per_mode_latencies.setdefault(mode, []).append(latency)

    summary: Dict[str, Any] = {}
    for mode, vals in per_mode_latencies.items():
        summary[mode] = {
            "mean_latency": statistics.mean(vals),
            "stdev_latency": statistics.pstdev(vals),
            "p50_latency": _percentile(vals, 50),
            "p95_latency": _percentile(vals, 95),
            "n": len(vals),
        }
    return summary


def compute_token_usage_stats(results_path: str = "data/benchmark_results.jsonl") -> Dict[str, Any]:
    """Compute mean token usage per mode."""
    per_mode: Dict[str, Dict[str, List[int]]] = {}

    in_file = Path(results_path)
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ans = record["answer_result"]
            mode = ans["mode"]
            usage = ans.get("usage")

            if not usage:
                continue

            if mode not in per_mode:
                per_mode[mode] = {"prompt_tokens": [], "completion_tokens": [], "total_tokens": []}

            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = usage.get(key)
                if val is not None:
                    per_mode[mode][key].append(int(val))

    summary: Dict[str, Any] = {}
    for mode, token_lists in per_mode.items():
        summary[mode] = {}
        for key, vals in token_lists.items():
            if vals:
                summary[mode][f"mean_{key}"] = statistics.mean(vals)
                summary[mode][f"stdev_{key}"] = statistics.pstdev(vals)
            else:
                summary[mode][f"mean_{key}"] = 0.0
                summary[mode][f"stdev_{key}"] = 0.0
        summary[mode]["n"] = len(token_lists.get("total_tokens", []))

    return summary


# Approximate per-token pricing (USD per 1M tokens)
PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "anthropic/claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
}


def _normalize_model_key(model_name: str) -> str:
    """Strip known wrapper prefixes (e.g. 'dspy/') from a model name to
    match keys in the PRICING table. DSPy records model_name as
    'dspy/openai/gpt-4o'; we strip the wrapper so the underlying model
    is priced correctly."""
    if not model_name:
        return model_name
    for prefix in ("dspy/",):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def compute_cost_accuracy_table(
    results_path: str = "data/benchmark_results.jsonl",
    judgements_path: str = "data/benchmark_judgements.jsonl",
) -> List[Dict[str, Any]]:
    """
    Build a cost-accuracy tradeoff table: for each mode, compute estimated
    cost, mean latency, and mean factual_correctness_ratio.
    """
    # Collect per-mode: latency, tokens, model_name
    mode_data: Dict[str, Dict[str, list]] = {}

    in_file = Path(results_path)
    if in_file.exists():
        with in_file.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ans = rec["answer_result"]
                mode = ans["mode"]
                if mode not in mode_data:
                    mode_data[mode] = {
                        "latencies": [], "prompt_tokens": [],
                        "completion_tokens": [], "model_name": [],
                    }
                mode_data[mode]["latencies"].append(ans.get("latency_sec", 0))
                mode_data[mode]["model_name"].append(ans.get("model_name", ""))
                usage = ans.get("usage") or {}
                mode_data[mode]["prompt_tokens"].append(usage.get("prompt_tokens") or 0)
                mode_data[mode]["completion_tokens"].append(usage.get("completion_tokens") or 0)

    # Collect per-mode factual scores from judgements
    mode_factual: Dict[str, List[float]] = {}
    jpath = Path(judgements_path)
    if jpath.exists():
        with jpath.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ans = rec.get("answer_result", {})
                mode = ans.get("mode", "")
                sf = rec.get("scores_factual", {}) or {}
                val = sf.get("factual_correctness_ratio")
                if val is not None:
                    mode_factual.setdefault(mode, []).append(float(val))

    rows = []
    for mode, data in mode_data.items():
        model_name = data["model_name"][0] if data["model_name"] else ""
        pricing_key = _normalize_model_key(model_name)
        pricing = PRICING.get(pricing_key, {"input": 0, "output": 0})

        mean_prompt = statistics.mean(data["prompt_tokens"]) if data["prompt_tokens"] else 0
        mean_completion = statistics.mean(data["completion_tokens"]) if data["completion_tokens"] else 0

        est_cost_per_call = (
            mean_prompt * pricing["input"] / 1_000_000
            + mean_completion * pricing["output"] / 1_000_000
        )

        factual_scores = mode_factual.get(mode, [])

        rows.append({
            "mode": mode,
            "model": model_name,
            "pricing_model": pricing_key,
            "mean_latency_sec": round(statistics.mean(data["latencies"]), 3) if data["latencies"] else 0,
            "mean_prompt_tokens": round(mean_prompt),
            "mean_completion_tokens": round(mean_completion),
            "est_cost_per_call_usd": round(est_cost_per_call, 6),
            "factual_correctness_ratio": round(statistics.mean(factual_scores), 3) if factual_scores else None,
            "n_results": len(data["latencies"]),
            "n_judgements": len(factual_scores),
        })

    rows.sort(key=lambda r: r.get("factual_correctness_ratio") or 0, reverse=True)
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute benchmark metrics")
    ap.add_argument("--results", default="data/benchmark_results.jsonl")
    ap.add_argument("--judgements", default="data/benchmark_judgements.jsonl")
    ap.add_argument("--outdir", default="data/analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Retrieval Metrics ===")
    retrieval = compute_retrieval_metrics(args.results)
    for mode, metrics in retrieval.items():
        print(f"\n  {mode}:")
        for k, v in metrics.items():
            print(f"    {k}: mean={v['mean']:.3f} stdev={v['stdev']:.3f} n={v['n']}")

    print("\n=== Latency Stats ===")
    latency = compute_latency_stats(args.results)
    for mode, stats in latency.items():
        print(f"  {mode}: mean={stats['mean_latency']:.3f}s "
              f"p50={stats['p50_latency']:.3f}s p95={stats['p95_latency']:.3f}s")

    print("\n=== Token Usage ===")
    tokens = compute_token_usage_stats(args.results)
    for mode, stats in tokens.items():
        print(f"  {mode}: prompt={stats['mean_prompt_tokens']:.0f} "
              f"completion={stats['mean_completion_tokens']:.0f} "
              f"total={stats['mean_total_tokens']:.0f}")

    print("\n=== Cost-Accuracy Tradeoff ===")
    cost_table = compute_cost_accuracy_table(args.results, args.judgements)
    for row in cost_table:
        print(f"  {row['mode']}: cost=${row['est_cost_per_call_usd']:.5f}/call "
              f"latency={row['mean_latency_sec']:.2f}s "
              f"factual={row['factual_correctness_ratio']}")

    # Save to JSON
    all_metrics = {
        "retrieval": retrieval,
        "latency": latency,
        "token_usage": tokens,
        "cost_accuracy": cost_table,
    }
    out_path = outdir / "metrics_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] Wrote metrics to {out_path}")
