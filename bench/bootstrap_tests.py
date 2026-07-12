"""
Paired bootstrap tests for Graph vs Classic RAG on three metrics.

Reads data/analysis/per_item_long.csv and outputs
data/analysis/bootstrap_graph_vs_classic.json with one record per
(generator, k, metric) cell containing mean delta, 95% CI, and
two-sided p-value.

The test is a paired bootstrap: for each benchmark item, compute the
judge-averaged metric under classic RAG and graph RAG, take the
per-item difference, resample the difference vector B times, and
summarize.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

METRICS = ("grounding_ratio", "factual_correctness_ratio", "hallucination_rate")
GENERATORS = ("openai", "anthropic", "deepseek")
K_VALUES = (1, 3, 5)


def _parse_pipeline(pipeline: str) -> Tuple[str, str, int]:
    """Parse 'rag_classic:openai:k=3' -> ('rag_classic', 'openai', 3)."""
    parts = pipeline.split(":")
    base = parts[0]
    gen = parts[1] if len(parts) > 1 else ""
    k_match = re.search(r"k=(\d+)", pipeline)
    k = int(k_match.group(1)) if k_match else 0
    return base, gen, k


def _item_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average each metric across judges for every (item_id, pipeline)."""
    return (
        df.groupby(["item_id", "pipeline"], as_index=False)[list(METRICS)]
        .mean()
    )


def _paired_deltas(
    items: pd.DataFrame, classic_mode: str, graph_mode: str, metric: str
) -> np.ndarray:
    """Return the per-item delta vector (graph - classic) for items present in both."""
    c = items[items["pipeline"] == classic_mode][["item_id", metric]].rename(
        columns={metric: "classic"}
    )
    g = items[items["pipeline"] == graph_mode][["item_id", metric]].rename(
        columns={metric: "graph"}
    )
    merged = c.merge(g, on="item_id", how="inner").dropna(
        subset=["classic", "graph"]
    )
    return (merged["graph"] - merged["classic"]).to_numpy()


def bootstrap_ci(
    deltas: np.ndarray, n_boot: int = 10_000, seed: int = 42
) -> Dict[str, float]:
    """Return mean, 95% CI, and two-sided p-value under H0: delta=0."""
    if len(deltas) == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n": 0,
        }
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = deltas[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    frac_le_0 = float((boots <= 0).mean())
    frac_ge_0 = float((boots >= 0).mean())
    p_value = float(min(1.0, 2 * min(frac_le_0, frac_ge_0)))
    return {
        "delta": float(deltas.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n": int(n),
    }


def run_tests(per_item_long: str, out_path: str, n_boot: int = 10_000) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]
    items = _item_means(df)

    results: Dict[str, Dict[str, float]] = {}
    for gen in GENERATORS:
        for k in K_VALUES:
            classic_mode = f"rag_classic:{gen}:k={k}"
            graph_mode = f"rag_graph:{gen}:k={k}:mn=10"
            if not (items["pipeline"] == classic_mode).any():
                continue
            if not (items["pipeline"] == graph_mode).any():
                continue
            for metric in METRICS:
                deltas = _paired_deltas(items, classic_mode, graph_mode, metric)
                key = f"{gen}|k={k}|{metric}"
                results[key] = bootstrap_ci(deltas, n_boot=n_boot)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Paired bootstrap (B={n_boot}) ===")
    print(
        f"{'cell':<50} {'delta':>8} {'ci_low':>8} {'ci_high':>8} "
        f"{'p':>8} {'n':>5}"
    )
    for key, r in results.items():
        star = " *" if r["p_value"] < 0.05 else "  "
        print(
            f"{key:<50} "
            f"{r['delta']:>+8.3f} {r['ci_low']:>+8.3f} {r['ci_high']:>+8.3f} "
            f"{r['p_value']:>8.3f}{star} {r['n']:>5}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_item_long", default="data/analysis/per_item_long.csv")
    ap.add_argument(
        "--out", default="data/analysis/bootstrap_graph_vs_classic.json"
    )
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    run_tests(args.per_item_long, args.out, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
