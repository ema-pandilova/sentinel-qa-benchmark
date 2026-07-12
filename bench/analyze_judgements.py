from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_IN = "data/benchmark_judgements.jsonl"
DEFAULT_OUTDIR = "data/analysis"


METRICS = [
    "factual_correctness_ratio",
    "grounding_ratio",
    "hallucination_rate",
]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _get_pipeline_name(answer_result: Dict[str, Any]) -> str:
    """
    Try to infer pipeline name from your AnswerResult.
    Examples:
      mode = "rag_classic:openai"
      mode = "zeroshot:deepseek"
      mode = "rag_dspy:openai"
    """
    mode = answer_result.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    # fallback
    model_name = answer_result.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return "unknown_pipeline"


def _get_item_id(bench: Dict[str, Any], ans: Dict[str, Any]) -> str:
    """
    Normalize a stable item id.
    Prefer benchmark_item.id (if present), else answer_result.question_id.
    """
    if isinstance(bench.get("id"), str) and bench["id"].strip():
        return bench["id"].strip()
    if isinstance(ans.get("question_id"), str) and ans["question_id"].strip():
        return ans["question_id"].strip()
    # last resort: hash-ish from question
    q = bench.get("question") or ans.get("question") or ""
    return f"q_{abs(hash(q))}"


def load_judgements(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Judgements file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {line_no}: JSON parse failed: {e}")
                continue

            bench = rec.get("benchmark_item", {}) or {}
            ans = rec.get("answer_result", {}) or {}
            scores_ground = rec.get("scores_ground", {}) or {}
            scores_factual = rec.get("scores_factual", {}) or {}
            scores = {**scores_ground, **scores_factual}

            item_id = _get_item_id(bench, ans)
            pipeline = _get_pipeline_name(ans)

            row: Dict[str, Any] = {
                "item_id": item_id,
                "pipeline": pipeline,
                "judge_model": rec.get("judge_model"),
                "judge_provider": rec.get("judge_provider"),
                "difficulty": bench.get("difficulty"),
                "language": bench.get("language"),
                "audience": bench.get("audience"),
                "source": bench.get("source") or bench.get("base_source") or ans.get("retrieved_docs", [{}])[0].get("source"),
            }

            for m in METRICS:
                row[m] = _safe_float(scores.get(m))

            row["comments_ground"] = scores_ground.get("comments")
            row["comments_factual"] = scores_factual.get("comments")
            row["judge_error"] = any(
                isinstance(c, str) and c.startswith("JUDGE_ERROR:")
                for c in [row["comments_ground"], row["comments_factual"]]
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    # keep only numeric metrics as floats
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


def summarize(df: pd.DataFrame, hallucination_high_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (summary_by_pipeline, summary_by_pipeline_and_judge)
    """

    # Drop judge errors for metric aggregation
    df_ok = df[~df["judge_error"]].copy()

    # hallucination_high_rate: fraction of items with hallucination_rate >= threshold
    df_ok["hallucination_high"] = df_ok["hallucination_rate"].apply(
        lambda x: float(x >= hallucination_high_threshold) if pd.notna(x) else None
    )

    # Summary per pipeline (averaging over judges)
    grp_p = df_ok.groupby(["pipeline"], dropna=False)

    summary_p = grp_p.agg(
        n=("item_id", "count"),
        factual_correctness_ratio_mean=("factual_correctness_ratio", "mean"),
        grounding_ratio_mean=("grounding_ratio", "mean"),
        hallucination_rate_mean=("hallucination_rate", "mean"),
        hallucination_high_rate=("hallucination_high", "mean"),
        factual_correctness_ratio_std=("factual_correctness_ratio", "std"),
        grounding_ratio_std=("grounding_ratio", "std"),
        hallucination_rate_std=("hallucination_rate", "std"),
    ).reset_index()

    # Summary per pipeline x judge
    grp_pj = df_ok.groupby(["pipeline", "judge_provider", "judge_model"], dropna=False)
    summary_pj = grp_pj.agg(
        n=("item_id", "count"),
        factual_correctness_ratio_mean=("factual_correctness_ratio", "mean"),
        grounding_ratio_mean=("grounding_ratio", "mean"),
        hallucination_rate_mean=("hallucination_rate", "mean"),
        hallucination_high_rate=("hallucination_high", "mean"),
    ).reset_index()

    return summary_p, summary_pj


def _extract_pipeline_facets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract k_value, generator, and base_mode from pipeline strings like 'rag_classic:openai:k=3'."""
    import re

    df = df.copy()
    df["base_mode"] = df["pipeline"].apply(lambda x: x.split(":")[0] if pd.notna(x) else x)
    df["generator"] = df["pipeline"].apply(
        lambda x: x.split(":")[1] if pd.notna(x) and len(x.split(":")) > 1 else None
    )
    df["k_value"] = df["pipeline"].apply(
        lambda x: int(m.group(1)) if pd.notna(x) and (m := re.search(r"k=(\d+)", x)) else None
    )
    return df


def summarize_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by pipeline × difficulty."""
    df_ok = df[~df["judge_error"]].copy()
    grp = df_ok.groupby(["pipeline", "difficulty"], dropna=False)
    return grp.agg(
        n=("item_id", "count"),
        factual_correctness_ratio_mean=("factual_correctness_ratio", "mean"),
        grounding_ratio_mean=("grounding_ratio", "mean"),
        hallucination_rate_mean=("hallucination_rate", "mean"),
    ).reset_index()


def summarize_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by base_mode × k_value (for k-ablation analysis)."""
    df_ext = _extract_pipeline_facets(df)
    df_ok = df_ext[~df_ext["judge_error"]].copy()
    df_ok = df_ok[df_ok["k_value"].notna()]

    if df_ok.empty:
        return pd.DataFrame()

    grp = df_ok.groupby(["base_mode", "generator", "k_value"], dropna=False)
    return grp.agg(
        n=("item_id", "count"),
        factual_correctness_ratio_mean=("factual_correctness_ratio", "mean"),
        grounding_ratio_mean=("grounding_ratio", "mean"),
        hallucination_rate_mean=("hallucination_rate", "mean"),
    ).reset_index()


def summarize_by_generator(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by base_mode × generator model."""
    df_ext = _extract_pipeline_facets(df)
    df_ok = df_ext[~df_ext["judge_error"]].copy()

    grp = df_ok.groupby(["base_mode", "generator"], dropna=False)
    return grp.agg(
        n=("item_id", "count"),
        factual_correctness_ratio_mean=("factual_correctness_ratio", "mean"),
        grounding_ratio_mean=("grounding_ratio", "mean"),
        hallucination_rate_mean=("hallucination_rate", "mean"),
    ).reset_index()


def judge_agreement_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlations between judges per pipeline+metric
    on common items. This is useful for "judge agreement" in the paper.
    """
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        print("[WARN] scipy not installed; skipping Spearman agreement. `pip install scipy` to enable.")
        return pd.DataFrame([])

    df_ok = df[~df["judge_error"]].copy()

    # Create a judge id column
    df_ok["judge_id"] = df_ok["judge_provider"].astype(str) + "/" + df_ok["judge_model"].astype(str)

    rows: List[Dict[str, Any]] = []
    for pipeline in sorted(df_ok["pipeline"].dropna().unique()):
        dpf = df_ok[df_ok["pipeline"] == pipeline]

        judges = sorted(dpf["judge_id"].dropna().unique())
        if len(judges) < 2:
            continue

        for metric in METRICS:
            # pivot: item_id x judge -> metric
            pv = dpf.pivot_table(index="item_id", columns="judge_id", values=metric, aggfunc="mean")
            # drop items with any missing in the pair when correlating
            for i in range(len(judges)):
                for j in range(i + 1, len(judges)):
                    ja, jb = judges[i], judges[j]
                    if ja not in pv.columns or jb not in pv.columns:
                        continue
                    sub = pv[[ja, jb]].dropna()
                    if len(sub) < 5:
                        continue
                    corr, p = spearmanr(sub[ja], sub[jb])
                    rows.append(
                        {
                            "pipeline": pipeline,
                            "metric": metric,
                            "judge_a": ja,
                            "judge_b": jb,
                            "n_common_items": len(sub),
                            "spearman_r": float(corr),
                            "p_value": float(p),
                        }
                    )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Path to benchmark_judgements.jsonl")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for CSVs")
    ap.add_argument(
        "--hallucination-high-threshold",
        type=float,
        default=0.3,
        help="Threshold to count 'high hallucination' rate (>= this score).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_judgements(args.in_path)
    if df.empty:
        raise RuntimeError("No judgement rows loaded. Check your meta_eval output file.")

    # Save long form for debugging
    df.to_csv(outdir / "per_item_long.csv", index=False)

    summary_p, summary_pj = summarize(df, hallucination_high_threshold=args.hallucination_high_threshold)
    summary_p.to_csv(outdir / "summary_by_pipeline.csv", index=False)
    summary_pj.to_csv(outdir / "summary_by_pipeline_and_judge.csv", index=False)

    agree = judge_agreement_spearman(df)
    if not agree.empty:
        agree.to_csv(outdir / "judge_agreement_spearman.csv", index=False)

    # Additional breakdowns
    by_diff = summarize_by_difficulty(df)
    by_diff.to_csv(outdir / "summary_by_difficulty.csv", index=False)

    by_k = summarize_by_k(df)
    if not by_k.empty:
        by_k.to_csv(outdir / "summary_by_k.csv", index=False)

    by_gen = summarize_by_generator(df)
    by_gen.to_csv(outdir / "summary_by_generator.csv", index=False)

    # Print summaries
    print("\n=== Loaded judgements ===")
    print(f"rows={len(df)}  pipelines={df['pipeline'].nunique()}  judges={df['judge_model'].nunique()}")

    print("\n=== Summary by pipeline (averaged over judges) ===")
    print(summary_p.sort_values(by="factual_correctness_ratio_mean", ascending=False).to_string(index=False))

    print("\n=== Summary by pipeline + judge ===")
    print(summary_pj.sort_values(by=["pipeline", "judge_provider", "judge_model"]).to_string(index=False))

    if not by_diff.empty:
        print("\n=== Summary by difficulty ===")
        print(by_diff.to_string(index=False))

    if not by_k.empty:
        print("\n=== Summary by k value ===")
        print(by_k.to_string(index=False))

    if not by_gen.empty:
        print("\n=== Summary by generator ===")
        print(by_gen.to_string(index=False))

    if not agree.empty:
        print("\n=== Judge agreement (Spearman) ===")
        print(
            agree.sort_values(by="spearman_r", ascending=False)
            .head(25)
            .to_string(index=False)
        )

    print(f"\n[OK] Wrote analysis CSVs to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
