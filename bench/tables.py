"""
LaTeX table generation for the 'Grounded != Correct' journal paper.

Reads:
    data/analysis/summary_by_pipeline.csv
    data/analysis/summary_by_difficulty.csv
    data/analysis/metrics_summary.json
    data/analysis/judge_agreement_spearman.csv
    data/analysis/bootstrap_graph_vs_classic.json
    data/benchmark.jsonl
    data/analysis/per_item_long.csv

Writes:
    paper/tables/tableN.tex  (N = 1..5 + S1, S2)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "--"
    try:
        if pd.isna(x):
            return "--"
    except (TypeError, ValueError):
        pass
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Table 1 - Benchmark statistics
# ---------------------------------------------------------------------------

def table1_benchmark_stats(benchmark_path: str, out: Path) -> None:
    langs: collections.Counter = collections.Counter()
    diffs: collections.Counter = collections.Counter()
    sources: set = set()
    n = 0
    ref_chunk_counts: list[int] = []

    with open(benchmark_path) as f:
        for line in f:
            o = json.loads(line)
            n += 1
            langs[o.get("language", "?")] += 1
            diffs[o.get("difficulty", "?")] += 1
            src = o.get("source")
            if src:
                sources.add(src)
            rd = o.get("reference_doc_ids") or []
            ref_chunk_counts.append(len(rd))

    mean_ref = sum(ref_chunk_counts) / len(ref_chunk_counts) if ref_chunk_counts else 0

    rows = [
        ("Total questions", str(n)),
        ("English questions", str(langs.get("en", 0))),
        ("Macedonian questions", str(langs.get("mk", 0))),
        ("Easy questions", str(diffs.get("easy", 0))),
        ("Medium questions", str(diffs.get("medium", 0))),
        ("Hard questions", str(diffs.get("hard", 0))),
        ("Unique source PDFs", str(len(sources))),
        # reference_doc_ids lists every chunk from the source doc (used for
        # loose Hit@K computation), so the mean reflects source-doc size
        (r"Mean chunks per source doc", f"{mean_ref:.1f}"),
    ]

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Property & Value \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 2 - Main results grid
# ---------------------------------------------------------------------------

def table2_main_results(summary_path: str, metrics_json: str, out: Path) -> None:
    df = pd.read_csv(summary_path)
    with open(metrics_json) as f:
        m = json.load(f)
    latency = m.get("latency", {})
    cost_rows = {r["mode"]: r for r in m.get("cost_accuracy", [])}

    def _order_key(pipeline: str):
        base_order = {"zeroshot": 0, "rag_classic": 1, "rag_dspy": 2, "rag_graph": 3}
        parts = pipeline.split(":")
        base = parts[0]
        gen = parts[1] if len(parts) > 1 else ""
        k = 0
        km = re.search(r"k=(\d+)", pipeline)
        if km:
            k = int(km.group(1))
        gen_order = {"openai": 0, "anthropic": 1, "deepseek": 2}.get(gen, 9)
        return (base_order.get(base, 99), gen_order, k)

    df = df.sort_values("pipeline", key=lambda s: s.map(_order_key))

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Pipeline & $n$ & Factual & Grounding & Halluc. & Lat. (s) & Cost (\$) \\",
        r"\midrule",
    ]
    current_family = None
    for _, row in df.iterrows():
        mode = row["pipeline"]
        base = mode.split(":")[0]
        if current_family and base != current_family:
            lines.append(r"\midrule")
        current_family = base

        lat_stats = latency.get(mode, {})
        lat_val = lat_stats.get("mean_latency")
        cost_row = cost_rows.get(mode, {})
        cost = cost_row.get("est_cost_per_call_usd")

        fact = f"{row['factual_correctness_ratio_mean']:.3f}"
        grnd = (
            f"{row['grounding_ratio_mean']:.3f}"
            if pd.notna(row.get("grounding_ratio_mean")) else "--"
        )
        hal = (
            f"{row['hallucination_rate_mean']:.3f}"
            if pd.notna(row.get("hallucination_rate_mean")) else "--"
        )
        lat_s = f"{lat_val:.2f}" if lat_val is not None else "--"
        cost_s = f"{cost:.4f}" if cost else "--"

        pipeline_tex = mode.replace("_", r"\_")
        lines.append(
            f"{pipeline_tex} & {int(row['n'])} & {fact} & {grnd} & {hal} "
            f"& {lat_s} & {cost_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 3 - Retrieval ceiling
# ---------------------------------------------------------------------------

def table3_retrieval_ceiling(metrics_json: str, out: Path) -> None:
    with open(metrics_json) as f:
        m = json.load(f)
    ret = m.get("retrieval", {})

    picks = [
        ("openai", "rag_classic:openai:k=5"),
        ("anthropic", "rag_classic:anthropic:k=5"),
        ("deepseek", "rag_classic:deepseek:k=5"),
    ]

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Generator & P@1 & P@3 & P@5 & Hit@1 & Hit@3 & Hit@5 \\",
        r"\midrule",
    ]
    for gen, mode in picks:
        r = ret.get(mode)
        if not r:
            continue
        lines.append(
            f"{gen} & "
            f"{r['P@1']['mean']:.3f} & {r['P@3']['mean']:.3f} & "
            f"{r['P@5']['mean']:.3f} & "
            f"{r['Hit@1']['mean']:.3f} & {r['Hit@3']['mean']:.3f} & "
            f"{r['Hit@5']['mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 4 - Judge agreement summary
# ---------------------------------------------------------------------------

def table4_judge_agreement(spearman_path: str, out: Path) -> None:
    df = pd.read_csv(spearman_path)

    def _short(x: str) -> str:
        s = x.split("/")[-1]
        return s.replace("claude-sonnet-4-20250514", "claude-4")

    df["judge_a_short"] = df["judge_a"].apply(_short)
    df["judge_b_short"] = df["judge_b"].apply(_short)

    agg = (
        df.groupby(["judge_a_short", "judge_b_short", "metric"])["spearman_r"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["judge_a_short", "judge_b_short"],
        columns="metric",
        values="spearman_r",
    ).reset_index()

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Judge A & Judge B & Grounding $\rho$ & Factual $\rho$ \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        a = row["judge_a_short"].replace("_", r"\_")
        b = row["judge_b_short"].replace("_", r"\_")
        gr = row.get("grounding_ratio")
        fr = row.get("factual_correctness_ratio")
        lines.append(f"{a} & {b} & {_fmt(gr)} & {_fmt(fr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 5 - Bootstrap tests on Graph vs Classic
# ---------------------------------------------------------------------------

def table5_bootstrap(bootstrap_json: str, out: Path) -> None:
    with open(bootstrap_json) as f:
        boot = json.load(f)

    # Hallucination-rate tests are omitted: hallucination rate = 1 - grounding,
    # so its paired difference is exactly -Delta_grounding (CI reflected about
    # zero, identical p and n). Reporting it alongside grounding is redundant.
    metric_display = {
        "grounding_ratio": "Grounding",
        "factual_correctness_ratio": "Factual",
    }
    metric_order = list(metric_display.keys())

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Generator & $k$ & Metric & $\Delta$ & 95\% CI & $p$ & $n$ \\",
        r"\midrule",
    ]

    def _key_sort(key: str):
        gen, kpart, metric = key.split("|")
        k = int(kpart.replace("k=", ""))
        gen_order = {"openai": 0, "anthropic": 1, "deepseek": 2}.get(gen, 9)
        met_order = metric_order.index(metric) if metric in metric_order else 9
        return (gen_order, k, met_order)

    current_gen = None
    for key in sorted(boot.keys(), key=_key_sort):
        gen, kpart, metric = key.split("|")
        if metric not in metric_display:
            continue  # skip hallucination-rate rows (redundant with grounding)
        r = boot[key]
        if current_gen and gen != current_gen:
            lines.append(r"\midrule")
        current_gen = gen
        k = kpart.replace("k=", "")
        star = r"$^*$" if r.get("p_value", 1) < 0.05 else ""
        lines.append(
            f"{gen} & {k} & {metric_display.get(metric, metric)} & "
            f"{_fmt(r['delta'])}{star} & "
            f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] & "
            f"{_fmt(r['p_value'])} & "
            f"{int(r['n'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{0.3em}",
        r"\noindent\footnotesize $^*$ denotes $p<0.05$ under a paired "
        r"bootstrap with $B=10{,}000$ resamples. Hallucination-rate tests "
        r"are omitted: since hallucination rate $=1-$ grounding, its paired "
        r"difference equals $-\Delta_{\text{grounding}}$, with the CI "
        r"reflected about zero and identical $p$ and $n$.",
    ]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table S1 - Per-difficulty breakdown
# ---------------------------------------------------------------------------

def tableS1_difficulty(summary_diff_path: str, out: Path) -> None:
    df = pd.read_csv(summary_diff_path)
    df["base"] = df["pipeline"].apply(lambda x: x.split(":")[0])
    df["generator"] = df["pipeline"].apply(
        lambda x: x.split(":")[1] if len(x.split(":")) > 1 else ""
    )
    agg = df.groupby(["base", "difficulty"], as_index=False).agg(
        factual=("factual_correctness_ratio_mean", "mean"),
        grounding=("grounding_ratio_mean", "mean"),
    )
    pivot_f = agg.pivot(index="base", columns="difficulty", values="factual")
    pivot_g = agg.pivot(index="base", columns="difficulty", values="grounding")

    diffs = ["easy", "medium", "hard"]
    lines = [
        r"\begin{tabular}{l" + "r" * 6 + "}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Factual} & \multicolumn{3}{c}{Grounding} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Pipeline & Easy & Med. & Hard & Easy & Med. & Hard \\",
        r"\midrule",
    ]
    for base in ["zeroshot", "rag_classic", "rag_dspy", "rag_graph"]:
        if base not in pivot_f.index:
            continue
        f_row = [pivot_f.loc[base].get(d) for d in diffs]
        g_row = [
            pivot_g.loc[base].get(d) if base in pivot_g.index else None
            for d in diffs
        ]
        all_vals = f_row + g_row
        cells = " & ".join(_fmt(v) for v in all_vals)
        base_tex = base.replace("_", r"\_")
        lines.append(f"{base_tex} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table S2 - Per-language breakdown
# ---------------------------------------------------------------------------

def tableS2_language(per_item_long: str, out: Path) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]
    df["base"] = df["pipeline"].apply(lambda x: x.split(":")[0])
    df = df[df["language"].isin(["en", "mk"])]

    agg = df.groupby(["base", "language"], as_index=False).agg(
        factual=("factual_correctness_ratio", "mean"),
        grounding=("grounding_ratio", "mean"),
    )
    pivot_f = agg.pivot(index="base", columns="language", values="factual")
    pivot_g = agg.pivot(index="base", columns="language", values="grounding")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Factual} & \multicolumn{2}{c}{Grounding} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Pipeline & EN & MK & EN & MK \\",
        r"\midrule",
    ]
    for base in ["zeroshot", "rag_classic", "rag_dspy", "rag_graph"]:
        if base not in pivot_f.index:
            continue
        f_en = pivot_f.loc[base].get("en")
        f_mk = pivot_f.loc[base].get("mk")
        g_en = pivot_g.loc[base].get("en") if base in pivot_g.index else None
        g_mk = pivot_g.loc[base].get("mk") if base in pivot_g.index else None
        base_tex = base.replace("_", r"\_")
        lines.append(
            f"{base_tex} & {_fmt(f_en)} & {_fmt(f_mk)} & "
            f"{_fmt(g_en)} & {_fmt(g_mk)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="data/analysis")
    ap.add_argument("--benchmark", default="data/benchmark.jsonl")
    ap.add_argument("--outdir", default="paper/tables")
    args = ap.parse_args()

    adir = Path(args.analysis_dir)
    outdir = Path(args.outdir)

    table1_benchmark_stats(args.benchmark, outdir / "table1_benchmark_stats.tex")
    print("[ok] table1")
    table2_main_results(
        str(adir / "summary_by_pipeline.csv"),
        str(adir / "metrics_summary.json"),
        outdir / "table2_main_results.tex",
    )
    print("[ok] table2")
    table3_retrieval_ceiling(
        str(adir / "metrics_summary.json"),
        outdir / "table3_retrieval_ceiling.tex",
    )
    print("[ok] table3")
    table4_judge_agreement(
        str(adir / "judge_agreement_spearman.csv"),
        outdir / "table4_judge_agreement.tex",
    )
    print("[ok] table4")
    table5_bootstrap(
        str(adir / "bootstrap_graph_vs_classic.json"),
        outdir / "table5_bootstrap_tests.tex",
    )
    print("[ok] table5")
    tableS1_difficulty(
        str(adir / "summary_by_difficulty.csv"),
        outdir / "tableS1_difficulty.tex",
    )
    print("[ok] tableS1")
    tableS2_language(
        str(adir / "per_item_long.csv"),
        outdir / "tableS2_language.tex",
    )
    print("[ok] tableS2")
    print(f"[done] all tables written to {outdir}")


if __name__ == "__main__":
    main()
