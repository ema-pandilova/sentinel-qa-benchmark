"""
Paper figure generation for the 'Grounded != Correct' journal paper.

Reads:
    data/analysis/per_item_long.csv
    data/analysis/summary_by_pipeline.csv
    data/analysis/summary_by_k.csv
    data/analysis/summary_by_difficulty.csv
    data/analysis/metrics_summary.json
    data/analysis/judge_agreement_spearman.csv
    data/analysis/bootstrap_graph_vs_classic.json

Writes:
    paper/figures/fig1..fig6, figS1..figS4  (PDF + PNG each)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

PIPELINE_COLORS = {
    "zeroshot":    "#888888",
    "rag_classic": "#1f77b4",
    "rag_dspy":    "#d62728",
    "rag_graph":   "#2ca02c",
}
PIPELINE_LABELS = {
    "zeroshot":    "Zeroshot",
    "rag_classic": "Classic RAG",
    "rag_dspy":    "DSPy RAG",
    "rag_graph":   "Graph RAG",
}
GENERATOR_MARKERS = {
    "openai":    "o",
    "anthropic": "s",
    "deepseek":  "^",
}
PIPELINE_ORDER = ["zeroshot", "rag_classic", "rag_dspy", "rag_graph"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


def _parse_pipeline(pipeline: str) -> Tuple[str, str, int, int]:
    """Return (base, generator, k, max_nodes)."""
    parts = pipeline.split(":")
    base = parts[0]
    gen = parts[1] if len(parts) > 1 else ""
    k_match = re.search(r"k=(\d+)", pipeline)
    k = int(k_match.group(1)) if k_match else 0
    mn_match = re.search(r"mn=(\d+)", pipeline)
    mn = int(mn_match.group(1)) if mn_match else 0
    return base, gen, k, mn


def _facet_summary(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    facets = df["pipeline"].apply(_parse_pipeline)
    df["base"] = [f[0] for f in facets]
    df["generator"] = [f[1] for f in facets]
    df["k"] = [f[2] for f in facets]
    df["mn"] = [f[3] for f in facets]
    return df


# ---------------------------------------------------------------------------
# Figure 0 - Pipelines schematic (sec. 4)
# ---------------------------------------------------------------------------

def fig0_pipelines_schematic(outdir: Path) -> None:
    """Four-pipeline architecture diagram.

    Rows (top -> bottom): Zero-shot, Classic RAG, DSPy RAG, GraphRAG.
    Shared retriever R_k is drawn in the same blue box across the three
    RAG rows. Shared generator G(.) is drawn in the same red box across
    Zero-shot / Classic RAG / GraphRAG (DSPy uses a compiled signature
    predictor Sigma in place of G+template).
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    rows = [
        ("Zero-shot",   "#888888"),
        ("Classic RAG", "#1f77b4"),
        ("DSPy RAG",    "#d62728"),
        ("GraphRAG",    "#2ca02c"),
    ]
    n_rows = len(rows)
    row_h = 1.6
    row_gap = 0.66
    bh = 0.92          # box height
    total_h = n_rows * row_h + (n_rows - 1) * row_gap

    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, total_h)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    def box(xy, y, h, text, face, edge="black", text_color="black",
            fontsize=10.5, italic=False, weight="normal"):
        x_left, x_right = xy
        w = x_right - x_left
        patch = FancyBboxPatch(
            (x_left, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.1, edgecolor=edge, facecolor=face, zorder=3,
        )
        ax.add_patch(patch)
        style = "italic" if italic else "normal"
        ax.text(x_left + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, style=style, weight=weight,
                zorder=4, color=text_color)

    def arrow(x1, y, x2, color="#444"):
        ax.add_patch(FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle="-|>", mutation_scale=14,
            lw=1.3, color=color, zorder=2,
        ))

    # Colour palette for box types (kept muted so text stays legible).
    color_query = "#fff2cc"
    color_retr  = "#dbe9ff"   # shared blue for retriever R_k
    color_ctx   = "#ececec"   # context assembly / concat
    color_exp   = "#e3d5f0"   # GraphRAG expansion E_{m_n}
    color_sig   = "#fde0c8"   # DSPy signature container
    color_gen   = "#f8d0d0"   # shared red for generator G
    color_ans   = "#d6ead5"

    # X-coordinates of box columns. Each row is laid out in left-to-right
    # pipeline order; arrows are drawn between successive boxes.
    #    col:           Q    R_k   ctx  G    A      (simple)
    #                   Q    R_k   exp  ctx G  A   (graph)
    # Column x positions:
    xs = {
        "Q":    (0.2, 2.4),
        "R":    (3.2, 6.4),
        "C1":   (7.1, 10.3),  # context assembly (concat) OR DSPy signature container
        "C2":   (10.9, 14.1), # GraphRAG second stage (concat after expansion)
        "G":    (14.8, 18.0),
        "A":    (18.7, 21.2),
    }

    def row_y(i):
        # Top row = row 0
        return total_h - (i + 1) * row_h - i * row_gap

    def row_box_y(i):
        # bottom y of a box vertically centred in row i's band
        return row_y(i) + (row_h - bh) / 2

    # ---- Build each row ------------------------------------------------
    # 0: Zero-shot  (no retrieval: the question goes straight to the generator)
    y = row_box_y(0)
    box(xs["Q"], y, bh, "Question $q$", color_query)
    box(xs["G"], y, bh, "Generator $G(\\cdot)$", color_gen)
    box(xs["A"], y, bh, "Answer $a$", color_ans)
    ycen = y + bh / 2
    # single long arrow spanning the bypassed retrieval stages
    arrow(xs["Q"][1], ycen, xs["G"][0], color="#888")
    ax.text((xs["Q"][1] + xs["G"][0]) / 2, ycen + 0.30,
            "no retrieval: answered directly",
            ha="center", va="bottom", fontsize=10,
            color="#888", style="italic", zorder=4)
    arrow(xs["G"][1], ycen, xs["A"][0])

    # 1: Classic RAG  (no graph expansion: concat feeds the generator directly)
    y = row_box_y(1)
    box(xs["Q"], y, bh, "Question $q$", color_query)
    box(xs["R"], y, bh, "Retriever\n$R_k(q)$", color_retr)
    box(xs["C1"], y, bh, "Concat\n$\\bigoplus_{c\\in R_k(q)} t(c)$", color_ctx)
    box(xs["G"], y, bh, "Generator\n$G(\\cdot)$", color_gen)
    box(xs["A"], y, bh, "Answer $a$", color_ans)
    ycen = y + bh / 2
    arrow(xs["Q"][1], ycen, xs["R"][0])
    arrow(xs["R"][1], ycen, xs["C1"][0])
    # single arrow spanning the (absent) graph-expansion stage
    arrow(xs["C1"][1], ycen, xs["G"][0], color="#888")
    ax.text((xs["C1"][1] + xs["G"][0]) / 2, ycen + 0.30,
            "no graph expansion",
            ha="center", va="bottom", fontsize=10,
            color="#888", style="italic", zorder=4)
    arrow(xs["G"][1], ycen, xs["A"][0])

    # 2: DSPy RAG
    y = row_box_y(2)
    box(xs["Q"], y, bh, "Question $q$", color_query)
    box(xs["R"], y, bh, "Retriever\n$R_k(q)$", color_retr)
    # Merge C1+C2+G into a single DSPy Sigma compiled predictor box.
    dspy_x0 = xs["C1"][0]
    dspy_x1 = xs["G"][1]
    ax.add_patch(FancyBboxPatch(
        (dspy_x0, y), dspy_x1 - dspy_x0, bh,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor="#7d3c98", facecolor=color_sig, zorder=3,
    ))
    ycen = y + bh / 2
    ax.text((dspy_x0 + dspy_x1) / 2, ycen + 0.17,
            "Compiled signature predictor",
            ha="center", va="center", fontsize=11, weight="bold", zorder=4)
    ax.text((dspy_x0 + dspy_x1) / 2, ycen - 0.17,
            r"$\Sigma:(\mathrm{context},\mathrm{question})\!\to\!\mathrm{answer}$",
            ha="center", va="center", fontsize=11, zorder=4)
    box(xs["A"], y, bh, "Answer $a$", color_ans)
    arrow(xs["Q"][1], ycen, xs["R"][0])
    arrow(xs["R"][1], ycen, dspy_x0)
    arrow(dspy_x1, ycen, xs["A"][0])

    # 3: GraphRAG
    y = row_box_y(3)
    box(xs["Q"], y, bh, "Question $q$", color_query)
    box(xs["R"], y, bh, "Retriever\n$R_k(q)$", color_retr)
    box(xs["C1"], y, bh, "Graph expansion\n$E_{m_n}(R_k(q);\\mathcal{G})$",
        color_exp)
    box(xs["C2"], y, bh, "Concat\n$\\bigoplus_{c\\in E_{m_n}} t(c)$", color_ctx)
    box(xs["G"], y, bh, "Generator\n$G(\\cdot)$", color_gen)
    box(xs["A"], y, bh, "Answer $a$", color_ans)
    ycen = y + bh / 2
    arrow(xs["Q"][1], ycen, xs["R"][0])
    arrow(xs["R"][1], ycen, xs["C1"][0])
    arrow(xs["C1"][1], ycen, xs["C2"][0])
    arrow(xs["C2"][1], ycen, xs["G"][0])
    arrow(xs["G"][1], ycen, xs["A"][0])

    # Row labels (left) + colored line on the left side keyed to pipeline
    for i, (label, col) in enumerate(rows):
        y = row_y(i) + row_h / 2
        ax.text(-0.1, y, label, ha="right", va="center",
                fontsize=12, weight="bold", color=col)
        # vertical coloured tick to the left of the row
        ax.plot([-0.02, -0.02], [y - bh / 2, y + bh / 2],
                color=col, lw=3.6, solid_capstyle="round", zorder=5)

    # Shared-substrate emphasis bracket: R_k across rows 2-4
    bracket_y = total_h + 0.36
    ax.plot(
        [xs["R"][0] - 0.1, xs["R"][1] + 0.1],
        [bracket_y, bracket_y],
        color="#2e4a7b", lw=1.2,
    )
    ax.plot(
        [xs["R"][0] - 0.1, xs["R"][0] - 0.1],
        [bracket_y, bracket_y - 0.15],
        color="#2e4a7b", lw=1.2,
    )
    ax.plot(
        [xs["R"][1] + 0.1, xs["R"][1] + 0.1],
        [bracket_y, bracket_y - 0.15],
        color="#2e4a7b", lw=1.2,
    )
    ax.text(
        (xs["R"][0] + xs["R"][1]) / 2, bracket_y + 0.08,
        "shared retriever $R_k$ (Classic, DSPy, GraphRAG)",
        ha="center", va="bottom", fontsize=11,
        color="#2e4a7b", style="italic",
    )

    # Shared-generator emphasis bracket: G across rows 1, 2, 4
    ax.plot(
        [xs["G"][0] - 0.1, xs["G"][1] + 0.1],
        [bracket_y, bracket_y],
        color="#7a2e2e", lw=1.2,
    )
    ax.plot(
        [xs["G"][0] - 0.1, xs["G"][0] - 0.1],
        [bracket_y, bracket_y - 0.15],
        color="#7a2e2e", lw=1.2,
    )
    ax.plot(
        [xs["G"][1] + 0.1, xs["G"][1] + 0.1],
        [bracket_y, bracket_y - 0.15],
        color="#7a2e2e", lw=1.2,
    )
    ax.text(
        (xs["G"][0] + xs["G"][1]) / 2, bracket_y + 0.08,
        "shared generator $G(\\cdot)$ (Zero-shot, Classic, GraphRAG)",
        ha="center", va="bottom", fontsize=11,
        color="#7a2e2e", style="italic",
    )

    # Mark that the shared generator G is instantiated with three models.
    ax.text(
        (xs["G"][0] + xs["G"][1]) / 2, -0.34,
        "each $G(\\cdot)$ run with 3 generators:\n"
        "GPT-4o, Claude Sonnet 4, DeepSeek-Chat",
        ha="center", va="top", fontsize=10.5,
        color="#7a2e2e", style="italic", zorder=4,
    )

    ax.set_xlim(-3.4, 22.2)
    ax.set_ylim(-1.4, total_h + 1.05)
    _save(fig, outdir, "fig0_pipelines_schematic")


# ---------------------------------------------------------------------------
# Figure 1 - The Plane (teaser figure)
# ---------------------------------------------------------------------------

def fig1_grounding_correctness_plane(summary_path: str, outdir: Path) -> None:
    df = _facet_summary(summary_path)
    df = df.dropna(subset=["grounding_ratio_mean", "factual_correctness_ratio_mean"])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # Quadrant shading
    x_mid = 0.60
    y_mid = 0.27
    ax.axhspan(y_mid, 1.0, xmin=x_mid, xmax=1.0,
               color="#e9f5e9", alpha=0.6, zorder=0)
    ax.axhspan(0.0, y_mid, xmin=x_mid, xmax=1.0,
               color="#fdecea", alpha=0.6, zorder=0)
    ax.axvline(x_mid, color="#999", lw=0.8, ls="--", zorder=1)
    ax.axhline(y_mid, color="#999", lw=0.8, ls="--", zorder=1)

    # Quadrant labels
    ax.text(0.81, 0.42, "grounded\n& correct", ha="center", va="center",
            fontsize=9, color="#2a7a2a", alpha=0.7, style="italic")
    ax.text(0.81, 0.10, "grounded\nbut wrong", ha="center", va="center",
            fontsize=9, color="#a02020", alpha=0.7, style="italic")
    ax.text(0.30, 0.42, "ungrounded\n& correct", ha="center", va="center",
            fontsize=9, color="#888", alpha=0.7, style="italic")
    ax.text(0.30, 0.10, "ungrounded\n& wrong", ha="center", va="center",
            fontsize=9, color="#666", alpha=0.7, style="italic")

    for _, row in df.iterrows():
        base = row["base"]
        gen = row["generator"]
        color = PIPELINE_COLORS.get(base, "#000000")
        marker = GENERATOR_MARKERS.get(gen, "o")
        ax.scatter(
            row["grounding_ratio_mean"],
            row["factual_correctness_ratio_mean"],
            s=140,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )

    # Annotate the two headline phenomena
    ax.annotate(
        "GraphRAG: shifts right, not up",
        xy=(0.60, 0.21), xytext=(0.32, 0.05),
        fontsize=9, color="#1a5a1a",
        arrowprops=dict(arrowstyle="->", color="#1a5a1a",
                        lw=1.0, alpha=0.7),
    )
    ax.annotate(
        "DSPy: the only pipeline\nthat improves both",
        xy=(0.80, 0.35), xytext=(0.35, 0.38),
        fontsize=9, color="#8a1a1a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#8a1a1a",
                        lw=1.0, alpha=0.8),
    )

    ax.set_xlabel("Grounding ratio (higher = more source-faithful)")
    ax.set_ylabel("Factual correctness ratio (higher = more accurate)")
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(0.0, 0.45)
    ax.set_title(
        "The grounded-vs-correct plane:\n"
        "GraphRAG shifts right, DSPy is alone in the upper-right"
    )

    from matplotlib.lines import Line2D
    pipeline_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PIPELINE_COLORS[p],
               markeredgecolor="black",
               markersize=10, label=PIPELINE_LABELS[p])
        for p in PIPELINE_ORDER
    ]
    gen_handles = [
        Line2D([0], [0], marker=m, color="w",
               markerfacecolor="#555555",
               markeredgecolor="black",
               markersize=9, label=g)
        for g, m in GENERATOR_MARKERS.items()
    ]
    leg1 = ax.legend(
        handles=pipeline_handles, loc="upper left",
        title="Pipeline", framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=gen_handles, loc="lower right",
        title="Generator", framealpha=0.9,
    )

    _save(fig, outdir, "fig1_grounding_correctness_plane")


# ---------------------------------------------------------------------------
# Figure 2 - Pipeline scorecard
# ---------------------------------------------------------------------------

def fig2_pipeline_scorecard(per_item_long: str, outdir: Path) -> None:
    """Grouped bars per pipeline (at k=3, or zeroshot) with item-clustered SE.

    SE is computed over items: within each (pipeline_base, item_id) we first
    average across judges, then report mean +/- SE of the item-level means.
    This yields the correct uncertainty for the headline pipeline score and
    matches the spec's 'judge-level standard error from per_item_long.csv'.
    """
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]
    df["base"] = df["pipeline"].apply(lambda x: x.split(":")[0])
    df["k"] = df["pipeline"].apply(
        lambda x: int(re.search(r"k=(\d+)", x).group(1))
        if re.search(r"k=(\d+)", x) else 0
    )
    # Use k=3 for RAG pipelines; zeroshot has no k.
    df = df[(df["base"] == "zeroshot") | (df["k"] == 3)]

    metrics = [
        ("factual_correctness_ratio", "Factual correctness", "#d62728"),
        ("grounding_ratio",           "Grounding",           "#1f77b4"),
    ]

    means: dict[str, dict[str, float]] = {}
    ses: dict[str, dict[str, float]] = {}
    bases_present = [b for b in PIPELINE_ORDER if b in df["base"].unique()]
    for base in bases_present:
        sub = df[df["base"] == base]
        means[base] = {}
        ses[base] = {}
        for col, _, _ in metrics:
            item_means = sub.groupby("item_id")[col].mean().dropna()
            if item_means.empty:
                means[base][col] = np.nan
                ses[base][col] = np.nan
                continue
            means[base][col] = float(item_means.mean())
            ses[base][col] = float(item_means.std(ddof=1)
                                   / np.sqrt(len(item_means)))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(bases_present))
    width = 0.34
    offsets = [-width / 2, width / 2]

    for (col, label, color), off in zip(metrics, offsets):
        vals = [means[b][col] for b in bases_present]
        errs = [ses[b][col] for b in bases_present]
        # For zeroshot, grounding/halluc are NaN -> skip those bars visually
        bar_x = [xi + off for xi, v in zip(x, vals) if pd.notna(v)]
        bar_v = [v for v in vals if pd.notna(v)]
        bar_e = [e for v, e in zip(vals, errs) if pd.notna(v)]
        ax.bar(bar_x, bar_v, width, yerr=bar_e, capsize=3, label=label,
               color=color, edgecolor="black", linewidth=0.5,
               error_kw={"elinewidth": 0.9, "ecolor": "black"})
        for xi, v, e in zip(bar_x, bar_v, bar_e):
            ax.text(xi, v + e + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS[b] for b in bases_present])
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Pipeline scorecard at k=3 (item-clustered mean $\\pm$ SE)"
    )
    ax.legend(loc="upper left", framealpha=0.9, ncol=1)

    _save(fig, outdir, "fig2_pipeline_scorecard")


# ---------------------------------------------------------------------------
# Figure 3 - The GraphRAG illusion
# ---------------------------------------------------------------------------

def fig3_graph_illusion(
    per_item_long: str, bootstrap_path: str, outdir: Path
) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]

    agg = (
        df.groupby(["item_id", "pipeline"], as_index=False)[
            ["grounding_ratio", "factual_correctness_ratio"]
        ].mean()
    )

    # Headline comparison: OpenAI k=1 (where the decoupling is sharpest)
    classic_mode = "rag_classic:openai:k=1"
    graph_mode = "rag_graph:openai:k=1:mn=10"

    classic = agg[agg["pipeline"] == classic_mode].rename(
        columns={
            "grounding_ratio": "grounding_c",
            "factual_correctness_ratio": "factual_c",
        }
    )[["item_id", "grounding_c", "factual_c"]]
    graph = agg[agg["pipeline"] == graph_mode].rename(
        columns={
            "grounding_ratio": "grounding_g",
            "factual_correctness_ratio": "factual_g",
        }
    )[["item_id", "grounding_g", "factual_g"]]
    pairs = classic.merge(graph, on="item_id", how="inner").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.2), sharey=False)

    # Grounding panel
    ax = axes[0]
    for _, row in pairs.iterrows():
        ax.plot(
            [0, 1], [row["grounding_c"], row["grounding_g"]],
            color="#1f77b4", alpha=0.28, lw=0.9,
        )
    ax.scatter(
        [0] * len(pairs), pairs["grounding_c"],
        color="#1f77b4", s=22, zorder=3,
        edgecolors="black", linewidths=0.3,
    )
    ax.scatter(
        [1] * len(pairs), pairs["grounding_g"],
        color="#2ca02c", s=22, zorder=3,
        edgecolors="black", linewidths=0.3,
    )
    mean_c = pairs["grounding_c"].mean()
    mean_g = pairs["grounding_g"].mean()
    ax.plot([0, 1], [mean_c, mean_g], color="black", lw=2.5, zorder=4,
            marker="o", markersize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Classic", "Graph"])
    ax.set_ylabel("Grounding ratio")
    ax.set_title("Grounding lifts with GraphRAG")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, 1.3)

    # Factual panel
    ax = axes[1]
    for _, row in pairs.iterrows():
        ax.plot(
            [0, 1], [row["factual_c"], row["factual_g"]],
            color="#d62728", alpha=0.28, lw=0.9,
        )
    ax.scatter(
        [0] * len(pairs), pairs["factual_c"],
        color="#1f77b4", s=22, zorder=3,
        edgecolors="black", linewidths=0.3,
    )
    ax.scatter(
        [1] * len(pairs), pairs["factual_g"],
        color="#2ca02c", s=22, zorder=3,
        edgecolors="black", linewidths=0.3,
    )
    mean_c = pairs["factual_c"].mean()
    mean_g = pairs["factual_g"].mean()
    ax.plot([0, 1], [mean_c, mean_g], color="black", lw=2.5, zorder=4,
            marker="o", markersize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Classic", "Graph"])
    ax.set_ylabel("Factual correctness ratio")
    ax.set_title("Factual correctness does not")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, 1.3)

    try:
        with open(bootstrap_path) as f:
            boot = json.load(f)
        key_g = "openai|k=1|grounding_ratio"
        key_f = "openai|k=1|factual_correctness_ratio"
        if key_g in boot:
            g = boot[key_g]
            axes[0].text(
                0.5, 0.97,
                f"Δ = {g['delta']:+.3f}  [{g['ci_low']:+.3f}, "
                f"{g['ci_high']:+.3f}]\np = {g['p_value']:.3f}",
                transform=axes[0].transAxes, ha="center", va="top",
                fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.9,
                          edgecolor="#bbb", boxstyle="round,pad=0.35"),
            )
        if key_f in boot:
            fb = boot[key_f]
            axes[1].text(
                0.5, 0.97,
                f"Δ = {fb['delta']:+.3f}  [{fb['ci_low']:+.3f}, "
                f"{fb['ci_high']:+.3f}]\np = {fb['p_value']:.3f}",
                transform=axes[1].transAxes, ha="center", va="top",
                fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.9,
                          edgecolor="#bbb", boxstyle="round,pad=0.35"),
            )
    except FileNotFoundError:
        pass

    fig.suptitle(
        "The GraphRAG illusion (OpenAI, k=1): "
        "the same retrieval shift that lifts grounding leaves correctness flat",
        y=1.02, fontsize=11,
    )
    _save(fig, outdir, "fig3_graph_illusion")


# ---------------------------------------------------------------------------
# Figure 4 - Judge reliability split
# ---------------------------------------------------------------------------

def fig4_judge_reliability(spearman_path: str, outdir: Path) -> None:
    df = pd.read_csv(spearman_path)
    judges = sorted(set(df["judge_a"]).union(df["judge_b"]))

    def _aggregate(metric: str) -> np.ndarray:
        sub = df[df["metric"] == metric]
        agg = (
            sub.groupby(["judge_a", "judge_b"])["spearman_r"]
            .mean()
            .reset_index()
        )
        m = np.full((len(judges), len(judges)), np.nan)
        for _, row in agg.iterrows():
            ja = judges.index(row["judge_a"])
            jb = judges.index(row["judge_b"])
            m[ja, jb] = row["spearman_r"]
            m[jb, ja] = row["spearman_r"]
        np.fill_diagonal(m, 1.0)
        return m

    mg = _aggregate("grounding_ratio")
    mf = _aggregate("factual_correctness_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plt.subplots_adjust(wspace=0.55)
    vmin, vmax = 0.0, 1.0

    # shorten names to avoid overlap
    def _short(j: str) -> str:
        s = j.split("/")[-1]
        return s.replace("claude-sonnet-4-20250514", "claude-sonnet-4")

    short_names = [_short(j) for j in judges]

    for ax, mat, title in [
        (axes[0], mg, "Grounding agreement"),
        (axes[1], mf, "Factual correctness agreement"),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
        ax.set_xticks(range(len(judges)))
        ax.set_yticks(range(len(judges)))
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(title)
        for i in range(len(judges)):
            for j in range(len(judges)):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black",
                        fontsize=9,
                    )
        ax.grid(False)

    fig.colorbar(im, ax=axes, shrink=0.75, label=r"Spearman $\rho$", pad=0.02)
    fig.suptitle(
        "Judges agree on grounding; they disagree on factual correctness",
        y=1.02, fontsize=11,
    )
    _save(fig, outdir, "fig4_judge_reliability")


# ---------------------------------------------------------------------------
# Figure 5 - Cost-accuracy Pareto frontier
# ---------------------------------------------------------------------------

def fig5_pareto(metrics_json: str, outdir: Path) -> None:
    with open(metrics_json) as f:
        m = json.load(f)
    rows = m.get("cost_accuracy", [])

    df = pd.DataFrame(rows).dropna(subset=["factual_correctness_ratio"])
    df["base"] = df["mode"].apply(lambda x: x.split(":")[0])
    df["cost"] = df["est_cost_per_call_usd"].replace(0, np.nan)
    df = df.dropna(subset=["cost"])

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    for base in PIPELINE_ORDER:
        sub = df[df["base"] == base]
        if sub.empty:
            continue
        ax.scatter(
            sub["cost"], sub["factual_correctness_ratio"],
            label=PIPELINE_LABELS[base],
            color=PIPELINE_COLORS[base],
            s=110, edgecolors="black", linewidths=0.6, alpha=0.9,
            zorder=3,
        )
        for _, row in sub.iterrows():
            label = row["mode"].replace("rag_", "").replace(":mn=10", "")
            ax.annotate(
                label,
                (row["cost"], row["factual_correctness_ratio"]),
                fontsize=6.5, alpha=0.75,
                xytext=(4, 3), textcoords="offset points",
            )

    # Pareto frontier (max factual for min cost)
    clean = df.sort_values("cost")
    frontier_x, frontier_y = [], []
    best_y = -1.0
    for _, row in clean.iterrows():
        if row["factual_correctness_ratio"] > best_y:
            frontier_x.append(row["cost"])
            frontier_y.append(row["factual_correctness_ratio"])
            best_y = row["factual_correctness_ratio"]
    ax.plot(
        frontier_x, frontier_y, "k--", lw=1.4, alpha=0.7,
        label="Pareto frontier", zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost per call (USD, log scale)")
    ax.set_ylabel("Factual correctness ratio")
    ax.set_title(
        "Cost-accuracy Pareto frontier: DSPy dominates the upper-left"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    _save(fig, outdir, "fig5_pareto_frontier")


# ---------------------------------------------------------------------------
# Figure 6 - k-ablation small multiples
# ---------------------------------------------------------------------------

def fig6_k_ablation(summary_k_path: str, outdir: Path) -> None:
    df = pd.read_csv(summary_k_path)
    df = df[df["generator"] == "openai"]
    df["k_value"] = df["k_value"].astype(float)

    # Zeroshot has no k dimension, so it isn't in summary_by_k.csv. Pull the
    # OpenAI zeroshot factual mean from summary_by_pipeline.csv to plot a
    # flat k-independent reference in the fourth panel.
    pipeline_path = Path(summary_k_path).parent / "summary_by_pipeline.csv"
    zs_fact: float | None = None
    if pipeline_path.exists():
        pdf = pd.read_csv(pipeline_path)
        zs_row = pdf[pdf["pipeline"] == "zeroshot:openai"]
        if not zs_row.empty:
            zs_fact = float(zs_row["factual_correctness_ratio_mean"].iloc[0])

    fig, axes = plt.subplots(
        2, 2, figsize=(7.8, 5.8), sharex=True, sharey=True,
    )
    # Zeroshot first (top-left) so it anchors the reader as the baseline.
    order = ["zeroshot", "rag_classic", "rag_dspy", "rag_graph"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for base, (i, j) in zip(order, positions):
        ax = axes[i, j]
        if base == "zeroshot":
            if zs_fact is None:
                ax.set_title("Zeroshot (no data)")
                ax.set_axis_off()
                continue
            xs = [1, 3, 5]
            ax.plot(xs, [zs_fact] * 3, "-o", color="#d62728",
                    lw=2.0, ms=8, label="Factual (k-independent)")
            ax.text(3, zs_fact + 0.03, f"{zs_fact:.2f}",
                    ha="center", fontsize=8, color="#d62728")
            ax.set_title(PIPELINE_LABELS[base])
            ax.set_xticks([1, 3, 5])
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel("Score")
            continue
        sub = df[df["base_mode"] == base].sort_values("k_value")
        if sub.empty:
            ax.set_title(PIPELINE_LABELS.get(base, base) + " (no data)")
            ax.set_axis_off()
            continue
        ax.plot(
            sub["k_value"], sub["factual_correctness_ratio_mean"],
            "-o", color="#d62728", lw=2.0, ms=8, label="Factual",
        )
        if sub["grounding_ratio_mean"].notna().any():
            ax.plot(
                sub["k_value"], sub["grounding_ratio_mean"],
                "--s", color="#1f77b4", lw=2.0, ms=8, label="Grounding",
            )
        ax.set_title(PIPELINE_LABELS.get(base, base))
        ax.set_xticks([1, 3, 5])
        ax.set_ylim(0, 1.0)
        if i == 1:
            ax.set_xlabel("k (retrieved chunks)")
        if j == 0:
            ax.set_ylabel("Score")
        if (i, j) == (0, 1):
            ax.legend(loc="lower right", fontsize=8)

    # Make sure Zeroshot panel also shows axis labels and xlabel on bottom row.
    if axes[1, 0].get_xlabel() == "":
        axes[1, 0].set_xlabel("k (retrieved chunks)")

    fig.suptitle("k-ablation (OpenAI generator)", y=1.00, fontsize=11)
    _save(fig, outdir, "fig6_k_ablation")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------

def figS1_language(per_item_long: str, outdir: Path) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]
    df["base"] = df["pipeline"].apply(lambda x: x.split(":")[0])
    df = df[df["language"].isin(["en", "mk"])]

    agg = (
        df.groupby(["base", "language"], as_index=False)[
            ["factual_correctness_ratio", "grounding_ratio"]
        ].mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for metric, ax, title in [
        ("factual_correctness_ratio", axes[0], "Factual correctness by language"),
        ("grounding_ratio", axes[1], "Grounding by language"),
    ]:
        bases = [b for b in PIPELINE_ORDER if b in agg["base"].values]
        x = np.arange(len(bases))
        width = 0.38
        en = [
            float(agg[(agg["base"] == b) & (agg["language"] == "en")][metric].mean())
            for b in bases
        ]
        mk = [
            float(agg[(agg["base"] == b) & (agg["language"] == "mk")][metric].mean())
            for b in bases
        ]
        ax.bar(x - width / 2, en, width, label="English",
               color="#1f77b4", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, mk, width, label="Macedonian",
               color="#ff7f0e", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([PIPELINE_LABELS[b] for b in bases], rotation=20)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Language robustness: does the decoupling hold in low-resource MK?",
        y=1.02, fontsize=11,
    )
    _save(fig, outdir, "figS1_language")


def figS2_difficulty(per_item_long: str, outdir: Path) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]
    df["base"] = df["pipeline"].apply(lambda x: x.split(":")[0])
    df = df[df["difficulty"].isin(["easy", "medium", "hard"])]

    bases = [b for b in PIPELINE_ORDER if b in df["base"].unique()]
    diffs = ["easy", "medium", "hard"]
    mat = np.full((len(bases), len(diffs)), np.nan)
    for i, b in enumerate(bases):
        for j, d in enumerate(diffs):
            sub = df[(df["base"] == b) & (df["difficulty"] == d)]
            if not sub.empty:
                mat[i, j] = sub["factual_correctness_ratio"].mean()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(mat, vmin=0, vmax=0.5, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels(diffs)
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels([PIPELINE_LABELS[b] for b in bases])
    for i in range(len(bases)):
        for j in range(len(diffs)):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mat[i, j] < 0.28 else "black",
                    fontsize=10,
                )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Factual correctness")
    ax.set_title("Factual correctness by pipeline × difficulty")
    _save(fig, outdir, "figS2_difficulty")


def figS4_judge_leniency(per_item_long: str, outdir: Path) -> None:
    df = pd.read_csv(per_item_long)
    if "judge_error" in df.columns:
        df = df[~df["judge_error"].astype(bool)]

    df["judge_id"] = (
        df["judge_provider"].astype(str) + "/" + df["judge_model"].astype(str)
    )

    agg = (
        df.groupby("judge_id", as_index=False)[
            ["factual_correctness_ratio", "grounding_ratio"]
        ].mean()
        .sort_values("factual_correctness_ratio", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(agg))
    width = 0.38
    ax.bar(
        x - width / 2, agg["factual_correctness_ratio"], width,
        label="Factual", color="#d62728",
        edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, agg["grounding_ratio"], width,
        label="Grounding", color="#1f77b4",
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([j.split("/")[-1] for j in agg["judge_id"]],
                       rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score across all pipelines")
    ax.set_title("Judge leniency: factual outlier = gpt-4o-mini")
    ax.legend()

    # value labels
    for xi, col in zip(x - width / 2, agg["factual_correctness_ratio"]):
        ax.text(xi, col + 0.01, f"{col:.2f}", ha="center", fontsize=8)
    for xi, col in zip(x + width / 2, agg["grounding_ratio"]):
        if pd.notna(col):
            ax.text(xi, col + 0.01, f"{col:.2f}", ha="center", fontsize=8)

    _save(fig, outdir, "figS4_judge_leniency")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="data/analysis")
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()

    adir = Path(args.analysis_dir)
    outdir = Path(args.outdir)

    fig0_pipelines_schematic(outdir)
    print("[ok] fig0")
    fig1_grounding_correctness_plane(
        str(adir / "summary_by_pipeline.csv"), outdir
    )
    print("[ok] fig1")
    fig2_pipeline_scorecard(
        str(adir / "per_item_long.csv"), outdir
    )
    print("[ok] fig2")
    fig3_graph_illusion(
        str(adir / "per_item_long.csv"),
        str(adir / "bootstrap_graph_vs_classic.json"),
        outdir,
    )
    print("[ok] fig3")
    fig4_judge_reliability(
        str(adir / "judge_agreement_spearman.csv"), outdir
    )
    print("[ok] fig4")
    fig5_pareto(str(adir / "metrics_summary.json"), outdir)
    print("[ok] fig5")
    fig6_k_ablation(str(adir / "summary_by_k.csv"), outdir)
    print("[ok] fig6")
    figS1_language(str(adir / "per_item_long.csv"), outdir)
    print("[ok] figS1")
    figS2_difficulty(str(adir / "per_item_long.csv"), outdir)
    print("[ok] figS2")
    # figS3 (max_nodes ablation) intentionally dropped: only mn=10 is judged,
    # so the figure is not publication-quality. max_nodes is discussed in
    # section 7.2 prose instead, with the threshold/mn sweep listed as future
    # work.
    figS4_judge_leniency(str(adir / "per_item_long.csv"), outdir)
    print("[ok] figS4")
    print(f"[done] all figures written to {outdir}")


if __name__ == "__main__":
    main()
