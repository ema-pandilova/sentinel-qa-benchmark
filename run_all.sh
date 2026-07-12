#!/usr/bin/env bash
#
# Run the complete benchmark pipeline end-to-end.
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --skip-gen   # Skip question generation (use existing benchmark.jsonl)
#   ./run_all.sh --dry-run    # Dry run (2 items only, for verification)
#
set -euo pipefail

# Suppress noisy Chroma telemetry warnings
export ANONYMIZED_TELEMETRY=false

SKIP_GEN=false
DRY_RUN=""

for arg in "$@"; do
  case $arg in
    --skip-gen)  SKIP_GEN=true ;;
    --dry-run)   DRY_RUN="--dry-run" ;;
    *)           echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

SECONDS=0
step=0
step() { step=$((step+1)); echo ""; echo "========================================"; echo "[$step] $1"; echo "========================================"; }

# ──────────────────────────────────────────────────
# PHASE 1: Data preparation
# ──────────────────────────────────────────────────

if [ "$SKIP_GEN" = false ]; then
  step "Generating 15 hard-difficulty questions"
  python bench/generate_questions.py \
    --n_questions=15 \
    --difficulty=hard \
    --append \
    --out=data/benchmark.jsonl
fi

step "Populating reference_doc_ids in benchmark.jsonl"
python bench/populate_reference_doc_ids.py

# ──────────────────────────────────────────────────
# PHASE 2: Build knowledge graphs (3 thresholds)
# ──────────────────────────────────────────────────

step "Building knowledge graphs (thresholds: 0.7, 0.8, 0.9)"
python -m bench.run_benchmark --experiment build-graphs

# ──────────────────────────────────────────────────
# PHASE 3: Run experiments
# ──────────────────────────────────────────────────

step "Running multi-model experiment (3 generators x 4 modes x k={1,3,5})"
python -m bench.run_benchmark --experiment multi-model $DRY_RUN

step "Running graph ablation experiment (GPT-4o x rag_graph x k=3 x 9 configs)"
python -m bench.run_benchmark --experiment graph-ablation $DRY_RUN

# Merge results into a single file for unified evaluation
step "Merging result files"
cat data/benchmark_results.jsonl data/benchmark_results_graph_ablation.jsonl \
  > data/benchmark_results_all.jsonl
echo "Merged: $(wc -l < data/benchmark_results_all.jsonl) total result records"

# ──────────────────────────────────────────────────
# PHASE 4: LLM-as-judge evaluation
# ──────────────────────────────────────────────────

step "Running LLM-as-judge evaluation (4 judges x 2 metrics)"
python -m bench.meta_eval \
  --results data/benchmark_results_all.jsonl \
  --output  data/benchmark_judgements.jsonl

# ──────────────────────────────────────────────────
# PHASE 5: Analysis
# ──────────────────────────────────────────────────

step "Analyzing judgements (summaries, breakdowns, judge agreement)"
python -m bench.analyze_judgements \
  --in data/benchmark_judgements.jsonl \
  --outdir data/analysis

step "Computing retrieval metrics, latency, token usage, cost-accuracy"
python -m bench.metrics \
  --results    data/benchmark_results_all.jsonl \
  --judgements data/benchmark_judgements.jsonl \
  --outdir     data/analysis

step "Running qualitative error analysis (GraphRAG vs classic RAG)"
python -m bench.error_analysis \
  --judgements data/benchmark_judgements.jsonl \
  --results    data/benchmark_results_all.jsonl \
  --outdir     data/analysis

# ──────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────

elapsed=$SECONDS
echo ""
echo "========================================"
echo "ALL DONE in $((elapsed / 60))m $((elapsed % 60))s"
echo "========================================"
echo ""
echo "Results:     data/benchmark_results.jsonl"
echo "             data/benchmark_results_graph_ablation.jsonl"
echo "             data/benchmark_results_all.jsonl"
echo "Judgements:  data/benchmark_judgements.jsonl"
echo "Analysis:    data/analysis/"
echo ""
ls -lh data/analysis/ 2>/dev/null || true
