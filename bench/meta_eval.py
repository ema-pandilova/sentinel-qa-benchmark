import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from bench.judge_prompt_factual import JUDGE_PROMPT_FACTUAL
from bench.judge_prompt_ground import JUDGE_PROMPT
from rag.llm_factory import make_chat_model


JUDGE_MODELS: List[Dict[str, str]] = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "deepseek", "model": "deepseek-chat"},
]


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"Invalid judge JSON:\n{text}")


def _load_done_judgements(path: Path) -> Tuple[Set[Tuple[str, str, str]], int]:
    """
    Load already-completed (item_id, mode, judge_model) triples from output.
    Only counts non-error judgements as done (errors will be retried).
    Returns (done_keys, total_lines_to_keep).
    """
    done = set()
    keep_lines = []

    if not path.exists():
        return done, 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                item_id = rec["benchmark_item"].get("id", "")
                mode = rec["answer_result"].get("mode", "")
                judge_model = rec.get("judge_model", "")

                # Check if this was an error — don't count it as done
                sf = rec.get("scores_factual", {}) or {}
                sg = rec.get("scores_ground", {}) or {}
                is_error = (
                    (isinstance(sf.get("comments"), str) and sf["comments"].startswith("JUDGE_ERROR:"))
                    or (isinstance(sg.get("comments"), str) and sg["comments"].startswith("JUDGE_ERROR:"))
                )

                if not is_error:
                    done.add((item_id, mode, judge_model))
                    keep_lines.append(line)
                # Drop error lines — they'll be retried
            except Exception:
                continue

    # Rewrite file without error lines
    if keep_lines:
        with path.open("w", encoding="utf-8") as f:
            for line in keep_lines:
                f.write(line)

    return done, len(keep_lines)


def run_meta_eval(
    results_path: str = "data/benchmark_results.jsonl",
    output_path: str = "data/benchmark_judgements.jsonl",
):
    in_path = Path(results_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Resume support: load existing successful judgements, strip errors
    done_keys, kept = _load_done_judgements(out_path)
    if done_keys:
        print(f"  Resuming: {kept} successful judgements kept, errors will be retried")

    total = 0
    skipped = 0
    errors = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "a", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)
            bench = record["benchmark_item"]
            ans = record["answer_result"]

            item_id = bench.get("id", "")
            mode = ans.get("mode", "unknown")
            is_zeroshot = mode.startswith("zeroshot")

            prompt_factual = (
                JUDGE_PROMPT_FACTUAL
                + f"""
QUESTION:
{bench["question"]}

REFERENCE_ANSWER:
{bench["reference_answer"]}

SYSTEM_ANSWER:
{ans["answer"]}
"""
            )

            if not is_zeroshot:
                prompt_ground = (
                    JUDGE_PROMPT
                    + f"""
QUESTION:
{bench["question"]}

REFERENCE_ANSWER:
{bench["reference_answer"]}

CONTEXT:
{ans["context"]}

SYSTEM_ANSWER:
{ans["answer"]}
"""
                )

            for jm in JUDGE_MODELS:
                # Skip if already successfully judged
                if (item_id, mode, jm["model"]) in done_keys:
                    skipped += 1
                    continue

                try:
                    judge = make_chat_model(
                        provider=jm["provider"],
                        model_name=jm["model"],
                        temperature=0.0,
                    )

                    resp_factual = judge.invoke(
                        [{"role": "system", "content": prompt_factual}]
                    )
                    scores_factual = parse_json(resp_factual.content)

                    if is_zeroshot:
                        scores_ground = {
                            "grounding_ratio": None,
                            "hallucination_rate": None,
                            "comments": "SKIPPED: grounding not applicable for zeroshot mode (no retrieved context)",
                        }
                    else:
                        resp = judge.invoke(
                            [{"role": "system", "content": prompt_ground}]
                        )
                        scores_ground = parse_json(resp.content)
                        if scores_ground.get("grounding_ratio") is not None:
                            scores_ground["hallucination_rate"] = 1.0 - scores_ground["grounding_ratio"]

                    total += 1
                    print(f"  [{total}] {mode} | judge={jm['provider']}/{jm['model']} | {item_id[:40]}")

                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] {mode} | judge={jm['provider']}/{jm['model']} | {item_id[:40]} | {e}")
                    scores_ground = {
                        "grounding_ratio": None,
                        "hallucination_rate": None,
                        "comments": f"JUDGE_ERROR: {type(e).__name__}: {e}",
                    }
                    scores_factual = {
                        "factual_correctness_ratio": None,
                        "comments": f"JUDGE_ERROR: {type(e).__name__}: {e}",
                    }

                fout.write(
                    json.dumps(
                        {
                            "benchmark_item": bench,
                            "answer_result": ans,
                            "judge_provider": jm["provider"],
                            "judge_model": jm["model"],
                            "scores_ground": scores_ground,
                            "scores_factual": scores_factual,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"\n[OK] Meta-eval: {total} new + {skipped} skipped ({errors} errors)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    ap.add_argument("--results", default="data/benchmark_results.jsonl",
                    help="Path to benchmark results JSONL")
    ap.add_argument("--output", default="data/benchmark_judgements.jsonl",
                    help="Path to output judgements JSONL")
    args = ap.parse_args()
    run_meta_eval(results_path=args.results, output_path=args.output)
