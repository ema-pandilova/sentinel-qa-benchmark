from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from rag.index import load_db


GEN_MODEL_DEFAULT = "gpt-4o"
OUT_DEFAULT = "data/benchmark.jsonl"


PROMPT_TEMPLATE = """
You are helping build a cybersecurity education benchmark.

You are given SOURCE_TEXT from an authoritative cybersecurity document.
Your task: generate EXACTLY {n} question-answer pair(s) that can be answered purely from SOURCE_TEXT.

Rules:
- The answer MUST be grounded in SOURCE_TEXT (no extra facts).
- Keep language consistent with SOURCE_TEXT.
{difficulty_instruction}
- Vary audience: child / parent / student / general.
- Output MUST be valid JSON only (no markdown, no commentary).
- Output format: a JSON list of objects, each with:
  - id_suffix (short string, like "a")
  - question
  - reference_answer
  - difficulty  (easy|medium|hard)
  - audience    (child|parent|student|general)
  - language    (e.g., en, mk, sq, sr, etc. or full language name)

IMPORTANT: Output JSON ONLY.
"""


def extract_json(text: str) -> Any:
    """
    Try to parse JSON robustly, even if the model adds extra text.
    Supports:
      - list JSON: [ ... ]
      - object JSON: { ... }  (we'll wrap into list)
    """
    text = (text or "").strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try find a JSON array
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return json.loads(m.group(0))

    # Try find a JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        obj = json.loads(m.group(0))
        return [obj]

    raise ValueError("Could not extract JSON from model output")


def pick_representative_chunks(
    docs: List[str],
    metas: List[Dict[str, Any]],
    prefer_unique_by: str = "source",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build a list of (text, meta) selecting 1 representative chunk per unique key.

    prefer_unique_by:
      - "source" (best: one per original file)
      - "doc_id" (fallback)
    Strategy:
      - group by key
      - pick the longest chunk as representative (usually more informative)
    """
    groups: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for text, meta in zip(docs, metas):
        if not text or not meta:
            continue

        key = meta.get(prefer_unique_by) or meta.get("doc_id") or meta.get("source")
        if not key:
            continue
        key = str(key)

        # choose the longest chunk per key
        prev = groups.get(key)
        if prev is None or len(text) > len(prev[0]):
            groups[key] = (text, meta)

    # return deterministic order (then shuffle later)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="Output benchmark jsonl path")
    ap.add_argument("--model", default=GEN_MODEL_DEFAULT, help="Generator model (OpenAI)")
    ap.add_argument(
        "--n_questions",
        type=int,
        default=100,
        help="How many benchmark items to generate in total",
    )
    ap.add_argument(
        "--unique_by",
        choices=["source", "doc_id"],
        default="source",
        help="Ensure diversity by unique source file (preferred) or doc_id",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling sources",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=4000,
        help="Truncate SOURCE_TEXT to this many characters to avoid huge prompts",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generator temperature",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per document if JSON parse fails",
    )
    ap.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Force all generated questions to this difficulty level",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = load_db()

    # pull full corpus from chroma
    data = db.get(include=["metadatas", "documents"])
    docs: List[str] = data.get("documents") or []
    metas: List[Dict[str, Any]] = data.get("metadatas") or []

    if not docs:
        raise RuntimeError("Chroma DB returned 0 documents. Is your CHROMA_PATH correct?")

    # pick 1 representative chunk per unique source/doc_id
    reps = pick_representative_chunks(docs, metas, prefer_unique_by=args.unique_by)
    if not reps:
        raise RuntimeError("Could not find representative chunks with usable metadata.")

    # shuffle and take the first n
    random.shuffle(reps)

    if args.n_questions > len(reps):
        print(
            f"[WARN] Requested n_questions={args.n_questions}, but only have {len(reps)} unique '{args.unique_by}'. "
            f"Will generate only {len(reps)} questions (one per unique doc)."
        )
        target = len(reps)
    else:
        target = args.n_questions

    reps = reps[:target]

    model = ChatOpenAI(model=args.model, temperature=args.temperature)

    written = 0
    file_mode = "a" if args.append else "w"
    with out_path.open(file_mode, encoding="utf-8") as f_out:
        for idx, (text, meta) in enumerate(reps, start=1):
            source = meta.get("source")
            doc_id = meta.get("doc_id")
            chunk_id = meta.get("chunk_id")

            # create a stable benchmark id
            # prefer source filename stem if possible
            src_key = str(source or doc_id or f"doc_{idx}")
            safe_key = re.sub(r"[^a-zA-Z0-9]+", "-", src_key)[-80:].strip("-")
            bench_id_prefix = f"{safe_key}"

            source_text = (text or "")[: args.max_chars]

            if args.difficulty:
                difficulty_instruction = f"- ALL questions MUST have difficulty: {args.difficulty}."
            else:
                difficulty_instruction = "- Vary difficulty: easy / medium / hard."

            prompt = PROMPT_TEMPLATE.format(n=1, difficulty_instruction=difficulty_instruction) + f"\n\nSOURCE_TEXT:\n{source_text}\n"
            qa_list: Optional[List[Dict[str, Any]]] = None

            for attempt in range(args.retries + 1):
                resp = model.invoke([{"role": "system", "content": prompt}])
                try:
                    parsed = extract_json(resp.content)
                    if isinstance(parsed, dict):
                        qa_list = [parsed]
                    elif isinstance(parsed, list):
                        qa_list = parsed
                    else:
                        qa_list = None
                    if qa_list:
                        break
                except Exception:
                    qa_list = None

                if attempt < args.retries:
                    # slightly nudge on retry
                    prompt_retry = prompt + "\nREMINDER: Output JSON ONLY. No markdown.\n"
                    prompt = prompt_retry

            if not qa_list:
                print(f"[WARN] Skipping source={source} doc_id={doc_id} (could not parse JSON)")
                continue

            qa = qa_list[0]

            item = {
                "id": f"{bench_id_prefix}-a",
                "question": qa.get("question", "").strip(),
                "reference_answer": qa.get("reference_answer", "").strip(),
                "difficulty": args.difficulty or (qa.get("difficulty") or "medium").strip(),
                "audience": (qa.get("audience") or "general").strip(),
                "language": (qa.get("language") or "").strip(),
                "source": source,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "unique_by": args.unique_by,
                "generator_model": args.model,
            }

            # basic validation
            if not item["question"] or not item["reference_answer"]:
                print(f"[WARN] Invalid QA (empty question/answer) for source={source} doc_id={doc_id}")
                continue

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Wrote {written} benchmark items to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
