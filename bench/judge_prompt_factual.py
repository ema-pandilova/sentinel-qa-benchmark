JUDGE_PROMPT_FACTUAL =  """
You are evaluating the factual correctness of an answer produced by a language model.

You are given:
- QUESTION
- SYSTEM_ANSWER (the generated answer)
- REFERENCE_ANSWER (an ideal answer based on authoritative sources)

IMPORTANT CONSTRAINT:
- You MUST NOT assume access to any retrieved context.
- You MUST evaluate factual correctness using ONLY the QUESTION and REFERENCE_ANSWER.
- Do NOT use external knowledge beyond what is reasonably implied by the REFERENCE_ANSWER.

Your task:

STEP 1: Claim Extraction
- Split the SYSTEM_ANSWER into individual sentences.
- Treat each sentence as a distinct factual claim.
- Include all sentences as claims for verification, without exclusion.


STEP 2: Claim Evaluation
For EACH extracted claim:
- Decide whether the claim is factually correct according to the REFERENCE_ANSWER.
- A claim is correct ONLY if it is explicitly supported or clearly implied by the REFERENCE_ANSWER.
- If the REFERENCE_ANSWER does not mention the claim, mark it as incorrect.

STEP 3: Scoring
- Count the total number of factual claims.
- Count how many claims are factually correct.
- Compute:
  factual_correctness_ratio = correct_claims / total_claims

Return a STRICT JSON object with the following schema and NOTHING else:

{
  "claims": [
    {
      "claim": "<string>",
      "correct": <true|false>
    }
  ],
  "total_claims": <int>,
  "correct_claims": <int>,
  "factual_correctness_ratio": <float between 0 and 1>,
  "comments": "<brief explanation of major errors or strengths, 1–2 sentences>"
}

IMPORTANT RULES:
- Do NOT consider whether the answer is grounded in any retrieved context.
- Penalize unsupported or fabricated claims, even if they sound plausible.
- If there are zero factual claims, set factual_correctness_ratio = 1.0.
"""
