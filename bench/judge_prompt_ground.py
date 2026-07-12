JUDGE_PROMPT = """
You are evaluating the faithfulness of an answer in a Retrieval-Augmented Generation (RAG) system.

You are given:
- QUESTION
- SYSTEM_ANSWER (the generated answer)
- CONTEXT (retrieved text passages used by the system)

Your task is to evaluate how well the SYSTEM_ANSWER is grounded in the provided CONTEXT.

Follow these steps STRICTLY:

STEP 1: Claim Extraction
- Split the SYSTEM_ANSWER into individual sentences.
- Treat each sentence as a distinct factual claim.
- Include all sentences as claims for verification, without exclusion.


STEP 2: Claim Verification
For EACH extracted claim:
- Decide whether the claim is explicitly supported by the CONTEXT.
- A claim is supported ONLY if the information is clearly stated or directly implied in the CONTEXT.
- World knowledge outside the CONTEXT must NOT be used.

STEP 3: Scoring
- Count the total number of claims.
- Count how many claims are supported by the CONTEXT.
- Compute:
  grounding_ratio = supported_claims / total_claims
  hallucination_rate = 1 - grounding_ratio

Return a STRICT JSON object with the following schema and NOTHING else:

{
  "claims": [
    {
      "claim": "<string>",
      "supported": <true|false>
    }
  ],
  "total_claims": <int>,
  "supported_claims": <int>,
  "grounding_ratio": <float between 0 and 1>,
  "hallucination_rate": <float between 0 and 1>
}

IMPORTANT RULES:
- Use ONLY the provided CONTEXT to judge support.
- Do NOT assess factual correctness beyond the CONTEXT.
- If there are zero factual claims, set grounding_ratio = 1.0 and hallucination_rate = 0.0.
"""
