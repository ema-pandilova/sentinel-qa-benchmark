from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time

import dspy
import tiktoken
from .config import ZEROSHOT_MODEL, RAG_MODEL, DSPY_MODEL, DEFAULT_K
from .prompts import build_prompt_regular
from .retrieval import retrieve_context, docs_to_metadata, fetch_docs_by_ids
from .llm_factory import make_chat_model
from rag.graph.expand import expand_chunks
from rag.graph.graph_store import load_graph

@dataclass
class AnswerResult:
    question_id: str
    mode: str
    question: str
    context: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    latency_sec: float
    model_name: str
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _invoke_chat(model, prompt: str) -> tuple[str, Optional[Dict[str, Any]]]:
    resp = model.invoke([{"role": "system", "content": prompt}])

    usage = None
    if hasattr(resp, "response_metadata") and resp.response_metadata is not None:
        token_usage = resp.response_metadata.get("token_usage")
        if token_usage:
            usage = {
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            }

    return resp.content, usage



def answer_graph_rag(
    question_id: str,
    question: str,
    k: int = DEFAULT_K,
    provider: str = "openai",
    model_name: Optional[str] = None,
    graph_path: Optional[str] = None,
    max_nodes: int = 10,
) -> AnswerResult:
    chosen_model = model_name or RAG_MODEL
    model = make_chat_model(
        provider=provider,
        model_name=chosen_model,
        temperature=0.1,
    )

    # 1. Vector retrieval
    docs_with_scores = retrieve_context(question, k=k, return_scores=True)

    if not docs_with_scores:
        expanded_text = ""
        retrieved_docs: List[Dict[str, Any]] = []
    else:
        # 2. Extract chunk IDs (Chroma stores chunk_id in "id" metadata field)
        retrieved_ids = [
            doc.metadata.get("chunk_id") or doc.metadata.get("id")
            for doc, _ in docs_with_scores
        ]

        retrieved_docs = docs_to_metadata(docs_with_scores)

        # 3. Graph expansion (ID-based)
        graph = load_graph(path=graph_path)
        expanded_ids = expand_chunks(
            retrieved_ids,
            graph,
            max_nodes=max_nodes,
        )

        # 4. Fetch text for expanded nodes
        expanded_docs = fetch_docs_by_ids(expanded_ids)
        expanded_text = "\n\n---\n\n".join(
            doc.page_content for doc in expanded_docs
        )
    # 5. Prompt
    prompt = build_prompt_regular(
        context=expanded_text,
        question=question,
        history="",
    )

    start = time.time()
    answer, usage = _invoke_chat(model, prompt)
    latency = time.time() - start

    return AnswerResult(
        question_id=question_id,
        mode=f"rag_graph:{provider}:k={k}:mn={max_nodes}",
        question=question,
        context=expanded_text,
        answer=answer,
        retrieved_docs=retrieved_docs,
        latency_sec=latency,
        model_name=f"{provider}/{chosen_model}",
        usage=usage,
    )

    

def answer_zeroshot(
    question_id: str,
    question: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> AnswerResult:
    chosen_model = model_name or ZEROSHOT_MODEL
    model = make_chat_model(provider=provider, model_name=chosen_model, temperature=0.1)

    prompt = build_prompt_regular(
        context="No external context was retrieved. Answer using your general knowledge and safe cybersecurity practices.",
        question=question,
        history="",
    )

    start = time.time()
    answer, usage = _invoke_chat(model, prompt)
    latency = time.time() - start

    return AnswerResult(
        question_id=question_id,
        mode=f"zeroshot:{provider}",
        question=question,
        context="",
        answer=answer,
        retrieved_docs=[],
        latency_sec=latency,
        model_name=f"{provider}/{chosen_model}",
        usage=usage,
    )


# --- RAG Classic ---


def answer_rag_classic(
    question_id: str,
    question: str,
    k: int = DEFAULT_K,
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> AnswerResult:
    chosen_model = model_name or RAG_MODEL
    model = make_chat_model(provider=provider, model_name=chosen_model, temperature=0.1)

    docs_with_scores = retrieve_context(question, k=k, return_scores=True)  # type: ignore[arg-type]

    if not docs_with_scores:
        context_text = ""
        retrieved_docs: List[Dict[str, Any]] = []
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in docs_with_scores])  # type: ignore[index]
        retrieved_docs = docs_to_metadata(docs_with_scores)  # type: ignore[arg-type]
    prompt = build_prompt_regular(context=context_text, question=question, history="")

    start = time.time()
    answer, usage = _invoke_chat(model, prompt)
    latency = time.time() - start

    return AnswerResult(
        question_id=question_id,
        mode=f"rag_classic:{provider}:k={k}",
        question=question,
        context=context_text,
        answer=answer,
        retrieved_docs=retrieved_docs,
        latency_sec=latency,
        model_name=f"{provider}/{chosen_model}",
        usage=usage,
    )


# DSPy 
class RAGQuery(dspy.Signature):
    context = dspy.InputField(desc="Retrieved cybersecurity context from the knowledge base.")
    question = dspy.InputField(desc="User question about cybersecurity or digital safety.")
    answer = dspy.OutputField(desc="Helpful, safe, grounded answer in the user's language.")


class RAGModel(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RAGQuery)

    def forward(self, context: str, question: str):
        return self.predict(context=context, question=question)


def configure_dspy_openai(model_name: str, temperature: float = 0.1) -> None:
    lm = dspy.LM(model=f"openai/{model_name}", temperature=temperature)
    dspy.configure(lm=lm)


configure_dspy_openai(DSPY_MODEL, temperature=0.1)
_dspy_rag_model = RAGModel()

def estimate_usage(prompt: str, answer: str, model_name: str) -> Dict[str, Any]:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(answer))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

def answer_rag_dspy(
    question_id: str,
    question: str,
    k: int = DEFAULT_K,
    dspy_model_name: Optional[str] = None,
) -> AnswerResult:
    if dspy_model_name and dspy_model_name != DSPY_MODEL:
        configure_dspy_openai(dspy_model_name, temperature=0.1)

    docs_with_scores = retrieve_context(question, k=k, return_scores=True)  # type: ignore[arg-type]

    if not docs_with_scores:
        context_text = ""
        retrieved_docs: List[Dict[str, Any]] = []
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in docs_with_scores])  # type: ignore[index]
        retrieved_docs = docs_to_metadata(docs_with_scores)  # type: ignore[arg-type]

    start = time.time()
    out = _dspy_rag_model(context=context_text, question=question)
    latency = time.time() - start

    answer_text = getattr(out, "answer", "") or ""
    usage = estimate_usage(
        prompt=context_text+question,
        answer=answer_text,
        model_name=dspy_model_name or DSPY_MODEL,
    )

    return AnswerResult(
        question_id=question_id,
        mode=f"rag_dspy:openai:k={k}",
        question=question,
        context=context_text,
        answer=answer_text,
        retrieved_docs=retrieved_docs,
        latency_sec=latency,
        model_name=f"dspy/openai/{dspy_model_name or DSPY_MODEL}",
        usage=usage,
    )
