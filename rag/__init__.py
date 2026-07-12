"""
Core RAG package for CyberRAG-Bench experiments.

This module exposes convenience imports for pipelines and retrieval.
"""

from .pipelines import (
    AnswerResult,
    answer_zeroshot,
    answer_rag_classic,
    answer_rag_dspy,
)
from .retrieval import retrieve_context
