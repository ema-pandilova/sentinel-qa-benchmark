from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any

Audience = Literal["child", "parent", "student", "general"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class BenchmarkItem:
    id: str
    question: str
    language: str        # e.g., "en", "mk", "al", "sr", etc.
    difficulty: Difficulty
    audience: Audience   # target audience
    reference_answer: str
    reference_doc_ids: List[str]  # list of doc_ids that support the reference answer
    tags: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None
