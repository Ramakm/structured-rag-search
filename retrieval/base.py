from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrievalResult:
    strategy: str
    query: str
    answer: str
    context_texts: List[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    chunks_used: int

    SYSTEM_PROMPT = (
        "You are a precise question-answering assistant. "
        "Answer the question using ONLY the provided context. "
        "Be concise. If the answer is not in the context, say 'Not found in context.'"
    )
