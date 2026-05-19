from __future__ import annotations

import time
from pathlib import Path
from typing import List

import tiktoken

import config
from llm.openrouter import OpenRouterClient
from retrieval.base import RetrievalResult

_enc = tiktoken.get_encoding("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens]) + "\n\n[... truncated ...]"


class FullContextRetriever:
    """
    Strategy 1: Concatenate all raw document text into the prompt.
    No retrieval step — just brute-force the entire corpus into context.
    """

    STRATEGY = "full_context"

    def __init__(self, llm: OpenRouterClient | None = None):
        self.llm = llm or OpenRouterClient()
        self._corpus: str = ""
        self._corpus_tokens: int = 0

    def load_corpus(self, raw_dir: Path = config.RAW_DIR) -> None:
        texts: List[str] = []
        for path in sorted(raw_dir.glob("*.txt")):
            texts.append(f"=== {path.stem.replace('_', ' ').upper()} ===\n\n{path.read_text()}")
        self._corpus = "\n\n".join(texts)
        self._corpus_tokens = len(_enc.encode(self._corpus))

    def answer(self, query: str) -> RetrievalResult:
        if not self._corpus:
            self.load_corpus()

        context = _truncate_to_tokens(self._corpus, config.MAX_CONTEXT_TOKENS)

        system = RetrievalResult.SYSTEM_PROMPT
        user = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        t0 = time.perf_counter()
        resp = self.llm.chat(system, user)
        total_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            strategy=self.STRATEGY,
            query=query,
            answer=resp.content,
            context_texts=[context],
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            total_tokens=resp.total_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=resp.latency_ms,
            retrieval_ms=0.0,
            generation_ms=resp.latency_ms,
            chunks_used=1,
        )
