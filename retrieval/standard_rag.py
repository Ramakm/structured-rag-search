from __future__ import annotations

import time

from qdrant_client import QdrantClient

import config
from llm.openrouter import OpenRouterClient
from pipeline.embedder import embed_query
from retrieval.base import RetrievalResult


class StandardRAG:
    """
    Strategy 2: Flat vector search over all chunks.
    Embed query → top-K chunks → answer.
    """

    STRATEGY = "standard_rag"

    def __init__(
        self,
        llm: OpenRouterClient | None = None,
        qdrant: QdrantClient | None = None,
        top_k: int = config.TOP_K_STANDARD,
    ):
        self.llm = llm or OpenRouterClient()
        self.qdrant = qdrant or QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.top_k = top_k

    def answer(self, query: str) -> RetrievalResult:
        # Retrieval step
        t_ret0 = time.perf_counter()
        q_vec = embed_query(query)
        hits = self.qdrant.search(
            collection_name=config.COLLECTION_CHUNKS,
            query_vector=q_vec,
            limit=self.top_k,
            with_payload=True,
        )
        retrieval_ms = (time.perf_counter() - t_ret0) * 1000

        context_texts = [h.payload["text"] for h in hits]
        context = "\n\n---\n\n".join(
            f"[Section: {h.payload['section_title']}]\n{h.payload['text']}" for h in hits
        )

        # Generation step
        system = RetrievalResult.SYSTEM_PROMPT
        user = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        t_gen0 = time.perf_counter()
        resp = self.llm.chat(system, user)
        generation_ms = (time.perf_counter() - t_gen0) * 1000

        return RetrievalResult(
            strategy=self.STRATEGY,
            query=query,
            answer=resp.content,
            context_texts=context_texts,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            total_tokens=resp.total_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=retrieval_ms + generation_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            chunks_used=len(hits),
        )
