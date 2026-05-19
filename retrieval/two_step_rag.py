from __future__ import annotations

import time
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config
from llm.openrouter import OpenRouterClient
from pipeline.embedder import embed_query
from retrieval.base import RetrievalResult


class TwoStepRAG:
    """
    Strategy 3: Coarse-to-fine hierarchical retrieval.

    Step 1 (Coarse): embed query → search sections_coarse → get top-K section IDs.
    Step 2 (Fine):   embed query → search chunks WITH payload filter on section_id
                     → get top-K fine chunks from relevant sections only.

    Benefits:
    - Reduces noise: only chunks within semantically relevant sections are considered.
    - Enables payload filtering in Qdrant (structured retrieval).
    - Lower token usage vs. full-context; better precision vs. standard RAG.
    """

    STRATEGY = "two_step_rag"

    def __init__(
        self,
        llm: OpenRouterClient | None = None,
        qdrant: QdrantClient | None = None,
        top_k_coarse: int = config.TOP_K_COARSE_SECTIONS,
        top_k_fine: int = config.TOP_K_FINE_CHUNKS,
    ):
        self.llm = llm or OpenRouterClient()
        self.qdrant = qdrant or QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.top_k_coarse = top_k_coarse
        self.top_k_fine = top_k_fine

    def answer(self, query: str) -> RetrievalResult:
        t_ret0 = time.perf_counter()
        q_vec = embed_query(query)

        # Step 1 — Coarse: find relevant sections
        coarse_hits = self.qdrant.search(
            collection_name=config.COLLECTION_SECTIONS,
            query_vector=q_vec,
            limit=self.top_k_coarse,
            with_payload=True,
        )
        section_ids: List[str] = [h.payload["section_id"] for h in coarse_hits]

        # Step 2 — Fine: search chunks filtered to those sections
        section_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="section_id",
                    match=qmodels.MatchAny(any=section_ids),
                )
            ]
        )
        fine_hits = self.qdrant.search(
            collection_name=config.COLLECTION_CHUNKS,
            query_vector=q_vec,
            query_filter=section_filter,
            limit=self.top_k_fine,
            with_payload=True,
        )
        retrieval_ms = (time.perf_counter() - t_ret0) * 1000

        context_texts = [h.payload["text"] for h in fine_hits]
        context = "\n\n---\n\n".join(
            f"[Section: {h.payload['section_title']}]\n{h.payload['text']}" for h in fine_hits
        )

        # Generation
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
            chunks_used=len(fine_hits),
        )
