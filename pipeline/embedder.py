from __future__ import annotations

import time
from typing import List

import openai

import config

_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(texts: List[str], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> List[List[float]]:
    """Embed a list of texts in batches. Returns list of float vectors."""
    all_vectors: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate each text to avoid token limit
        batch = [t[:8000] for t in batch]
        response = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        # Results are returned in the same order as input
        batch_vecs = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
        all_vectors.extend(batch_vecs)
        if i + batch_size < len(texts):
            time.sleep(0.05)  # light rate-limit guard

    return all_vectors


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
