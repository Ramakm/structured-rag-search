from __future__ import annotations

import tiktoken
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

import config
from pipeline.ingestion import Document, Section, Chunk
from pipeline.embedder import embed_texts

_enc = tiktoken.get_encoding("cl100k_base")


def _truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


class Indexer:
    def __init__(self, client: QdrantClient | None = None):
        self.client = client or QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

    def _ensure_collection(self, name: str) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=name,
                field_name="section_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def reset_collections(self) -> None:
        for name in (config.COLLECTION_SECTIONS, config.COLLECTION_CHUNKS):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self._ensure_collection(config.COLLECTION_SECTIONS)
        self._ensure_collection(config.COLLECTION_CHUNKS)

    def index_sections(self, sections: List[Section]) -> None:
        self._ensure_collection(config.COLLECTION_SECTIONS)
        texts = [
            f"{s.title}\n\n{_truncate_tokens(s.content, config.SECTION_EMBED_MAX_TOKENS)}"
            for s in sections
        ]
        print(f"  Embedding {len(sections)} sections...")
        vectors = embed_texts(texts)

        points = [
            qmodels.PointStruct(
                id=abs(hash(s.section_id)) % (2**63),
                vector=vec,
                payload={
                    "section_id": s.section_id,
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "text": _truncate_tokens(s.content, config.SECTION_EMBED_MAX_TOKENS),
                    "token_count": s.token_count,
                },
            )
            for s, vec in zip(sections, vectors)
        ]
        self.client.upsert(collection_name=config.COLLECTION_SECTIONS, points=points)
        print(f"  Upserted {len(points)} section vectors.")

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        self._ensure_collection(config.COLLECTION_CHUNKS)
        print(f"  Embedding {len(chunks)} chunks in batches of {batch_size}...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="  Indexing chunks"):
            batch = chunks[i : i + batch_size]
            vectors = embed_texts([c.text for c in batch])
            points = [
                qmodels.PointStruct(
                    id=abs(hash(c.chunk_id)) % (2**63),
                    vector=vec,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "section_id": c.section_id,
                        "section_title": c.section_title,
                        "text": c.text,
                        "token_count": c.token_count,
                        "position": c.position,
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=config.COLLECTION_CHUNKS, points=points)

        print(f"  Indexed {len(chunks)} chunk vectors.")

    def index_documents(self, docs: List[Document]) -> None:
        all_sections: List[Section] = []
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_sections.extend(doc.sections)
            all_chunks.extend(doc.chunks)

        print(f"\nIndexing {len(all_sections)} sections and {len(all_chunks)} chunks...")
        self.index_sections(all_sections)
        self.index_chunks(all_chunks)
        print("Indexing complete.")
