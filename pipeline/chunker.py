from __future__ import annotations

import hashlib
from typing import List

import tiktoken

import config
from pipeline.ingestion import Chunk, Document, Section

_enc = tiktoken.get_encoding("cl100k_base")


def _make_chunk_id(*parts: str) -> str:
    return hashlib.md5("||".join(parts).encode()).hexdigest()[:16]


class TokenChunker:
    """Splits sections into fixed-size token chunks with overlap."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE_TOKENS,
        overlap: int = config.CHUNK_OVERLAP_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_section(self, section: Section) -> List[Chunk]:
        tokens = _enc.encode(section.content)
        chunks: List[Chunk] = []
        start = 0
        pos = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            token_slice = tokens[start:end]
            text = _enc.decode(token_slice)
            cid = _make_chunk_id(section.section_id, str(pos))
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=section.doc_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    text=text,
                    token_count=len(token_slice),
                    position=pos,
                )
            )
            if end == len(tokens):
                break
            start = end - self.overlap
            pos += 1

        return chunks

    def chunk_document(self, doc: Document) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for section in doc.sections:
            all_chunks.extend(self.chunk_section(section))
        doc.chunks = all_chunks
        return all_chunks
