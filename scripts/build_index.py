#!/usr/bin/env python3
"""
Build the Qdrant index from raw documents.

Usage:
    python scripts/build_index.py [--reset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.ingestion import load_documents
from pipeline.chunker import TokenChunker
from pipeline.indexer import Indexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant index from raw documents")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collections")
    parser.add_argument("--raw-dir", default=str(config.RAW_DIR), help="Path to raw docs directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not any(raw_dir.glob("*.txt")):
        print(f"ERROR: No .txt files found in {raw_dir}")
        sys.exit(1)

    print("=" * 60)
    print("STRUCTURED RAG SEARCH — Index Builder")
    print("=" * 60)

    # Step 1: Load and parse documents
    print(f"\n[1/4] Loading documents from {raw_dir}...")
    docs = load_documents(raw_dir)
    total_sections = sum(len(d.sections) for d in docs)
    print(f"  Loaded {len(docs)} documents, {total_sections} sections")
    for doc in docs:
        print(f"  - {doc.title}: {doc.token_count:,} tokens, {len(doc.sections)} sections")

    # Step 2: Chunk documents
    print(f"\n[2/4] Chunking (size={config.CHUNK_SIZE_TOKENS}, overlap={config.CHUNK_OVERLAP_TOKENS} tokens)...")
    chunker = TokenChunker()
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        print(f"  - {doc.title}: {len(chunks)} chunks")
    total_chunks = sum(len(d.chunks) for d in docs)
    print(f"  Total chunks: {total_chunks}")

    # Step 3: Set up Qdrant collections
    print(f"\n[3/4] Setting up Qdrant at {config.QDRANT_URL}...")
    indexer = Indexer()
    if args.reset:
        print("  Resetting collections...")
        indexer.reset_collections()
    else:
        indexer._ensure_collection(config.COLLECTION_SECTIONS)
        indexer._ensure_collection(config.COLLECTION_CHUNKS)

    # Step 4: Embed + index
    print("\n[4/4] Embedding and indexing...")
    indexer.index_documents(docs)

    print("\nIndex build complete.")
    print(f"  Collections: {config.COLLECTION_SECTIONS}, {config.COLLECTION_CHUNKS}")
    print(f"  Sections indexed : {total_sections}")
    print(f"  Chunks indexed   : {total_chunks}")


if __name__ == "__main__":
    main()
