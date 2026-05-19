#!/usr/bin/env python3
"""
Run the benchmark across all three retrieval strategies.

Usage:
    python scripts/run_benchmark.py [--strategies full_context standard_rag two_step_rag]
                                    [--qa-path data/qa_pairs.json]
                                    [--run-id my_run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from benchmark.runner import BenchmarkRunner, load_qa_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full_context", "standard_rag", "two_step_rag"],
        choices=["full_context", "standard_rag", "two_step_rag"],
        help="Which strategies to evaluate",
    )
    parser.add_argument(
        "--qa-path",
        default=str(config.DATA_DIR / "qa_pairs.json"),
        help="Path to QA pairs JSON",
    )
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    args = parser.parse_args()

    print("=" * 60)
    print("STRUCTURED RAG SEARCH — Benchmark Runner")
    print("=" * 60)

    qa_pairs = load_qa_pairs(Path(args.qa_path))
    runner = BenchmarkRunner()
    runner.run(qa_pairs=qa_pairs, strategies=args.strategies, run_id=args.run_id)


if __name__ == "__main__":
    main()
