from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import config
from benchmark.evaluator import score_answer
from llm.openrouter import OpenRouterClient
from pipeline.embedder import embed_query
from retrieval.full_context import FullContextRetriever
from retrieval.standard_rag import StandardRAG
from retrieval.two_step_rag import TwoStepRAG
from tracking.tracker import BenchmarkTracker, RunRecord


@dataclass
class QAPair:
    question: str
    expected_answer: str
    keywords: List[str]
    category: str


def load_qa_pairs(path: Path = config.DATA_DIR / "qa_pairs.json") -> List[QAPair]:
    data = json.loads(path.read_text())
    return [
        QAPair(
            question=d["question"],
            expected_answer=d["expected_answer"],
            keywords=d.get("keywords", []),
            category=d.get("category", "general"),
        )
        for d in data
    ]


class BenchmarkRunner:
    def __init__(self, tracker: BenchmarkTracker | None = None):
        self.tracker = tracker or BenchmarkTracker()
        llm = OpenRouterClient()

        self.strategies = {
            "full_context": FullContextRetriever(llm=llm),
            "standard_rag": StandardRAG(llm=llm),
            "two_step_rag": TwoStepRAG(llm=llm),
        }

    def run(
        self,
        qa_pairs: List[QAPair] | None = None,
        strategies: List[str] | None = None,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        if qa_pairs is None:
            qa_pairs = load_qa_pairs()
        if strategies is None:
            strategies = list(self.strategies.keys())
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        print(f"\nBenchmark run: {run_id}")
        print(f"Strategies: {strategies}")
        print(f"QA pairs:   {len(qa_pairs)}")
        print(f"Total calls: {len(qa_pairs) * len(strategies)}\n")

        records: List[RunRecord] = []

        for qa in tqdm(qa_pairs, desc="QA pairs"):
            for strat_name in strategies:
                retriever = self.strategies[strat_name]
                try:
                    result = retriever.answer(qa.question)
                    acc = score_answer(result.answer, qa.expected_answer, qa.keywords)

                    rec = RunRecord(
                        run_id=run_id,
                        strategy=strat_name,
                        query=qa.question,
                        answer=result.answer,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        total_tokens=result.total_tokens,
                        cost_usd=result.cost_usd,
                        latency_ms=result.latency_ms,
                        retrieval_ms=result.retrieval_ms,
                        generation_ms=result.generation_ms,
                        accuracy_score=acc,
                        chunks_used=result.chunks_used,
                    )
                    self.tracker.log(rec)
                    records.append(rec)
                except Exception as e:
                    print(f"\n  ERROR [{strat_name}] '{qa.question[:50]}': {e}")

        summary = self.tracker.get_summary()
        print("\n=== BENCHMARK SUMMARY ===")
        print(summary.to_string(index=False))
        return summary
