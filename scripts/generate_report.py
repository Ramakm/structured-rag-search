#!/usr/bin/env python3
"""
Generate dashboard charts from stored benchmark results.

Usage:
    python scripts/generate_report.py [--out-dir reports/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from dashboard.visualizer import BenchmarkVisualizer
from tracking.tracker import BenchmarkTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--out-dir", default=str(config.REPORTS_DIR), help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("STRUCTURED RAG SEARCH — Report Generator")
    print("=" * 60)

    tracker = BenchmarkTracker()
    viz = BenchmarkVisualizer(reports_dir=Path(args.out_dir))
    viz.generate_full_report(tracker)


if __name__ == "__main__":
    main()
