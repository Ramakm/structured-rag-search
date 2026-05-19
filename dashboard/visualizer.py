from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import config

STRATEGY_COLORS = {
    "full_context": "#e74c3c",
    "standard_rag": "#3498db",
    "two_step_rag": "#2ecc71",
}
STRATEGY_LABELS = {
    "full_context": "Full Context",
    "standard_rag": "Standard RAG",
    "two_step_rag": "2-Step RAG",
}


class BenchmarkVisualizer:
    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.out = reports_dir
        self.out.mkdir(parents=True, exist_ok=True)

    def _bar(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        col: str,
        title: str,
        ylabel: str,
    ) -> None:
        strategies = df["strategy"].tolist()
        values = df[col].tolist()
        colors = [STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]
        labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def plot_token_usage(self, summary: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Token Usage Comparison", fontsize=15, fontweight="bold")

        self._bar(axes[0], summary, "avg_input_tokens", "Avg Input Tokens / Query", "Tokens")
        self._bar(axes[1], summary, "avg_total_tokens", "Avg Total Tokens / Query", "Tokens")

        plt.tight_layout()
        path = self.out / "token_usage.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_cost(self, summary: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Cost Analysis", fontsize=15, fontweight="bold")

        self._bar(axes[0], summary, "avg_cost_usd", "Avg Cost / Query (USD)", "USD")
        self._bar(axes[1], summary, "total_cost_usd", "Total Cost Across All Queries (USD)", "USD")

        plt.tight_layout()
        path = self.out / "cost_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_latency(self, raw_df: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Latency Distribution", fontsize=15, fontweight="bold")

        strategies = raw_df["strategy"].unique().tolist()
        colors = [STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]
        labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

        for ax, col, title in zip(
            axes,
            ["latency_ms", "retrieval_ms", "generation_ms"],
            ["Total Latency (ms)", "Retrieval Latency (ms)", "Generation Latency (ms)"],
        ):
            data = [raw_df[raw_df["strategy"] == s][col].dropna().tolist() for s in strategies]
            bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(labels, rotation=10, ha="right")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("ms")
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        path = self.out / "latency_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_accuracy(self, summary: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        self._bar(ax, summary, "avg_accuracy", "Retrieval Accuracy Score (0–1)", "Score")
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        path = self.out / "accuracy.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_roi_scatter(self, summary: pd.DataFrame) -> Path:
        """Cost vs Accuracy scatter — bottom-left is worse, top-right is better."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("ROI: Cost vs. Accuracy", fontsize=15, fontweight="bold")

        for _, row in summary.iterrows():
            s = row["strategy"]
            color = STRATEGY_COLORS.get(s, "#95a5a6")
            label = STRATEGY_LABELS.get(s, s)
            ax.scatter(
                row["avg_cost_usd"] * 1000,  # millidollars for readability
                row["avg_accuracy"],
                s=200,
                color=color,
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )
            ax.annotate(
                label,
                (row["avg_cost_usd"] * 1000, row["avg_accuracy"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=10,
            )

        ax.set_xlabel("Avg Cost per Query (mUSD, lower is better)")
        ax.set_ylabel("Avg Accuracy Score (higher is better)")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate quadrants
        ax.text(0.02, 0.97, "Low cost, High accuracy\n(IDEAL)", transform=ax.transAxes,
                fontsize=8, color="green", va="top", alpha=0.6)
        ax.text(0.70, 0.05, "High cost, Low accuracy\n(WORST)", transform=ax.transAxes,
                fontsize=8, color="red", va="bottom", alpha=0.6)

        plt.tight_layout()
        path = self.out / "roi_scatter.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_token_savings(self, summary: pd.DataFrame) -> Path:
        """Show token reduction relative to full-context baseline."""
        baseline = summary[summary["strategy"] == "full_context"]["avg_total_tokens"].values
        if len(baseline) == 0:
            return None
        baseline_val = baseline[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        strategies = summary["strategy"].tolist()
        savings = [
            (baseline_val - row["avg_total_tokens"]) / baseline_val * 100
            for _, row in summary.iterrows()
        ]
        colors = [STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]
        labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

        bars = ax.bar(labels, savings, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title("Token Savings vs. Full Context Baseline (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Token Reduction (%)")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, savings):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        path = self.out / "token_savings.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_full_report(self, tracker) -> None:
        """Generate all charts and print a summary report."""
        raw_df = tracker.get_all()
        summary = tracker.get_summary()

        if raw_df.empty:
            print("No benchmark data found. Run the benchmark first.")
            return

        print("\nGenerating dashboard charts...")
        paths = []
        paths.append(self.plot_token_usage(summary))
        paths.append(self.plot_cost(summary))
        paths.append(self.plot_latency(raw_df))
        paths.append(self.plot_accuracy(summary))
        paths.append(self.plot_roi_scatter(summary))
        p = self.plot_token_savings(summary)
        if p:
            paths.append(p)

        print("\n=== REPORT GENERATED ===")
        for p in paths:
            if p:
                print(f"  {p}")

        print("\n=== FINAL SUMMARY ===")
        cols = [
            "strategy", "queries", "avg_total_tokens", "avg_cost_usd",
            "avg_latency_ms", "avg_accuracy", "avg_chunks_used",
        ]
        available = [c for c in cols if c in summary.columns]
        print(summary[available].to_string(index=False))

        # Print token savings insight
        fc_row = summary[summary["strategy"] == "full_context"]
        ts_row = summary[summary["strategy"] == "two_step_rag"]
        if not fc_row.empty and not ts_row.empty:
            tok_save = (
                1 - ts_row["avg_total_tokens"].values[0] / fc_row["avg_total_tokens"].values[0]
            ) * 100
            cost_save = (
                1 - ts_row["avg_cost_usd"].values[0] / fc_row["avg_cost_usd"].values[0]
            ) * 100
            print(f"\n  2-Step RAG saves {tok_save:.1f}% tokens and {cost_save:.1f}% cost vs Full Context.")
