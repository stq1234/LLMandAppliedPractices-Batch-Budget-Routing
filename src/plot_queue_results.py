"""Plot shared-queue deadline outcomes separately from simulated service time."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
POLICIES = ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off")
COLORS = {"fifo": "#7b8794", "value_first": "#ed8b00", "adaptive": "#1f77b4",
          "adaptive_noisy_signal_off": "#2e8b57", "first_round_only": "#9b59b6"}


def read_rows() -> list[dict]:
    with (RESULTS / "queue_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_on_time_quality() -> None:
    rows = read_rows()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True, sharey=True)
    for ax, (split, condition) in zip(axes.ravel(),
                                     (("test", "standard"), ("test", "tight"),
                                      ("shift", "standard"), ("shift", "tight"))):
        current = [r for r in rows if r["experiment"] == split and r["deadline_condition"] == condition]
        for policy in POLICIES:
            points = sorted((r for r in current if r["policy"] == policy),
                            key=lambda r: int(r["budget_units_per_batch"]))
            ax.plot([int(r["budget_units_per_batch"]) for r in points],
                    [float(r["on_time_exact_value_coverage"]) for r in points],
                    marker="o", linewidth=1.8, color=COLORS[policy], label=policy)
        first = next((r for r in current if r["policy"] == "first_round_only"), None)
        if first:
            ax.axhline(float(first["on_time_exact_value_coverage"]), color=COLORS["first_round_only"],
                       linestyle="--", linewidth=1.4, label="first_round_only")
        ax.set_title(f"{split} / {condition} deadlines")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Budget units per 8-request batch")
    for ax in axes[:, 0]:
        ax.set_ylabel("On-time exact value coverage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exact value delivered before response deadlines\n"
                 "6 synchronous mini-batches × 8 requests; 4 parallel service slots")
    fig.tight_layout()
    fig.savefig(FIGURES / "queue_deadline_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_waiting_and_timeouts() -> None:
    rows = read_rows()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    focus = [r for r in rows if r["experiment"] == "test" and r["policy"] in POLICIES]
    for condition in ("standard", "tight"):
        for policy in ("fifo", "value_first", "adaptive"):
            points = sorted((r for r in focus if r["deadline_condition"] == condition
                             and r["policy"] == policy),
                            key=lambda r: int(r["budget_units_per_batch"]))
            label = f"{policy} / {condition}"
            axes[0].plot([int(r["budget_units_per_batch"]) for r in points],
                         [float(r["mean_queue_wait_ms_per_request"]) for r in points],
                         marker="o", linewidth=1.6, label=label)
            axes[1].plot([int(r["budget_units_per_batch"]) for r in points],
                         [float(r["timeout_rate"]) for r in points],
                         marker="o", linewidth=1.6, label=label)
    axes[0].set_ylabel("Mean queue waiting time (simulated ms)")
    axes[1].set_ylabel("Timed-out requests / all requests")
    for ax in axes:
        ax.set_xlabel("Budget units per mini-batch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Queue delay and deadline failures (baseline test)")
    fig.tight_layout()
    fig.savefig(FIGURES / "queue_waiting_timeouts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_on_time_quality()
    plot_waiting_and_timeouts()
    print("Wrote shared-queue figures to results/figures")

