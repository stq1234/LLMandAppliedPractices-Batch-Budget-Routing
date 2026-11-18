"""Render summary figures from results/summary.csv and allocation diagnostics."""

from __future__ import annotations

import csv
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_quality() -> None:
    rows = read_csv(RESULTS / "summary.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {"fifo": "#7b8794", "value_first": "#ed8b00", "adaptive": "#1f77b4",
              "adaptive_noisy_signal_off": "#2e8b57"}
    for ax, split in zip(axes, ("test", "shift")):
        for policy in ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off"):
            points = [r for r in rows if r["experiment"] == split
                      and r["variant"] == "normal" and r["policy"] == policy]
            points.sort(key=lambda r: int(r["budget_units"]))
            ax.plot([int(r["budget_units"]) for r in points],
                    [float(r["value_weighted_exact_rate"]) for r in points],
                    marker="o", linewidth=2, color=colors[policy], label=policy)
        ax.set_title("Baseline test" if split == "test" else "Distribution-shift test")
        ax.set_xlabel("Batch policy budget (resource units)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Value-weighted exact-answer rate")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Quality under a shared batch budget")
    fig.tight_layout()
    fig.savefig(FIGURES / "budget_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_allocation_sensitivity() -> None:
    rows = read_csv(RESULTS / "summary.csv")
    focus = [r for r in rows if r["experiment"] == "test"
             and r["variant"] in ("normal", "reversed_order", "value_order_swap")
             and int(r["budget_units"]) == 96]
    variants = ["normal", "reversed_order", "value_order_swap"]
    labels = {"normal": "Original order/value", "reversed_order": "Reversed order",
              "value_order_swap": "Values swapped + order changed"}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    policies = ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off")
    width = 0.18
    xs = list(range(len(variants)))
    for j, policy in enumerate(policies):
        vals = []
        for variant in variants:
            item = next((r for r in focus if r["variant"] == variant and r["policy"] == policy), None)
            vals.append(float(item["spend_on_top_quartile_value_fraction"]) if item else 0)
        ax.bar([x + (j - 1.5) * width for x in xs], vals, width=width,
               label=policy, color={"fifo": "#7b8794", "value_first": "#ed8b00",
                                    "adaptive": "#1f77b4", "adaptive_noisy_signal_off": "#2e8b57"}[policy])
    ax.set_xticks(xs, [labels[v] for v in variants])
    ax.set_ylabel("Share of action units spent on highest-value quartile")
    ax.set_ylim(0, 1)
    ax.set_title("Value and order intervention at 96 resource units")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "allocation_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_resources() -> None:
    rows = read_csv(RESULTS / "summary.csv")
    focus = [r for r in rows if r["variant"] == "normal" and int(r["budget_units"]) == 144]
    policies = ("fifo", "value_first", "adaptive", "adaptive_noisy_signal_off")
    actions = ("generate", "query", "execute", "validate", "revise", "rollback")
    colors = {"generate": "#4c78a8", "query": "#f2a541", "execute": "#72b7b2",
              "validate": "#e45756", "revise": "#54a24b", "rollback": "#b279a2"}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels, bottom = [], [0.0] * (2 * len(policies))
    split_order = ("test", "shift")
    all_rows = []
    for split in split_order:
        for policy in policies:
            item = next((r for r in focus if r["experiment"] == split and r["policy"] == policy), None)
            all_rows.append(item)
            labels.append(f"{split}\n{policy.replace('_', ' ')}")
    xs = list(range(len(all_rows)))
    for action in actions:
        vals = []
        for row in all_rows:
            counts = ast.literal_eval(row["action_counts"]) if row else {}
            total = sum(counts.values())
            vals.append(counts.get(action, 0) / total if total else 0)
        ax.bar(xs, vals, bottom=bottom, label=action, color=colors[action], width=0.72)
        bottom = [x + y for x, y in zip(bottom, vals)]
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("Share of policy resource units")
    ax.set_title("Action mix under distribution shift (budget 144) ")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_quality()
    plot_allocation_sensitivity()
    plot_resources()
    print(f"Wrote figures to {FIGURES.relative_to(ROOT)}")

