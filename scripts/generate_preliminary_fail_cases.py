#!/usr/bin/env python3
"""Generate fail-case image for Preliminary and Motivation section."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LINE_COLOR = "#3868c0"
GRID_COLOR = "#d8e0f0"
BAD_COLOR = "#cc332e"
GOOD_COLOR = "#36873a"


def _series(case: int, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(120 + case)
    x = np.arange(n)
    y = np.full(n, 0.62) + rng.normal(0, 0.002, size=n)

    if case == 1:
        # Minor jitter: should be ignored but is falsely retained.
        y[42:48] += np.array([0.01, 0.015, 0.02, 0.013, 0.008, 0.003])
    elif case == 2:
        # Historical comparison case: current event stronger but retrieval misses context.
        y[70:86] += np.concatenate([np.linspace(0.0, 0.045, 5), np.full(6, 0.045), np.linspace(0.045, 0.0, 5)])
    elif case == 3:
        # Distractor case: one target + one distractor.
        y[30:44] += np.concatenate([np.linspace(0.0, 0.028, 4), np.full(6, 0.028), np.linspace(0.028, 0.0, 4)])
        y[82:94] += np.concatenate([np.linspace(0.0, 0.042, 4), np.full(4, 0.042), np.linspace(0.042, 0.0, 4)])
    return x, y


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / "preliminary_fail_cases"

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": "#9aa8c0",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.7), sharey=True)
    titles = ["Case A: Relative Under-grounding", "Case B: Missing Context Retrieval", "Case C: Distractor Retention"]
    notes = [
        "Intent: ignore brief changes\nFailure: minor jitter kept",
        "Intent: compare with recent similar events\nFailure: wrong keep/discard due to missed context",
        "Intent: keep only target pattern\nFailure: distractor retained",
    ]

    for i, ax in enumerate(axes, start=1):
        x, y = _series(i)
        ax.plot(x, y, color=LINE_COLOR, lw=1.7)
        ax.grid(True, color=GRID_COLOR, alpha=0.55, linewidth=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i - 1], fontsize=8.5, pad=4)

        if i == 1:
            ax.axvspan(42, 48, color=BAD_COLOR, alpha=0.20, lw=0)
        elif i == 2:
            ax.axvspan(70, 86, color=BAD_COLOR, alpha=0.20, lw=0)
        else:
            ax.axvspan(30, 44, color=GOOD_COLOR, alpha=0.22, lw=0)
            ax.axvspan(82, 94, color=BAD_COLOR, alpha=0.20, lw=0)

        ax.text(
            0.02,
            0.03,
            notes[i - 1],
            transform=ax.transAxes,
            fontsize=7.0,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.18", "fc": "#f8f8f8", "ec": "#dddddd", "lw": 0.6},
        )

    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_stem}.png/.pdf")


if __name__ == "__main__":
    main()
