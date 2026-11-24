#!/usr/bin/env python3
"""Generate a deployment-case KPI figure from a local input file.

The script exports normalized values and generic labels.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "generated"

LINE_COLOR = "#2F5E91"
DROP_COLOR = "#D79627"
CURRENT_COLOR = "#C83D3D"
SPIKE_COLOR = "#7B6FAE"
GRID_COLOR = "#DDE5F0"
TEXT_COLOR = "#243447"
AXIS_COLOR = "#7D8A9C"


def parse_array(source_text: str, key: str) -> list[float]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(\[[\s\S]*?\])', source_text)
    if not match:
        raise ValueError(f"Cannot find array for {key!r}.")
    return ast.literal_eval(match.group(1))


def detect_repeated_drops(values: np.ndarray) -> list[int]:
    """Detect sharp local drops, then merge adjacent points into events."""
    candidates: list[tuple[int, float, float]] = []
    for idx, value in enumerate(values):
        lo = max(0, idx - 10)
        hi = min(len(values), idx + 11)
        local_median = float(np.median(values[lo:hi]))
        local_slice = values[max(0, idx - 1) : min(len(values), idx + 2)]
        if (
            value < 0.68 * local_median
            and value <= np.percentile(values, 12)
            and value <= float(np.min(local_slice))
        ):
            candidates.append((idx, float(value), local_median))

    merged: list[tuple[int, float, float]] = []
    for candidate in candidates:
        if merged and candidate[0] - merged[-1][0] <= 3:
            if candidate[1] < merged[-1][1]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return [idx for idx, _, _ in merged]


def export_anonymized_csv(
    output_path: Path,
    values: np.ndarray,
    start_time: datetime,
    repeated_drops: set[int],
    intent_window_start: int,
    spike_window: range,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    normalized = (values - value_min) / (value_max - value_min)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "minute_index",
                "time_label",
                "normalized_value",
                "in_two_hour_window",
                "repeated_drop_event",
                "current_alert",
                "distinct_upward_event",
            ],
        )
        writer.writeheader()
        for idx, value in enumerate(normalized):
            writer.writerow(
                {
                    "minute_index": idx,
                    "time_label": (start_time + timedelta(minutes=idx)).strftime(
                        "%H:%M"
                    ),
                    "normalized_value": f"{value:.6f}",
                    "in_two_hour_window": idx >= intent_window_start,
                    "repeated_drop_event": idx in repeated_drops,
                    "current_alert": idx == len(values) - 1,
                    "distinct_upward_event": idx in spike_window,
                }
            )


def make_figure(
    values: np.ndarray,
    start_time: datetime,
    repeated_drops: list[int],
    output_dir: Path,
    stem: str,
) -> None:
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    y = (values - value_min) / (value_max - value_min)
    x = np.arange(len(values))
    intent_window_start = len(values) - 120
    current_idx = len(values) - 1
    spike_window = range(255, 258)  # 19:55--19:57 after mapping 15:40 start.

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.6,
            "axes.titlesize": 8.8,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 0.75,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    ax.plot(x, y, color=LINE_COLOR, linewidth=1.85, zorder=3)

    prior_drops = [idx for idx in repeated_drops if idx >= intent_window_start]
    for idx in prior_drops:
        is_current = idx == current_idx
        ax.scatter(
            [idx],
            [y[idx]],
            s=44 if is_current else 30,
            color=CURRENT_COLOR if is_current else DROP_COLOR,
            edgecolor="white",
            linewidth=0.75,
            zorder=4,
        )

    ax.scatter(
        [257],
        [y[257]],
        s=34,
        color=SPIKE_COLOR,
        edgecolor="white",
        linewidth=0.75,
        zorder=4,
    )

    tick_minutes = [0, 120, 240, 359, 479]
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels(
        [(start_time + timedelta(minutes=i)).strftime("%H:%M") for i in tick_minutes]
    )
    ax.set_xlim(0, current_idx)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=8.1, length=2.6, width=0.65, pad=1.5)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.65, alpha=0.75)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    export_anonymized_csv(
        output_dir / f"{stem}.csv",
        values,
        start_time,
        set(repeated_drops),
        intent_window_start,
        spike_window,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Local KPI input file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="deployment_frequent_pattern_kpi")
    args = parser.parse_args()

    source_text = args.source.read_text()
    values = np.array(parse_array(source_text, "hist_values"), dtype=float)[-480:]
    repeated_drops = detect_repeated_drops(values)
    start_time = datetime(2026, 4, 15, 15, 40)
    make_figure(values, start_time, repeated_drops, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
