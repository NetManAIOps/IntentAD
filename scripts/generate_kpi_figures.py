#!/usr/bin/env python3
"""Generate KPI plot assets requested for paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Shared style (picked to match the existing pastel + blue/red/green figure tone).
LINE_COLOR = "#3868c0"
HISTORY_LINE_COLOR = "#7f8a9f"
GRID_COLOR = "#d8e0f0"
DATA_ANOMALY_COLOR = "#cc332e"
INTENT_ANOMALY_COLOR = "#36873a"
HISTORY_ANOMALY_COLOR = "#d18b22"

DEEP_THINKING_METRIC_NAME = "metadata-store##replication_apply_latency"
DEEP_THINKING_METRIC_TYPE = "LatencySecond"
DEEP_THINKING_SAMPLING_INTERVAL = "5 minutes"

BASELINE = 0.629
HISTORY_BASELINE_OFFSET = 0.000
N_POINTS = 226
NOISE_LEVEL = 0.005
PRIMARY_PERIOD_POINTS = 60
SECONDARY_PERIOD_POINTS = 30
PERIOD_AMPLITUDES = (0.0368, 0.0114)
PERIOD_PHASE_SHIFT = np.pi / 4
# PERIOD_PHASE_SHIFT = 0
PERIOD_FP_START, PERIOD_FP_END = 8, 40
PERIODIC_SUFFIX = "_with_period"
SPIKE_SUFFIX = "_with_spike"
SPIKE_INDEX = 110
SPIKE_MAG = 0.064
LLM_DETECTION_SPANS = [
    (55, 70, "sustained_elevation"),
    (109, 110, "spike"),
    (128, 144, "sustained_elevation"),
    (145, 147, "sudden_drop"),
]
DetectionSpan = tuple[float, float, str, float]

A1_START, A1_END, A1_MAG = 50, 75, 0.0562
A2_START, A2_END, A2_MAG = 128, 147, 0.0989
HISTORY_ANOMALY_REL_START, HISTORY_ANOMALY_REL_END = -203, -176
HISTORY_ANOMALY_MAG = 0.0640

OF_SPIKE_START, OF_SPIKE_END, OF_SPIKE_MAG = 48, 64, 0.062
OF_DIP_START, OF_DIP_END, OF_DIP_MAG = 132, 152, 0.071

MAIN_LINE_WIDTH = 1.9
HISTORY_LINE_WIDTH = 1.5
HIGHLIGHT_LINE_WIDTH = 2.2


def _setup_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": "#9aa8c0",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _resolve_relative_window(
    length: int, start_offset: int, end_offset: int
) -> tuple[int, int]:
    """Convert end-relative offsets to an absolute inclusive window."""
    end_idx = length - 1
    start = end_idx + start_offset
    end = end_idx + end_offset
    if start < 0 or end >= length or start > end:
        raise ValueError(
            f"Invalid relative window ({start_offset}, {end_offset}) for length {length}."
        )
    return start, end


def _build_sudden_increase_shape(
    window_len: int, magnitude: float, rise_points: int, fall_points: int
) -> np.ndarray:
    if window_len <= rise_points + fall_points:
        raise ValueError("Window length must be larger than rise_points + fall_points.")
    rise = np.linspace(0.0, magnitude, rise_points, endpoint=False)
    plateau = np.full(window_len - rise_points - fall_points, magnitude)
    fall = np.linspace(magnitude, 0.0, fall_points)
    return np.concatenate([rise, plateau, fall])


def _build_smooth_periodic_wave(
    length: int,
    amplitudes: tuple[float, float] = PERIOD_AMPLITUDES,
    primary_period_points: int = PRIMARY_PERIOD_POINTS,
    secondary_period_points: int = SECONDARY_PERIOD_POINTS,
    phase_shift: float = PERIOD_PHASE_SHIFT,
) -> np.ndarray:
    x = np.arange(length, dtype=float)
    primary_amplitude, secondary_amplitude = amplitudes
    primary_wave = primary_amplitude * np.cos(2.0 * np.pi * x / primary_period_points)
    secondary_wave = secondary_amplitude * np.sin(
        2.0 * np.pi * x / secondary_period_points + phase_shift
    )
    return primary_wave + secondary_wave


def _deep_thinking_stem(base_stem: str, add_period: bool, add_spike: bool) -> str:
    stem = base_stem
    if add_period:
        stem += PERIODIC_SUFFIX
    if add_spike:
        stem += SPIKE_SUFFIX
    return stem


def _single_point_window(index: int) -> tuple[float, float]:
    return index - 0.5, index + 0.5


def generate_deep_thinking_kpi(
    seed: int = 2026, add_period: bool = True, add_spike: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Build the deep-thinking KPI with two sudden increases and optional smooth seasonality."""
    rng = np.random.default_rng(seed)
    x = np.arange(N_POINTS)
    y = np.full(N_POINTS, BASELINE, dtype=float)
    y += rng.normal(0.0, NOISE_LEVEL, size=N_POINTS)  # low-noise baseline
    if add_period:
        y += _build_smooth_periodic_wave(length=N_POINTS)
    if add_spike:
        y[SPIKE_INDEX] += SPIKE_MAG

    # Anomaly 1: sudden increase event over 39 points [50, 88], mag 0.0362.
    rise_1 = np.linspace(0.0, A1_MAG, 6, endpoint=False)
    plateau_1 = np.full((A1_END - A1_START + 1) - 12, A1_MAG)
    fall_1 = np.linspace(A1_MAG, 0.0, 6)
    shape_1 = np.concatenate([rise_1, plateau_1, fall_1])
    y[A1_START : A1_END + 1] += shape_1

    # Anomaly 2: abrupt upward level-shift over 20 points [128, 147], mag 0.0789.
    rise_2 = np.array([A2_MAG / 3.0, 2.0 * A2_MAG / 3.0, A2_MAG])
    plateau_2 = np.full((A2_END - A2_START + 1) - 6, A2_MAG)
    fall_2 = np.linspace(A2_MAG, 0.0, 3)
    shape_2 = np.concatenate([rise_2, plateau_2, fall_2])
    y[A2_START : A2_END + 1] += shape_2

    return x, y


def generate_deep_thinking_history_kpi(
    seed: int = 2042, add_period: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Build a historical KPI curve aligned with the current pattern and one past spike."""
    rng = np.random.default_rng(seed)
    x = np.arange(N_POINTS)
    y = np.full(N_POINTS, BASELINE - HISTORY_BASELINE_OFFSET, dtype=float)
    y += rng.normal(0.0, NOISE_LEVEL, size=N_POINTS)
    if add_period:
        y += _build_smooth_periodic_wave(length=N_POINTS)

    history_start, history_end = _resolve_relative_window(
        N_POINTS, HISTORY_ANOMALY_REL_START, HISTORY_ANOMALY_REL_END
    )
    y[history_start : history_end + 1] += _build_sudden_increase_shape(
        history_end - history_start + 1,
        HISTORY_ANOMALY_MAG,
        rise_points=5,
        fall_points=5,
    )

    return x, y


def generate_overall_framework_kpi(seed: int = 2031) -> tuple[np.ndarray, np.ndarray]:
    """Build a distinct KPI for overall framework: one spike + one dip."""
    rng = np.random.default_rng(seed)
    x = np.arange(N_POINTS)
    y = np.full(N_POINTS, BASELINE - 0.01, dtype=float)
    y += rng.normal(0.0, 0.0024, size=N_POINTS)

    # Upward spike event (data anomaly only).
    up_rise = np.array([0.016, 0.039, OF_SPIKE_MAG])
    up_fall = np.array([0.038, 0.018, 0.0])
    spike_mid_len = (OF_SPIKE_END - OF_SPIKE_START + 1) - len(up_rise) - len(up_fall)
    spike_mid = np.full(spike_mid_len, OF_SPIKE_MAG)
    spike_shape = np.concatenate([up_rise, spike_mid, up_fall])
    y[OF_SPIKE_START : OF_SPIKE_END + 1] += spike_shape

    # Downward dip event (intent anomaly in this scenario).
    dip_down = np.array([0.014, 0.042, OF_DIP_MAG])
    dip_up = np.array([0.044, 0.020, 0.0])
    dip_mid_len = (OF_DIP_END - OF_DIP_START + 1) - len(dip_down) - len(dip_up)
    dip_mid = np.full(dip_mid_len, OF_DIP_MAG)
    dip_shape = np.concatenate([dip_down, dip_mid, dip_up])
    y[OF_DIP_START : OF_DIP_END + 1] -= dip_shape

    return x, y


def _apply_axis_style(ax: plt.Axes, show_axes: bool = True, show_ticks: bool = True) -> None:
    if show_axes:
        ax.grid(True, color=GRID_COLOR, alpha=0.65, linewidth=0.8)
        ax.set_xlim(0, N_POINTS - 1)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_deep_thinking_kpi_data(
    x: np.ndarray,
    y: np.ndarray,
    history_x: np.ndarray,
    history_y: np.ndarray,
    output_dir: Path,
    add_period: bool,
    add_spike: bool,
) -> str:
    history_start, history_end = _resolve_relative_window(
        len(history_x), HISTORY_ANOMALY_REL_START, HISTORY_ANOMALY_REL_END
    )
    current_data_anomalies = [
        {
            "id": "A1",
            "start": A1_START,
            "end": A1_END,
            "pattern": "sudden_increase",
            "magnitude": A1_MAG,
        },
        {
            "id": "A2",
            "start": A2_START,
            "end": A2_END,
            "pattern": "sudden_increase",
            "magnitude": A2_MAG,
        },
    ]
    if add_spike:
        current_data_anomalies.insert(
            0,
            {
                "id": "S1",
                "start": SPIKE_INDEX,
                "end": SPIKE_INDEX,
                "pattern": "point_spike",
                "magnitude": SPIKE_MAG,
            },
        )

    payload = {
        "metric_name": DEEP_THINKING_METRIC_NAME,
        "metric_type": DEEP_THINKING_METRIC_TYPE,
        "sampling_interval": DEEP_THINKING_SAMPLING_INTERVAL,
        "add_period": add_period,
        "add_spike": add_spike,
        "current_window": {
            "x_index": x.astype(int).tolist(),
            "y": [float(value) for value in y],
            "background_period": {
                "enabled": add_period,
                "pattern": "smooth_sin_cos",
                "primary_period_points": PRIMARY_PERIOD_POINTS if add_period else None,
                "secondary_period_points": SECONDARY_PERIOD_POINTS if add_period else None,
                "amplitudes": list(PERIOD_AMPLITUDES) if add_period else None,
                "phase_shift": PERIOD_PHASE_SHIFT if add_period else None,
            },
            "point_spike": {
                "enabled": add_spike,
                "index": SPIKE_INDEX if add_spike else None,
                "magnitude": SPIKE_MAG if add_spike else None,
            },
            "all_data_anomalies": current_data_anomalies,
            "intent_aligned_anomalies": [
                {
                    "id": "A2",
                    "start": A2_START,
                    "end": A2_END,
                    "pattern": "sudden_increase",
                    "magnitude": A2_MAG,
                }
            ],
        },
        "history_window": {
            "x_index": history_x.astype(int).tolist(),
            "x_relative_to_end": [int(value - (len(history_x) - 1)) for value in history_x],
            "y": [float(value) for value in history_y],
            "background_period": {
                "enabled": add_period,
                "pattern": "smooth_sin_cos",
                "primary_period_points": PRIMARY_PERIOD_POINTS if add_period else None,
                "secondary_period_points": SECONDARY_PERIOD_POINTS if add_period else None,
                "amplitudes": list(PERIOD_AMPLITUDES) if add_period else None,
                "phase_shift": PERIOD_PHASE_SHIFT if add_period else None,
            },
            "all_data_anomalies": [
                {
                    "id": "H1",
                    "start": history_start,
                    "end": history_end,
                    "start_offset": HISTORY_ANOMALY_REL_START,
                    "end_offset": HISTORY_ANOMALY_REL_END,
                    "pattern": "sudden_increase",
                    "magnitude": HISTORY_ANOMALY_MAG,
                }
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{_deep_thinking_stem('deep_thinking_kpi_data', add_period, add_spike)}.json"
    )
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path.name


def draw_overall_framework_input_kpi(x: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)
    _apply_axis_style(ax, show_axes=True, show_ticks=False)
    _save_figure(fig, output_dir, "overall_framework_input_kpi")


def draw_overall_framework_output_ad_result(
    x: np.ndarray, y: np.ndarray, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)

    # Scenario: spike is data anomaly, dip is final intent anomaly.
    ax.axvspan(OF_SPIKE_START, OF_SPIKE_END, color=DATA_ANOMALY_COLOR, alpha=0.16, lw=0)
    ax.axvspan(OF_DIP_START, OF_DIP_END, color=INTENT_ANOMALY_COLOR, alpha=0.28, lw=0)

    _apply_axis_style(ax, show_axes=True, show_ticks=False)
    legend_items = [
        Patch(facecolor=DATA_ANOMALY_COLOR, alpha=0.16, label="Data anomaly"),
        Patch(facecolor=INTENT_ANOMALY_COLOR, alpha=0.28, label="Intent anomaly"),
    ]
    # ax.legend(handles=legend_items, loc="upper left", frameon=True, framealpha=0.95)
    ax.legend(handles=legend_items, frameon=True, framealpha=0.95, fontsize=10.5)
    _save_figure(fig, output_dir, "overall_framework_output_ad_result")


def draw_deep_thinking_top_kpi(
    x: np.ndarray, y: np.ndarray, output_dir: Path, add_period: bool, add_spike: bool
) -> str:
    base_scale = 0.5
    l_w = (20, 3)
    fig, ax = plt.subplots(figsize=(l_w[0] * base_scale, l_w[1] * base_scale))
    # A1 is filtered (data anomaly color); A2 is kept (intent anomaly color).
    if add_spike:
        spike_start, spike_end = _single_point_window(SPIKE_INDEX)
        ax.axvspan(spike_start, spike_end, color=DATA_ANOMALY_COLOR, alpha=0.18, lw=0)
    ax.axvspan(A1_START, A1_END, color=DATA_ANOMALY_COLOR, alpha=0.12, lw=0)
    ax.axvspan(A2_START, A2_END, color=INTENT_ANOMALY_COLOR, alpha=0.24, lw=0)
    ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)
    if add_spike:
        ax.scatter(x[SPIKE_INDEX], y[SPIKE_INDEX], color=DATA_ANOMALY_COLOR, s=10, zorder=4)

    _apply_axis_style(ax, show_axes=True)
    stem = _deep_thinking_stem("deep_thinking_top_kpi", add_period, add_spike)
    _save_figure(fig, output_dir, stem)
    return stem


def draw_deep_thinking_plain_kpi(
    x: np.ndarray, y: np.ndarray, output_dir: Path, add_period: bool, add_spike: bool
) -> str:
    base_scale = 0.5
    l_w = (20, 3)
    fig, ax = plt.subplots(figsize=(l_w[0] * base_scale, l_w[1] * base_scale))
    ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)

    _apply_axis_style(ax, show_axes=True)
    stem = _deep_thinking_stem("deep_thinking_plain_kpi", add_period, add_spike)
    _save_figure(fig, output_dir, stem)
    return stem


def draw_deep_thinking_history_kpi(
    x: np.ndarray, y: np.ndarray, output_dir: Path, add_period: bool, add_spike: bool
) -> str:
    history_start, history_end = _resolve_relative_window(
        len(x), HISTORY_ANOMALY_REL_START, HISTORY_ANOMALY_REL_END
    )
    base_scale = 0.5
    l_w = (20, 3)
    fig, ax = plt.subplots(figsize=(l_w[0] * base_scale, l_w[1] * base_scale))
    ax.axvspan(history_start, history_end, color=HISTORY_ANOMALY_COLOR, alpha=0.14, lw=0)
    ax.plot(
        x,
        y,
        color=HISTORY_LINE_COLOR,
        linewidth=HISTORY_LINE_WIDTH,
        # linestyle=(0, (4, 2)),
    )
    ax.plot(
        x[history_start : history_end + 1],
        y[history_start : history_end + 1],
        color=HISTORY_ANOMALY_COLOR,
        linewidth=HIGHLIGHT_LINE_WIDTH,
    )

    _apply_axis_style(ax, show_axes=True)
    history_ticks = np.arange(0, len(x), 25)
    ax.set_xticks(history_ticks)
    ax.set_xticklabels([str(tick - (len(x) - 1)) for tick in history_ticks])
    stem = _deep_thinking_stem("deep_thinking_history_kpi", add_period, add_spike)
    _save_figure(fig, output_dir, stem)
    return stem


def draw_deep_thinking_model_detection_result(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    title: str,
    detections: list[DetectionSpan],
    stem: str,
    add_period: bool,
    add_spike: bool,
) -> str:
    base_scale = 0.5
    l_w = (13, 3)
    fig, ax = plt.subplots(figsize=(l_w[0] * base_scale, l_w[1] * base_scale))
    ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)
    for start, end, color, alpha in detections:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    _apply_axis_style(ax, show_axes=True)
    # ax.set_title(title, fontsize=9.0, pad=4)
    final_stem = _deep_thinking_stem(stem, add_period, add_spike)
    _save_figure(fig, output_dir, final_stem)
    return final_stem


def draw_deep_thinking_model_detection_results(
    x: np.ndarray, y: np.ndarray, output_dir: Path, add_period: bool, add_spike: bool
) -> list[str]:
    def detection_span(
        start: float,
        end: float,
        is_correct: bool,
        data_alpha: float = 0.14,
        intent_alpha: float = 0.22,
    ) -> DetectionSpan:
        if is_correct:
            return start, end, INTENT_ANOMALY_COLOR, intent_alpha
        return start, end, DATA_ANOMALY_COLOR, data_alpha

    def split_detection_by_intent_overlap(
        start: float,
        end: float,
        data_alpha: float = 0.14,
        intent_alpha: float = 0.22,
    ) -> list[DetectionSpan]:
        spans: list[DetectionSpan] = []
        left_end = min(end, A2_START)
        if start < left_end:
            spans.append(
                detection_span(start, left_end, is_correct=False, data_alpha=data_alpha)
            )

        intent_start = max(start, A2_START)
        intent_end = min(end, A2_END)
        if intent_start < intent_end:
            spans.append(
                detection_span(
                    intent_start,
                    intent_end,
                    is_correct=True,
                    intent_alpha=intent_alpha,
                )
            )

        right_start = max(start, A2_END)
        if right_start < end:
            spans.append(
                detection_span(right_start, end, is_correct=False, data_alpha=data_alpha)
            )
        return spans

    statistical_detections: list[DetectionSpan] = []
    # if add_period:
    #     statistical_detections.extend(
    #         split_detection_by_intent_overlap(PERIOD_FP_START, PERIOD_FP_END)
    #     )
    if add_spike:
        statistical_detections.extend(
            split_detection_by_intent_overlap(*_single_point_window(SPIKE_INDEX))
        )
    statistical_detections.extend(
        split_detection_by_intent_overlap(A1_START, A1_END)
        + split_detection_by_intent_overlap(A2_START, A2_END)
    )
    llm_detections: list[DetectionSpan] = []
    for start, end, _ in LLM_DETECTION_SPANS:
        llm_detections.extend(
            split_detection_by_intent_overlap(start, end, data_alpha=0.22)
        )

    model_specs = [
        {
            "title": "Statistical & DL",
            "detections": statistical_detections,
            "stem": "deep_thinking_detection_statistical_dl",
        },
        {
            "title": "Rule-based",
            "detections": [detection_span(A2_START, A2_END, is_correct=True)],
            "stem": "deep_thinking_detection_rule_based",
        },
        {
            "title": "IntentAD",
            "detections": [detection_span(A2_START, A2_END, is_correct=True)],
            "stem": "deep_thinking_detection_intentad",
        },
        {
            "title": "LLM",
            "detections": llm_detections,
            "stem": "deep_thinking_detection_llm",
        },
    ]

    stems: list[str] = []
    for spec in model_specs:
        stems.append(
            draw_deep_thinking_model_detection_result(
                x=x,
                y=y,
                output_dir=output_dir,
                title=spec["title"],
                detections=spec["detections"],
                stem=spec["stem"],
                add_period=add_period,
                add_spike=add_spike,
            )
        )
    return stems


def _build_anomaly_series(
    kind: str, length: int = 120, baseline: float = 0.55, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(length)
    y = np.full(length, baseline) + rng.normal(0.0, 0.003, size=length)

    if kind == "spike":
        c = 60
        y[c - 1 : c + 2] += np.array([0.04, 0.13, 0.03])
    elif kind == "dip":
        c = 60
        y[c - 1 : c + 2] -= np.array([0.03, 0.11, 0.04])
    elif kind == "slow_increase":
        s, e = 30, 95
        y[s:e] += np.linspace(0.0, 0.12, e - s)
    elif kind == "slow_decrease":
        s, e = 30, 95
        y[s:e] -= np.linspace(0.0, 0.12, e - s)
    else:
        raise ValueError(f"Unsupported anomaly kind: {kind}")
    return x, y


def draw_data_engine_anomaly_examples(output_dir: Path) -> list[str]:
    kinds = ["spike", "dip", "slow_increase", "slow_decrease"]
    stems: list[str] = []

    for i, kind in enumerate(kinds):
        x, y = _build_anomaly_series(kind=kind, seed=11 + i)
        fig, ax = plt.subplots(figsize=(6.8, 1.8))
        ax.plot(x, y, color=LINE_COLOR, linewidth=MAIN_LINE_WIDTH)
        if kind in {"spike", "dip"}:
            ax.plot(
                x[57:64],
                y[57:64],
                color=DATA_ANOMALY_COLOR,
                linewidth=HIGHLIGHT_LINE_WIDTH,
            )
        else:
            ax.plot(
                x[30:95],
                y[30:95],
                color=DATA_ANOMALY_COLOR,
                linewidth=HIGHLIGHT_LINE_WIDTH,
            )
        _apply_axis_style(ax, show_axes=False)
        stem = f"data_engine_{kind}"
        stems.append(stem)
        _save_figure(fig, output_dir, stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KPI drawing assets for paper figures."
    )
    default_out = Path(__file__).resolve().parents[1] / "generated"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--add-period",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a smooth sin/cos periodic component to the deep thinking KPI.",
    )
    parser.add_argument(
        "--add-spike",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a one-point spike at x=110 to the deep thinking KPI.",
    )
    args = parser.parse_args()

    _setup_rcparams()
    output_dir = args.output_dir.resolve()

    overall_x, overall_y = generate_overall_framework_kpi(seed=2031)
    deep_x, deep_y = generate_deep_thinking_kpi(
        seed=2026, add_period=args.add_period, add_spike=args.add_spike
    )
    deep_history_x, deep_history_y = generate_deep_thinking_history_kpi(
        seed=2042, add_period=args.add_period
    )
    deep_json_name = export_deep_thinking_kpi_data(
        deep_x,
        deep_y,
        deep_history_x,
        deep_history_y,
        output_dir,
        add_period=args.add_period,
        add_spike=args.add_spike,
    )
    draw_overall_framework_input_kpi(overall_x, overall_y, output_dir)
    draw_overall_framework_output_ad_result(overall_x, overall_y, output_dir)
    deep_top_stem = draw_deep_thinking_top_kpi(
        deep_x, deep_y, output_dir, add_period=args.add_period, add_spike=args.add_spike
    )
    deep_plain_stem = draw_deep_thinking_plain_kpi(
        deep_x, deep_y, output_dir, add_period=args.add_period, add_spike=args.add_spike
    )
    deep_history_stem = draw_deep_thinking_history_kpi(
        deep_history_x,
        deep_history_y,
        output_dir,
        add_period=args.add_period,
        add_spike=args.add_spike,
    )
    model_detection_stems = draw_deep_thinking_model_detection_results(
        deep_x,
        deep_y,
        output_dir,
        add_period=args.add_period,
        add_spike=args.add_spike,
    )
    data_engine_stems = draw_data_engine_anomaly_examples(output_dir)

    print(f"Generated KPI assets in: {output_dir}")
    for stem in [
        "overall_framework_input_kpi",
        "overall_framework_output_ad_result",
        deep_top_stem,
        deep_plain_stem,
        deep_history_stem,
    ] + model_detection_stems + data_engine_stems:
        print(f" - {stem}.png")
        print(f" - {stem}.pdf")
    print(f" - {deep_json_name}")


if __name__ == "__main__":
    main()
