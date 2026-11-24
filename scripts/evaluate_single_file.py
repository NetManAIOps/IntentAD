#!/usr/bin/env python3
"""Evaluate one IntentAD prediction JSONL file.

This wrapper keeps the original experiment evaluator intact while providing a
small command for smoke tests and reproducible evaluation.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single prediction JSONL file.")
    parser.add_argument("--input", required=True, help="Prediction JSONL file.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    parser.add_argument("--delay", type=int, default=15, help="Delay threshold for delay F1.")
    parser.add_argument(
        "--pattern-average",
        choices=["micro", "macro", "weighted", "binary"],
        default="macro",
    )
    parser.add_argument("--no-position", action="store_true")
    parser.add_argument("--no-pattern", action="store_true")
    parser.add_argument("--per-sample", action="store_true")
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_file(
        args.input,
        evaluate_position=not args.no_position,
        evaluate_pattern=not args.no_pattern,
        delay_threshold=args.delay,
        pattern_average=args.pattern_average,
        per_sample=args.per_sample,
        margin=args.margin,
        plot_dir=args.plot_dir,
        profile=args.profile,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
