import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


TAXONOMY_KEYS = [
	"constraint_count",
	"intent_category",
	"relative_expression",
	"history_dependency",
	"distractor_presence",
]
DISPLAY_NAMES = {
	"constraint_count": "Constraint Count",
	"intent_category": "Intent Category",
	"relative_expression": "Relative Expression",
	"history_dependency": "History Dependency",
	"distractor_presence": "Distractor Presence",
}
DISPLAY_ABBR = {
	"constraint_count": "CC",
	"intent_category": "IC",
	"relative_expression": "RE",
	"history_dependency": "HD",
	"distractor_presence": "DP",
}
TYPE_DISPLAY_ORDER = {
	"constraint_count": ["none", "single", "multiple"],
	"relative_expression": ["with", "without"],
	"history_dependency": ["yes", "no"],
	"distractor_presence": ["yes", "no"],
	"intent_category": ["type", "duration", "magnitude", "frequency", "others"],
}


def _normalize_key(text: str) -> str:
	return str(text).strip().lower()


def _to_display_label(raw_label: str) -> str:
	parts = str(raw_label).strip().split("_")
	return " ".join(part.capitalize() for part in parts if part)


def _ordered_display_items(
	taxonomy_key: str,
	taxonomy_meta: Dict[str, Dict[str, Any]],
) -> List[tuple[str, int]]:
	raw_map: Dict[str, int] = {}
	for raw_label, payload in taxonomy_meta.items():
		label_key = _normalize_key(raw_label)
		raw_map[label_key] = int(payload.get("count", 0))

	ordered_items: List[tuple[str, int]] = []
	for key in TYPE_DISPLAY_ORDER.get(taxonomy_key, []):
		if key in raw_map:
			ordered_items.append((_to_display_label(key), raw_map[key]))

	remaining = sorted(
		[(k, v) for k, v in raw_map.items() if k not in TYPE_DISPLAY_ORDER.get(taxonomy_key, [])],
		key=lambda x: x[1],
		reverse=True,
	)
	for key, value in remaining:
		ordered_items.append((_to_display_label(key), value))

	return ordered_items


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute taxonomy statistics from additional_info JSONL and "
			"save as statistics.json."
		)
	)
	parser.add_argument(
		"--dataset",
		"-d",
		required=True,
		help="Dataset name.",
	)
	parser.add_argument(
		"--round",
		type=int,
		default=2,
		help="Decimal places for percentage values. Default: 2",
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Whether to draw taxonomy count charts.",
	)
	parser.add_argument(
		"--plot-style",
		choices=["bars", "donut", "pie"],
		default="bars",
		help="Plot style: bars, donut or pie. Default: bars",
	)
	return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as file:
		for line_no, line in enumerate(file, start=1):
			raw = line.strip()
			if not raw:
				continue
			try:
				records.append(json.loads(raw))
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
	return records


def _safe_get_taxonomy(record: Dict[str, Any]) -> Dict[str, Any]:
	additional_info = record.get("additional_info", {})
	taxonomy = additional_info.get("intent_taxonomy") if isinstance(additional_info, dict) else None
	if taxonomy is None:
		taxonomy = record.get("intent_taxonomy", {})
	return taxonomy if isinstance(taxonomy, dict) else {}


def _safe_get_human_intent_structured(record: Dict[str, Any]) -> Dict[str, Any]:
	additional_info = record.get("additional_info", {})
	structured = additional_info.get("human_intent_structured") if isinstance(additional_info, dict) else None
	if structured is None:
		structured = record.get("human_intent_structured", {})
	return structured if isinstance(structured, dict) else {}


def _normalize_intent_category(value: Any) -> List[str]:
	if isinstance(value, list):
		cleaned = [str(item).strip() for item in value if str(item).strip()]
		return cleaned if cleaned else ["none"]
	if value is None:
		return ["none"]
	text = str(value).strip()
	return [text] if text else ["none"]


def _normalize_scalar(value: Any) -> str:
	if value is None:
		return "unknown"
	text = str(value).strip()
	return text if text else "unknown"


def _is_non_constraint_empty_value(taxonomy_key: str, value: Any) -> bool:
	if taxonomy_key == "constraint_count":
		return False
	value_key = _normalize_key(value)
	return value_key in {"none", "unknown", ""}


def _has_constraint(constraint_count_value: Any) -> bool:
	return _normalize_key(constraint_count_value) != "none"


def _has_filtered_anomalies(record: Dict[str, Any]) -> bool:
	structured = _safe_get_human_intent_structured(record)
	rules = structured.get("rules", [])
	if not isinstance(rules, list):
		return False

	for rule in rules:
		if not isinstance(rule, dict):
			continue
		filtered = rule.get("filtered_anomalies", [])
		if isinstance(filtered, list) and len(filtered) > 0:
			return True
	return False


def compute_statistics(records: List[Dict[str, Any]], round_digits: int = 2) -> Dict[str, Any]:
	total_cases = len(records)
	constrained_case_ids: List[int] = []
	distractor_by_case_id: Dict[int, bool] = {}

	bucket: Dict[str, Dict[str, List[int]]] = {
		"constraint_count": defaultdict(list),
		"intent_category": defaultdict(list),
		"relative_expression": defaultdict(list),
		"history_dependency": defaultdict(list),
		"distractor_presence": defaultdict(list),
	}

	for case_id, record in enumerate(records):
		taxonomy = _safe_get_taxonomy(record)

		constraint_count = _normalize_scalar(taxonomy.get("constraint_count"))
		bucket["constraint_count"][constraint_count].append(case_id)
		has_constraint = _has_constraint(constraint_count)
		if has_constraint:
			constrained_case_ids.append(case_id)

		has_distractor = has_constraint and _has_filtered_anomalies(record)
		distractor_by_case_id[case_id] = has_distractor
		bucket["distractor_presence"]["yes" if has_distractor else "no"].append(case_id)

		if has_constraint:
			for category in _normalize_intent_category(taxonomy.get("intent_category")):
				if _is_non_constraint_empty_value("intent_category", category):
					continue
				bucket["intent_category"][category].append(case_id)

			relative_expression = _normalize_scalar(taxonomy.get("relative_expression"))
			if not _is_non_constraint_empty_value("relative_expression", relative_expression):
				bucket["relative_expression"][relative_expression].append(case_id)

			history_dependency = _normalize_scalar(taxonomy.get("history_dependency"))
			if not _is_non_constraint_empty_value("history_dependency", history_dependency):
				bucket["history_dependency"][history_dependency].append(case_id)

	denominators = {
		"constraint_count": total_cases,
		"intent_category": len(constrained_case_ids),
		"relative_expression": len(constrained_case_ids),
		"history_dependency": len(constrained_case_ids),
		"distractor_presence": total_cases,
	}

	statistics: Dict[str, Any] = {
		"meta": {
			"total_cases": total_cases,
			"constrained_cases": len(constrained_case_ids),
			"denominators": denominators,
			"taxonomy_keys": TAXONOMY_KEYS,
		}
	}
	# Cross-tabulate each taxonomy type by distractor presence (yes/no).
	statistics["cross"] = {"distractor_by_taxonomy_type": {}}
	for taxonomy_key in TAXONOMY_KEYS:
		statistics["cross"]["distractor_by_taxonomy_type"][taxonomy_key] = {}
		for value_key in sorted(bucket[taxonomy_key].keys()):
			case_ids = set(bucket[taxonomy_key][value_key])
			with_distractor = sum(1 for cid in case_ids if distractor_by_case_id.get(cid, False))
			without_distractor = len(case_ids) - with_distractor
			statistics["cross"]["distractor_by_taxonomy_type"][taxonomy_key][value_key] = {
				"with_distractor": with_distractor,
				"without_distractor": without_distractor,
			}

	statistics["details"] = {}
	for taxonomy_key in TAXONOMY_KEYS:
		statistics["details"][taxonomy_key] = {}
		statistics["meta"][taxonomy_key] = {}
		denominator = denominators.get(taxonomy_key, total_cases)
		for value_key in sorted(bucket[taxonomy_key].keys()):
			case_ids = sorted(bucket[taxonomy_key][value_key])
			count = len(case_ids)
			percentage = round((count / denominator * 100.0), round_digits) if denominator else 0.0
			statistics["meta"][taxonomy_key][value_key] = {
				"count": count,
				"percentage": percentage,
			}
			statistics["details"][taxonomy_key][value_key] = case_ids

	return statistics


def draw_taxonomy_charts(statistics: Dict[str, Any], plot_style: str, output_path: Path) -> None:
	if plot_style == "bars":
		draw_taxonomy_bar_layout(statistics, output_path)
		return

	draw_taxonomy_ring_layout(statistics, plot_style, output_path)


def _get_denominator(statistics: Dict[str, Any], taxonomy_key: str) -> int:
	meta = statistics.get("meta", {})
	denominators = meta.get("denominators", {})
	if isinstance(denominators, dict) and taxonomy_key in denominators:
		return int(denominators[taxonomy_key])
	return int(meta.get("total_cases", 0))


def draw_taxonomy_bar_layout(statistics: Dict[str, Any], output_path: Path) -> None:
	sns.set_theme(style="whitegrid")

	plot_keys = [
		"constraint_count",
		"relative_expression",
		"history_dependency",
		"intent_category",
		"distractor_presence",
	]

	palette_by_key = {
		"constraint_count": "Blues_d",
		"relative_expression": "Greens_d",
		"history_dependency": "Purples_d",
		"intent_category": "Oranges_r",
		"distractor_presence": "Reds_d",
	}

	n_cols = 2
	n_rows = (len(plot_keys) + n_cols - 1) // n_cols
	fig, axes_grid = plt.subplots(figsize=(11.2, 4.0 * n_rows), nrows=n_rows, ncols=n_cols)
	if hasattr(axes_grid, "flatten"):
		axes = list(axes_grid.flatten())
	else:
		axes = [axes_grid]

	for idx, taxonomy_key in enumerate(plot_keys):
		ax = axes[idx]
		taxonomy_meta = statistics.get("meta", {}).get(taxonomy_key, {})
		items = _ordered_display_items(taxonomy_key, taxonomy_meta)
		if taxonomy_key != "constraint_count":
			items = [item for item in items if item[0] not in {"None", "Unknown"}]

		labels = [item[0] for item in items]
		counts = [item[1] for item in items]
		denominator = _get_denominator(statistics, taxonomy_key)

		if taxonomy_key == "intent_category":
			if counts:
				sns.barplot(y=labels, x=counts, ax=ax, palette=palette_by_key[taxonomy_key])
				ax.set_xlim(0, max(counts) * 1.18)
			else:
				ax.text(0.5, 0.5, "No data", ha="center", va="center")
				ax.set_xlim(0, 1)
			ax.set_xlabel("Case Count", fontsize=11)
			ax.set_ylabel("")

			for patch in ax.patches:
				count = int(round(patch.get_width()))
				pct = (count / denominator) if denominator else 0.0
				ax.annotate(
					f"  {count} ({pct:.1%})",
					(count, patch.get_y() + patch.get_height() / 2.0),
					ha="left",
					va="center",
					fontsize=11,
				)
		else:
			if counts:
				sns.barplot(x=labels, y=counts, ax=ax, palette=palette_by_key[taxonomy_key])
				ax.set_ylim(0, max(counts) * 1.25)
			else:
				ax.text(0.5, 0.5, "No data", ha="center", va="center")
				ax.set_ylim(0, 1)
			ax.set_ylabel("Case Count", fontsize=11)
			ax.set_xlabel("")

			for patch in ax.patches:
				count = int(round(patch.get_height()))
				pct = (count / denominator) if denominator else 0.0
				ax.annotate(
					f"{count}\n({pct:.1%})",
					(patch.get_x() + patch.get_width() / 2.0, count),
					ha="center",
					va="bottom",
					fontsize=10,
					xytext=(0, 4),
					textcoords="offset points",
				)

		ax.set_title(DISPLAY_NAMES[taxonomy_key], fontsize=13, pad=10, fontweight="bold")
		ax.tick_params(axis="both", labelsize=11)

	for ax in axes[len(plot_keys):]:
		ax.axis("off")

	for ax in axes:
		for spine in ax.spines.values():
			spine.set_edgecolor("black")
			spine.set_linewidth(0.8)

	# fig.suptitle("IntentAD Taxonomy Distribution", fontsize=16, fontweight="bold", y=0.9)
	# fig.text(
	# 	0.01,
	# 	0.005,
	# 	"Note: IC is multi-label. One case can map to multiple intent categories, "
	# 	"so IC counts can sum to more than total cases.",
	# 	fontsize=10,

	# plt.tight_layout(rect=(0, 0.03, 1, 0.96))
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
	plt.close(fig)
	print(f"Saved bar chart to: {output_path}")


def draw_taxonomy_ring_layout(statistics: Dict[str, Any], plot_style: str, output_path: Path) -> None:
	fig, axis = plt.subplots(figsize=(10, 8))

	palette_by_key = {
		"constraint_count": plt.get_cmap("Blues"),
		"intent_category": plt.get_cmap("Greens"),
		"relative_expression": plt.get_cmap("Oranges"),
		"history_dependency": plt.get_cmap("Purples"),
		"distractor_presence": plt.get_cmap("Reds"),
	}

	base_radius = 1.25
	ring_width = 0.24
	ring_gap = 0.03

	for ring_index, taxonomy_key in enumerate(TAXONOMY_KEYS):
		taxonomy_meta = statistics.get("meta", {}).get(taxonomy_key, {})
		if not taxonomy_meta:
			continue

		ordered_items = _ordered_display_items(taxonomy_key, taxonomy_meta)
		if taxonomy_key != "constraint_count":
			ordered_items = [item for item in ordered_items if item[0] not in {"None", "Unknown"}]

		labels = [item[0] for item in ordered_items]
		sizes = [item[1] for item in ordered_items]
		ring_total = sum(sizes)
		if ring_total == 0:
			continue

		radius = base_radius - ring_index * (ring_width + ring_gap)
		cmap = palette_by_key[taxonomy_key]
		colors = [cmap(0.45 + 0.45 * (idx / max(len(labels), 1))) for idx in range(len(labels))]

		wedges, _ = axis.pie(
			sizes,
			radius=radius,
			startangle=90,
			counterclock=False,
			labels=None,
			colors=colors,
			wedgeprops={
				"width": ring_width,
				"edgecolor": "white",
				"linewidth": 1.2,
			},
		)

		legend_labels = []
		taxonomy_denominator = _get_denominator(statistics, taxonomy_key)
		for label, count in zip(labels, sizes):
			share_cases = (count / taxonomy_denominator * 100.0) if taxonomy_denominator else 0.0
			share_ring = (count / ring_total * 100.0) if ring_total else 0.0
			legend_labels.append(
				f"{label}: {count} | coverage {share_cases:.1f}% | ring {share_ring:.1f}%"
			)

		legend_anchor_y = 1.02 - ring_index * 0.25
		axis.legend(
			wedges,
			legend_labels,
			title=f"{DISPLAY_ABBR[taxonomy_key]}: {DISPLAY_NAMES[taxonomy_key]}",
			loc="upper left",
			bbox_to_anchor=(1.02, legend_anchor_y),
			frameon=False,
			fontsize=11,
			title_fontsize=13,
		)

		axis.text(
			-1.72,
			radius - ring_width / 2,
			f"{DISPLAY_ABBR[taxonomy_key]}: {DISPLAY_NAMES[taxonomy_key]}",
			ha="left",
			va="center",
			fontsize=12,
			fontweight="bold",
		)

	if plot_style == "donut":
		inner_radius = base_radius - len(TAXONOMY_KEYS) * (ring_width + ring_gap) + ring_gap
		inner_radius = max(0.2, inner_radius)
		center_circle = plt.Circle((0, 0), inner_radius, fc="white")
		axis.add_artist(center_circle)

	axis.text(
		0,
		0,
		f"Total\n{statistics.get('meta', {}).get('total_cases', 0)}",
		ha="center",
		va="center",
		fontsize=16,
		fontweight="bold",
	)

	axis.set_title("Intent Taxonomy Distribution", fontsize=18, pad=20)
	fig.text(
		0.02,
		0.02,
		"Note: Intent Category is multi-label. One case can belong to multiple categories, "
		"so category counts in that ring can sum to more than total cases.",
		fontsize=11,
	)
	axis.axis("equal")
	plt.tight_layout(rect=(0, 0.06, 0.83, 1.0))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = parse_args()

	input_path = Path(f"../data/llm_sft/generated/full/additional_info/{args.dataset}_test_byllm.jsonl")
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	output_path = Path(f"../data/intentad_bench/{args.dataset}/statistics.jsonl")

	records = load_jsonl(input_path)
	statistics = compute_statistics(records, round_digits=args.round)
	statistics["meta"]["input_file"] = str(input_path)
	print(f"statistics meta:\n{statistics['meta']}\n\nstatistics cross:\n{statistics['cross']}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as file:
		json.dump(statistics, file, ensure_ascii=False, indent=2)

	chart_output_path = None
	if args.plot:
		chart_output_path = output_path.parent / f"taxonomy_distribution_{args.plot_style}.pdf"
		draw_taxonomy_charts(statistics, args.plot_style, chart_output_path)

	print(f"Processed {statistics['meta']['total_cases']} cases")
	print(f"Saved statistics to: {output_path}")
	if chart_output_path is not None:
		print(f"Saved chart to: {chart_output_path}")


if __name__ == "__main__":
	main()
