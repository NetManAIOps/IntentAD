import json
import argparse
import re
import textwrap
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

# Optional plotting support.
_MATPLOT_OK = True
try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
	_MATPLOT_OK = False

from parser.parser import LLMOutputParser
from evaluator.position_evaluator import PositionEvaluator
from evaluator.pattern_evaluator import PatternEvaluator

SEQ_LEN_REGEX = re.compile(r"Sequence Length:\s*(\d+)")
METRIC_NAME_REGEX = re.compile(r"Metric Name:\s*([^\n\r]+)")
METRIC_NAME_VITS_STYLE_REGEX = re.compile(r"metric called ([^\n\r\s]+)")
GENERATED_DATASET_PATTERNS = [
	re.compile(r"^(\d{8}_\d{6})_test_byllm$"),
	re.compile(r"^test_(\d{8}_\d{6})_byllm$"),
]


def extract_sequence_length(input_text: str) -> Optional[int]:
	match = SEQ_LEN_REGEX.search(input_text)
	if match:
		try:
			return int(match.group(1))
		except ValueError:
			return None
	return None


def extract_metric_name(input_text: str) -> Optional[str]:
	match1 = METRIC_NAME_REGEX.search(input_text)
	if match1:
		try:
			return match1.group(1).strip()
		except ValueError:
			return None
	match2 = METRIC_NAME_VITS_STYLE_REGEX.search(input_text)
	if match2:
		try:
			return match2.group(1).strip()
		except ValueError:
			return None
	return None


def extract_human_intent(input_text: str) -> Optional[str]:
	human_intent = input_text.split("[User Intent]")[-1].split("<|im_end|>")[0].split("##Important")[0].strip() if "[User Intent]" in input_text else None
	return human_intent


def filter_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
	"""Remove invalid sentinel segments like (-1,-1) and ensure start<=end."""
	cleaned = []
	for s, e in segments:
		if s is None or e is None:
			continue
		if s < 0 or e < 0:
			continue
		if e < s:
			s, e = e, s
		cleaned.append((s, e))
	return cleaned


def infer_statistics_path_for_generated(dataname: str) -> Optional[Path]:
	for pattern in GENERATED_DATASET_PATTERNS:
		match = pattern.match(str(dataname))
		if match:
			bench_id = match.group(1)
			base_dir = Path(__file__).resolve().parent.parent
			return base_dir / 'data' / 'intentad_bench' / bench_id / 'statistics.jsonl'
	return None


def load_statistics_details(statistics_path: Path) -> Optional[Dict[str, Dict[str, List[int]]]]:
	if not statistics_path.exists():
		return None
	try:
		with statistics_path.open('r', encoding='utf-8') as f:
			raw = f.read().strip()
			if not raw:
				return None
			data = json.loads(raw)
	except json.JSONDecodeError:
		#      jsonl      json    
		try:
			with statistics_path.open('r', encoding='utf-8') as f:
				first_non_empty = None
				for line in f:
					line = line.strip()
					if line:
						first_non_empty = line
						break
				if first_non_empty is None:
					return None
				data = json.loads(first_non_empty)
		except Exception:
			return None
	except Exception:
		return None

	if not isinstance(data, dict):
		return None
	details = data.get('details', {})
	if not isinstance(details, dict):
		return None

	result: Dict[str, Dict[str, List[int]]] = {}
	for taxonomy_name, type_map in details.items():
		if not isinstance(type_map, dict):
			continue
		result[taxonomy_name] = {}
		for taxonomy_type, case_ids in type_map.items():
			if not isinstance(case_ids, list):
				continue
			normalized = []
			for cid in case_ids:
				try:
					normalized.append(int(cid))
				except Exception:
					continue
			result[taxonomy_name][str(taxonomy_type)] = normalized
	return result if result else None


def load_source_splits(dataset_path: Path) -> Optional[List[Optional[str]]]:
	"""Load source_split for each case from raw dataset jsonl.
	Case order is expected to align with model output raw_records order.
	"""
	if not dataset_path.exists():
		return None

	splits: List[Optional[str]] = []
	try:
		with dataset_path.open('r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					splits.append(None)
					continue
				try:
					obj = json.loads(line)
				except json.JSONDecodeError:
					# Keep index alignment even if one line is malformed.
					splits.append(None)
					continue

				if isinstance(obj, dict):
					source_split = obj.get('source_split')
					splits.append(source_split if isinstance(source_split, str) else None)
				else:
					splits.append(None)
	except Exception:
		return None

	return splits if splits else None


def build_taxonomy_subset_results(
	group_eval_records: List[Dict[str, Any]],
	statistics_details: Dict[str, Dict[str, List[int]]],
	evaluate_position: bool,
	evaluate_pattern: bool,
	pattern_average: str,
) -> Dict[str, Any]:
	def _aggregate_pattern_detection(records: List[Dict[str, Any]]) -> Dict[str, Any]:
		per_pattern: Dict[str, Dict[str, int]] = {}
		total_tp = 0
		total_fp = 0
		for rec in records:
			group_map = rec.get('pattern_detection_per_pred', {})
			if not isinstance(group_map, dict):
				continue
			for pat, stats in group_map.items():
				if not isinstance(stats, dict):
					continue
				tp = int(stats.get('tp', 0))
				fp = int(stats.get('fp', 0))
				entry = per_pattern.setdefault(str(pat), {'tp': 0, 'fp': 0})
				entry['tp'] += tp
				entry['fp'] += fp
				total_tp += tp
				total_fp += fp
		for _, stats in per_pattern.items():
			d = stats['tp'] + stats['fp']
			stats['precision'] = (stats['tp'] / d) if d > 0 else 0.0
		per_pattern = dict(sorted(per_pattern.items(), key=lambda item: item[1]['precision'], reverse=True))
		den = total_tp + total_fp
		return {
			'per_pattern': per_pattern,
			'total': {
				'tp': total_tp,
				'fp': total_fp,
				'precision': (total_tp / den) if den > 0 else 0.0,
			},
		}

	result: Dict[str, Any] = {}
	n_total_groups = len(group_eval_records)

	for taxonomy_name, type_map in statistics_details.items():
		if not isinstance(type_map, dict):
			continue
		taxonomy_out: Dict[str, Any] = {}
		for taxonomy_type, case_ids in type_map.items():
			if not isinstance(case_ids, list):
				continue
			valid_ids = [cid for cid in case_ids if 0 <= cid < n_total_groups]
			subset_records = [group_eval_records[cid] for cid in valid_ids]

			subset_metrics: Dict[str, Any] = {
				'num_samples': len(subset_records),
				'num_case_ids': len(case_ids),
				'num_valid_case_ids': len(valid_ids),
			}

			if evaluate_position and subset_records:
				metric_types = list(subset_records[0].get('position_metrics', {}).keys())
				agg_position = {
					k: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
					for k in metric_types
				}
				for rec in subset_records:
					pos = rec.get('position_metrics', {})
					for k in metric_types:
						cur = pos.get(k, {})
						agg_position[k]['precision'] += float(cur.get('precision', 0.0))
						agg_position[k]['recall'] += float(cur.get('recall', 0.0))
						agg_position[k]['f1'] += float(cur.get('f1', 0.0))
				for k in metric_types:
					agg_position[k]['precision'] /= len(subset_records)
					agg_position[k]['recall'] /= len(subset_records)
					agg_position[k]['f1'] /= len(subset_records)
				subset_metrics['position_metrics_avg'] = agg_position

			if evaluate_pattern:
				all_true_patterns: List[Any] = []
				all_pred_patterns: List[Any] = []
				for rec in subset_records:
					all_true_patterns.extend(rec.get('true_patterns_aligned', []))
					all_pred_patterns.extend(rec.get('pred_patterns_aligned', []))
				if all_true_patterns or all_pred_patterns:
					unique_labels = sorted(set(p for p in all_true_patterns + all_pred_patterns if p is not None))
					if any(p is None for p in all_true_patterns + all_pred_patterns):
						unique_labels.append(None)
					pattern_evaluator = PatternEvaluator(average=pattern_average, labels=unique_labels)
					subset_metrics['pattern_metrics'] = pattern_evaluator.evaluate(all_true_patterns, all_pred_patterns)
					subset_metrics['pattern_labels'] = unique_labels
				else:
					subset_metrics['pattern_metrics'] = None
					subset_metrics['pattern_labels'] = []

			subset_metrics['pattern_detection_stats'] = _aggregate_pattern_detection(subset_records)
			taxonomy_out[taxonomy_type] = subset_metrics

		result[taxonomy_name] = taxonomy_out

	return result


def match_patterns(true_segments: List[Tuple[int, int]], true_patterns: List[Any],
				   pred_segments: List[Tuple[int, int]], pred_patterns: List[Any]) -> Tuple[List[Any], List[Any]]:
	"""
	Match predicted anomalies to ground-truth anomalies using IoU of segments.
	Returns aligned y_true_patterns, y_pred_patterns lists which can differ from original ordering.
	Unmatched segments are paired with None on the opposite side.
	"""
	# Build list of true and pred with indices
	true_items = list(zip(range(len(true_segments)), true_segments, true_patterns))
	pred_items = list(zip(range(len(pred_segments)), pred_segments, pred_patterns))

	def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
		s1, e1 = a
		s2, e2 = b
		inter = max(0, min(e1, e2) - max(s1, s2) + 1)
		if inter == 0:
			return 0.0
		union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
		if union <= 0:
			return 0.0
		return inter / union

	matched_true = set()
	matched_pred = set()
	pairs: List[Tuple[Optional[Any], Optional[Any]]] = []

	# Greedy matching: for each pred, choose best IoU true not yet matched
	for p_idx, p_seg, p_pat in pred_items:
		best_t = None
		best_iou = 0.0
		for t_idx, t_seg, t_pat in true_items:
			if t_idx in matched_true:
				continue
			i = iou(p_seg, t_seg)
			if i > 0 and i > best_iou:
				best_iou = i
				best_t = (t_idx, t_seg, t_pat)
		if best_t is not None:
			matched_true.add(best_t[0])
			matched_pred.add(p_idx)
			pairs.append((best_t[2], p_pat))
		else:
			# unmatched prediction
			pairs.append(("unmatched", p_pat))

	# Remaining unmatched true anomalies
	for t_idx, t_seg, t_pat in true_items:
		if t_idx not in matched_true:
			pairs.append((t_pat, "unmatched"))

	y_true_aligned = [pt for pt, _ in pairs]
	y_pred_aligned = [pp for _, pp in pairs]
	return y_true_aligned, y_pred_aligned


def plot_sample(seq_len: int,
			 true_segments: List[Tuple[int, int]],
			 pred_segments: List[Tuple[int, int]],
			 true_patterns: List[Any],
			 pred_patterns: List[Any],
			 true_magnitudes: List[float],
			 pred_magnitudes: List[float],
			 timeseries: Any,
			 out_dir: Path,
			 metric_name: str = "sample",
			 position_metrics: Optional[Dict[str, Any]] = None,
			 human_intent: Optional[str] = None):
	"""          
	1.        time series       
	2.          axvspan       +          scatter      
	3.  “    ”                     pattern        
	"""
	if not _MATPLOT_OK:
		return
	out_dir.mkdir(parents=True, exist_ok=True)

	ts = None
	if timeseries is not None and isinstance(timeseries, list) and len(timeseries) > 0:
		if isinstance(timeseries[0], list):
			ts = timeseries[0]
		else:
			ts = timeseries

	if ts is None:
		#        (0)          
		ts = [0.0] * seq_len
	elif len(ts) < seq_len:
		seq_len = len(ts)

	x = np.arange(seq_len)
	fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
	config = [
		("Ground Truth", true_segments, true_patterns, False, true_magnitudes),
		("Model Prediction", pred_segments, pred_patterns, True, pred_magnitudes),
	]

	for ax, (title, segs, patterns, annotate_patterns, magnitudes) in zip(axes, config):
		ax.plot(x, ts[:seq_len], color='#1f77b4', linewidth=0.8, label='value')
		has_anom = False
		text_positition_flag = 1
		for idx, (s, e) in enumerate(segs):
			if e < 0 or s >= seq_len:
				continue
			s2 = max(0, s)
			e2 = min(seq_len - 1, e)
			if s2 > e2:
				continue
			has_anom = True
			ax.axvspan(s2, e2, color='red', alpha=0.20)
			try:
				seg_idx = range(s2, e2 + 1)
				ax.scatter(seg_idx, [ts[i] for i in seg_idx], color='red', s=2, marker='o', edgecolors='black', linewidths=0.25, zorder=5)
			except Exception:
				pass
			if annotate_patterns and patterns and idx < len(patterns):
				p = patterns[idx]
				magnitude = magnitudes[idx]
				if p is not None and str(p).strip() != '':
					mid = (s2 + e2) // 2
					y_min, y_max = ax.get_ylim()
					pattern_mid = y_max
					magnitude_mid = y_min
					pattern_abbr = ''.join([word[0] for word in p.split('_')])
					ax.text(mid, pattern_mid, f"{pattern_abbr}", color='black', fontsize=7, ha='center', va='top' if text_positition_flag else 'bottom', rotation=0)
					ax.text(mid, magnitude_mid, f"{magnitude}", color='black', fontsize=5, ha='center', va='bottom' if text_positition_flag else 'top', rotation=0)
					text_positition_flag = 0 if text_positition_flag else 1
		ax.set_title(title, fontsize=10)
		ax.grid(alpha=0.3, linestyle='--', linewidth=0.4)
		if has_anom:
			from matplotlib.lines import Line2D
			custom_handles = [
				Line2D([0],[0], marker='o', color='red', markerfacecolor='red', markeredgecolor='black', markersize=5, linestyle='None', label='anomaly points')
			]
			ax.legend(handles=custom_handles, loc='upper right', fontsize=7, frameon=False)

	x_label_prefix = ""
	if human_intent:
		wrapped_intent = textwrap.fill(str(human_intent), width=90)
		x_label_prefix = f"Intent: {wrapped_intent}"

	if position_metrics:
		x_labels = []
		metrics_shown_l1 = ['range', 'standard', 'delay']
		for metric_type in metrics_shown_l1:
			if metric_type in position_metrics:
				metric = position_metrics[metric_type]
				x_labels.append(f'{metric_type}({metric["precision"]:.2f} P / {metric["recall"]:.2f} R / {metric["f1"]:.2f} F)')
		x_label_l1 = ", ".join(x_labels)
		x_labels = []
		metrics_shown_l2 = ['point', 'event', 'temporal']
		for metric_type in metrics_shown_l2:
			if metric_type in position_metrics:
				metric = position_metrics[metric_type]
				x_labels.append(f'{metric_type}({metric["precision"]:.2f} P / {metric["recall"]:.2f} R / {metric["f1"]:.2f} F)')
		x_label_l2 = ", ".join(x_labels)
		x_label = f'{x_label_l1}\n{x_label_l2}'
		if x_label_prefix:
			x_label = f'{x_label}\n{x_label_prefix}'
		axes[-1].set_xlabel(x_label, labelpad=8)
	elif x_label_prefix:
		axes[-1].set_xlabel(x_label_prefix, labelpad=8)
	fig.tight_layout()
	fig.savefig(out_dir / f'{metric_name}.pdf', dpi=300, bbox_inches='tight')
	plt.close(fig)


def evaluate_file(path: str,
			  evaluate_position: bool = True,
			  evaluate_pattern: bool = True,
			  delay_threshold: int = 7,
			  pattern_average: str = 'micro',
			  per_sample: bool = False,
			  include_group_records: bool = False,
			  format: str = 'interval',
			  margin: int = 0,
			  plot_dir: Optional[str] = None,
			  plot_anomaly: bool = True,
			  best_f1: Optional[str] = None,
			  profile: bool = False,
			  profile_topk: int = 10,
			  source_splits: Optional[List[Optional[str]]] = None,
			  filter_train: bool = False):
	
	def parse_group_key(metric: Optional[str]) -> Tuple[str, int, bool]:
		if not metric:
			return f"sample_group", 0, False
		parts = str(metric).split('##')
		if len(parts) >= 2 and parts[-1].isdigit():
			try:
				return '##'.join(parts[:-1]), int(parts[-1]), True
			except ValueError:
				return metric, 0, False
		return metric, 0, False

	#           TP/FP   
	def _overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
		return not (a[1] < b[0] or b[1] < a[0])

	def add_offset(segs: List[Tuple[int,int]], off: int) -> List[Tuple[int,int]]:
		return [(s + off, e + off) for s, e in segs]
	
	def add_margin(segs: List[Tuple[int,int]], margin: int, seq_len: int) -> List[Tuple[int,int]]:
		return [(max(s - margin, 0), min(e + margin, seq_len-1)) for s, e in segs]

	def merge_segments_with_attrs(
		segs: List[Tuple[int,int]],
		patterns: Optional[List[Any]] = None,
		magnitudes: Optional[List[float]] = None
	) -> Tuple[List[Tuple[int,int]], Optional[List[Any]], Optional[List[float]]]:
		if not segs:
			return [], ([] if patterns is not None else None), ([] if magnitudes is not None else None)
		items = list(zip(segs,
						 patterns if patterns is not None else [None]*len(segs),
						 magnitudes if magnitudes is not None else [None]*len(segs)))
		items.sort(key=lambda x: (x[0][0], x[0][1]))
		merged_segs: List[Tuple[int,int]] = []
		merged_pats: List[Any] = [] if patterns is not None else None
		merged_amps: List[float] = [] if magnitudes is not None else None
		cur_s, cur_e = items[0][0]
		cur_pat = items[0][1]
		cur_amp = items[0][2]
		for (s,e), p, a in items[1:]:
			if s <= cur_e+1:  # overlap merge only (      overlap)
				cur_e = max(cur_e, e)
				# pattern: keep the first non-empty; if same keep; else keep existing
				if merged_pats is not None:
					if cur_pat is None or (isinstance(cur_pat, str) and cur_pat.strip() == ''):
						cur_pat = p
				# magnitude: keep max numeric
				if merged_amps is not None:
					try:
						fa = float(a) if a is not None else None
						fc = float(cur_amp) if cur_amp is not None else None
						if fc is None:
							cur_amp = fa
						elif fa is not None and fa > fc:
							cur_amp = fa
					except (TypeError, ValueError):
						pass
			else:
				merged_segs.append((cur_s, cur_e))
				if merged_pats is not None:
					merged_pats.append(cur_pat)
				if merged_amps is not None:
					merged_amps.append(cur_amp)
				cur_s, cur_e = s, e
				cur_pat = p
				cur_amp = a
		# append last
		merged_segs.append((cur_s, cur_e))
		if merged_pats is not None:
			merged_pats.append(cur_pat)
		if merged_amps is not None:
			merged_amps.append(cur_amp)
		return merged_segs, merged_pats, merged_amps

	def preprocess_segments(true_positions_abs, true_patterns_all, true_magnitudes_all,
						 pred_positions_abs, pred_patterns_all, pred_magnitudes_all):
		if true_positions_abs:
			combined = list(zip(true_positions_abs, true_patterns_all, true_magnitudes_all))
			combined_sorted = sorted(combined, key=lambda x: x[0][0])
			true_positions_abs, true_patterns_all, true_magnitudes_all = zip(*combined_sorted)
			true_positions_abs, true_patterns_all, true_magnitudes_all = list(true_positions_abs), list(true_patterns_all), list(true_magnitudes_all)
		if pred_positions_abs:
			combined = list(zip(pred_positions_abs, pred_patterns_all, pred_magnitudes_all))
			combined_sorted = sorted(combined, key=lambda x: x[0][0])
			pred_positions_abs, pred_patterns_all, pred_magnitudes_all = zip(*combined_sorted)
			pred_positions_abs, pred_patterns_all, pred_magnitudes_all = list(pred_positions_abs), list(pred_patterns_all), list(pred_magnitudes_all)

		true_positions_abs, true_patterns_all, true_magnitudes_all = merge_segments_with_attrs(
			true_positions_abs, true_patterns_all if true_patterns_all else None, true_magnitudes_all if true_magnitudes_all else None
		)
		pred_positions_abs, pred_patterns_all, pred_magnitudes_all = merge_segments_with_attrs(
			pred_positions_abs, pred_patterns_all if pred_patterns_all else None, pred_magnitudes_all if pred_magnitudes_all else None
		)
		return true_positions_abs, true_patterns_all, true_magnitudes_all, pred_positions_abs, pred_patterns_all, pred_magnitudes_all
	
	position_evaluator = PositionEvaluator(delay_threshold=delay_threshold, margin=margin) if evaluate_position else None

	all_true_patterns: List[Any] = [] if evaluate_pattern else []
	all_pred_patterns: List[Any] = [] if evaluate_pattern else []
	per_sample_results = []

	#        pattern                 pattern    TP/FP      pattern      
	pattern_detection_per_pred: Dict[Any, Dict[str, int]] = {}
	total_tp = 0
	total_fp = 0
	group_eval_records: List[Dict[str, Any]] = []

	agg_position: Dict[str, Dict[str, float]] = { metric_type: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for metric_type in position_evaluator.metric_types } if evaluate_position else {}
	n_samples = 0

	if plot_dir is not None:
		plot_dir.mkdir(parents=True, exist_ok=True)

	# profiling containers
	t_total_start = time.perf_counter() if profile else None
	t_read_start = time.perf_counter() if profile else None
	# --- helpers for grouping & merging ---

	with open(path, 'r', encoding='utf-8') as f:
		raw_records = [json.loads(line.strip()) for line in f]
	if filter_train:
		if source_splits is None:
			print("[WARNING] --filter-train enabled but source_split metadata is unavailable; skip filtering.")
		else:
			n_raw = len(raw_records)
			n_split = len(source_splits)
			n_use = min(n_raw, n_split)
			if n_raw != n_split:
				print(f"[WARNING] raw_records ({n_raw}) and source_splits ({n_split}) length mismatch; use first {n_use} cases.")
			filtered_records: List[Dict[str, Any]] = []
			for idx in range(n_use):
				if source_splits[idx] == 'train':
					continue
				filtered_records.append(raw_records[idx])
			raw_records = filtered_records
			print(f"[INFO] filter_train enabled: kept {len(raw_records)} / {n_use} cases (excluded source_split=='train').")
	t_read = (time.perf_counter() - t_read_start) if profile else 0.0

	# Group records by base metric_name (without the last ##idx)
	t_group_start = time.perf_counter() if profile else None
	groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for rec in raw_records:
		input_text = rec.get('input', '')
		timeseries = rec.get('timeseries')
		metric_name_rec = rec.get('metric', None)
		seq_len = extract_sequence_length(input_text)
		human_inent = extract_human_intent(input_text)
		if seq_len is None and timeseries:
			if isinstance(timeseries, list) and timeseries:
				if isinstance(timeseries[0], list):
					seq_len = len(timeseries[0])
				else:
					seq_len = len(timeseries)
		# If still None, skip this record entirely
		if seq_len is None:
			continue
		metric_name = extract_metric_name(input_text) or metric_name_rec or "sample"
		gkey, gidx, parse_success = parse_group_key(metric_name)
		if gkey in groups and not parse_success:
			gkey = f"{gkey}_extra_{len(groups)}"
			gidx = 0

		rec['_metric_name'] = metric_name
		rec['_group_key'] = gkey
		rec['_group_idx'] = gidx
		rec['_seq_len'] = seq_len
		rec['_human_intent'] = human_inent
		groups[gkey].append(rec)
	t_group = (time.perf_counter() - t_group_start) if profile else 0.0

	# per-group profiling accumulators
	t_parse_offset_total = 0.0
	t_merge_total = 0.0
	t_position_eval_total = 0.0
	t_pattern_match_total = 0.0
	t_pattern_detect_total = 0.0
	t_plot_total = 0.0
	slow_groups: List[Dict[str, Any]] = []

	# Iterate groups (each represents a full underlying sequence combined from fragments)
	for gkey, recs in tqdm(groups.items(), desc=f"Calculating metrics for {path} (grouped):"):
		# sort fragments by their index
		recs_sorted = sorted(recs, key=lambda r: r.get('_group_idx', 0))
		offset = 0
		# aggregated containers (absolute positions)
		all_timeseries: List[Any] = []
		true_positions_abs: List[Tuple[int,int]] = []
		pred_positions_abs: List[Tuple[int,int]] = []
		true_patterns_all: List[Any] = []
		pred_patterns_all: List[Any] = []
		true_magnitudes_all: List[float] = []
		pred_magnitudes_all: List[float] = []

		t_parse_offset_start = time.perf_counter() if profile else None
		for rec in recs_sorted:
			output_text = rec.get('output', '')
			label_text = rec.get('label', '')
			seq_len = rec.get('_seq_len', 0)
			human_intent = rec.get('_human_intent', None)
			timeseries = rec.get('timeseries')

			pred_pos, pred_pats, pred_amps = LLMOutputParser.parse(output_text, format, answer_tag=True)
			true_pos, true_pats, true_amps = LLMOutputParser.parse(label_text, answer_tag=False)

			pred_pos = filter_segments(pred_pos)
			true_pos = filter_segments(true_pos)

			# convert to absolute by adding current offset
			if pred_pos:
				abs_pred = add_offset(pred_pos, offset)
				pred_positions_abs.extend(abs_pred)
				if pred_pats:
					pred_patterns_all.extend(pred_pats)
				if pred_amps:
					pred_magnitudes_all.extend(pred_amps)
			if true_pos:
				abs_true = add_offset(true_pos, offset)
				true_positions_abs.extend(abs_true)
				if true_pats:
					true_patterns_all.extend(true_pats)
				if true_amps:
					true_magnitudes_all.extend(true_amps)
			if timeseries is not None:
				all_timeseries.append(timeseries[0] if isinstance(timeseries[0], list) else timeseries)

			offset += int(seq_len)
		t_parse_offset = (time.perf_counter() - t_parse_offset_start) if profile else 0.0
		t_parse_offset_total += t_parse_offset

		total_seq_len = offset
		# true_positions_abs[-1] = (min(true_positions_abs[-1][0],total_seq_len-1), min(true_positions_abs[-1][1], total_seq_len-1))

		# Merge overlapping segments at group level (   overlap   )
		t_merge_start = time.perf_counter() if profile else None
		# true_positions_abs = add_margin(true_positions_abs, margin, total_seq_len)

		true_positions_abs, true_patterns_all, true_magnitudes_all, pred_positions_abs, pred_patterns_all, pred_magnitudes_all = preprocess_segments(true_positions_abs, true_patterns_all, true_magnitudes_all, pred_positions_abs, pred_patterns_all, pred_magnitudes_all)

		all_timeseries = sum(all_timeseries, [])
		t_merge = (time.perf_counter() - t_merge_start) if profile else 0.0
		t_merge_total += t_merge

		# Position evaluation (optional)
		position_metrics = None
		t_pos_eval = 0.0
		if evaluate_position and position_evaluator is not None:
			# Prefer using magnitudes to sweep thresholds if available
			t_pos_start = time.perf_counter() if profile else None
			if best_f1 and best_f1=='magnitude' and pred_magnitudes_all is not None and len(pred_magnitudes_all) == len(pred_positions_abs) and len(pred_positions_abs) > 0:
				position_metrics_1 = position_evaluator.evaluate(true_positions_abs, pred_positions_abs, pred_magnitudes_all, total_seq_len, metric_types = ['standard', 'temporal'], best=True)
			else:
				position_metrics_1 = position_evaluator.evaluate(true_positions_abs, pred_positions_abs, total_seq_len, metric_types = ['standard', 'temporal'])
			true_positions_abs = add_margin(true_positions_abs, margin, total_seq_len)
			true_positions_abs, true_patterns_all, true_magnitudes_all, pred_positions_abs, pred_patterns_all, pred_magnitudes_all = preprocess_segments(true_positions_abs, true_patterns_all, true_magnitudes_all, pred_positions_abs, pred_patterns_all, pred_magnitudes_all)
			if best_f1 and best_f1=='magnitude' and pred_magnitudes_all is not None and len(pred_magnitudes_all) == len(pred_positions_abs) and len(pred_positions_abs) > 0:
				position_metrics_2 = position_evaluator.evaluate(true_positions_abs, pred_positions_abs, pred_magnitudes_all, total_seq_len, metric_types = ['range', 'point', 'delay', 'event'], best=True)
			else:
				position_metrics_2 = position_evaluator.evaluate(true_positions_abs, pred_positions_abs, total_seq_len, metric_types = ['range', 'point', 'delay', 'event'])
			position_metrics = position_metrics_1 | position_metrics_2
			# position_metrics = position_evaluator.evaluate(true_positions_abs, pred_positions_abs, total_seq_len)
			t_pos_eval = (time.perf_counter() - t_pos_start) if profile else 0.0
			t_position_eval_total += t_pos_eval
			for k in position_evaluator.metric_types:
				for m in ['precision', 'recall', 'f1']:
					agg_position[k][m] += position_metrics[k][m]

		# Pattern matching & accumulation (only for pattern classification metrics)
		t_pat_match = 0.0
		if evaluate_pattern:
			# pad patterns if lengths mismatch (conservative)
			if pred_patterns_all is None:
				pred_patterns_all = []
			if true_patterns_all is None:
				true_patterns_all = []
			# Ensure pattern list lengths equal to segment lengths by padding with None
			if len(pred_patterns_all) < len(pred_positions_abs):
				pred_patterns_all.extend([None] * (len(pred_positions_abs) - len(pred_patterns_all)))
			if len(true_patterns_all) < len(true_positions_abs):
				true_patterns_all.extend([None] * (len(true_positions_abs) - len(true_patterns_all)))
			t_pat_start = time.perf_counter() if profile else None
			y_true_pat_aligned, y_pred_pat_aligned = match_patterns(true_positions_abs, true_patterns_all, pred_positions_abs, pred_patterns_all)
			t_pat_match = (time.perf_counter() - t_pat_start) if profile else 0.0
			t_pattern_match_total += t_pat_match
			all_true_patterns.extend(y_true_pat_aligned)
			all_pred_patterns.extend(y_pred_pat_aligned)
		else:
			y_true_pat_aligned, y_pred_pat_aligned = [], []
		
		t_pd_start = time.perf_counter() if profile else None
		group_pattern_detection: Dict[Any, Dict[str, int]] = {}
		for idx, seg in enumerate(pred_positions_abs):
			p_pat = None
			if pred_patterns_all is not None and idx < len(pred_patterns_all):
				p_pat = pred_patterns_all[idx]
			if p_pat is None or (isinstance(p_pat, str) and p_pat.strip() == ''):
				continue
			is_tp = any(_overlap(seg, tseg) for tseg in true_positions_abs)
			stats = pattern_detection_per_pred.setdefault(p_pat, {'tp': 0, 'fp': 0})
			group_stats = group_pattern_detection.setdefault(p_pat, {'tp': 0, 'fp': 0})
			if is_tp:
				stats['tp'] += 1
				group_stats['tp'] += 1
				total_tp += 1
			else:
				stats['fp'] += 1
				group_stats['fp'] += 1
				total_fp += 1
		t_pd = (time.perf_counter() - t_pd_start) if profile else 0.0
		t_pattern_detect_total += t_pd

		if include_group_records:
			group_eval_records.append({
				'group_index': n_samples,
				'metric_group': gkey,
				'position_metrics': position_metrics if evaluate_position else {},
				'true_patterns_aligned': y_true_pat_aligned if evaluate_pattern else [],
				'pred_patterns_aligned': y_pred_pat_aligned if evaluate_pattern else [],
				'pattern_detection_per_pred': group_pattern_detection,
			})

		if per_sample:
			entry = {
				'metric_group': gkey,
				'sequence_length': total_seq_len,
				'true_positions': true_positions_abs,
				'pred_positions': pred_positions_abs,
			}
			if evaluate_position:
				entry['position_metrics'] = position_metrics
			if evaluate_pattern:
				entry['true_patterns_aligned'] = y_true_pat_aligned
				entry['pred_patterns_aligned'] = y_pred_pat_aligned
			per_sample_results.append(entry)

		if plot_dir is not None:
			if (plot_anomaly and (true_positions_abs or pred_positions_abs)) or (not plot_anomaly):
				# Plot with placeholder timeseries (None)
				t_plot_start = time.perf_counter() if profile else None
				plot_sample(total_seq_len, true_positions_abs, pred_positions_abs,
							true_patterns_all or [], pred_patterns_all or [],
							true_magnitudes_all or [], pred_magnitudes_all or [],
							all_timeseries, plot_dir, gkey, position_metrics, human_intent)
				if profile:
					t_plot_total += (time.perf_counter() - t_plot_start)
		n_samples += 1

		# collect per-group summary (useful to identify slow ones)
		if profile:
			num_pred = len(pred_positions_abs)
			num_true = len(true_positions_abs)
			# estimate thresholds count
			if pred_magnitudes_all is not None and len(pred_magnitudes_all) == len(pred_positions_abs) and len(pred_positions_abs) > 0:
				try:
					k_th = len(set(float(a) for a in pred_magnitudes_all)) + 1
				except Exception:
					k_th = len(pred_positions_abs) + 1
			else:
				k_th = 1
			slow_groups.append({
				'group': gkey,
				'seq_len': total_seq_len,
				'num_pred_segments': num_pred,
				'num_true_segments': num_true,
				'thresholds_swept': k_th,
				't_parse_offset': t_parse_offset,
				't_merge': t_merge,
				't_position_eval': t_pos_eval,
				't_pattern_match': t_pat_match,
				't_pattern_detect': t_pd,
			})

	if n_samples == 0:
		return {'error': 'No valid samples found.'}

	# Average position metrics if evaluated (n_samples now equals number of aggregated sequences)
	if evaluate_position and n_samples > 0:
		for k in agg_position:
			for m in agg_position[k]:
				agg_position[k][m] /= n_samples

	pattern_metrics = None
	unique_labels: List[Any] = []
	if evaluate_pattern:
		unique_labels = sorted(set(p for p in all_true_patterns + all_pred_patterns if p is not None))
		if any(p is None for p in all_true_patterns + all_pred_patterns):
			unique_labels.append(None)
		pattern_evaluator = PatternEvaluator(average=pattern_average, labels=unique_labels)
		pattern_metrics = pattern_evaluator.evaluate(all_true_patterns, all_pred_patterns)

	result: Dict[str, Any] = {
		'num_samples': n_samples,
	}
	if evaluate_position:
		result['position_metrics_avg'] = agg_position
	if evaluate_pattern:
		result['pattern_metrics'] = pattern_metrics
		result['pattern_labels'] = unique_labels
	if include_group_records:
		result['_group_eval_records'] = group_eval_records

	for stats in pattern_detection_per_pred.values():
		stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0.0
	#   pattern_detection_per_pred value precision      
	pattern_detection_per_pred = dict(sorted(pattern_detection_per_pred.items(), key=lambda item: item[1]['precision'], reverse=True))
	pattern_detection_stats = {
		'per_pattern': pattern_detection_per_pred,
		'total': {'tp': total_tp, 'fp': total_fp, 'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0}
	}
	result['pattern_detection_stats'] = pattern_detection_stats
	if per_sample:
		result['per_sample'] = per_sample_results

	# profiling summary
	if profile:
		# sort top slow groups by position eval time then by seq_len
		top_groups = sorted(slow_groups, key=lambda x: (x['t_position_eval'], x['seq_len']), reverse=True)[:max(1, profile_topk)]
		t_total = (time.perf_counter() - t_total_start) if t_total_start is not None else 0.0
		result['profile'] = {
			'total_seconds': t_total,
			'read_seconds': t_read,
			'grouping_seconds': t_group,
			'parse_offset_seconds': t_parse_offset_total,
			'merge_seconds': t_merge_total,
			'position_eval_seconds': t_position_eval_total,
			'pattern_match_seconds': t_pattern_match_total,
			'pattern_detect_seconds': t_pattern_detect_total,
			'plot_seconds': t_plot_total,
			'num_groups': len(groups),
			'top_slow_groups': top_groups,
		}
	return result


def main():
	parser = argparse.ArgumentParser(description='Evaluate LLM anomaly detection outputs.')
	parser.add_argument('-i', '--input-dir', type=str, default="../outputs", help='Path of model output path.')
	parser.add_argument('-o', '--result-dir', type=str, default="../results-all/results", help='Path to save aggregated metrics JSON.')
	parser.add_argument('--split', type=str, default="default", choices=["default", "short", "season"], help='Dataset split/length mode.')
	parser.add_argument('--rule', type=str, default=None, choices=['v1', None], help='Enable extra rule on datasets.')
	parser.add_argument('--plot', action='store_true', help='Generate comparison plots (ground truth vs prediction).')
	parser.add_argument('--plot-anomaly', action='store_true', help='Plot only anomalies.')
	parser.add_argument('--plot-dir', type=str, default='../plots', help='Directory to save plots if --plot is set.')
	parser.add_argument('--delay', type=int, default=15, help='Delay threshold for delay F1.')
	parser.add_argument('--pattern-average', type=str, default='macro', choices=['micro', 'macro', 'weighted', 'binary'], help='Average method for pattern evaluation.')
	parser.add_argument('--per-sample', action='store_true', help='Include per-sample metrics in output.')
	parser.add_argument('--no-position', action='store_true', help='Disable position evaluation.')
	parser.add_argument('--no-pattern', action='store_true', help='Disable pattern evaluation.')
	parser.add_argument('--margin', action='store_true', help='Enable margin on groundtruth.')
	parser.add_argument('--best-f1', type=str, default=None, choices=['magnitude', None], help='Enable best f1 calculation.')
	parser.add_argument('--profile', action='store_true', help='Enable timing profile during evaluation.')
	parser.add_argument('--profile-topk', type=int, default=10, help='Show top-K slow sequences in profile report.')
	parser.add_argument('--filter-train', action='store_true', help="Filter out cases with source_split == 'train' using raw QA dataset metadata.")
	args = parser.parse_args()
	
	margin_dict = {
		'AIOPS': 5, 'NAB': 0, 'TODS': 5, 'UCR': 50, 'WSD': 3, 'Yahoo': 5, 'Service': 5, 'Dodgers': 3, 'Power': 5,
		'TAO': 5, 'SMD': 5, 'Stock': 3, 'MSL': 5, 'SMAP': 5, 'OPPORTUNITY': 5, 'SED': 5
	}
	datanames = ['AIOPS', 'NAB', 'TODS', 'UCR', 'WSD', 'Yahoo', 'Service']
	generated_datanames = ['20251004_163407_test_1000', '20251010_071208_test_1000', '20251010_071208_test_1000_magnitude', '20251021_082525_test_1000', '20251022_022214_test_1000']
	datanames = ['NAB', 'AIOPS', 'WSD', 'Yahoo', 'Service', 'TODS', 'UCR', 'Dodgers']
	datanames = ['NAB', 'AIOPS', 'WSD', 'Yahoo', 'Service', 'TODS', 'UCR', 'Dodgers', "Power", "TAO", "SMD", "Stock", "MSL", "SMAP", "OPPORTUNITY", "SED"]
	datanames = []
	generated_datanames = ["intentad_bench"]
	datanames = ['AIOPS']	# 'AIOPS', 'WSD', 'Yahoo', 'Service'
	generated_datanames = []
	all_datanames = datanames + generated_datanames
	
	best_f1 = args.best_f1	# None, "magnitude"
	split = args.split	# default, short, season
	rule_version = args.rule # None, 'v1'
	models = {
		# "TSLLM-AD-VITS-v1.0-20251028": "interval",
		# "TSLLM-AD-v1.1-20251001": "interval",
		# "TSLLM-AD-v1.2-20251009": "interval",
		# "TSLLM-AD-v1.3-20251011": "interval",
		# "TSLLM-AD-v1.4-20251015": "interval",
		# "TSLLM-AD-v1.5-20251021": "interval",
		# "TSLLM-AD-v2.0-20251027": "interval",
		# "TSLLM-AD-v3.0-20251109": "interval",
		# "TSLLM-AD-v3.1-20251110": "interval",
		# "TSLLM-AD-v3.2-20251110": "interval",
		# "TSLLM-AD-v3.3-20251110": "interval",
		# "TSLLM-AD-v3.4-20251111": "interval",
		# "TSLLM-AD-v3.5-20251111": "interval",
        # "TSLLM-AD-v3.6-20251112": "interval",
		# "TSLLM-AD-v3.7-20251113": "interval",
		# "TSLLM-AD-v3.8-20251115": "interval",
		# "TSLLM-AD-v3.9-20251115": "interval",
		# "TSLLM-AD-v3.10-20251117": "interval",
		# "TSLLM-AD-v3.10-rl-20251117": "interval",
		# "TSLLM-AD-v3.11-20251117": "interval",
		# "TSLLM-AD-v3.11-rl-20251117": "interval",
		# "TSLLM-AD-v3.11-rl-2-20251117": "interval",
		# "TSLLM-AD-v4.0-20251120": "interval",
		# "TSLLM-AD-v4.0-20251120-DAPO-v1-20251121": "interval",
		# "TSLLM-AD-v4.1-20251125": "interval",
		# "TSLLM-AD-v4.1-20251125-DAPO-v1-20251125": "interval",
		# "TSLLM-AD-v5.2-20251204": "interval",
		# "TSLLM-AD-v5.5-20251209": "interval",
		# "TSLLM-AD-v5.5-20251209-DAPO-v1-20251209": "interval",
		# "TSLLM-AD-v5.6-20251210": "interval",
		# "TSLLM-AD-v5.8-20251210": "interval",
		# "TSLLM-AD-v5.10-20251210": "interval",
		# "TSLLM-AD-v5.10-20251210-DAPO-v1-20251210": "interval",
		# "TSLLM-AD-v5.10-20251210-DAPO-v1.1-20251210": "interval",
		# "TSLLM-AD-v5.10-20251210-DAPO-v1.3-20251210": "interval",
		# "TSLLM-AD-v5.11-20251215": "interval",
		# "TSLLM-AD-v5.12-20251215": "interval",
		# "TSLLM-AD-v5.12-20251215-DAPO-v1.1-20251215": "interval",
		# "TSLLM-AD-v6.0-20251217": "interval",
		# "TSLLM-AD-v6.1-20251217": "interval",
		# "TSLLM-AD-v6.2-20251217": "interval",
		# "TSLLM-AD-v6.2-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v6.3-20251217": "interval",
		# "TSLLM-AD-v6.3-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v6.4-20251217": "interval",
		# "TSLLM-AD-v6.5-20251217": "interval",
		# "TSLLM-AD-v6.6-20251217": "interval",
		# "TSLLM-AD-v7.0-20251217": "interval",
		# "TSLLM-AD-v7.1-20251217": "interval",
		# "TSLLM-AD-v7.2-20251217": "interval",
        # "TSLLM-AD-v7.1-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v7.2-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v7.3-20251217": "interval",
		# "TSLLM-AD-v7.3-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v7.4-20251217": "interval",
		# "TSLLM-AD-v7.5-20251217": "interval",
		# "TSLLM-AD-v7.5-20251217-DAPO-v1.0-20251217": "interval",
		# "TSLLM-AD-v8.0-20251225": "interval",
        # "TSLLM-AD-v8.1-20251225": "interval",
		# "TSLLM-AD-v8.1-20251225-DAPO-v1.0-20251225": "interval",
		# "TSLLM-AD-v8.2-20251225": "interval",
        # "TSLLM-AD-v8.3-20251225": "interval",
		# "TSLLM-AD-v8.3-20251225-DAPO-v1.0-20251225": "interval",
		# "TSLLM-AD-v8.4-20251225": "interval",
        # "TSLLM-AD-v8.5-20251225": "interval",
		# "TSLLM-AD-v8.5-20251225-DAPO-v1.0-20251225": "interval",
		# "TSLLM-AD-v8.6-20251225": "interval",
        # "TSLLM-AD-v8.7-20251225": "interval",
		# "TSLLM-AD-v8.7-20251225-DAPO-v1.0-20251225": "interval",

		# "TSLLM-AD-TEXT-v1.0-20251028": "interval",
		# "TSLLM-AD-VITS-TEXT-v1.0-20251029": "interval",
		# "TSLLM-AD-VITS-TEXT-v1.1-20251031": "interval",
		# "TSLLM-AD-VITS-TEXT-v1.2-20251104": "interval",
		# "TSLLM-AD-VITS-TEXT-v1.3-20251105": "interval",
		# "TSLLM-AD-VITS-TEXT-v2.0-20251105": "index",


		# "HUMAN-AD-v1.0-20260106": "interval",
		# "HUMAN-AD-ASE-v1.1-20260307-DAPO-v1.0-20260307": "interval",
		# "HUMAN-AD-ASE-v1.1-20260307-DAPO-v1.1-20260307": "interval",
		# "HUMAN-AD-ASE-v2.2-20260323": "interval",
		# "HUMAN-AD-ASE-v2.2-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.2-MEMORY-20260323-DAPO-v2.0-20260323": "interval",
		# "HUMAN-AD-ASE-v2.2-MEMORY-20260323-DAPO-v2.1-20260323": "interval",
		# "HUMAN-AD-ASE-v2.3-20260407": "interval",
		# "HUMAN-AD-ASE-v2.3-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.3-MEMORY-20260408-DAPO-v3.0-20260408": "interval",
		# "HUMAN-AD-ASE-v2.4-20260411": "interval",
		# "HUMAN-AD-ASE-v2.4-20260411-wo-DAP": "interval",
		# "HUMAN-AD-ASE-v2.4-20260411-wo-SIP": "interval",
		# "HUMAN-AD-ASE-v2.4-20260411-wo-thinking": "interval",
		# "HUMAN-AD-ASE-v2.4-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.4-wo-DAP-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.4-wo-SIP-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.4-wo-DAP-MEMORY-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.4-wo-SIP-MEMORY-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.4-AMR-noretrieve-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.4-MEMORY-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.4-MEMORY-20260412-DAPO-v4.1-20260412": "interval",
		# "HUMAN-AD-ASE-v2.4-MEMORY-WO-STAGE1": "interval",
		# "HUMAN-AD-ASE-v2.4-MEMORY-WO-STAGE1-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.5-20260411": "interval",
		# "HUMAN-AD-ASE-v2.5-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.6-20260411": "interval",
		# "HUMAN-AD-ASE-v2.6-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.6.1-MEMORY": "interval",
		# "HUMAN-AD-ASE-v2.6-MEMORY-20260412-DAPO-v4.0-20260412": "interval",
		# "HUMAN-AD-ASE-v2.6.1-MEMORY-20260412-DAPO-v4.0-20260412": "interval",


		# "ChatTS-14B": "interval",
		# "Qwen2.5-14B-Instruct": "interval",
		# "Qwen2.5-VL-7B-Instruct": "interval",
		# "GPT-5.2-Text": "interval",
		# "GPT-5.2-Vision": "interval",
		"GPT-5.4-Text": "interval",
		# "GPT-5.4-Vision": "interval",
		# "DeepSeek-R1": "interval",

		# "ChatTS-14B-Agent": "interval",
		# "Qwen2.5-14B-Instruct-Agent": "interval",
		# "Qwen2.5-VL-7B-Instruct-Agent": "interval",
		# "GPT-5.4-Text-Agent": "interval",
		# "GPT-5.4-Vision-Agent": "interval",
	}
	output_root = args.input_dir if split == 'default' else args.input_dir + f'_{split}'
	plot_root = args.plot_dir if split == 'default' else args.plot_dir + f'_{split}'
	result_root = args.result_dir if split == 'default' else args.result_dir + f'_{split}'
	result_root = result_root if best_f1 is None else result_root + f'_bestby_{best_f1}'
	output_root = Path(output_root)
	plot_root = Path(plot_root)
	result_root = Path(result_root)

	for model_name, format in models.items():
		for dataname in all_datanames:
			print(f"[INFO] data: {dataname}...")
			statistics_path = infer_statistics_path_for_generated(dataname)
			statistics_details = None
			source_splits = None
			if statistics_path is not None:
				statistics_details = load_statistics_details(statistics_path)
				if statistics_details is None:
					print(f"[WARNING] statistics file not found or invalid for generated dataset: {statistics_path}")
				else:
					print(f"[INFO] loaded statistics details from {statistics_path}")
			else:
				base_dir = Path(__file__).resolve().parent.parent
				raw_dataset_path = base_dir / 'data' / 'llm_sft' / 'no_cot' / 'filter' / f'QA_{dataname}.jsonl'
				source_splits = load_source_splits(raw_dataset_path)
				if source_splits is None:
					print(f"[WARNING] source_split metadata not available: {raw_dataset_path}")
				else:
					print(f"[INFO] loaded source_split metadata from {raw_dataset_path} ({len(source_splits)} cases)")
			if rule_version:
				DATA_NAME = f"QA_{dataname}_rule_{rule_version}"
			else:
				DATA_NAME = f"QA_{dataname}"
			model_output_path = output_root / DATA_NAME / f"{model_name}.jsonl"
			if not model_output_path.exists():
				print(f"[WARNING] Model output path {model_output_path} does not exist. Skipping...")
				continue
			plot_dir = plot_root / DATA_NAME / model_name if args.plot else None
			try:
				metrics = evaluate_file(
					model_output_path,
					evaluate_position=not args.no_position,
					evaluate_pattern=not args.no_pattern,
					delay_threshold=args.delay,
					pattern_average=args.pattern_average,
					per_sample=args.per_sample,
					include_group_records=(statistics_details is not None),
					format=format,
					margin=margin_dict.get(dataname, 3)  if args.margin else 0,
					plot_dir=plot_dir,
					plot_anomaly=args.plot_anomaly,
					best_f1=args.best_f1,
					profile=args.profile,
					profile_topk=args.profile_topk,
					source_splits=source_splits,
					filter_train=args.filter_train,
				)
			except Exception as e:
				print(f"[ERROR] {e} when processing {model_name} on {dataname}")
				continue

			if statistics_details is not None:
				group_eval_records = metrics.pop('_group_eval_records', [])
				if isinstance(group_eval_records, list) and group_eval_records:
					metrics['taxonomy_type_results'] = {
						'statistics_path': str(statistics_path),
						'case_mapping': {
							'total_groups': len(group_eval_records),
							'mapping_rule': 'case_id maps to grouped sample index by evaluation order',
						},
						'by_taxonomy': build_taxonomy_subset_results(
							group_eval_records=group_eval_records,
							statistics_details=statistics_details,
							evaluate_position=not args.no_position,
							evaluate_pattern=not args.no_pattern,
							pattern_average=args.pattern_average,
						),
					}
				else:
					metrics['taxonomy_type_results'] = {
						'statistics_path': str(statistics_path),
						'warning': 'group-level records unavailable, taxonomy subset results skipped',
					}
			if args.result_dir:
				result_dir = result_root / DATA_NAME
				result_dir.mkdir(parents=True, exist_ok=True)
				result_path = result_dir / f"{model_name}.json"
				with open(result_path, 'w', encoding='utf-8') as wf:
					json.dump(metrics, wf, ensure_ascii=False, indent=2)
			print(f"[INFO] metric calculated for model [{model_name}] on dataset {dataname}...")
			print(f"[INFO] results saved to {result_path}")
			# print(json.dumps(metrics, ensure_ascii=False, indent=2))
			# print()


if __name__ == '__main__':
	main()
