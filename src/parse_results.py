import os
import re
import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

METRIC_TYPES = ["range", "standard", "delay"]

def remove_dates(text):
    return re.sub(r'-\d{8}', '', text)


def find_project_root(start: str) -> str:
	"""Best-effort: use the directory containing this file as project root."""
	# In this repo, src lives directly under the project root
	return os.path.abspath(os.path.join(start, os.pardir))


def load_metrics(json_path: str, metric_types: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
	f"""Load F1 values for different metric types from a result json.

	Reads data["position_metrics_avg"][metric_type]["f1"] for each metric_type
	in metric_types (default: {METRIC_TYPES}).

	Returns:
		Dict[str, float]: mapping metric_type -> f1
		or None on failure.
	"""
	metric_types = metric_types or METRIC_TYPES
	try:
		with open(json_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		pos_metrics = data.get("position_metrics_avg", {})
		if not isinstance(pos_metrics, dict):
			return None

		out: Dict[str, float] = {}
		for mt in metric_types:
			mt_val = pos_metrics.get(mt, None)
			if isinstance(mt_val, dict) and "f1" in mt_val:
				try:
					out[mt] = float(mt_val["f1"])
				except Exception:
					# skip bad value
					pass
		return out if out else None
	except Exception:
		return None


def scan_results(results_dir: str, allowed_datasets: Optional[List[str]] = None):
	"""Scan results directory.

	Returns:
		datasets: List[str] - dataset folder names
		models: Set[str] - union of model names across datasets
		values: Dict[metric][model][dataset] -> float
	"""
	# --- Sort models by parsed version if matches TSLLM-AD-vX[.Y[.Z]]-DATE ---
	def _extract_version_tuple(name):
		pattern = r'v(\d+)\.(\d+)(?:.*?DAPO-v(\d+)(?:\.(\d+))?)?'
		match = re.search(pattern, name)
		
		if not match:
			return None
		
		versions = []
		versions.append(int(match.group(1)))
		versions.append(int(match.group(2)))
		
		# DAPO  
		if match.group(3):  #  DAPO   
			versions.append(int(match.group(3)))
			if match.group(4):  #  DAPO   
				versions.append(int(match.group(4)))
			else:  #        0
				versions.append(0)
		
		return tuple(versions)
		# mobj = re.search(r"TSLLM-AD-v(\d+(?:\.\d+)*)", name)
		# if not mobj:
		# 	return None
		# ver_str = mobj.group(1)
		# try:
		# 	return tuple(int(p) for p in ver_str.split('.'))
		# except Exception:
		# 	return None
		
	if not os.path.isdir(results_dir):
		raise FileNotFoundError(f"Results directory not found: {results_dir}")

	datasets: List[str] = [
		d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
	]
	# sort according to allowed_datasets
	datasets.sort(key=lambda d: allowed_datasets.index(d.split("QA_")[-1].replace('_', '-')) if allowed_datasets and d.split("QA_")[-1].replace('_', '-') in allowed_datasets else len(allowed_datasets) + 1)
	# datasets.sort()
	dataset_dict = {d.split("QA_")[-1].replace('_', '-'): d for d in datasets}
	print(dataset_dict.keys(), allowed_datasets)

	# We only care about F1 for each metric type
	metrics = METRIC_TYPES
	models: Set[str] = set()
	values: Dict[str, Dict[str, Dict[str, float]]] = {
		m: defaultdict(dict) for m in metrics
	}

	final_datasets = []
	for dataset_name, dataset in dataset_dict.items():
		if dataset_name not in allowed_datasets:
			continue
		final_datasets.append(dataset_name)
		dpath = os.path.join(results_dir, dataset)
		for fname in sorted(os.listdir(dpath)):
			if not fname.endswith(".json"):
				continue
			model = os.path.splitext(fname)[0]
			fpath = os.path.join(dpath, fname)
			m = load_metrics(fpath, metrics)
			if m is None:
				continue
			models.add(model)
			for k in metrics:
				values[k][model][dataset_name] = m.get(k)

	parsed: List[Tuple[Tuple[int, ...], str]] = []
	unparsed: List[str] = []
	for name in models:
		vt = _extract_version_tuple(name)
		if vt is None:
			unparsed.append(name)
		else:
			parsed.append((vt, name))

	parsed.sort(key=lambda x: x[0])
	sorted_models = [n for _, n in parsed] + sorted(unparsed)

	return final_datasets, sorted_models, values


def fmt(value: Optional[float], decimals: int = 3) -> str:
	if value is None:
		return "--"
	try:
		return f"{value:.{decimals}f}"
	except Exception:
		return "--"


def render_latex_table(
	datasets: List[str],
	models: List[str],
	values: Dict[str, Dict[str, Dict[str, float]]],
	decimals: int = 3,
	label: str = "tbl:f1_all",
	title_highlight: str = "",
) -> str:
	"""Render a LaTeX tabular where:
	- Columns: Metric | Model | <dataset...>
	- Rows: for each metric_type in [standard, delay], for each model, show F1 only.
	"""
	# Decide row order for metric types
	desired_order = METRIC_TYPES
	metrics_in_values = [m for m in desired_order if m in values.keys()] or sorted(values.keys())

	# Column spec: 2 left columns (Metric, Model) + len(datasets) centered columns
	col_spec = "ll" + ("|c" * len(datasets))
	header_cols = ["Metric", "Model", *datasets]

	# Pre-compute max values per (metric_type, dataset)
	max_map: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in metrics_in_values}
	for metric in metrics_in_values:
		for ds in datasets:
			all_vals = [values.get(metric, {}).get(model, {}).get(ds) for model in models]
			numeric_vals = [v for v in all_vals if isinstance(v, (int, float))]
			max_map[metric][ds] = max(numeric_vals) if numeric_vals else None

	lines: List[str] = []
	lines.append("\\begin{table*}[htbp]")
	lines.append("\\centering")
	lines.append("\\captionsetup{skip=0pt}")
	lines.append(f"\\caption{{Performance comparison of TSAD on {title_highlight}. F1 (standard/delay) is reported.}}")
	lines.append(f"\\label{{{label}}}")
	lines.append("\\small")
	lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
	lines.append("\\toprule")
	lines.append(" & ".join(header_cols) + " \\\\")
	lines.append("\\midrule")

	for metric in metrics_in_values:
		for model in models:
			row: List[str] = [metric, remove_dates(model.split("TSLLM-AD-")[-1])]
			for ds in datasets:
				v = values.get(metric, {}).get(model, {}).get(ds)
				cell = fmt(v, decimals)
				max_v = max_map.get(metric, {}).get(ds)
				if isinstance(v, (int, float)) and isinstance(max_v, (int, float)):
					if abs(v - max_v) <= 1e-12:
						cell = f"\\boldres{{{cell}}}"
				row.append(cell)
			lines.append(" & ".join(row) + " \\\\")
		if metric != metrics_in_values[-1]:
			lines.append("\\hline")

	lines.append("\\bottomrule")
	lines.append("\\end{tabular}")
	lines.append("\\end{table*}")

	return "\n".join(lines) + "\n"


def load_baselines(csv_path: str) -> Tuple[Set[str], Dict[str, Dict[str, Dict[str, float]]]]:
	"""Load baseline results from a wide CSV file.

	Expected headers (case-insensitive):
	  - "F1 Type" (metric type: standard/delay). Empty cells inherit the last non-empty metric in previous rows.
	  - "Model"
	  - dataset columns: e.g., "AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo"

	Returns:
	  models: Set[str]
	  model_types: Dict[model] -> str
	  values: Dict[metric][model][dataset] -> float
	"""
	models: Set[str] = set()
	model_types: Dict[str, str] = {}
	values: Dict[str, Dict[str, Dict[str, float]]] = {key: defaultdict(dict) for key in METRIC_TYPES}

	if not os.path.isfile(csv_path):
		return models, model_types, values

	try:
		with open(csv_path, "r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			if not reader.fieldnames:
				return models, values

			# Normalize header names
			headers_lower = [h.lower() for h in reader.fieldnames]
			# Identify key columns
			try:
				idx_metric = headers_lower.index("f1 type")
			except ValueError:
				idx_metric = None
			try:
				idx_model = headers_lower.index("model")
			except ValueError:
				idx_model = None
			try:
				idx_model_type = headers_lower.index("model type")
			except ValueError:
				idx_model_type = None

			# Dataset columns are the remaining ones
			dataset_cols: List[Tuple[str, int]] = []
			for i, h in enumerate(reader.fieldnames):
				hl = h.lower()
				if i == idx_metric or i == idx_model:
					continue
				if hl.strip() == "":
					continue
				dataset_cols.append((h.strip(), i))

			if idx_model is None or not dataset_cols:
				return models, model_types, values

			current_metric: Optional[str] = None

			for row in reader:
				# Skip empty/separator rows
				if all((v is None or str(v).strip() == "") for v in row.values()):
					continue

				metric_raw = None
				if idx_metric is not None:
					# DictReader keeps original headers as keys
					metric_header = reader.fieldnames[idx_metric]
					metric_raw = (row.get(metric_header) or "").strip().lower()
				if metric_raw:
					# Update current metric
					current_metric = metric_raw
				# Use current metric if this row doesn't specify it
				metric = current_metric or ""
				if metric not in METRIC_TYPES:
					# Unknown metric; skip row
					continue

				model_header = reader.fieldnames[idx_model]
				model = (row.get(model_header) or "").strip()
				if not model:
					# No model name, skip
					continue
				model_type_header = reader.fieldnames[idx_model_type]
				model_type = (row.get(model_type_header) or "").strip()
				if not model_type:
					# No model type, skip
					continue

				models.add(model)
				model_types[model] = model_type

				# Read dataset values
				for ds_name, ds_idx in dataset_cols:
					ds_header = reader.fieldnames[ds_idx]
					val_str = (row.get(ds_header) or "").strip()
					if val_str == "":
						continue
					try:
						# Remove potential trailing annotations
						cleaned = val_str.split()[0]
						f1_val = float(cleaned) / 100.0  # CSV has percentages
					except Exception:
						continue
					values[metric][model][ds_name] = f1_val
	except Exception:
		# Fail silently; leave values empty if parsing fails
		pass

	return list(models), model_types, values


def main():
	parser = argparse.ArgumentParser(description="Parse results to LaTeX table (F1 across metric types)")
	parser.add_argument("--result-dir", type=str, default="../results-all/results", help="Path to results directory")
	parser.add_argument("--out", type=str, default="../tables-all", help="Optional output directory to save the LaTeX table")
	parser.add_argument("--decimals", type=int, default=3, help="Decimal places for numbers")
	parser.add_argument('--dataset-type', type=str, default='standard', choices=['standard', 'generated', 'mixed'], help='Which dataset to process.')
	parser.add_argument('--best-f1', type=str, default=None, choices=['magnitude', None], help='Enable best f1 calculation.')
	parser.add_argument('--split', type=str, default="default", choices=["default", "short", "season"], help='Dataset split/length mode.')
	parser.add_argument('--rule', type=str, default=None, choices=['v1', None], help='Enable extra rule on datasets.')
	parser.add_argument('--baseline', action='store_true', help='Include baseline results from CSV.')
	args = parser.parse_args()

	dataset_type = args.dataset_type  # 'standard', 'generated', 'mixed'
	split = args.split	# default, short, season
	rule_version = args.rule # None, 'v1'

	baselines_to_filter = ["FITS"]
	models_to_filter = ["TSLLM-AD-v1.3-20251011", "TSLLM-AD-v1.4-20251015", "TSLLM-AD-v1.5-20251021", "TSLLM-AD-v5.2-20251204"]
	# models_to_include = ["TSLLM-AD-v4.0-20251120", "TSLLM-AD-v5.5-20251209", "TSLLM-AD-v5.5-20251209-DAPO-v1-20251209", "TSLLM-AD-v5.6-20251210", "TSLLM-AD-v5.10-20251210", "TSLLM-AD-v5.10-20251210-DAPO-v1-20251210", "TSLLM-AD-v5.10-20251210-DAPO-v1.3-20251210", "TSLLM-AD-v5.11-20251215", "TSLLM-AD-v5.12-20251215", "TSLLM-AD-v5.12-20251215-DAPO-v1.1-20251215"]
	# models_to_include = ["TSLLM-AD-v5.6-20251210", "TSLLM-AD-v5.10-20251210", "TSLLM-AD-v5.10-20251210-DAPO-v1-20251210", "TSLLM-AD-v6.0-20251217", "TSLLM-AD-v6.1-20251217", "TSLLM-AD-v6.2-20251217", "TSLLM-AD-v6.2-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v6.3-20251217", "TSLLM-AD-v6.3-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.0-20251217", "TSLLM-AD-v7.2-20251217", "TSLLM-AD-v7.2-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.3-20251217"]
	# models_to_include = ["TSLLM-AD-v5.6-20251210", "TSLLM-AD-v5.10-20251210", "TSLLM-AD-v5.10-20251210-DAPO-v1-20251210", "TSLLM-AD-v6.0-20251217", "TSLLM-AD-v6.1-20251217", "TSLLM-AD-v6.2-20251217", "TSLLM-AD-v6.2-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v6.3-20251217", "TSLLM-AD-v6.3-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.0-20251217", "TSLLM-AD-v7.2-20251217", "TSLLM-AD-v7.2-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.3-20251217", "TSLLM-AD-v7.3-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.4-20251217", "TSLLM-AD-v7.5-20251217", "TSLLM-AD-v7.5-20251217-DAPO-v1.0-20251217"]
	# models_to_include = ["TSLLM-AD-v5.6-20251210", "TSLLM-AD-v5.10-20251210", "TSLLM-AD-v5.10-20251210-DAPO-v1-20251210"]
	# models_to_include = ["TSLLM-AD-v7.0-20251217", "TSLLM-AD-v7.3-20251217", "TSLLM-AD-v7.3-20251217-DAPO-v1.0-20251217"]
	models_to_include = ["TSLLM-AD-v6.0-20251217", "TSLLM-AD-v6.2-20251217", "TSLLM-AD-v6.2-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v7.0-20251217", "TSLLM-AD-v7.3-20251217", "TSLLM-AD-v7.3-20251217-DAPO-v1.0-20251217", "TSLLM-AD-v8.0-20251225", "TSLLM-AD-v8.1-20251225", "TSLLM-AD-v8.1-20251225-DAPO-v1.0-20251225", "TSLLM-AD-v8.2-20251225", "TSLLM-AD-v8.3-20251225", "TSLLM-AD-v8.3-20251225-DAPO-v1.0-20251225", "TSLLM-AD-v8.4-20251225", "TSLLM-AD-v8.5-20251225", "TSLLM-AD-v8.5-20251225-DAPO-v1.0-20251225"]

	datanames = ['AIOPS', 'WSD', 'Yahoo', 'NAB', 'TODS', 'UCR']
	generated_datanames = ['20251004_163407_test_1000', '20251010_071208_test_1000', '20251010_071208_test_1000_magnitude']
	datanames = ['AIOPS', 'WSD', 'Yahoo', 'Service', 'NAB', 'TODS', 'UCR']
	datanames = ['AIOPS', 'WSD', 'Yahoo', 'Service', 'NAB', 'TODS', 'UCR', 'Dodgers', "Power", "TAO", "SMD", "Stock", "MSL", "SMAP", "OPPORTUNITY", "SED"]
	generated_datanames = []
	all_datanames = generated_datanames if dataset_type == 'generated' else datanames if dataset_type == 'standard' else datanames + generated_datanames

	rule_postfix = f"_rule_{rule_version}" if rule_version else ""
	all_datanames = [f'{name}{rule_postfix}'.replace("_", "-") for name in all_datanames]

	label = "tbl:f1_standard" if dataset_type == 'standard' else "tbl:f1_generated" if dataset_type == 'generated' else "tbl:f1_mixed"
	label += f"_{split}"
	label += f"_bestby_{args.best_f1}" if args.best_f1 else ""
	label += f"_rule_{rule_version}" if rule_version else ""

	title_highlight = 'Generated Datasets' if dataset_type == 'generated' else 'Standard Datasets' if dataset_type == 'standard' else 'All Datasets'
	title_highlight += f", Length: {split}"
	title_highlight += f", Best F1 by: {args.best_f1}" if args.best_f1 else ""
	title_highlight += (f" (Rule: {rule_version})" if rule_version else "")


	result_root = args.result_dir if split == 'default' else args.result_dir + f'_{split}'
	results_dir = result_root if args.best_f1 is None else result_root + f'_bestby_{args.best_f1}'

	datasets, models, values = scan_results(results_dir, all_datanames)
	models = [m for m in models if m not in models_to_filter and m in models_to_include]
	
	if args.baseline:
		# Merge baseline results from CSV into models and values
		baseline_csv = "baseline_res.csv"
		baseline_models, baseline_model_types, baseline_values = load_baselines(baseline_csv)
		baseline_models = [m for m in baseline_models if m not in baselines_to_filter]
		if baseline_models:
			# sort baseline_models by two factor: 1. baseline_model_types, 2. str order
			baseline_models.sort(key=lambda m: (baseline_model_types.get(m, ""), m))
			models.extend(baseline_models)
			# # Append new baseline models to the end, keeping existing order
			# for bm in sorted(baseline_models):
			# 	if bm not in models:
			# 		models.append(bm)
			# Merge baseline values, but only for datasets we will render
			for metric in METRIC_TYPES:
				bv_m = baseline_values.get(metric, {})
				for model_name, ds_map in bv_m.items():
					for ds_name, f1_val in ds_map.items():
						if ds_name in datasets:
							values.setdefault(metric, defaultdict(dict))
							values[metric][model_name][ds_name] = f1_val

	latex = render_latex_table(datasets, models, values, decimals=args.decimals, label=label, title_highlight=title_highlight)

	# print to stdout
	print(latex)

	# optionally write to file
	if args.out:
		out_dir = args.out
		out_path = os.path.join(out_dir, f"f1_table_{dataset_type}{'' if split == 'default' else f'_{split}'}{'' if args.best_f1 is None else f'_bestby_{args.best_f1}'}{rule_postfix}.tex")
		out_dir = os.path.dirname(out_path)
		if out_dir and not os.path.exists(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			f.write(latex)


if __name__ == "__main__":
	main()
