import json
import argparse
import random
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

from ts_generator import AnomalyDetectionDataGenerator
from ts_generator.anomaly_detection_generator import visualize_sample
from ts_generator.util import save_ts_json
from util import merge_sample


def write_jsonl(records: List[Dict], output_file: Path) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(input_file: Path) -> List[Dict]:
	records = []
	with input_file.open("r", encoding="utf-8") as f:
		for line in f:
			try:
				rec = json.loads(line.strip())
				records.append(rec)
			except json.JSONDecodeError:
				continue
	return records


def generate_llm_test_dataset(test_records: List[Dict], output_file: Path, query_memory: bool = False) -> None:
	processed_data = []
	for test_sample in tqdm(test_records, desc="Processing test samples for LLM dataset"):
		ts_info = test_sample.get("timeseries_info", {})
		ts_additional_info = ts_info.get("additional_info", {}) if isinstance(ts_info, dict) else {}
		history_anomalies = ts_additional_info.get("history_anomalies", []) if isinstance(ts_additional_info, dict) else []
		anomaly_series = test_sample.get("timeseries", [])
		human_intent = test_sample.get("human_intent", {})
		anomaly_patterns = test_sample.get("anomalies_detected", [])
		final_results = test_sample.get("anomalies_final", [])
		latent_anomalies = test_sample.get("anomalies_latent", [])
		label = test_sample.get("label", 0)
		ts_case = {
            "timeseries_info": ts_info,
            "timeseries_data": anomaly_series,
            "human_intent": human_intent,
            "anomalies_detected": [ pattern for pattern in anomaly_patterns ],
            "anomalies_final": [ pattern for pattern in final_results ],
            "label": label
        }
		if query_memory:
			ts_case["history_anomalies"] = history_anomalies
		processed_data.append(ts_case)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	save_ts_json(processed_data, output_file)
	print(f"Save dataset to {output_file}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate paired IntentAD benchmark cases.")
	parser.add_argument("--num-cases", type=int, default=1000)
	parser.add_argument("--length-mode", choices=["dynamic", "fixed"], default="dynamic")
	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--no-visualize", action="store_true")
	parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "intentad_bench")
	parser.add_argument("--output-llm-dir", type=Path, default=PROJECT_ROOT / "data" / "tsad" / "generated" / "full")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	num_cases = args.num_cases
	length_mode = args.length_mode
	fixed_train_len = 512
	fixed_test_len = 256
	typical_seq_len = [256, 512, 1024]
	typical_prob = 0.1
	seq_len_range = (256, 1024)

	pair_train_ratio_range = (0.5, 0.5)
	pair_min_train_len = 128
	pair_min_test_len = 128
	output_dir = args.output_dir
	output_llm_dir = args.output_llm_dir
	sampling_interval = "random"  # or int seconds, e.g. 300
	train_anomaly_count_range = (0, 3)
	test_anomaly_count_range = (0, 3)
	anomaly_type_range = (1, 3)
	human_intent_count_range = (0, 3)
	#    5/6/7    
	# 5: (normal_case_ratio, has_final_anomaly_ratio, no_final_but_detected_ratio, has_filtered_anomaly_ratio, requires_history_case_ratio)
	# 6: + baseline_zero_case_ratio
	# 7: + no_intent_case_ratio (human_intent.rules   )
	sample_type_ratio_tuple = (1, 0.75, 0.15, 0.5, 0.2)
	seed = args.seed
	is_filter = True
	is_visualize = not args.no_visualize
	query_memory = True

	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)

	if sampling_interval != "random":
		sampling_interval = int(sampling_interval)

	generator_config = {
		"sampling_interval": sampling_interval,
		"typical_seq_len": typical_seq_len,
		"typical_prob": typical_prob,
		"seq_len_range": [int(seq_len_range[0]), int(seq_len_range[1])],
		"pair_train_ratio_range": [float(pair_train_ratio_range[0]), float(pair_train_ratio_range[1])],
		"pair_min_train_len": int(pair_min_train_len),
		"pair_min_test_len": int(pair_min_test_len),
		"human_intent_count_range": [human_intent_count_range[0], human_intent_count_range[1]],
		"anomaly_count_range": [test_anomaly_count_range[0], test_anomaly_count_range[1]],
		"anomaly_type_range": [anomaly_type_range[0], anomaly_type_range[1]],
		"is_train": True,
		"is_filter": is_filter,
		"is_short": False,
		"use_tool": False,
		"query_memory": query_memory,
		"sample_type_ratio_tuple": sample_type_ratio_tuple,
		"enable_multiple_anomalies": True,
		"enable_complex_patterns": True,
		"visualize": is_visualize,
	}

	generator = AnomalyDetectionDataGenerator(generator_config)
	train_records, test_records = generator.generate_pair_dataset_for_bench(
		num_samples=num_cases,
		train_len=(int(fixed_train_len) if length_mode == "fixed" else None),
		test_len=(int(fixed_test_len) if length_mode == "fixed" else None),
		train_anomaly_count_range=train_anomaly_count_range,
		test_anomaly_count_range=test_anomaly_count_range,
		anomaly_type_range=anomaly_type_range,
	)
	dataset_name = datetime.now().strftime("%Y%m%d_%H%M%S")
	train_path = output_dir / dataset_name / "train.jsonl"
	test_path = output_dir / dataset_name / "test.jsonl"
	write_jsonl(train_records, train_path)
	write_jsonl(test_records, test_path)

	generate_llm_test_dataset(
		test_records,
		output_llm_dir / f"{dataset_name}_test.json",
		query_memory=bool(generator_config.get("query_memory", False)),
	)

	if generator_config["query_memory"]:
		history_anomalies_dict = {}
		for idx, data in enumerate(test_records):
			ts_info = data.get("timeseries_info", {})
			metric_name = ts_info.get("metric_name", f"metric_{idx}")
			history_anomalies = ts_info.get("additional_info", {}).get("history_anomalies", [])
			history_anomalies_dict[metric_name] = history_anomalies
		memory_dir = output_llm_dir / "memory"
		memory_dir.mkdir(parents=True, exist_ok=True)
		memory_file = memory_dir / f"{dataset_name}_test.json"
		with open(memory_file, "w", encoding="utf-8") as f:
			json.dump(history_anomalies_dict, f, ensure_ascii=False, indent=2)
		print(f"Save memory file to {memory_file}")

	print(f"Generated pair dataset: train={len(train_records)}, test={len(test_records)}")
	print(f"Train JSONL: {train_path}")
	print(f"Test JSONL: {test_path}")
	if train_records:
		train_lens = [len(x.get("timeseries", [])) for x in train_records]
		test_lens = [len(x.get("timeseries", [])) for x in test_records]
		print(
			"Length stats | "
			f"train min/max={min(train_lens)}/{max(train_lens)}, "
			f"test min/max={min(test_lens)}/{max(test_lens)}"
		)

	if generator_config["visualize"]:
		visualize_root = PROJECT_ROOT / "data" / "visualizations" / "intentad_bench"
		visualize_dir = visualize_root / dataset_name
		visualize_dir.mkdir(parents=True, exist_ok=True)
		for idx in tqdm(range(len(train_records)), desc="Visualizing samples"):
			data = merge_sample(train_records[idx], test_records[idx])
			output_vis_file = f"{visualize_dir}/{idx}.pdf"
			visualize_sample(data, output_vis_file, split_train_test=True)
		print(f"Visualize generated dataset in {visualize_dir}")


if __name__ == "__main__":
	main()
