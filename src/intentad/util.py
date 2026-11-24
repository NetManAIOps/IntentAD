import json
import random
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm


def merge_sample(train_sample: Dict, test_sample: Dict) -> Dict:
	train_case_id = int(train_sample.get("case_id", -1))
	test_case_id = int(test_sample.get("case_id", -1))
	if train_case_id != test_case_id:
		raise ValueError(f"case_id mismatch: train={train_case_id}, test={test_case_id}")

	train_ts = list(train_sample.get("timeseries", []) or [])
	test_ts = list(test_sample.get("timeseries", []) or [])
	train_len = len(train_ts)
	test_offset = train_len

	def _shift_pattern_items(items: List[Dict], offset: int) -> List[Dict]:
		shifted: List[Dict] = []
		for item in items or []:
			if not isinstance(item, dict):
				continue
			new_item = deepcopy(item)
			try:
				new_item["start_pos"] = int(new_item.get("start_pos", 0)) + offset
				new_item["end_pos"] = int(new_item.get("end_pos", 0)) + offset
			except Exception:
				continue
			shifted.append(new_item)
		return shifted

	def _merge_human_intent(train_intent: Dict, test_intent: Dict, test_offset: int) -> Dict:
		if not isinstance(train_intent, dict) and not isinstance(test_intent, dict):
			return {}
		train_intent = train_intent if isinstance(train_intent, dict) else {}
		test_intent = test_intent if isinstance(test_intent, dict) else {}

		merged_categories: List = []
		for c in list(train_intent.get("categories", []) or []) + list(test_intent.get("categories", []) or []):
			if c not in merged_categories:
				merged_categories.append(c)

		def _rule_key(rule: Dict) -> str:
			name = str(rule.get("name", ""))
			category = str(rule.get("category", ""))
			params = json.dumps(rule.get("parameters", {}) or {}, ensure_ascii=False, sort_keys=True)
			return f"{name}||{category}||{params}"

		def _merge_rule_list(base_rules: List, incoming_rules: List, offset: int) -> List:
			lookup: Dict[str, Dict] = {}
			ordered_keys: List[str] = []

			for rule in base_rules or []:
				if not isinstance(rule, dict):
					continue
				key = _rule_key(rule)
				if key not in lookup:
					new_rule = deepcopy(rule)
					new_rule["filtered_anomalies"] = _shift_pattern_items(
						list(rule.get("filtered_anomalies", []) or []),
						0,
					)
					lookup[key] = new_rule
					ordered_keys.append(key)

			for rule in incoming_rules or []:
				if not isinstance(rule, dict):
					continue
				key = _rule_key(rule)
				if key not in lookup:
					new_rule = deepcopy(rule)
					new_rule["filtered_anomalies"] = []
					lookup[key] = new_rule
					ordered_keys.append(key)
				lookup[key]["filtered_anomalies"].extend(
					_shift_pattern_items(list(rule.get("filtered_anomalies", []) or []), offset)
				)

			#      start/end/type        
			for key in ordered_keys:
				rule = lookup[key]
				seen = set()
				deduped = []
				for a in rule.get("filtered_anomalies", []) or []:
					if not isinstance(a, dict):
						continue
					k = (int(a.get("start_pos", -1)), int(a.get("end_pos", -1)), str(a.get("pattern_type", "")))
					if k in seen:
						continue
					seen.add(k)
					deduped.append(a)
				rule["filtered_anomalies"] = deduped

			return [lookup[k] for k in ordered_keys]

		merged_rules = _merge_rule_list(
			list(train_intent.get("rules", []) or []),
			list(test_intent.get("rules", []) or []),
			test_offset,
		)
		return {
			"categories": merged_categories,
			"rules": merged_rules,
		}

	def _shift_label_items(items: List, offset: int) -> List:
		shifted: List = []
		for item in items or []:
			if isinstance(item, (list, tuple)) and len(item) >= 2:
				s = int(item[0]) + offset
				e = int(item[1]) + offset
				if len(item) == 2:
					shifted.append((s, e))
				elif len(item) == 4:
					shifted.append((s, e, item[2], float(item[3])))
				else:
					shifted.append(tuple([s, e] + list(item[2:])))
		return shifted

	merged_info = deepcopy(train_sample.get("timeseries_info", {}) or {})
	test_info = deepcopy(test_sample.get("timeseries_info", {}) or {})
	if not merged_info and test_info:
		merged_info = test_info
	merged_info["seq_len"] = int(train_len + len(test_ts))
	if "metric_name" not in merged_info and "metric_name" in test_info:
		merged_info["metric_name"] = test_info.get("metric_name")
	if "metric_type" not in merged_info and "metric_type" in test_info:
		merged_info["metric_type"] = test_info.get("metric_type")
	if "sampling_interval" not in merged_info and "sampling_interval" in test_info:
		merged_info["sampling_interval"] = test_info.get("sampling_interval")

	merged_human_intent = _merge_human_intent(
		deepcopy(train_sample.get("human_intent") or {}),
		deepcopy(test_sample.get("human_intent") or {}),
		test_offset,
	)

	merged = {
		"case_id": train_case_id,
		"split": "merged",
		"timeseries_data": train_ts + test_ts,
		"timeseries_info": merged_info,
		"human_intent": merged_human_intent,
		"anomalies_detected": list(train_sample.get("anomalies_detected", []) or [])
		+ _shift_pattern_items(list(test_sample.get("anomalies_detected", []) or []), test_offset),
		"anomalies_final": list(train_sample.get("anomalies_final", []) or [])
		+ _shift_pattern_items(list(test_sample.get("anomalies_final", []) or []), test_offset),
		"label": list(train_sample.get("label", []) or [])
		+ _shift_label_items(list(test_sample.get("label", []) or []), test_offset),
		"additional_info": {
			"split_meta": {
				"train_len": train_len,
				"test_len": len(test_ts),
				"total_len": int(train_len + len(test_ts)),
			},
		},
	}
	return merged