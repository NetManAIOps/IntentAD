"""Generate human-intent rules and apply them to detected anomalies."""

import random
from typing import Any, Dict, List, Optional, Tuple

from .util import AnomalyPattern, TimeSeriesInfo
from .anomaly_injector import all_anomaly_patterns, AnomalyInjector


class HumanIntentGenerator:
	"""Generate and apply intent rules for anomaly detection."""

	def __init__(self):
		anomalies = all_anomaly_patterns()
		temporal_anomalies = [
			"slow_increase",
			"slow_decrease",
			"sudden_increase",
			"sudden_decrease",
			"spike",
			"dip",
			"multi_spike",
			"multi_dip",
			"baseline_shift_up_small",
			"baseline_shift_down_small"
		]

		self.rule_order = [
			"focus",
			"threshold",
			"ignore_frequency",
			"ignore_duration",
			"truncate_duration",
			"scope",
			"others",
			"merge",
		]

		#             filter_rule_configs    
		# -       recent_reoccurrence     
		# -    ignore_short_duration / merge_close_anomalies
		# - “scope”                      slow_increase/slow_decrease/saturation_values 
		self.human_intent_configs: Dict[str, Dict[str, Any]] = {
			# focus
			"focus_on_increase_only": {
				"probability": 0.25,
				"category": "focus",
				"anomalies": ["sudden_increase", "slow_increase", "spike", "multi_spike"],
				"description_template": "Focus on increase-type anomalies only",
			},
			"focus_on_decrease_only": {
				"probability": 0.25,
				"category": "focus",
				"anomalies": ["sudden_decrease", "slow_decrease", "dip", "multi_dip", "zero_values"],
				"description_template": "Focus on decrease-type anomalies only",
			},
			"focus_on_zero_only": {
				"probability": 0.15,
				"category": "focus",
				"anomalies": ["zero_values"],
				"description_template": "Focus on zero-value anomalies only",
			},

			# threshold
			"absolute_magnitude_threshold": {
				"probability": 0.25,
				"category": "threshold",
				"anomalies": temporal_anomalies + ["volatility_increase", "volatility_decrease"],
				"parameter_ranges": {
					"threshold_value": (0.3, 1.5),
				},
				"description_template": "Report anomalies only if magnitude >= {threshold_value_str}",
			},

			# ignore_frequency
			"ignore_frequent_patterns": {
				"probability": 0.30,
				# "probability": 0.90,
				"category": "ignore_frequency",
				"anomalies": temporal_anomalies + ["zero_values", "volatility_increase", "volatility_decrease"],
				"parameter_ranges": {
					"window_size": [200, 300, 400, 500, 600, 800, 1000, 1440],
					"min_occurrences": (3, 5),
				},
				"description_template": "Ignore patterns that are periodic or occur >= {min_occurrences} times within a window of {window_size} points",
			},
			"ignore_rare_patterns": {
				"probability": 0.18,
				# "probability": 0.54,
				"category": "ignore_frequency",
				"anomalies": temporal_anomalies + ["zero_values", "volatility_increase", "volatility_decrease"],
				"parameter_ranges": {
					"window_size": [200, 300, 400, 500, 600, 800, 1000, 1440],
					"max_occurrences": (1, 2),
				},
				"description_template": "Ignore occasional patterns that occur <= {max_occurrences} times within a window of {window_size} points",
			},

			# ignore_duration (  )
			"ignore_short_duration": {
				"probability": 0.25,
				"category": "ignore_duration",
				"anomalies": anomalies,
				"parameter_ranges": {
					"min_duration": [2, 3, 5, 8, 15],
				},
				"description_template": "Ignore anomalies with duration < {min_duration} points",
			},

			# truncate_duration
			"truncate_overlong_duration": {
				"probability": 0.30,
				"category": "truncate_duration",
				"anomalies": anomalies,
				"parameter_ranges": {
					"max_duration": [20, 30, 40, 50],
				},
				"description_template": "If an anomaly lasts over {max_duration} points, only keep the first {max_duration} points",
			},

			# merge (  )
			"merge_close_anomalies": {
				"probability": 0.25,
				"category": "merge",
				"anomalies": anomalies,
				"parameter_ranges": {
					"max_gap": [1, 2, 3, 5, 10],
					"same_pattern_only": [True, False],
				},
				"description_template": "Merge anomalies that are close (gap <= {max_gap})",
			},

			# others
			"limit_total_reports": {
				"probability": 0.18,
				"category": "others",
				"anomalies": anomalies,
				"parameter_ranges": {
					"max_reports": [1, 2, 3, 4],
					"rank_by": ["magnitude", "duration", "hybrid"],
				},
				"description_template": "Keep at most {max_reports} anomaly reports, prioritized by {rank_by}",
			},
			"ignore_recent_reoccurrence": {
				"probability": 0.30,
				# "probability": 0.90,
				"category": "others",
				"anomalies": anomalies,
				"parameter_ranges": {
					"window_size": [200, 400, 500, 600, 800, 1000, 1440, 4320, 7200],
					"mode": ["magnitude", "duration", "both"],
				},
				"description_template": "Ignore recent reoccurrence in a window of {window_size} points by {mode} comparison",
			},

			# scope                  
			"include_slow_trends": {
				"probability": 0.12,
				"category": "scope",
				"anomalies": ["slow_increase", "slow_decrease"],
				# "parameter_ranges": {
				# 	"count": [1, 2],
				"description_template": "Also detect slow trend change anomalies",
			},
			# "include_saturation_values": {
			# 	"probability": 0.10,
			# 	"category": "scope",
			# 	"anomalies": ["saturation_values"],
			# 	"parameter_ranges": {
			# 		"count": [1, 2],
			# 	"description_template": "Also detect saturation (upper-limit) anomalies",

			# scope              /        
			"include_small_baseline_shifts": {
				"probability": 0.0,
				"category": "scope",
				"anomalies": ["baseline_shift_up_small", "baseline_shift_down_small"],
				"parameter_ranges": {
					# "count": [1, 2],
					"min_duration": [20, 40, 80],
					"min_offset_ratio": [0.03, 0.05, 0.1],
				},
				"description_template": "Also detect subtle but sustained baseline shifts (min_duration={min_duration}, min_offset_ratio={min_offset_ratio})",
			},
			"include_volatility_changes": {
				"probability": 0.10,
				"category": "scope",
				"anomalies": ["volatility_increase", "volatility_decrease"],
				"parameter_ranges": {
					# "count": [1, 2],
					# "min_duration": [20, 40, 80],
					"variance_ratio": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
					# "window_size": [30, 60, 120],
				},
				"description_template": "Also detect volatility changes with variance ratio no smaller than {variance_ratio}",
			},
		}

	def generate_human_intent(self, ts_info: TimeSeriesInfo, num_rule_range: List[int]) -> Dict[str, Any]:
		"""                        """
		selected_rules: List[str] = []
		selected_categories: set = set()

		rule_names = list(self.human_intent_configs.keys())
		rule_weights = [self.human_intent_configs[name]["probability"] for name in rule_names]

		# 0.2  num_rules = 0, 0.4  num_rules = 1
		rand_prob = random.random()
		if rand_prob < 0.2:
			num_rules = 0
		elif rand_prob < 0.6:
			num_rules = 1
		else:
			num_rules = random.randint(int(num_rule_range[0]), int(num_rule_range[1]))
		for _ in range(num_rules):
			if not rule_names:
				break

			#         focus_on_zero_only       threshold
			if "focus_on_zero_only" in selected_rules:
				indices = [
					i
					for i, name in enumerate(rule_names)
					if self.human_intent_configs[name]["category"] == "threshold"
				]
				for idx in sorted(indices, reverse=True):
					rule_names.pop(idx)
					rule_weights.pop(idx)

			#    focus        focus      
			if "focus" in selected_categories:
				indices = [
					i
					for i, name in enumerate(rule_names)
					if self.human_intent_configs[name]["category"] == "focus"
				]
				for idx in sorted(indices, reverse=True):
					rule_names.pop(idx)
					rule_weights.pop(idx)
				# pop include_volatility_changes, include_small_baseline_shifts
				to_filter_rules = ["include_volatility_changes", "include_small_baseline_shifts"]
				if "focus_on_zero_only" in selected_rules:
					to_filter_rules.append("include_slow_trends")
				indices = [
					i
					for i, name in enumerate(rule_names)
					if name in to_filter_rules
				]
				for idx in sorted(indices, reverse=True):
					rule_names.pop(idx)
					rule_weights.pop(idx)
			
			#    ignore_frequency        ignore_frequency      
			if "ignore_frequency" in selected_categories:
				indices = [
					i
					for i, name in enumerate(rule_names)
					if self.human_intent_configs[name]["category"] == "ignore_frequency"
				]
				for idx in sorted(indices, reverse=True):
					rule_names.pop(idx)
					rule_weights.pop(idx)

			selected_rule = random.choices(rule_names, weights=rule_weights)[0]
			selected_rules.append(selected_rule)
			selected_categories.add(self.human_intent_configs[selected_rule]["category"])

			idx = rule_names.index(selected_rule)
			rule_names.pop(idx)
			rule_weights.pop(idx)

		categories = list(selected_categories)
		categories.sort(key=lambda c: self.rule_order.index(c) if c in self.rule_order else len(self.rule_order))

		human_intent: Dict[str, Any] = {
			"rules": [],
			"categories": categories,
		}

		for rule_name in selected_rules:
			cfg = self.human_intent_configs[rule_name]
			params = self._generate_rule_parameters(rule_name, cfg, ts_info)
			desc = self._generate_rule_description(rule_name, cfg, params)
			detail = {
				"name": rule_name,
				"category": cfg["category"],
				"parameters": params,
				"description": desc,
				"filtered_anomalies": [],
				"history_query_analysis": None,
			}
			human_intent["rules"].append(detail)

		return human_intent

	def apply_human_intent(
		self,
		anomalies: List[AnomalyPattern],
		human_intent: Dict[str, Any],
		ts_info: Optional[TimeSeriesInfo] = None,
	) -> List[AnomalyPattern]:
		"""         anomalies_detected    anomalies_final"""
		def _history_query_analysis(used_history: bool, required_count: int, no_history_count: int) -> Dict[str, Any]:
			total = required_count + no_history_count
			if total == 0:
				status = "not_filtered"
			elif required_count > 0 and no_history_count == 0:
				status = "required"
			elif required_count == 0:
				status = "not_required"
			else:
				status = "partially_required"
			return {
				"used_history": used_history,
				"filtered_count": total,
				"requires_history_count": required_count,
				"without_history_also_filters_count": no_history_count,
				"dependency_status": status,
			}

		if not human_intent or not human_intent.get("rules"):
			return anomalies

		rules = list(human_intent["rules"])
		rules.sort(key=lambda r: self.rule_order.index(r["category"]) if r["category"] in self.rule_order else len(self.rule_order))

		final_results = anomalies.copy()
		original_anomalies = anomalies

		for rule in rules:
			name = rule["name"]
			category = rule["category"]
			params = rule.get("parameters", {})

			if category == "scope":
				# scope                     
				continue

			if category == "focus":
				allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
				new_final = [a for a in final_results if a.pattern_type.lower() in allowed]
				rule["filtered_anomalies"].extend(list(set(final_results) - set(new_final)))
				final_results = new_final
				continue

			if category == "threshold":
				if name == "absolute_magnitude_threshold":
					threshold = float(params.get("threshold_value", 0.0))
					new_final: List[AnomalyPattern] = []
					for a in final_results:
						if any(r.get("name") == "focus_on_zero_only" for r in rules):
							new_final.append(a)
							continue
						allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
						if a.pattern_type.lower() in allowed and float(a.magnitude) < threshold:
							rule["filtered_anomalies"].append(a)
						else:
							new_final.append(a)
					final_results = new_final
				continue

			if category == "ignore_frequency":
				window_size = int(params.get("window_size", 100))
				min_occ = int(params.get("min_occurrences", 3))
				max_occ = int(params.get("max_occurrences", 1))
				allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))

				addi = (ts_info.additional_info if (ts_info and isinstance(ts_info.additional_info, dict)) else {})
				positions_by_pattern: Dict[str, List[int]] = {}
				history_positions_by_pattern: Dict[str, List[int]] = {}
				current_positions_by_pattern: Dict[str, List[int]] = {}
				history_tuples = addi.get("history_anomalies") if isinstance(addi, dict) else None
				if isinstance(history_tuples, list):
					for item in history_tuples:
						if not isinstance(item, (list, tuple)) or len(item) != 4:
							continue
						s, e, p, _ = item
						try:
							s_i = int(s)
							e_i = int(e)
						except Exception:
							continue
						if e_i < s_i:
							s_i, e_i = e_i, s_i
						p_norm = str(p).lower()
						if p_norm in allowed:
							history_positions_by_pattern.setdefault(p_norm, []).append(s_i)

				for p, pos_list in history_positions_by_pattern.items():
					positions_by_pattern.setdefault(p, []).extend(pos_list)

				for a in original_anomalies:
					p = a.pattern_type.lower()
					if p in allowed:
						current_positions_by_pattern.setdefault(p, []).append(int(a.start_pos))
						positions_by_pattern.setdefault(p, []).append(int(a.start_pos))

				def _calc_max_occ(input_positions: Dict[str, List[int]]) -> Dict[str, int]:
					res: Dict[str, int] = {}
					for p, pos_list in input_positions.items():
						if not pos_list:
							continue
						cur_pos = list(pos_list)
						cur_pos.sort()
						l = 0
						max_in_window = 0
						for r in range(len(cur_pos)):
							while l <= r and cur_pos[r] - cur_pos[l] > max(0, window_size - 1):
								l += 1
							max_in_window = max(max_in_window, r - l + 1)
						res[p] = max_in_window
					return res

				max_occ_in_window = _calc_max_occ(positions_by_pattern)
				max_occ_no_history = _calc_max_occ(current_positions_by_pattern)

				target_patterns_with_history: set = set()
				target_patterns_no_history: set = set()

				def _fill_target_patterns(max_occ_map: Dict[str, int], target_set: set):
					if name == "ignore_rare_patterns":
						for p, cnt in max_occ_map.items():
							if 0 < cnt <= max_occ:
								target_set.add(p)
					else:
						for p, cnt in max_occ_map.items():
							if cnt >= min_occ:
								target_set.add(p)

				_fill_target_patterns(max_occ_in_window, target_patterns_with_history)
				_fill_target_patterns(max_occ_no_history, target_patterns_no_history)

				history_required_count = 0
				no_history_count = 0
				has_history = any(len(v) > 0 for v in history_positions_by_pattern.values())

				new_final = []
				for a in final_results:
					p = a.pattern_type.lower()
					if p in target_patterns_with_history:
						rule["filtered_anomalies"].append(a)
						if p in target_patterns_no_history:
							no_history_count += 1
						else:
							history_required_count += 1
					else:
						new_final.append(a)
				rule["history_query_analysis"] = _history_query_analysis(
					used_history=has_history,
					required_count=history_required_count,
					no_history_count=no_history_count,
				)
				final_results = new_final
				continue

			if category == "ignore_duration":
				min_duration = int(params.get("min_duration", 2))
				allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
				new_final = []
				for a in final_results:
					p = a.pattern_type.lower()
					dur = int(a.end_pos) - int(a.start_pos) + 1
					if p in allowed and dur < min_duration:
						rule["filtered_anomalies"].append(a)
					else:
						new_final.append(a)
				final_results = new_final
				continue

			if category == "truncate_duration":
				max_dur = int(params.get("max_duration", 50))
				allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
				new_final: List[AnomalyPattern] = []
				for a in final_results:
					p = a.pattern_type.lower()
					if p in allowed:
						dur = int(a.end_pos) - int(a.start_pos) + 1
						if dur > max_dur:
							try:
								if not isinstance(a.additional_info, dict):
									a.additional_info = {}
							except Exception:
								a.additional_info = {}
							a.additional_info["action"] = "truncate"
							a.additional_info["adjusted_start_pos"] = int(a.start_pos)
							a.additional_info["adjusted_end_pos"] = int(a.start_pos) + max_dur - 1
							rule["filtered_anomalies"].append(a)

							new_dict = a.to_dict()
							new_dict["end_pos"] = int(a.start_pos) + max_dur - 1
							new_dict["description"] = AnomalyInjector._generate_anomaly_description(
								p,
								int(new_dict["start_pos"]),
								int(new_dict["end_pos"]),
								max_dur,
							)
							new_dict["additional_info"] = {}
							new_final.append(AnomalyPattern.from_dict(new_dict))
						else:
							new_final.append(a)
					else:
						new_final.append(a)
				final_results = new_final
				continue

			if category == "merge":
				max_gap = int(params.get("max_gap", 1))
				same_pattern_only = bool(params.get("same_pattern_only", True))
				allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
				final_results = self._merge_close_anomalies(
					final_results,
					max_gap=max_gap,
					same_pattern_only=same_pattern_only,
					allowed_patterns=allowed,
				)
				continue

			if category == "others":
				if name == "ignore_recent_reoccurrence":
					window_size = max(1, int(params.get("window_size", 100)))
					mode = str(params.get("mode", "both")).strip().lower()
					if mode not in {"magnitude", "duration", "both"}:
						mode = "both"

					allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))
					sorted_results = sorted(final_results, key=lambda a: (int(a.start_pos), int(a.end_pos)))

					# History seeds: accept tuple-list from ts_info.additional_info["history_anomalies"],
					# and fallback to summary dict ts_info.additional_info["historical_anomalies"].
					seed_events: List[Dict[str, Any]] = []
					addi = ts_info.additional_info if (ts_info and isinstance(ts_info.additional_info, dict)) else {}
					seed_tuples = addi.get("history_anomalies") if isinstance(addi, dict) else None
					if isinstance(seed_tuples, list):
						for item in seed_tuples:
							if not isinstance(item, (list, tuple)) or len(item) != 4:
								continue
							s, e, p, m = item
							try:
								s_i = int(s)
								e_i = int(e)
								m_f = float(m)
							except Exception:
								continue
							if e_i < s_i:
								s_i, e_i = e_i, s_i
							p_norm = str(p).lower()
							if p_norm not in allowed:
								continue
							seed_events.append(
								{
									"start": s_i,
									"end": e_i,
									"pattern": p_norm,
									"magnitude": m_f,
									"duration": e_i - s_i + 1,
								}
							)

					if not seed_events:
						hist_summary = addi.get("historical_anomalies") if isinstance(addi, dict) else None
						if isinstance(hist_summary, dict):
							for p, ref in hist_summary.items():
								if str(p).lower() not in allowed or not isinstance(ref, dict):
									continue
								mag = ref.get("magnitude")
								dur = ref.get("duration")
								try:
									mag_f = float(mag)
									dur_i = max(1, int(dur))
								except Exception:
									continue
								seed_events.append(
									{
										"start": -dur_i,
										"end": -1,
										"pattern": str(p).lower(),
										"magnitude": mag_f,
										"duration": dur_i,
									}
								)

					accepted_by_pattern: Dict[str, List[Dict[str, Any]]] = {}
					for ev in seed_events:
						ev["source"] = "history"
						accepted_by_pattern.setdefault(ev["pattern"], []).append(ev)

					history_required_count = 0
					no_history_count = 0
					has_history = len(seed_events) > 0

					new_final: List[AnomalyPattern] = []
					for a in sorted_results:
						p = a.pattern_type.lower()
						if p not in allowed:
							new_final.append(a)
							continue

						cur_start = int(a.start_pos)
						cur_end = int(a.end_pos)
						cur_mag = float(a.magnitude)
						cur_dur = cur_end - cur_start + 1

						recent_refs = []
						for prev in accepted_by_pattern.get(p, []):
							if cur_start - int(prev["end"]) <= window_size and int(prev["end"]) <= cur_start:
								recent_refs.append(prev)

						should_filter = False
						should_filter_without_history = False
						for prev in recent_refs:
							mag_ok = float(prev["magnitude"]) >= cur_mag
							dur_ok = int(prev["duration"]) >= cur_dur
							matched = False
							if mode == "magnitude" and mag_ok:
								matched = True
							if mode == "duration" and dur_ok:
								matched = True
							if mode == "both" and mag_ok and dur_ok:
								matched = True
							if matched:
								should_filter = True
								if prev.get("source") != "history":
									should_filter_without_history = True

						if should_filter:
							rule["filtered_anomalies"].append(a)
							if should_filter_without_history:
								no_history_count += 1
							else:
								history_required_count += 1
							continue

						new_final.append(a)
						accepted_by_pattern.setdefault(p, []).append(
							{
								"start": cur_start,
								"end": cur_end,
								"pattern": p,
								"magnitude": cur_mag,
								"duration": cur_dur,
								"source": "current",
							}
						)

					# Keep original chronological order in final_results.
					kept_ids = {id(a) for a in new_final}
					final_results = [a for a in final_results if id(a) in kept_ids]
					rule["history_query_analysis"] = _history_query_analysis(
						used_history=has_history,
						required_count=history_required_count,
						no_history_count=no_history_count,
					)
					continue

				if name == "limit_total_reports":
					max_reports = max(0, int(params.get("max_reports", 3)))
					rank_by = str(params.get("rank_by", "magnitude")).strip().lower()
					allowed = set(self.human_intent_configs[name].get("anomalies", all_anomaly_patterns()))

					if max_reports <= 0:
						rule["filtered_anomalies"].extend(final_results)
						final_results = []
						continue

					candidates = [a for a in final_results if a.pattern_type.lower() in allowed]
					non_candidates = [a for a in final_results if a.pattern_type.lower() not in allowed]

					def _duration(a: AnomalyPattern) -> int:
						return int(a.end_pos) - int(a.start_pos) + 1

					def _score(a: AnomalyPattern):
						if rank_by == "duration":
							return (_duration(a), float(a.magnitude), -int(a.start_pos))
						if rank_by == "hybrid":
							return (float(a.magnitude) * max(1, _duration(a)), float(a.magnitude), _duration(a), -int(a.start_pos))
						return (float(a.magnitude), _duration(a), -int(a.start_pos))

					sorted_candidates = sorted(candidates, key=_score, reverse=True)
					kept_candidates = sorted_candidates[:max_reports]
					kept_ids = {id(a) for a in kept_candidates}

					new_final: List[AnomalyPattern] = []
					for a in final_results:
						if id(a) in kept_ids:
							new_final.append(a)
						elif a in non_candidates:
							new_final.append(a)
						else:
							rule["filtered_anomalies"].append(a)
					final_results = new_final
				continue

		#   filtered_anomalies      dict       
		for rule in human_intent["rules"]:
			rule["filtered_anomalies"] = [a.to_dict() for a in rule.get("filtered_anomalies", [])]
		return final_results

	def get_available_intents(self) -> List[str]:
		return list(self.human_intent_configs.keys())

	def generate_specific_human_intent(self, intent_name: str, ts_info: TimeSeriesInfo) -> Dict[str, Any]:
		if intent_name not in self.human_intent_configs:
			raise ValueError(f"Unknown human intent rule: {intent_name}")
		cfg = self.human_intent_configs[intent_name]
		params = self._generate_rule_parameters(intent_name, cfg, ts_info)
		desc = self._generate_rule_description(intent_name, cfg, params)
		return {
			"categories": [cfg["category"]],
			"rules": [
				{
					"name": intent_name,
					"category": cfg["category"],
					"parameters": params,
					"description": desc,
					"filtered_anomalies": [],
					"history_query_analysis": None,
				}
			],
		}

	def _generate_rule_parameters(self, name: str, cfg: Dict[str, Any], ts_info: TimeSeriesInfo) -> Dict[str, Any]:
		params: Dict[str, Any] = {}
		if "parameter_ranges" in cfg:
			for param_name, param_range in cfg["parameter_ranges"].items():
				if isinstance(param_range, tuple) and len(param_range) == 2:
					if isinstance(param_range[0], int):
						params[param_name] = random.randint(int(param_range[0]), int(param_range[1]))
					else:
						params[param_name] = random.uniform(float(param_range[0]), float(param_range[1]))
				elif isinstance(param_range, list):
					# if param_name == "window_size" and name != "ignore_recent_reoccurrence":
					# 	param_range = [param for param in param_range if param < ts_info.seq_len]
					params[param_name] = random.choice(param_range)

				if param_name == "threshold_value":
					params[param_name] = float(params[param_name]) * float(ts_info.magnitude)
					if ts_info.value_type == "int":
						params[param_name] = int(params[param_name])

		if name == "absolute_magnitude_threshold" and "threshold_value" in params:
			val = params["threshold_value"]
			try:
				if ts_info.value_type == "int":
					params["threshold_value_str"] = str(int(val))
				else:
					base_magnitude = ts_info.magnitude
					# abs_val = abs(float(val))
					relative_magnitude = abs(float(val)-base_magnitude)
					if relative_magnitude == 0:
						params["threshold_value_str"] = "0"
					elif relative_magnitude < 1e-3:
						params["threshold_value_str"] = f"{val:.1e}"
					elif relative_magnitude < 0.01:
						params["threshold_value_str"] = f"{val:.3f}".rstrip("0").rstrip(".")
					elif relative_magnitude < 0.1:
						params["threshold_value_str"] = f"{val:.2f}".rstrip("0").rstrip(".")
					elif relative_magnitude < 1:
						params["threshold_value_str"] = f"{val:.1f}".rstrip("0").rstrip(".")
					elif relative_magnitude < 10:
						params["threshold_value_str"] = f"{round(val, 0):.0f}"
					elif relative_magnitude < 100:
						params["threshold_value_str"] = f"{round(val, -1):.0f}"
					elif relative_magnitude < 1000:
						params["threshold_value_str"] = f"{round(val, -2):.0f}"
					else:
						params["threshold_value_str"] = f"{round(val, -3):.0f}"
			except Exception:
				params["threshold_value_str"] = str(val)

		return params

	def _generate_rule_description(self, name: str, cfg: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> str:
		template = cfg.get("description_template", f"Human intent rule: {name}")
		if params and "parameter_ranges" in cfg:
			try:
				return template.format(**params)
			except KeyError as e:
				return f"Human intent rule: {name} (template formatting error: {e})"
		return template

	def _merge_close_anomalies(
		self,
		anomalies: List[AnomalyPattern],
		max_gap: int,
		same_pattern_only: bool,
		allowed_patterns: Optional[set] = None,
	) -> List[AnomalyPattern]:
		if not anomalies:
			return anomalies
		allowed_patterns = allowed_patterns or set(all_anomaly_patterns())

		items = sorted(anomalies, key=lambda a: (int(a.start_pos), int(a.end_pos)))
		merged: List[AnomalyPattern] = []
		cur_group: List[AnomalyPattern] = [items[0]]

		def _can_merge(prev: AnomalyPattern, nxt: AnomalyPattern) -> bool:
			if prev.pattern_type.lower() not in allowed_patterns or nxt.pattern_type.lower() not in allowed_patterns:
				return False
			gap = int(nxt.start_pos) - int(prev.end_pos) - 1
			if gap > max_gap:
				return False
			if same_pattern_only and prev.pattern_type.lower() != nxt.pattern_type.lower():
				return False
			return True

		def _flush(group: List[AnomalyPattern]):
			if not group:
				return
			if len(group) == 1:
				merged.append(group[0])
				return
			p = group[0].pattern_type
			start = min(int(a.start_pos) for a in group)
			end = max(int(a.end_pos) for a in group)
			mag = max(float(a.magnitude) for a in group)
			new = AnomalyPattern(
				name=f"merged_{p}_{start}_{end}",
				start_pos=start,
				end_pos=end,
				magnitude=mag,
				pattern_type=p,
				description=f"Merged anomaly ({p}): merged {len(group)} events from position {start} to {end}",
				additional_info={
					"action": "merge",
					"merged_count": len(group),
					"merged_items": [a.to_dict() for a in group],
				},
			)
			merged.append(new)

		for a in items[1:]:
			last = cur_group[-1]
			if _can_merge(last, a):
				cur_group.append(a)
			else:
				_flush(cur_group)
				cur_group = [a]

		_flush(cur_group)
		return merged


def all_human_intents() -> List[str]:
	gen = HumanIntentGenerator()
	return gen.get_available_intents()
