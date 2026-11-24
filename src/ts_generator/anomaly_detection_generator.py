"""Coordinate synthetic time-series, anomaly, and intent generation."""

import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Optional, Any

from .timeseries_generator import TimeSeriesGenerator
from .anomaly_injector import AnomalyInjector
from .human_intent_generator import HumanIntentGenerator
from .util import generate_statistics, save_ts_json, TimeSeriesInfo, AnomalyPattern, SAMPLING_INTERVAL_STR_DICT


DEFAULT_CONFIG = {
    "sampling_interval": 'random',
    "typical_seq_len": [128, 256, 512],
    "typical_prob": 0.1,
    "seq_len_range": [128, 512],
    "human_intent_count_range": [0, 3],
    "anomaly_count_range": [0, 5],
    "anomaly_type_range": [1, 3],
    "is_train": True,
    "enable_multiple_anomalies": True,
    "enable_complex_patterns": True,
    # slice tool    SFT warm-up 
    "use_tool": True,
    "query_memory": False,
    "slice_initial_ratio": 0.2,            #    slice           
    "slice_initial_padding_ratio": 0.2,    #    slice      padding  
    "slice_low_overlap_thr": 0.3,          #              IoU 
    "slice_max_stops": 4,                  #    slice   
    "hallucination_offset_frac_range": [0.03, 0.12],  #    LLM             
    "step_difficulty_probs": [0.70, 0.22, 0.08], # 2/3/4      
    "fake_chain_prob": 0.05,               #               

    #      latent                anomalies_detected
    #    human_intent     scope        “  /  ”   anomalies_detected/anomalies_final
    "latent_scope_injection_prob": 0.05,
    "latent_scope_max_patterns": 2,
    "emit_latent_anomalies": False,
    #           5/6/7   
    # 5    (normal_case_ratio, has_final_anomaly_ratio, no_final_but_detected_ratio, has_filtered_anomaly_ratio, requires_history_case_ratio)
    # 6      5         baseline_zero_case_ratio
    # 7      6         no_intent_case_ratio human_intent.rules    
    # normal_case_ratio         simple     (1 - normal_case_ratio)
    #        normal/simple        requires_history_case_ratio   sum(rule.history_query_analysis.requires_history_count) > 0   case   
    "sample_type_ratio_tuple": (0.99, 0.75, 0.15, 0.5, 0.2, 0.0, 0.0),
    #                “      ”        “     target”         
    "sampling_max_attempts_multiplier": 300,
    "sampling_priority_phase_ratio": 0.95,
    # baseline=0 case                  “    0”         
    "baseline_zero_min_zero_ratio": 0.55,
}


class AnomalyDetectionDataGenerator:
    """              -       """
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        for key in DEFAULT_CONFIG:
            if key not in self.config:
                self.config[key] = DEFAULT_CONFIG[key]
        print(f"[DEBUG] Config:\n{self.config}")
        self.ts_generator = TimeSeriesGenerator()
        self.anomaly_injector = AnomalyInjector()
        self.human_intent_generator = HumanIntentGenerator()

    def _inject_extra_patterns_by_intent(
        self,
        series: np.ndarray,
        ts_info: TimeSeriesInfo,
        anomalies: List[AnomalyPattern],
        human_intent: Dict[str, Any],
        latent_anomalies: Optional[List[AnomalyPattern]] = None,
        probability_fill: float = 0.5,
    ) -> Tuple[np.ndarray, List[AnomalyPattern], List[AnomalyPattern]]:
        """   human_intent    scope         “  ”                 

             
        -          slow_increase/slow_decrease/saturation_values      anomalies_detected 
        -                          
        """
        latent_anomalies = list(latent_anomalies or [])
        if not human_intent or not human_intent.get("rules"):
            return series, anomalies, latent_anomalies

        scope_rules = [r for r in human_intent.get("rules", []) if r.get("category") == "scope"]
        if not scope_rules:
            return series, anomalies, latent_anomalies

        occupied: set = set()
        for a in anomalies:
            for pos in range(int(a.start_pos), int(a.end_pos) + 1):
                occupied.add(pos)

        cur_series = series
        cur_anoms = list(anomalies)
        cur_latent = list(latent_anomalies)

        for rule in scope_rules:
            rule_name = rule.get("name")
            cfg = self.human_intent_generator.human_intent_configs.get(rule_name, {})
            patterns = [item for item in cfg.get("anomalies", [])]
            if ts_info.characteristics["trend"] not in ["keep steady", "increase"]:
                if "slow_increase" in patterns:
                    patterns.remove("slow_increase")
            if ts_info.characteristics["trend"] not in ["keep steady", "decrease"]:
                if "slow_decrease" in patterns:
                    patterns.remove("slow_decrease")
            if ts_info.characteristics["noise"] != "noisy":
                if "volatility_decrease" in patterns:
                    patterns.remove("volatility_decrease")
            p = random.choice(patterns) if patterns else None
            count = int(rule.get("parameters", {}).get("count", 1))
            rule_params = rule.get("parameters", {})
            
            # 1)      latent                 series 
            promoted = 0
            remain_latent: List[AnomalyPattern] = []
            for la in cur_latent:
                if la.pattern_type == p and promoted < count:
                    try:
                        if not isinstance(la.additional_info, dict):
                            la.additional_info = {}
                    except Exception:
                        la.additional_info = {}
                    la.additional_info["source"] = "latent_scope_promoted"
                    la.additional_info["intent_rule"] = rule_name
                    cur_anoms.append(la)
                    for pos in range(int(la.start_pos), int(la.end_pos) + 1):
                        occupied.add(pos)
                    promoted += 1
                else:
                    remain_latent.append(la)
            cur_latent = remain_latent
            
            # 2)               series 
            if p is not None:
                injected = promoted
                attempts = 0
                if random.random() >= probability_fill:
                    while injected < count and attempts < 50:
                        attempts += 1
                        ap = self.anomaly_injector._generate_anomaly_pattern(p, ts_info, occupied, rule_params)
                        if not ap:
                            continue
                        try:
                            if not isinstance(ap.additional_info, dict):
                                ap.additional_info = {}
                        except Exception:
                            ap.additional_info = {}
                        ap.additional_info["source"] = "scope_intent"
                        ap.additional_info["intent_rule"] = rule_name
                        #     inject_anomalies        
                        if p == "zero_values" and np.any(cur_series[ap.start_pos : ap.end_pos + 1] == 0):
                            continue
                        if p == "saturation_values" and np.any(cur_series[ap.start_pos : ap.end_pos + 1] >= ts_info.normal_range[1]):
                            continue
                        self.anomaly_injector._inject_single_anomaly(cur_series, ap, ts_info)
                        cur_anoms.append(ap)
                        for pos in range(int(ap.start_pos), int(ap.end_pos) + 1):
                            occupied.add(pos)
                        injected += 1

        cur_anoms.sort(key=lambda x: int(x.start_pos))
        cur_series = np.clip(cur_series, ts_info.normal_range[0], ts_info.normal_range[1])
        ts_info.statistics = generate_statistics(cur_series)
        return cur_series, cur_anoms, cur_latent

    def _inject_latent_scope_patterns(
        self,
        series: np.ndarray,
        ts_info: TimeSeriesInfo,
        detected_anomalies: List[AnomalyPattern],
    ) -> Tuple[np.ndarray, List[AnomalyPattern]]:
        """       latent           anomalies_detected 

              scope rule                      
          scope rule        _inject_extra_patterns_by_intent   “  ” 
        """
        prob = float(self.config.get("latent_scope_injection_prob", 0.0) or 0.0)
        max_patterns = int(self.config.get("latent_scope_max_patterns", 0) or 0)
        if prob <= 0.0 or max_patterns <= 0:
            return series, []
        if random.random() >= prob:
            return series, []

        # pool:     scope intent      anomalies   
        scope_pool: List[str] = []
        for cfg in self.human_intent_generator.human_intent_configs.values():
            if cfg.get("category") == "scope":
                for a in cfg.get("anomalies", []) or []:
                    if a not in scope_pool:
                        scope_pool.append(a)
        if not scope_pool:
            return series, []
        
        scope_pool.remove("baseline_shift_up_small")
        scope_pool.remove("baseline_shift_down_small")
        # if ts_info.characteristics["trend"] not in ["keep steady"] or ts_info.characteristics["seasonal"] not in ["no periodic fluctuation"]:
        #     scope_pool.remove("baseline_shift_up_small")
        #     scope_pool.remove("baseline_shift_down_small")
        if ts_info.characteristics["noise"] != "noisy":
        #     scope_pool.remove("volatility_increase")
            scope_pool.remove("volatility_decrease")
        if ts_info.characteristics["trend"] not in ["keep steady", "increase"]:
            scope_pool.remove("slow_increase")
        if ts_info.characteristics["trend"] not in ["keep steady", "decrease"]:
            scope_pool.remove("slow_decrease")

        occupied: set = set()
        for a in detected_anomalies:
            for pos in range(int(a.start_pos), int(a.end_pos) + 1):
                occupied.add(pos)

        k = random.randint(1, max_patterns)
        picks = random.sample(scope_pool, k=min(k, len(scope_pool)))

        cur_series = series
        latent: List[AnomalyPattern] = []
        for p in picks:
            ap = self.anomaly_injector._generate_anomaly_pattern(p, ts_info, occupied, params={})
            if not ap:
                continue
            try:
                if not isinstance(ap.additional_info, dict):
                    ap.additional_info = {}
            except Exception:
                ap.additional_info = {}
            ap.additional_info["source"] = "latent"
            ap.additional_info["latent"] = True

            #   inject_anomalies        
            if p == "zero_values" and np.any(cur_series[ap.start_pos : ap.end_pos + 1] == 0):
                continue
            if p == "saturation_values" and np.any(cur_series[ap.start_pos : ap.end_pos + 1] >= ts_info.normal_range[1]):
                continue

            self.anomaly_injector._inject_single_anomaly(cur_series, ap, ts_info)
            latent.append(ap)
            for pos in range(int(ap.start_pos), int(ap.end_pos) + 1):
                occupied.add(pos)

        cur_series = np.clip(cur_series, ts_info.normal_range[0], ts_info.normal_range[1])
        ts_info.statistics = generate_statistics(cur_series)
        latent.sort(key=lambda x: int(x.start_pos))
        return cur_series, latent
    
    def validate_simple(self, case_info: Dict) -> bool:
        ts_info = case_info["timeseries_info"]
        anomalies_detected = case_info.get("anomalies_detected", [])
        no_anomaly = len(anomalies_detected) == 0
        simple_ts = (ts_info["characteristics"]["trend"] in ["keep steady", "increase", "decrease"] and ts_info["characteristics"]["seasonal"] == "no periodic fluctuation" and ts_info["characteristics"]["noise"] == "almost no noise")
        is_simple_case = no_anomaly and simple_ts
        return is_simple_case

    def _extract_sample_type_flags(self, case: Dict[str, Any]) -> Dict[str, bool]:
        """            case    """
        anomalies_final = case.get("anomalies_final", []) or []
        anomalies_detected = case.get("anomalies_detected", []) or []
        human_intent = case.get("human_intent", {}) or {}
        ts_info = case.get("timeseries_info", {}) or {}

        has_final_anomaly = len(anomalies_final) > 0
        no_final_but_detected = (len(anomalies_final) == 0 and len(anomalies_detected) > 0)

        has_filtered_anomaly = False
        requires_history_count_sum = 0
        
        for rule in human_intent.get("rules", []) or []:
            filtered = rule.get("filtered_anomalies", []) or []
            if len(filtered) > 0:
                has_filtered_anomaly = True
            hqa = rule.get("history_query_analysis") if isinstance(rule, dict) else None
            if isinstance(hqa, dict):
                try:
                    requires_history_count_sum += int(hqa.get("requires_history_count", 0) or 0)
                except Exception:
                    continue

        requires_history_case = requires_history_count_sum > 0
        baseline_zero_case = False
        try:
            if "baseline_value" in ts_info:
                baseline_zero_case = abs(float(ts_info.get("baseline_value", 0.0))) <= 1e-9
        except Exception:
            baseline_zero_case = False

        no_intent_case = False
        try:
            if "human_intent" in case and isinstance(human_intent, dict):
                no_intent_case = len(human_intent.get("rules", []) or []) == 0
        except Exception:
            no_intent_case = False

        return {
            "has_final_anomaly": has_final_anomaly,
            "no_final_but_detected": no_final_but_detected,
            "has_filtered_anomaly": has_filtered_anomaly,
            "requires_history_case": requires_history_case,
            "baseline_zero_case": baseline_zero_case,
            "no_intent_case": no_intent_case,
        }

    def _compute_unmet_ratio_deficits(
        self,
        ratio_keys: List[str],
        current_counts: Dict[str, int],
        target_counts: Optional[Dict[str, int]],
    ) -> Dict[str, int]:
        """                """
        if target_counts is None:
            return {}
        deficits: Dict[str, int] = {}
        for k in ratio_keys:
            target = int(target_counts.get(k, 0))
            current = int(current_counts.get(k, 0))
            gap = target - current
            if gap > 0:
                deficits[k] = gap
        return deficits

    def _pick_priority_ratio_key(self, deficits: Dict[str, int], ratio_keys: List[str]) -> Optional[str]:
        """              """
        if not deficits:
            return None

        tie_break_order = [
            "requires_history_case",
            "baseline_zero_case",
            "no_intent_case",
            "has_filtered_anomaly",
            "no_final_but_detected",
            "has_final_anomaly",
        ]
        tie_break_rank = {k: i for i, k in enumerate(tie_break_order)}

        best_key: Optional[str] = None
        best_gap = -1
        best_rank = 10**9
        for k in ratio_keys:
            gap = int(deficits.get(k, 0))
            if gap <= 0:
                continue
            rank = int(tie_break_rank.get(k, 10**8))
            if gap > best_gap or (gap == best_gap and rank < best_rank):
                best_gap = gap
                best_rank = rank
                best_key = k
        return best_key

    def _is_ratio_overflow(
        self,
        ratio_keys: List[str],
        flags: Dict[str, bool],
        current_counts: Dict[str, int],
        target_counts: Optional[Dict[str, int]],
    ) -> bool:
        """                  target        """
        if target_counts is None:
            return False
        for k in ratio_keys:
            if not bool(flags.get(k, False)):
                continue
            if int(current_counts.get(k, 0)) >= int(target_counts.get(k, 0)):
                return True
        return False

    def _passes_baseline_zero_density_constraint(self, case: Dict[str, Any], min_zero_ratio: float) -> bool:
        """baseline=0 case     “    0”      """
        if min_zero_ratio <= 0.0:
            return True

        ts_info = case.get("timeseries_info", {}) or {}
        if "baseline_value" not in ts_info:
            return True
        if abs(float(ts_info.get("baseline_value", 1.0))) > 1e-9:
            return True

        values = case.get("timeseries_data", [])
        if not isinstance(values, list) or len(values) == 0:
            return True
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return True

        zero_ratio = float(np.mean(np.abs(arr) <= 1e-9))
        return zero_ratio >= float(min_zero_ratio)

    # def _build_history_anomalies(
    #     self,
    #     ts_info: TimeSeriesInfo,
    #     anomalies: List[AnomalyPattern],
    # ) -> List[Tuple[int, int, str, float]]:
    #     """Build per-case history anomaly tuples: (start_idx, end_idx, pattern, magnitude).

    #     Prefer independent history tuples from ts_info.additional_info["history_anomalies"].
    #     Positions are normalized to negative offsets within [-1000, -1].
    #     if not isinstance(ts_info.additional_info, dict):
    #         return []

    #     raw = ts_info.additional_info.get("history_anomalies", [])
    #     if not isinstance(raw, list):
    #         return []

    #     output: List[Tuple[int, int, str, float]] = []
    #     min_idx, max_idx = -1000, -1

    #     for item in raw:
    #         if not isinstance(item, (list, tuple)) or len(item) != 4:
    #             continue
    #         s, e, p, m = item
    #         try:
    #             s_i = int(s)
    #             e_i = int(e)
    #             m_f = float(m)
    #         except Exception:
    #             continue

    #         if s_i > e_i:
    #             s_i, e_i = e_i, s_i
    #         s_i = max(min_idx, min(max_idx, s_i))
    #         e_i = max(min_idx, min(max_idx, e_i))
    #         if e_i < s_i:
    #             e_i = s_i

    #         output.append((int(s_i), int(e_i), str(p).lower(), float(m_f)))

    #     output.sort(key=lambda x: (x[0], x[1], x[2]))
    #     return output

    def generate_ts_case(self):
        """      case"""
        seq_len_range = tuple(self.config["seq_len_range"])
        typical_seq_len = self.config.get("typical_seq_len", None)
        typical_prob = self.config.get("typical_prob", 0.1)
        ts_info, base_series = self.ts_generator.generate_base_ts(typical_seq_len=typical_seq_len, typical_prob=typical_prob, seq_len_range=seq_len_range, sampling_interval=self.config.sampling_interval, is_train=self.config["is_train"])
        ts_case = {
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": base_series.tolist()
        }
        return ts_case
    

    def generate_tsad_case(self, rejected = None, force_baseline_zero: bool = False) -> Dict[str, Any]:
        """        case"""
        # 1.                    
        typical_seq_len = self.config.get("typical_seq_len", None)
        typical_prob = self.config.get("typical_prob", 0.1)
        seq_len_range = tuple(self.config["seq_len_range"])
        ts_info, base_series = self.ts_generator.generate_base_ts(
            typical_seq_len=typical_seq_len,
            typical_prob=typical_prob,
            seq_len_range=seq_len_range,
            sampling_interval=self.config["sampling_interval"],
            is_train=self.config["is_train"],
            force_baseline_zero=force_baseline_zero,
        )
        
        # Inject data-level anomalies.
        anomaly_count_range = (self.config["anomaly_count_range"][0], self.config["anomaly_count_range"][1])
        anomaly_series, anomaly_patterns = self.anomaly_injector.inject_anomalies(
            base_series, 
            ts_info, 
            anomaly_count_range=anomaly_count_range,
            anomaly_type_range=(self.config["anomaly_type_range"][0], self.config["anomaly_type_range"][1]),
            inject_history_anomalies=self.config["query_memory"],
        )
        if ts_info.value_type == "int":
            anomaly_series = np.round(anomaly_series)
        anomaly_series = np.clip(anomaly_series, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(anomaly_series)
        ts_info.statistics = statistics
        label = [(anomaly.start_pos, anomaly.end_pos, anomaly.pattern_type, anomaly.magnitude) for anomaly in anomaly_patterns]

        # if self.config["query_memory"]:
        #     history_anomalies = self._build_history_anomalies(ts_info, anomaly_patterns)
        #     if not isinstance(ts_info.additional_info, dict):
        #         ts_info.additional_info = {}
        #     ts_info.additional_info["history_anomalies"] = history_anomalies
        
        ts_case = {
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": anomaly_series.tolist(),
            "anomalies_detected": [ pattern.to_dict() for pattern in anomaly_patterns ],
            "label": label,
        }


        # ============ Slice Tool        ============
        def _interval_iou(a: Tuple[int,int], b: Tuple[int,int]) -> float:
            s1, e1 = int(a[0]), int(a[1]); s2, e2 = int(b[0]), int(b[1])
            inter = max(0, min(e1, e2) - max(s1, s2) + 1)
            if inter <= 0: return 0.0
            u = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
            return inter / u if u > 0 else 0.0

        def _clamp_interval(start: int, end: int, n: int) -> Tuple[int,int]:
            s = max(0, min(start, n-1)); e = max(s, min(end, n-1)); return int(s), int(e)

        def _center_of(iv: Tuple[int,int]) -> int:
            return int((iv[0] + iv[1]) // 2)

        def _make_window(center: int, win_len: int, n: int) -> Tuple[int,int]:
            half = max(1, win_len // 2)
            s = center - half; e = s + win_len - 1
            if s < 0: e += -s; s = 0
            if e > n - 1:
                shift = e - (n - 1); s -= shift; e = n - 1
                if s < 0: s = 0
            return int(s), int(e)

        def _noisy_pred_around_truth(truth: Tuple[int,int], base_win: Tuple[int,int], n: int, scale: float) -> Tuple[int,int]:
            tlen = truth[1] - truth[0] + 1
            wlen = base_win[1] - base_win[0] + 1
            max_sigma = max(1.0, (wlen * max(0.05, scale)) / 4.0)
            shift = int(np.random.normal(0, max_sigma))
            new_center = _center_of(truth) + shift
            pred_len = max(2, int(tlen * random.uniform(0.8, 1.25)))
            p_s = new_center - pred_len // 2
            p_e = p_s + pred_len - 1
            return _clamp_interval(p_s, p_e, n)

        def _initial_llm_pred(truth: Tuple[int,int], n: int) -> Tuple[int,int]:
            frac_lo, frac_hi = self.config["hallucination_offset_frac_range"]
            frac = random.uniform(frac_lo, frac_hi)
            max_shift = max(1, int(frac * n))
            shift = random.randint(-max_shift, max_shift)
            tlen = truth[1] - truth[0] + 1
            est_len = max(2, int(tlen * random.uniform(0.8, 1.3)))
            new_center = _center_of(truth) + shift
            s = new_center - est_len // 2
            e = s + est_len - 1
            return _clamp_interval(s, e, n)

        def _simulate_chain_for_truth(truth_iv: Tuple[int,int], seq_len: int, difficulty_probs: List[float]):
            gt_s, gt_e = truth_iv
            gt_len = gt_e - gt_s + 1
            r = random.random()
            if r < difficulty_probs[0]: target_stop = 2
            elif r < sum(difficulty_probs[:2]): target_stop = 3
            elif r < sum(difficulty_probs[:3]): target_stop = 4
            low_thr = float(self.config["slice_low_overlap_thr"])
            base_ratio = float(self.config["slice_initial_ratio"])
            padding_ratio = float(self.config["slice_initial_padding_ratio"])
            init_len = max(int(gt_len * (1+padding_ratio)), int(base_ratio * seq_len))
            steps = []
            if target_stop == 2:
                prev_pred = (gt_s, gt_e)
            else:
                prev_pred = _initial_llm_pred(truth_iv, seq_len)
            cur_win_len = init_len
            center = _center_of(prev_pred)
            s_in, e_in = _make_window(center, cur_win_len, seq_len)
            steps.append({
                "step": 1,
                "prediction": (int(prev_pred[0]), int(prev_pred[1])),
                "prediction_relative": (int(prev_pred[0]), int(prev_pred[1])),
                "overlap": None,
                "consistent": False,
                "slice_input": (s_in, e_in),
                "slice_len": e_in - s_in + 1,
            })
            for step in range(2, target_stop):
                if step == target_stop-1:
                    gt_s, gt_e = truth_iv
                    pred = (gt_s, gt_e)
                    iou = _interval_iou(prev_pred, pred)
                else:
                    scale = max(0.02, cur_win_len / max(8.0, seq_len))
                    n_attempts = 0
                    while n_attempts < 100:
                        n_attempts += 1
                        pred = _noisy_pred_around_truth(truth_iv, (s_in, e_in), seq_len, scale)
                        if pred[0] == gt_s and pred[1] == gt_e:
                            continue
                        if pred[0] < s_in or pred[1] > e_in:
                            continue
                        break
                    if n_attempts >= 100:
                        raise ValueError(f"[Simulate Chain] Failed to get valid pred after 100 attempts for truth {truth_iv} in slice {s_in}-{e_in}")
                    iou = _interval_iou(prev_pred, pred)
                last_s_in, last_e_in = s_in, e_in
                if iou >= low_thr:
                    padding_ratio /= 2
                    cur_win_len = max(int(gt_len * (1+padding_ratio)), cur_win_len // 2)
                center = _center_of(pred)
                s_in, e_in = _make_window(center, cur_win_len, seq_len)
                prev_pred = pred
                if pred[0] < last_s_in or pred[1] > last_e_in:
                    raise ValueError(f"[Simulate Chain] Invalid pred {pred} out of last slice {last_s_in}-{last_e_in}")
                steps.append({
                    "step": step,
                    "prediction": (pred[0], pred[1]),
                    "prediction_relative": (pred[0]-last_s_in, pred[1]-last_s_in),
                    "overlap": iou,
                    "consistent": False,
                    "slice_input": (s_in, e_in),
                    "slice_len": e_in - s_in + 1,
                })
            if gt_s < s_in or gt_e > e_in:
                raise ValueError(f"[Simulate Chain] Final ground truth {gt_s}-{gt_e} out of last slice {s_in}-{e_in}")
            steps.append({
                "step": target_stop,
                "prediction": (gt_s, gt_e),
                "prediction_relative": (gt_s-s_in, gt_e-s_in),
                "overlap": 1.0,
                "consistent": True,
                "slice_input": None,
                "slice_len": None
            })
            return {
                "stopped": "detected",
                "num_steps": len(steps),
                "steps_detail": steps,
            }

        def _simulate_fake_chain(fake_anomaly: AnomalyPattern, seq_len: int):
            gt_s, gt_e = fake_anomaly.start_pos, fake_anomaly.end_pos
            truth_iv = (gt_s, gt_e)
            gt_len = gt_e - gt_s + 1
            low_thr = float(self.config["slice_low_overlap_thr"])
            max_stops = int(self.config["slice_max_stops"])
            base_ratio = float(self.config["slice_initial_ratio"])
            padding_ratio = float(self.config["slice_initial_padding_ratio"])
            low_thr = float(self.config["slice_low_overlap_thr"])
            base_ratio = float(self.config["slice_initial_ratio"])
            padding_ratio = float(self.config["slice_initial_padding_ratio"])
            init_len = max(int(gt_len * (1+padding_ratio)), int(base_ratio * seq_len))
            steps = []
            prev_pred = _initial_llm_pred(truth_iv, seq_len)
            cur_win_len = init_len
            center = _center_of(prev_pred)
            s_in, e_in = _make_window(center, cur_win_len, seq_len)
            steps.append({
                "step": 1,
                "prediction": (int(prev_pred[0]), int(prev_pred[1])),
                "prediction_relative": (int(prev_pred[0]), int(prev_pred[1])),
                "overlap": None,
                "consistent": False,
                "slice_input": (s_in, e_in),
                "slice_len": e_in - s_in + 1,
            })
            for step in range(2, max_stops + 1):
                scale = max(0.02, cur_win_len / max(8.0, seq_len))
                n_attempts = 0
                while n_attempts < 100:
                    n_attempts += 1
                    pred = _noisy_pred_around_truth(truth_iv, (s_in, e_in), seq_len, scale)
                    if pred[0] == gt_s and pred[1] == gt_e:
                        continue
                    if pred[0] < s_in or pred[1] > e_in:
                        continue
                    break
                if n_attempts >= 100:
                    raise ValueError(f"[Simulate Fake Chain] Failed to get valid pred after 100 attempts for truth {truth_iv} in slice {s_in}-{e_in}")
                iou = _interval_iou(prev_pred, pred)
                last_s_in, last_e_in = s_in, e_in
                if iou >= low_thr:
                    padding_ratio /= 2
                    cur_win_len = max(int(gt_len * (1+padding_ratio)), cur_win_len // 2)
                center = _center_of(pred)
                s_in, e_in = _make_window(center, cur_win_len, seq_len)
                prev_pred = pred
                if pred[0] < last_s_in or pred[1] > last_e_in:
                    raise ValueError(f"[Simulate Chain] Invalid pred {pred} out of last slice {last_s_in}-{last_e_in}")
                steps.append({
                    "step": step,
                    "prediction": (pred[0], pred[1]),
                    "prediction_relative": (pred[0]-last_s_in, pred[1]-last_s_in),
                    "overlap": iou,
                    "consistent": False,
                    "slice_input": (s_in, e_in) if step != max_stops else None,
                    "slice_len": e_in - s_in + 1,
                })
            return {
                "stopped": "max",
                "num_steps": len(steps),
                "steps_detail": steps,
            }

        if self.config.get("use_tool", True):
            detection_chains = []
            seq_len = len(anomaly_series)

            containing_fake = False
            if random.random() < float(self.config.get("fake_chain_prob", 0.05)):
                containing_fake = True
                real_amps = [float(ap.magnitude) for ap in anomaly_patterns] or [0.0]
                mean_amp = float(sum(real_amps) / len(real_amps))
                truth_intervals = [(int(ap.start_pos), int(ap.end_pos)) for ap in anomaly_patterns]
                attempts = 0
                max_attempts = 100
                f_s, f_e = 0, 1
                pattern_type=random.choice(self.anomaly_injector.get_available_patterns())
                duration = self.anomaly_injector._select_duration(pattern_type, ts_info)
                while attempts < max_attempts:
                    f_s = random.randint(0, max(0, seq_len - 2))
                    f_e = min(seq_len - 1, f_s + duration)
                    if all(_interval_iou((f_s, f_e), ti) == 0.0 for ti in truth_intervals):
                        break
                    attempts += 1
                if attempts >= max_attempts:
                    f_e = seq_len - 1
                    f_s = max(0, f_e - 1)
                amp = max(0.0, mean_amp * random.uniform(0.9, 1.1)) if mean_amp > 0 else random.uniform(0.01, 0.05)
                fake_anomaly = AnomalyPattern(
                    name="fake_anomaly",
                    start_pos=int(f_s),
                    end_pos=int(f_e),
                    magnitude=float(amp),
                    pattern_type=random.choice(self.anomaly_injector.get_available_patterns()),
                    description="This is a fake anomaly for testing.",
                )
                fake_anomaly_chain = _simulate_fake_chain(fake_anomaly, seq_len)
                anomaly_patterns.append(fake_anomaly)
                # sort anomaly_patterns by start_pos, and get index of fake_anomaly
                anomaly_patterns.sort(key=lambda x: x.start_pos)
                ts_case["anomalies_detected"] = [ pattern.to_dict() for pattern in anomaly_patterns ]
                fake_index = anomaly_patterns.index(fake_anomaly)

            difficulty_probs = list(self.config["step_difficulty_probs"]) if "step_difficulty_probs" in self.config else [0.70,0.22,0.08]
            for idx, ap in enumerate(anomaly_patterns):
                truth_iv = (int(ap.start_pos), int(ap.end_pos))
                chain = _simulate_chain_for_truth(truth_iv, seq_len, difficulty_probs)
                chain_record = {
                    "anomaly_index": idx,
                    "stopped": chain["stopped"],
                    "num_steps": chain["num_steps"],
                    "steps_detail": chain["steps_detail"],
                }
                detection_chains.append(chain_record)
            
            if containing_fake:
                detection_chains[fake_index] = {
                    "anomaly_index": fake_index,
                    "stopped": fake_anomaly_chain["stopped"],
                    "num_steps": fake_anomaly_chain["num_steps"],
                    "steps_detail": fake_anomaly_chain["steps_detail"],
                }
            
            ts_case["detection_chains"] = detection_chains

        # 4.    rejected       
        if rejected is not None:
            try:
                rejected_set = set(map(str.lower, rejected)) if isinstance(rejected, (list, tuple, set)) else set()
            except Exception:
                rejected_set = set()

            seq_len = len(anomaly_series)
            #      anomalies_detected   dict                 
            rej_anomalies = [pattern.to_dict() for pattern in anomaly_patterns]

            # position      start/end              
            if "position" in rejected_set:
                max_shift = max(2, min(10, seq_len // 20))  #     5%     10    2
                for a in rej_anomalies:
                    if "start_pos" in a and "end_pos" in a:
                        orig_s = int(a["start_pos"])
                        orig_e = int(a["end_pos"])
                        #                   shift      
                        attempts = 0
                        while attempts < 5:
                            shift_choices = [i for i in range(-max_shift, max_shift + 1) if i != 0]
                            shift_s = random.choice(shift_choices)
                            shift_e = random.choice(shift_choices)
                            s = orig_s + shift_s
                            e = orig_e + shift_e
                            s = max(0, min(s, seq_len - 2)) if seq_len >= 2 else 0
                            e = max(s + 1, min(e, seq_len - 1)) if seq_len >= 2 else s
                            if s != orig_s or e != orig_e:
                                break
                            attempts += 1
                        if s == orig_s and e == orig_e:
                            if e + 1 < seq_len:
                                e += 1
                            elif s - 1 >= 0:
                                s -= 1
                        a["start_pos"], a["end_pos"] = int(s), int(e)

            # pattern           pattern_type
            if "pattern" in rejected_set:
                try:
                    all_types = self.anomaly_injector.get_available_patterns()
                except Exception:
                    all_types = None
                for a in rej_anomalies:
                    old_t = a.get("pattern_type")
                    if all_types:
                        candidates = [t for t in all_types if t != old_t]
                        if candidates:
                            a["pattern_type"] = random.choice(candidates)

            # magnitude             
            if "magnitude" in rejected_set:
                for a in rej_anomalies:
                    amp = float(a.get("magnitude", 0.0))
                    #         0              
                    if amp == 0.0:
                        new_amp = random.uniform(0.01, 0.05)
                        a["magnitude"] = float(new_amp)
                        continue
                    attempts = 0
                    new_amp = amp
                    while attempts < 5 and abs(new_amp - amp) < 1e-9:
                        scale = random.choice([random.uniform(0.7, 0.9), random.uniform(1.1, 1.3)])  #      1
                        new_amp = max(0.0, amp * scale)
                        attempts += 1
                    if abs(new_amp - amp) < 1e-9:
                        #           20%
                        new_amp = amp * 1.2
                    a["magnitude"] = float(new_amp)

            #      label         rejected label
            rejected_label = [
                (
                    int(a.get("start_pos", 0)),
                    int(a.get("end_pos", 0)),
                    str(a.get("pattern_type", "unknown")),
                    float(a.get("magnitude", 0.0)),
                )
                for a in rej_anomalies
            ]

            ts_case["rejected"] = {
                "anomalies_detected": rej_anomalies,
                "label": rejected_label,
            }
        return ts_case


    def generate_case(self, reject=None, force_baseline_zero: bool = False) -> Dict[str, Any]:
        """    case"""
        # 1.                    
        typical_seq_len = self.config.get("typical_seq_len", None)
        typical_prob = self.config.get("typical_prob", 0.1)
        seq_len_range = tuple(self.config["seq_len_range"])
        ts_info, base_series = self.ts_generator.generate_base_ts(
            typical_seq_len=typical_seq_len,
            typical_prob=typical_prob,
            seq_len_range=seq_len_range,
            sampling_interval=self.config["sampling_interval"],
            is_train=self.config["is_train"],
            force_baseline_zero=force_baseline_zero,
        )
        
        # 2.       
        anomaly_count_range = (self.config["anomaly_count_range"][0], self.config["anomaly_count_range"][1])
        anomaly_series, anomaly_patterns = self.anomaly_injector.inject_anomalies(
            base_series, 
            ts_info, 
            anomaly_count_range=anomaly_count_range,
            anomaly_type_range=(self.config["anomaly_type_range"][0], self.config["anomaly_type_range"][1]),
            inject_frequent_pattern=True,
            inject_history_anomalies=self.config["query_memory"],
        )
        statistics = generate_statistics(anomaly_series)
        ts_info.statistics = statistics

        # Inject latent scope anomalies that are not part of anomalies_detected by default.
        anomaly_series, latent_anomalies = self._inject_latent_scope_patterns(
            anomaly_series, ts_info, anomaly_patterns
        )
        
        # Generate human intent.
        intent_range = self.config.get("human_intent_count_range") or [0, 3]
        human_intent = self.human_intent_generator.generate_human_intent(ts_info, intent_range)

        # Apply scope intents by promoting or injecting relevant anomaly types.
        anomaly_series, anomaly_patterns, latent_anomalies = self._inject_extra_patterns_by_intent(
            anomaly_series, ts_info, anomaly_patterns, human_intent, latent_anomalies
        )

        # Apply human intent to obtain final anomalies.
        # if self.config["query_memory"]:
        #     history_anomalies = self._build_history_anomalies(ts_info, anomaly_patterns)
        #     if not isinstance(ts_info.additional_info, dict):
        #         ts_info.additional_info = {}
        #     ts_info.additional_info["history_anomalies"] = history_anomalies
        final_results = self.human_intent_generator.apply_human_intent(anomaly_patterns, human_intent, ts_info)
        label = [(a.start_pos, a.end_pos, a.pattern_type, a.magnitude) for a in final_results]
        
        # 4.      case
        ts_case = {
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": anomaly_series.tolist(),
            "human_intent": human_intent,
            "anomalies_detected": [ pattern.to_dict() for pattern in anomaly_patterns ],
            "anomalies_final": [ pattern.to_dict() for pattern in final_results ],
            "label": label
        }

        if bool(self.config.get("emit_latent_anomalies", False)):
            ts_case["anomalies_latent"] = [a.to_dict() for a in latent_anomalies]

        # 5.                   generate_tsad_case 
        if self.config.get("use_tool", True):
            def _interval_iou(a: Tuple[int,int], b: Tuple[int,int]) -> float:
                s1, e1 = int(a[0]), int(a[1]); s2, e2 = int(b[0]), int(b[1])
                inter = max(0, min(e1, e2) - max(s1, s2) + 1)
                if inter <= 0: return 0.0
                u = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
                return inter / u if u > 0 else 0.0

            def _clamp_interval(start: int, end: int, n: int) -> Tuple[int,int]:
                s = max(0, min(start, n-1)); e = max(s, min(end, n-1)); return int(s), int(e)

            def _center_of(iv: Tuple[int,int]) -> int:
                return int((iv[0] + iv[1]) // 2)

            def _make_window(center: int, win_len: int, n: int) -> Tuple[int,int]:
                half = max(1, win_len // 2)
                s = center - half; e = s + win_len - 1
                if s < 0: e += -s; s = 0
                if e > n - 1:
                    shift = e - (n - 1); s -= shift; e = n - 1
                    if s < 0: s = 0
                return int(s), int(e)

            def _noisy_pred_around_truth(truth: Tuple[int,int], base_win: Tuple[int,int], n: int, scale: float) -> Tuple[int,int]:
                tlen = truth[1] - truth[0] + 1
                wlen = base_win[1] - base_win[0] + 1
                max_sigma = max(1.0, (wlen * max(0.05, scale)) / 4.0)
                shift = int(np.random.normal(0, max_sigma))
                new_center = _center_of(truth) + shift
                pred_len = max(2, int(tlen * random.uniform(0.8, 1.25)))
                p_s = new_center - pred_len // 2
                p_e = p_s + pred_len - 1
                return _clamp_interval(p_s, p_e, n)

            def _initial_llm_pred(truth: Tuple[int,int], n: int) -> Tuple[int,int]:
                frac_lo, frac_hi = self.config["hallucination_offset_frac_range"]
                frac = random.uniform(frac_lo, frac_hi)
                max_shift = max(1, int(frac * n))
                shift = random.randint(-max_shift, max_shift)
                tlen = truth[1] - truth[0] + 1
                est_len = max(2, int(tlen * random.uniform(0.8, 1.3)))
                new_center = _center_of(truth) + shift
                s = new_center - est_len // 2
                e = s + est_len - 1
                return _clamp_interval(s, e, n)

            def _simulate_chain_for_truth(truth_iv: Tuple[int,int], seq_len: int, difficulty_probs: List[float]):
                gt_s, gt_e = truth_iv
                gt_len = gt_e - gt_s + 1
                r = random.random()
                if r < difficulty_probs[0]: target_stop = 2
                elif r < sum(difficulty_probs[:2]): target_stop = 3
                elif r < sum(difficulty_probs[:3]): target_stop = 4
                low_thr = float(self.config["slice_low_overlap_thr"])
                base_ratio = float(self.config["slice_initial_ratio"])
                padding_ratio = float(self.config["slice_initial_padding_ratio"])
                init_len = max(int(gt_len * (1+padding_ratio)), int(base_ratio * seq_len))
                steps = []
                if target_stop == 2:
                    prev_pred = (gt_s, gt_e)
                else:
                    prev_pred = _initial_llm_pred(truth_iv, seq_len)
                cur_win_len = init_len
                center = _center_of(prev_pred)
                s_in, e_in = _make_window(center, cur_win_len, seq_len)
                steps.append({
                    "step": 1,
                    "prediction": (int(prev_pred[0]), int(prev_pred[1])),
                    "prediction_relative": (int(prev_pred[0]), int(prev_pred[1])),
                    "overlap": None,
                    "consistent": False,
                    "slice_input": (s_in, e_in),
                    "slice_len": e_in - s_in + 1,
                })
                for step in range(2, target_stop):
                    if step == target_stop-1:
                        gt_s, gt_e = truth_iv
                        pred = (gt_s, gt_e)
                        iou = _interval_iou(prev_pred, pred)
                    else:
                        scale = max(0.02, cur_win_len / max(8.0, seq_len))
                        n_attempts = 0
                        while n_attempts < 100:
                            n_attempts += 1
                            pred = _noisy_pred_around_truth(truth_iv, (s_in, e_in), seq_len, scale)
                            if pred[0] == gt_s and pred[1] == gt_e:
                                continue
                            if pred[0] < s_in or pred[1] > e_in:
                                continue
                            break
                        if n_attempts >= 100:
                            raise ValueError(f"[Simulate Chain] Failed to get valid pred after 100 attempts for truth {truth_iv} in slice {s_in}-{e_in}")
                        iou = _interval_iou(prev_pred, pred)
                    last_s_in, last_e_in = s_in, e_in
                    if iou >= low_thr:
                        padding_ratio /= 2
                        cur_win_len = max(int(gt_len * (1+padding_ratio)), cur_win_len // 2)
                    center = _center_of(pred)
                    s_in, e_in = _make_window(center, cur_win_len, seq_len)
                    prev_pred = pred
                    if pred[0] < last_s_in or pred[1] > last_e_in:
                        raise ValueError(f"[Simulate Chain] Invalid pred {pred} out of last slice {last_s_in}-{last_e_in}")
                    steps.append({
                        "step": step,
                        "prediction": (pred[0], pred[1]),
                        "prediction_relative": (pred[0]-last_s_in, pred[1]-last_s_in),
                        "overlap": iou,
                        "consistent": False,
                        "slice_input": (s_in, e_in),
                        "slice_len": e_in - s_in + 1,
                    })
                if gt_s < s_in or gt_e > e_in:
                    raise ValueError(f"[Simulate Chain] Final ground truth {gt_s}-{gt_e} out of last slice {s_in}-{e_in}")
                steps.append({
                    "step": target_stop,
                    "prediction": (gt_s, gt_e),
                    "prediction_relative": (gt_s-s_in, gt_e-s_in),
                    "overlap": 1.0,
                    "consistent": True,
                    "slice_input": None,
                    "slice_len": None
                })
                return {
                    "stopped": "detected",
                    "num_steps": len(steps),
                    "steps_detail": steps,
                }

            detection_chains = []
            seq_len = len(anomaly_series)
            difficulty_probs = list(self.config["step_difficulty_probs"]) if "step_difficulty_probs" in self.config else [0.70,0.22,0.08]
            for idx, ap in enumerate(anomaly_patterns):
                truth_iv = (int(ap.start_pos), int(ap.end_pos))
                chain = _simulate_chain_for_truth(truth_iv, seq_len, difficulty_probs)
                chain_record = {
                    "anomaly_index": idx,
                    "stopped": chain["stopped"],
                    "num_steps": chain["num_steps"],
                    "steps_detail": chain["steps_detail"],
                }
                detection_chains.append(chain_record)
            ts_case["detection_chains"] = detection_chains
        
        return ts_case


    def generate_dataset(self, num_samples: int, reject=None, output_file: str = None, visualize_dir = None) -> List[Dict[str, Any]]:
        """          """
        dataset = []
        used_metric_names: set = set()
        generate_method = self.generate_case if self.config["is_filter"] else self.generate_tsad_case

        ratio_tuple = self.config.get("sample_type_ratio_tuple", None)
        ratio_keys = [
            "has_final_anomaly",
            "no_final_but_detected",
            "has_filtered_anomaly",
            "requires_history_case",
        ]
        target_counts: Optional[Dict[str, int]] = None
        normal_case_ratio = 0.99
        if ratio_tuple is not None:
            if not isinstance(ratio_tuple, (list, tuple)) or len(ratio_tuple) not in (5, 6, 7):
                raise ValueError("sample_type_ratio_tuple must be a list/tuple of length 5, 6 or 7")

            parsed_ratios: List[float] = []
            for v in ratio_tuple:
                r = float(v)
                if r < 0.0 or r > 1.0:
                    raise ValueError("Each item in sample_type_ratio_tuple must be in [0, 1]")
                parsed_ratios.append(r)
            if len(parsed_ratios) >= 6:
                ratio_keys.append("baseline_zero_case")
            if len(parsed_ratios) >= 7:
                ratio_keys.append("no_intent_case")
            normal_case_ratio = float(parsed_ratios[0])
            target_counts = {
                k: int(round(num_samples * parsed_ratios[i + 1]))
                for i, k in enumerate(ratio_keys)
            }
        else:
            normal_case_ratio = float(self.config.get("normal_case_ratio", 0.99))
        current_counts: Dict[str, int] = {k: 0 for k in ratio_keys}

        normal_case_ratio = max(0.0, min(1.0, normal_case_ratio))

        def _dedup_metric_name(name: str) -> str:
            base = str(name or "unknown_metric")
            if base not in used_metric_names:
                used_metric_names.add(base)
                return base
            idx = 2
            while True:
                candidate = f"{base}_{idx}"
                if candidate not in used_metric_names:
                    used_metric_names.add(candidate)
                    return candidate
                idx += 1

        tmp_num = 0
        num_samples_normal = int(num_samples * normal_case_ratio)
        attempts_multiplier = int(self.config.get("sampling_max_attempts_multiplier", 200) or 200)
        attempts_multiplier = max(1, attempts_multiplier)
        max_attempts = int(num_samples * attempts_multiplier)
        priority_phase_ratio = float(self.config.get("sampling_priority_phase_ratio", 0.9) or 0.9)
        priority_phase_ratio = max(0.0, min(1.0, priority_phase_ratio))
        priority_phase_attempts = int(max_attempts * priority_phase_ratio)
        attempt_idx = 0
        baseline_zero_min_zero_ratio = float(self.config.get("baseline_zero_min_zero_ratio", 0.55) or 0.55)
        print(f"[DEBUG] {baseline_zero_min_zero_ratio=}")
        for _ in trange(max_attempts, desc="Generating dataset"):
            attempt_idx += 1
            # case = generate_method(reject)
            # dataset.append(case)
            try:
                force_baseline_zero = False
                if target_counts is not None and "baseline_zero_case" in ratio_keys:
                    deficits_hint = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
                    priority_key_hint = self._pick_priority_ratio_key(deficits_hint, ratio_keys)
                    baseline_zero_deficit = int(deficits_hint.get("baseline_zero_case", 0))
                    if baseline_zero_deficit > 0:
                        if attempt_idx <= priority_phase_attempts:
                            force_baseline_zero = (priority_key_hint == "baseline_zero_case")
                        else:
                            force_baseline_zero = True
                case = generate_method(force_baseline_zero=force_baseline_zero)
                if not self._passes_baseline_zero_density_constraint(case, baseline_zero_min_zero_ratio):
                    continue
                if tmp_num < num_samples_normal and self.validate_simple(case):
                    continue
                if tmp_num >= num_samples_normal and not self.validate_simple(case):
                    continue

                flags: Optional[Dict[str, bool]] = None
                if target_counts is not None:
                    flags = self._extract_sample_type_flags(case)

                    #            target    no_final_but_detected    
                    if self._is_ratio_overflow(ratio_keys, flags, current_counts, target_counts):
                        continue

                    deficits = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
                    priority_key = self._pick_priority_ratio_key(deficits, ratio_keys)
                    if (
                        priority_key is not None
                        and attempt_idx <= priority_phase_attempts
                        and not bool(flags.get(priority_key, False))
                    ):
                        continue
                
                case_info = case.get("timeseries_info", {})
                current_name = str(case_info.get("metric_name") or "unknown_metric")
                unique_name = _dedup_metric_name(current_name)
                case_info["metric_name"] = unique_name

                dataset.append(case)
                tmp_num += 1
                if target_counts is not None and flags is not None:
                    for k in ratio_keys:
                        if flags.get(k, False):
                            current_counts[k] += 1
                if tmp_num >= num_samples:
                    break
            except Exception as e:
                print(f"Error: {e}")
                continue
            if tmp_num >= num_samples:
                break

        if tmp_num < num_samples:
            print(
                f"[Sampling][WARN] Requested {num_samples}, generated {tmp_num}. "
                f"Try increasing sampling_max_attempts_multiplier or loosening ratio targets."
            )
        
        #      dataset       
        random.shuffle(dataset)

        print(f"Successfully generated {len(dataset)} cases.")
        if target_counts is not None and len(dataset) > 0:
            print("[Sampling Ratio] Target/Actual:")
            for k in ratio_keys:
                actual_ratio = float(current_counts[k]) / float(len(dataset))
                target_ratio = float(target_counts[k]) / float(num_samples) if num_samples > 0 else 0.0
                print(
                    f"  - {k}: target={target_counts[k]} ({target_ratio:.3f}), "
                    f"actual={current_counts[k]} ({actual_ratio:.3f})"
                )
            remaining = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
            if remaining:
                print("[Sampling Ratio][WARN] Unmet target counts:")
                for k in ratio_keys:
                    if k in remaining:
                        print(f"  - {k}: short_by={remaining[k]}")

        if output_file:
            save_ts_json(dataset, output_file)
            print(f"Save dataset to {output_file}")
        
        # save history anomalies to memory for later use
        if self.config["query_memory"]:
            history_anomalies_dict = {}
            for idx, data in enumerate(dataset):
                ts_info = data.get("timeseries_info", {})
                metric_name = ts_info.get("metric_name", f"metric_{idx}")
                history_anomalies = ts_info.get("additional_info", {}).get("history_anomalies", [])
                history_anomalies_dict[metric_name] = history_anomalies
            memory_dir = Path(output_file).parent / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            memory_file = memory_dir / Path(output_file).name
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(history_anomalies_dict, f, ensure_ascii=False, indent=2)
            print(f"Save memory file to {memory_file}")
        
        if visualize_dir is not None:
            Path(visualize_dir).mkdir(parents=True, exist_ok=True)
            idx = 0
            for data in tqdm(dataset, desc="Visualizing samples"):
                output_vis_file = f"{visualize_dir}/{idx}.pdf"
                visualize_sample(data, output_vis_file)
                idx += 1
            print(f"Visualize generated dataset in {visualize_dir}")
        
        return dataset

    def generate_pair_case_for_bench(
        self,
        train_len: Optional[int] = None,
        test_len: Optional[int] = None,
        train_anomaly_count_range: Tuple[int, int] = (0, 1),
        test_anomaly_count_range: Tuple[int, int] = (1, 3),
        anomaly_type_range: Optional[Tuple[int, int]] = None,
        is_filter: bool = True,
        force_baseline_zero: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """     bench    pair case train/test       """
        anomaly_type_range = anomaly_type_range or (
            int(self.config["anomaly_type_range"][0]),
            int(self.config["anomaly_type_range"][1]),
        )

        typical_seq_len = self.config.get("typical_seq_len", None)
        typical_prob = self.config.get("typical_prob", 0.1)

        # 1)    train_len/test_len ->     
        # 2)    ->    typical_seq_len + seq_len_range               
        if train_len is not None and test_len is not None:
            train_len = int(train_len)
            test_len = int(test_len)
            if train_len <= 0 or test_len <= 0:
                raise ValueError("train_len and test_len must be positive")
            total_len = int(train_len + test_len)
            ts_info = self.ts_generator.generate_ts_info(
                typical_seq_len=typical_seq_len,
                typical_prob=typical_prob,
                seq_len_range=(total_len, total_len),
                sampling_interval=self.config["sampling_interval"],
                force_baseline_zero=force_baseline_zero,
            )
        elif train_len is None and test_len is None:
            seq_len_range = tuple(self.config.get("seq_len_range", [128, 512]))
            ts_info = self.ts_generator.generate_ts_info(
                typical_seq_len=typical_seq_len,
                typical_prob=typical_prob,
                seq_len_range=seq_len_range,
                sampling_interval=self.config["sampling_interval"],
                force_baseline_zero=force_baseline_zero,
            )
            total_len = int(ts_info.seq_len)

            ratio_cfg = self.config.get("pair_train_ratio_range", [0.6, 0.8])
            ratio_lo = float(ratio_cfg[0])
            ratio_hi = float(ratio_cfg[1])
            if ratio_lo > ratio_hi:
                ratio_lo, ratio_hi = ratio_hi, ratio_lo
            ratio_lo = max(0.05, min(0.95, ratio_lo))
            ratio_hi = max(0.05, min(0.95, ratio_hi))
            if ratio_lo > ratio_hi:
                ratio_lo, ratio_hi = ratio_hi, ratio_lo

            min_train_len = int(self.config.get("pair_min_train_len", 32))
            min_test_len = int(self.config.get("pair_min_test_len", 32))
            if total_len < (min_train_len + min_test_len):
                min_train_len = max(1, total_len // 2)
                min_test_len = max(1, total_len - min_train_len)

            for _ in range(50):
                train_ratio = random.uniform(ratio_lo, ratio_hi)
                cand_train = int(round(total_len * train_ratio))
                cand_test = total_len - cand_train
                if cand_train >= min_train_len and cand_test >= min_test_len:
                    train_len = cand_train
                    test_len = cand_test
                    break

            if train_len is None or test_len is None:
                train_len = max(min_train_len, int(round(total_len * ratio_lo)))
                train_len = min(train_len, total_len - min_test_len)
                test_len = total_len - train_len
        else:
            raise ValueError("train_len and test_len should be both provided or both omitted")

        base_series = self.ts_generator.generate_ts_value(ts_info)
        if ts_info.value_type == "int":
            base_series = np.round(base_series)
        base_series = np.clip(base_series, ts_info.normal_range[0], ts_info.normal_range[1])
        ts_info.statistics = generate_statistics(base_series)

        def _compute_ratio(anomalies: List[AnomalyPattern], seq_len: int) -> float:
            if seq_len <= 0 or not anomalies:
                return 0.0
            mask = np.zeros(seq_len, dtype=np.int64)
            for ap in anomalies:
                s = max(0, int(ap.start_pos))
                e = min(seq_len - 1, int(ap.end_pos))
                if s <= e:
                    mask[s : e + 1] = 1
            return float(mask.mean())
        
        def _update_ts_info(ts_info: TimeSeriesInfo, segment: np.ndarray):
            seg_info = TimeSeriesInfo.from_dict(ts_info.to_dict())
            seg_info.seq_len = int(len(segment))
            seg_info.statistics = generate_statistics(segment)
            return seg_info

        def _normalize_history_tuple(
            s: Any,
            e: Any,
            p: Any,
            m: Any,
            min_idx: int = -1000,
            max_idx: int = -1,
        ) -> Optional[Tuple[int, int, str, float]]:
            try:
                s_i = int(s)
                e_i = int(e)
                m_f = float(m)
            except Exception:
                return None
            if s_i > e_i:
                s_i, e_i = e_i, s_i
            s_i = max(min_idx, min(max_idx, s_i))
            e_i = max(min_idx, min(max_idx, e_i))
            if e_i < s_i:
                e_i = s_i
            return (int(s_i), int(e_i), str(p).lower(), float(m_f))

        def _build_test_history_from_train(train_record: Dict[str, Any]) -> List[Tuple[int, int, str, float]]:
            merged: List[Tuple[int, int, str, float]] = []

            # 1) train anomalies_detected: train-index -> test-index     
            for a in train_record.get("anomalies_detected", []) or []:
                if not isinstance(a, dict):
                    continue
                try:
                    s_raw = int(a.get("start_pos", 0))
                    e_raw = int(a.get("end_pos", 0))
                except Exception:
                    continue
                ap = _normalize_history_tuple(
                    (s_raw - int(train_len)),
                    (e_raw - int(train_len)),
                    a.get("pattern_type", "unknown"),
                    a.get("magnitude", 0.0),
                )
                if ap is not None:
                    merged.append(ap)

            # 2) train additional_info.history_anomalies:    train    ->    test   
            train_info = train_record.get("timeseries_info", {}) or {}
            train_addi = train_info.get("additional_info", {}) if isinstance(train_info, dict) else {}
            raw_hist = train_addi.get("history_anomalies", []) if isinstance(train_addi, dict) else []
            for item in raw_hist:
                if not isinstance(item, (list, tuple)) or len(item) != 4:
                    continue
                hs, he, hp, hm = item
                try:
                    hs_i = int(hs)
                    he_i = int(he)
                except Exception:
                    continue
                ap = _normalize_history_tuple(
                    (hs_i - int(train_len)),
                    (he_i - int(train_len)),
                    hp,
                    hm,
                )
                if ap is not None:
                    merged.append(ap)

            uniq = {}
            for h in merged:
                uniq[(h[0], h[1], h[2], round(h[3], 8))] = h
            ret = list(uniq.values())
            ret.sort(key=lambda x: (x[0], x[1], x[2]))
            return ret

        def _build_segment_record(
            seg_info: TimeSeriesInfo,
            segment: np.ndarray,
            split_name: str,
            anomaly_count_range: Tuple[int, int],
            human_intent: Dict[str, Any],
            train_record: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            # 1.     
            inject_history_anomalies = bool(self.config.get("query_memory", False) and split_name == "train")
            seg_series, seg_patterns = self.anomaly_injector.inject_anomalies(
                segment,
                seg_info,
                anomaly_count_range=anomaly_count_range,
                anomaly_type_range=anomaly_type_range,
                inject_history_anomalies=inject_history_anomalies,
            )

            # test split:             train anomalies/history         
            if bool(self.config.get("query_memory", False)) and split_name == "test" and train_record is not None:
                history_anomalies = _build_test_history_from_train(train_record)
                if not isinstance(seg_info.additional_info, dict):
                    seg_info.additional_info = {}
                seg_info.additional_info["history_anomalies"] = history_anomalies

            if is_filter:
                # 2.       
                seg_series, latent_patterns = self._inject_latent_scope_patterns(
                    seg_series, seg_info, seg_patterns
                )
                # 3. scope:                   
                seg_series, seg_patterns, latent_patterns = self._inject_extra_patterns_by_intent(
                    seg_series, seg_info, seg_patterns, human_intent, latent_patterns
                )

            if seg_info.value_type == "int":
                seg_series = np.round(seg_series)
            seg_series = np.clip(seg_series, seg_info.normal_range[0], seg_info.normal_range[1])
            seg_info.statistics = generate_statistics(seg_series)

            final_patterns = self.human_intent_generator.apply_human_intent(seg_patterns, human_intent, seg_info) if is_filter else seg_patterns

            label = [
                (int(ap.start_pos), int(ap.end_pos), ap.pattern_type, float(ap.magnitude))
                for ap in final_patterns
            ]

            ret = {
                "case_id": -1,
                "split": split_name,
                "timeseries": seg_series.tolist(),
                "label": label,
                "timeseries_info": seg_info.to_dict(),
                "human_intent": human_intent,
                "anomalies_detected": [ap.to_dict() for ap in seg_patterns],
                "anomalies_final": [ap.to_dict() for ap in final_patterns],
            }
            if is_filter:
                ret["human_intent"] = human_intent
            ret["additional_info"]= {
                "split_meta": {
                    "train_len": int(train_len),
                    "test_len": int(test_len),
                    "total_len": int(total_len),
                },
                f"{split_name}_anomaly_ratio": _compute_ratio(final_patterns, len(seg_series)),
            }

            return ret

        train_base = base_series[:train_len]
        train_info = _update_ts_info(ts_info, train_base)
        test_base = base_series[train_len: train_len + test_len]
        test_info = _update_ts_info(ts_info, test_base)

        intent_range = self.config.get("human_intent_count_range") or [0, 3]
        human_intent = self.human_intent_generator.generate_human_intent(train_info, intent_range)
        human_intent_bak = copy.deepcopy(human_intent)

        train_record = _build_segment_record(
            seg_info=train_info,
            segment=train_base,
            split_name="train",
            anomaly_count_range=train_anomaly_count_range,
            human_intent=human_intent,
        )
        test_record = _build_segment_record(
            seg_info=test_info,
            segment=test_base,
            split_name="test",
            anomaly_count_range=test_anomaly_count_range,
            human_intent=human_intent_bak,
            train_record=train_record,
        )
        return {"train": train_record, "test": test_record}

    def generate_pair_dataset_for_bench(
        self,
        num_samples: int,
        train_len: Optional[int] = None,
        test_len: Optional[int] = None,
        train_anomaly_count_range: Tuple[int, int] = (0, 1),
        test_anomaly_count_range: Tuple[int, int] = (1, 3),
        anomaly_type_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """     bench    pair     train/test       """
        train_records: List[Dict[str, Any]] = []
        test_records: List[Dict[str, Any]] = []
        used_metric_names: set = set()

        ratio_tuple = self.config.get("sample_type_ratio_tuple", None)
        ratio_keys = [
            "has_final_anomaly",
            "no_final_but_detected",
            "has_filtered_anomaly",
            "requires_history_case",
        ]
        target_counts: Optional[Dict[str, int]] = None
        normal_case_ratio = 0.99
        if ratio_tuple is not None:
            if not isinstance(ratio_tuple, (list, tuple)) or len(ratio_tuple) not in (5, 6, 7):
                raise ValueError("sample_type_ratio_tuple must be a list/tuple of length 5, 6 or 7")

            parsed_ratios: List[float] = []
            for v in ratio_tuple:
                r = float(v)
                if r < 0.0 or r > 1.0:
                    raise ValueError("Each item in sample_type_ratio_tuple must be in [0, 1]")
                parsed_ratios.append(r)
            if len(parsed_ratios) >= 6:
                ratio_keys.append("baseline_zero_case")
            if len(parsed_ratios) >= 7:
                ratio_keys.append("no_intent_case")
            normal_case_ratio = float(parsed_ratios[0])
            target_counts = {
                k: int(round(num_samples * parsed_ratios[i + 1]))
                for i, k in enumerate(ratio_keys)
            }
        else:
            normal_case_ratio = float(self.config.get("normal_case_ratio", 0.99))
        current_counts: Dict[str, int] = {k: 0 for k in ratio_keys}

        normal_case_ratio = max(0.0, min(1.0, normal_case_ratio))

        def _dedup_metric_name(name: str) -> str:
            base = str(name or "unknown_metric")
            if base not in used_metric_names:
                used_metric_names.add(base)
                return base
            idx = 2
            while True:
                candidate = f"{base}_{idx}"
                if candidate not in used_metric_names:
                    used_metric_names.add(candidate)
                    return candidate
                idx += 1

        num_samples_normal = int(num_samples * normal_case_ratio)
        tmp_num = 0
        attempts_multiplier = int(self.config.get("sampling_max_attempts_multiplier", 200) or 200)
        attempts_multiplier = max(1, attempts_multiplier)
        max_attempts = int(num_samples * attempts_multiplier)
        priority_phase_ratio = float(self.config.get("sampling_priority_phase_ratio", 0.9) or 0.9)
        priority_phase_ratio = max(0.0, min(1.0, priority_phase_ratio))
        priority_phase_attempts = int(max_attempts * priority_phase_ratio)
        attempt_idx = 0
        baseline_zero_min_zero_ratio = float(self.config.get("baseline_zero_min_zero_ratio", 0.55) or 0.55)

        for _ in trange(max_attempts, desc="Generating intentad bench"):
            attempt_idx += 1
            try:
                force_baseline_zero = False
                if target_counts is not None and "baseline_zero_case" in ratio_keys:
                    deficits_hint = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
                    priority_key_hint = self._pick_priority_ratio_key(deficits_hint, ratio_keys)
                    baseline_zero_deficit = int(deficits_hint.get("baseline_zero_case", 0))
                    if baseline_zero_deficit > 0:
                        if attempt_idx <= priority_phase_attempts:
                            force_baseline_zero = (priority_key_hint == "baseline_zero_case")
                        else:
                            force_baseline_zero = True

                pair_case = self.generate_pair_case_for_bench(
                    train_len=train_len,
                    test_len=test_len,
                    train_anomaly_count_range=train_anomaly_count_range,
                    test_anomaly_count_range=test_anomaly_count_range,
                    anomaly_type_range=anomaly_type_range,
                    is_filter=self.config["is_filter"],
                    force_baseline_zero=force_baseline_zero,
                )
                if not self._passes_baseline_zero_density_constraint(pair_case.get("test", {}), baseline_zero_min_zero_ratio):
                    continue
                if tmp_num < num_samples_normal and self.validate_simple(pair_case["train"]) and self.validate_simple(pair_case["test"]):
                    continue
                if tmp_num >= num_samples_normal and not self.validate_simple(pair_case["train"]) and not self.validate_simple(pair_case["test"]):
                    continue

                # sample_type_ratio_tuple         pair       test split    
                test_flags: Optional[Dict[str, bool]] = None
                if target_counts is not None:
                    test_flags = self._extract_sample_type_flags(pair_case["test"])

                    #            target    test split 
                    if self._is_ratio_overflow(ratio_keys, test_flags, current_counts, target_counts):
                        continue

                    deficits = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
                    priority_key = self._pick_priority_ratio_key(deficits, ratio_keys)
                    if (
                        priority_key is not None
                        and attempt_idx <= priority_phase_attempts
                        and not bool(test_flags.get(priority_key, False))
                    ):
                        continue

                train_info = pair_case["train"].get("timeseries_info") or {}
                test_info = pair_case["test"].get("timeseries_info") or {}
                current_name = str(train_info.get("metric_name") or test_info.get("metric_name") or "unknown_metric")
                unique_name = _dedup_metric_name(current_name)
                train_info["metric_name"] = unique_name
                test_info["metric_name"] = unique_name

                train_records.append(pair_case["train"])
                test_records.append(pair_case["test"])
                tmp_num += 1
                if target_counts is not None and test_flags is not None:
                    for k in ratio_keys:
                        if test_flags.get(k, False):
                            current_counts[k] += 1
                if tmp_num >= num_samples:
                    break
            except Exception as e:
                print(f"Error generating case {tmp_num}: {e}")
                continue

        if tmp_num < num_samples:
            print(
                f"[Sampling][Pair/Test][WARN] Requested {num_samples}, generated {tmp_num}. "
                f"Try increasing sampling_max_attempts_multiplier or loosening ratio targets."
            )
            
        combined = list(zip(train_records, test_records))
        if target_counts is not None and len(test_records) > 0:
            print("[Sampling Ratio][Pair/Test] Target/Actual:")
            for k in ratio_keys:
                actual_ratio = float(current_counts[k]) / float(len(test_records))
                target_ratio = float(target_counts[k]) / float(num_samples) if num_samples > 0 else 0.0
                print(
                    f"  - {k}: target={target_counts[k]} ({target_ratio:.3f}), "
                    f"actual={current_counts[k]} ({actual_ratio:.3f})"
                )
            remaining = self._compute_unmet_ratio_deficits(ratio_keys, current_counts, target_counts)
            if remaining:
                print("[Sampling Ratio][Pair/Test][WARN] Unmet target counts:")
                for k in ratio_keys:
                    if k in remaining:
                        print(f"  - {k}: short_by={remaining[k]}")
        random.shuffle(combined)
        train_records[:], test_records[:] = zip(*combined)
        for i in range(len(train_records)):
            train_records[i]["case_id"] = i
            test_records[i]["case_id"] = i
        return train_records, test_records

    def get_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """         """
        if not dataset:
            return {}
        
        stats = {
            "total_cases": len(dataset),
            "metric_types": {},
            "characteristics": {},
            "anomaly_patterns": {},
            "intent_categories": {},
            "seq_len_distribution": {},
            "anomaly_count_distribution": {}
        }
        
        for case in dataset:
            metric_type = case["timeseries_info"]["metric_type"]
            stats["metric_types"][metric_type] = stats["metric_types"].get(metric_type, 0) + 1
            
            #             anomalies_final    anomalies_detected
            patterns = case.get("anomalies_final") or case.get("anomalies_detected") or []
            for pattern in patterns:
                pattern_type = pattern.get("pattern_type")
                stats["anomaly_patterns"][pattern_type] = stats["anomaly_patterns"].get(pattern_type, 0) + 1
            
            for category in (case.get("human_intent") or {}).get("categories", []):
                stats["intent_categories"][category] = stats["intent_categories"].get(category, 0) + 1
            
            seq_len = case["timeseries_info"]["seq_len"]
            seq_len_range = f"{seq_len//100*100}-{seq_len//100*100+99}"
            stats["seq_len_distribution"][seq_len_range] = stats["seq_len_distribution"].get(seq_len_range, 0) + 1
            
            anomaly_count = len(patterns)
            stats["anomaly_count_distribution"][anomaly_count] = stats["anomaly_count_distribution"].get(anomaly_count, 0) + 1
        
        return stats

    def print_statistics(self, dataset: List[Dict[str, Any]]):
        """Print a compact summary of generated cases."""
        stats = self.get_statistics(dataset)
        
        print("=== Dataset Statistics ===")
        print(f"Total cases: {stats['total_cases']}")
        
        print("\nMetric types:")
        for metric_type, count in sorted(stats["metric_types"].items()):
            percentage = count / stats["total_cases"] * 100
            print(f"  {metric_type}: {count} ({percentage:.1f}%)")
        
        print("\nAnomaly patterns:")
        for pattern, count in sorted(stats["anomaly_patterns"].items()):
            total_anomalies = sum(stats["anomaly_patterns"].values())
            percentage = count / total_anomalies * 100
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
        
        print("\nIntent categories:")
        for category, count in sorted(stats["intent_categories"].items()):
            total_intents = sum(stats["intent_categories"].values()) or 1
            percentage = count / total_intents * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")



def generate_anomaly_pattern_samples() -> List[Dict[str, Any]]:
    """Generate one example case for each supported anomaly pattern."""
    print("Generating anomaly-pattern samples...")
    
    generator = AnomalyDetectionDataGenerator()
    all_pattern_types = generator.anomaly_injector.get_available_patterns()
    if "weekly_pattern_break" in all_pattern_types:
        all_pattern_types.remove("weekly_pattern_break")
    pattern_samples = []
    
    for pattern_type in all_pattern_types:
        print(f"Generating pattern sample: {pattern_type}")
        seq_len_range = DEFAULT_CONFIG["seq_len_range"]
        typical_seq_len = DEFAULT_CONFIG["typical_seq_len"]
        typical_prob = DEFAULT_CONFIG["typical_prob"]
        if "pattern_break" in pattern_type:
            while True:
                ts_info, base_series = generator.ts_generator.generate_base_ts(
                    typical_seq_len=typical_seq_len,
                    typical_prob=typical_prob,
                    seq_len_range=seq_len_range, 
                    sampling_interval=DEFAULT_CONFIG["sampling_interval"]
                )
                if "hourly" in pattern_type and ts_info.characteristics.get("frequency") != "hourly":
                    continue
                if "daily" in pattern_type and ts_info.characteristics.get("frequency") != "daily":
                    continue
                if "weekly" in pattern_type and ts_info.characteristics.get("frequency") != "weekly":
                    continue
                break
        elif "historical_pattern_mismatch" in pattern_type:
            while True:
                ts_info, base_series = generator.ts_generator.generate_base_ts(
                    typical_seq_len=typical_seq_len,
                    typical_prob=typical_prob,
                    seq_len_range=seq_len_range, 
                    sampling_interval=DEFAULT_CONFIG["sampling_interval"]
                )
                if ts_info.characteristics.get("seasonal") != "irregular pattern":
                    continue
                break
        else:
            ts_info, base_series = generator.ts_generator.generate_base_ts(
                typical_seq_len=typical_seq_len,
                typical_prob=typical_prob,
                seq_len_range=seq_len_range, 
                sampling_interval=DEFAULT_CONFIG["sampling_interval"],
                is_train=DEFAULT_CONFIG["is_train"]
            )
        
        anomaly_series, anomaly_patterns = generator.anomaly_injector.inject_specific_pattern(
            base_series, ts_info, pattern_type, count=1
        )
        
        if ts_info.value_type == "int":
            anomaly_series = np.round(anomaly_series)
        anomaly_series = np.clip(anomaly_series, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(anomaly_series)
        ts_info.statistics = statistics
        
        sample = {
            "pattern_type": pattern_type,
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": anomaly_series.tolist(),
            "anomalies_detected": [
                {
                    "name": pattern.name,
                    "start_pos": pattern.start_pos,
                    "end_pos": pattern.end_pos,
                    "magnitude": pattern.magnitude,
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description
                }
                for pattern in anomaly_patterns
            ],
            "metadata": {
                "target_pattern": pattern_type,
                "anomaly_count": len(anomaly_patterns),
                "seq_len": len(anomaly_series)
            }
        }
        
        pattern_samples.append(sample)
    
    print(f"Generated {len(pattern_samples)} anomaly-pattern samples.")
    output_dir = Path("data/time_series")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "anomaly_patterns.json"
    save_ts_json(pattern_samples, str(output_file))
    print(f"Saved anomaly-pattern samples to: {output_file}")
    return pattern_samples


def generate_human_intent_samples() -> List[Dict[str, Any]]:
    """Generate example cases for each supported human-intent rule."""
    print("Generating human-intent samples...")

    generator = AnomalyDetectionDataGenerator()
    all_intent_types = generator.human_intent_generator.get_available_intents()
    intent_samples: List[Dict[str, Any]] = []

    for intent_type in all_intent_types:
        print(f"Generating intent sample: {intent_type}")

        seq_len_range = DEFAULT_CONFIG["seq_len_range"]
        typical_seq_len = DEFAULT_CONFIG["typical_seq_len"]
        typical_prob = DEFAULT_CONFIG["typical_prob"]
        ts_info, base_series = generator.ts_generator.generate_base_ts(
            typical_seq_len=typical_seq_len,
            typical_prob=typical_prob,
            seq_len_range=seq_len_range,
            sampling_interval=DEFAULT_CONFIG["sampling_interval"],
        )

        anomaly_count_range = (4, 6)
        anomaly_type_range = (1, 3)
        anomaly_series, anomaly_patterns = generator.anomaly_injector.inject_anomalies(
            base_series,
            ts_info,
            anomaly_count_range=anomaly_count_range,
            anomaly_type_range=anomaly_type_range,
            inject_frequent_pattern=True,
        )

        human_intent = generator.human_intent_generator.generate_specific_human_intent(intent_type, ts_info)
        anomaly_series, anomaly_patterns, _latent = generator._inject_extra_patterns_by_intent(
            anomaly_series, ts_info, anomaly_patterns, human_intent, []
        )

        final_results = generator.human_intent_generator.apply_human_intent(anomaly_patterns, human_intent, ts_info)
        label = [(anomaly.start_pos, anomaly.end_pos) for anomaly in final_results]

        sample = {
            "intent_type": intent_type,
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": anomaly_series.tolist(),
            "human_intent": human_intent,
            "anomalies_detected": [pattern.to_dict() for pattern in anomaly_patterns],
            "anomalies_final": [pattern.to_dict() for pattern in final_results],
            "label": label,
        }

        intent_samples.append(sample)

    print(f"Generated {len(intent_samples)} human-intent samples.")
    output_dir = Path("data/time_series")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "human_intents.json"
    save_ts_json(intent_samples, str(output_file))
    print(f"Saved human-intent samples to: {output_file}")
    return intent_samples


def visualize_sample(sample, output_file, split_train_test: bool = False):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    timeseries = np.array(sample.get("timeseries_data", []))
    anomalies = sample.get("anomalies_detected", []) or []
    anomalies_final = sample.get("anomalies_final", []) or []
    anomalies_latent = sample.get("anomalies_latent", []) or []
    human_intent = sample.get("human_intent") or {}
    timeseries_info = sample.get("timeseries_info") or {}
    metric_type = timeseries_info.get("metric_type", "unknown")
    metric_name = timeseries_info.get("metric_name", "unknown")
    ts_info = TimeSeriesInfo.from_dict(timeseries_info) if timeseries_info else None
    history_anomalies = ts_info.additional_info.get("history_anomalies", []) if ts_info and isinstance(ts_info.additional_info, dict) else []
    
    ax.plot(timeseries, 'b-', alpha=0.7, linewidth=1.5, label='Time Series')

    if split_train_test:
        split_meta = (sample.get("additional_info") or {}).get("split_meta", {}) if isinstance(sample, dict) else {}
        try:
            train_len = int(split_meta.get("train_len", -1))
        except Exception:
            train_len = -1
        if 0 < train_len < len(timeseries):
            # Draw a boundary between train (left) and test (right).
            ax.axvline(
                x=train_len - 0.5,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="Train/Test Split",
                zorder=8,
            )

    #      detected         final/filtered/latent       
    detected_facecolor = "#4C78A8"  # muted blue

    #       latent / final / filtered
    #          “detected   ”         
    def _plot_spans(
        items,
        *,
        facecolor: str,
        alpha: float,
        label: str,
        hatch: Optional[str] = None,
        edgecolor: Optional[str] = None,
        linewidth: float = 0.8,
        zorder: int = 3,
    ):
        if not items:
            return
        first = True
        for it in items:
            try:
                s = int(it.get("start_pos"))
                e = int(it.get("end_pos"))
            except Exception:
                continue
            ax.axvspan(
                s,
                e,
                alpha=alpha,
                color=facecolor,
                hatch=hatch,
                edgecolor=edgecolor,
                linewidth=(linewidth if edgecolor or hatch else 0),
                zorder=zorder,
                label=(label if first else None),
            )
            first = False

    #   final/filtered    “    detected anomaly       ”       axvspan   
    def _key(d: Dict[str, Any]) -> Tuple[int, int, str]:
        try:
            s = int(d.get("start_pos", -1) or -1)
            e = int(d.get("end_pos", -1) or -1)
        except Exception:
            s, e = -1, -1
        t = str(d.get("pattern_type", "unknown"))
        return s, e, t
    
    #   human_intent.rules[*].filtered_anomalies               
    filtered_items: List[Dict[str, Any]] = []
    filtered_dict: Dict[str, str] = {}
    try:
        for r in human_intent.get("rules", []) or []:
            for fa in r.get("filtered_anomalies", []) or []:
                if isinstance(fa, dict):
                    filtered_items.append(fa)
                    filtered_dict[_key(fa)] = r.get("name", "unknown_rule")
    except Exception:
        filtered_items = []

    final_keys = {_key(d) for d in (anomalies_final or []) if isinstance(d, dict)}
    filtered_keys = {_key(d) for d in (filtered_items or []) if isinstance(d, dict)}

    # promoted   detected    additional_info.source=latent_scope_promoted
    promoted_items: List[Dict[str, Any]] = []
    for a in (anomalies or []):
        if not isinstance(a, dict):
            continue
        try:
            addi = a.get("additional_info", {})
            if isinstance(addi, dict) and addi.get("source") == "latent_scope_promoted":
                promoted_items.append(a)
        except Exception:
            continue

    # latent-only    final/filtered/detected    latent      
    detected_keys = {_key(d) for d in (anomalies or []) if isinstance(d, dict)}
    latent_only = [
        d
        for d in (anomalies_latent or [])
        if isinstance(d, dict) and _key(d) not in detected_keys and _key(d) not in final_keys and _key(d) not in filtered_keys
    ]
    
    for i, anomaly in enumerate(anomalies):
        start = anomaly["start_pos"]
        end = anomaly["end_pos"]
        anomaly_magnitude = anomaly.get("magnitude", 0)
        anomaly_type = anomaly.get("pattern_type", "unknown")
        # detected               
        ax.axvspan(
            int(start),
            int(end),
            alpha=0.22,
            color=detected_facecolor,
            zorder=2,
            label=("Detected" if i == 0 else None),
        )
        duration = end - start + 1
        mid_point = (start + end) // 2
        pattern_abbr = ''.join([word[0] for word in anomaly_type.split('_')])
        annotate_str = f'{pattern_abbr}\n{start}-{end}\n({anomaly_magnitude:.3f})'
        if _key(anomaly) in filtered_keys:
            filtered_by = filtered_dict.get(_key(anomaly), "unknown_rule")
            filtered_by_abbr = ''.join([word[0] for word in filtered_by.split('_')])
            annotate_str += f"\nX: {filtered_by_abbr}"
        ax.annotate(annotate_str, 
                    xy=(mid_point, timeseries[mid_point]), 
                    xytext=(mid_point, min(timeseries)+(max(timeseries)-min(timeseries)) * 1.05),
                    fontsize=10, ha='center', alpha=0.7,
                    arrowprops=dict(arrowstyle='->'))

    #          filtered    final    promoted     latent-only    
    #        zorder       detected     
    _plot_spans(filtered_items, facecolor="#E45756", alpha=0.18, label="Filtered out", hatch="xx", edgecolor="#B23A48", linewidth=1.0, zorder=4)
    _plot_spans(anomalies_final, facecolor="#54A24B", alpha=0.16, label="Final (kept)", hatch=None, edgecolor="#2E7D32", linewidth=1.4, zorder=5)
    _plot_spans(promoted_items, facecolor="#808080", alpha=0.12, label="Promoted (from latent)", hatch="//", edgecolor="#666666", linewidth=1.0, zorder=6)
    _plot_spans(latent_only, facecolor="#808080", alpha=0.10, label="Latent (not labeled)", hatch="//", edgecolor="#666666", linewidth=1.0, zorder=7)
    
    if ts_info is not None:
        sampling_interval_str = SAMPLING_INTERVAL_STR_DICT.get(ts_info.sampling_interval, str(ts_info.sampling_interval))
        ax.set_title(f"Sample Interval:{sampling_interval_str}", fontsize=14, fontweight='bold')
        ax.set_xlabel(
            f"{ts_info.characteristics.get('trend')}, "
            f"{ts_info.characteristics.get('seasonal')}({ts_info.characteristics.get('frequency')}, {getattr(ts_info, 'magnitude', None)}), "
            f"{ts_info.characteristics.get('noise')}",
            fontsize=12,
        )
    else:
        ax.set_title("Sample", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, alpha=0.3)

    if human_intent:
        try:
            rule_lines: List[str] = []
            for r in human_intent.get("rules", []) or []:
                name = r.get("name", "unknown")
                params = r.get("parameters", {}) or {}
                if params:
                    pstr = ", ".join([f"{k}={v}" for k, v in params.items()])
                    rule_lines.append(f"- {name} ({pstr})")
                else:
                    rule_lines.append(f"- {name}")
            history_lines: List[str] = []
            for h in history_anomalies[:5]:
                try:
                    hs, he, hp, hm = h
                    history_lines.append(f"- ({int(hs)},{int(he)},{str(hp)},{float(hm):.3f})")
                except Exception:
                    continue
            history_info = ""
            if history_lines:
                history_info = (
                    f"\nHistory ({len(history_anomalies)}):\n"
                    + "\n".join(history_lines)
                    + ("\n- ..." if len(history_anomalies) > len(history_lines) else "")
                )
            info = (
                f"Metric: {metric_name} ({metric_type})\n"
                f"Detected: {len(anomalies)} | Final: {len(anomalies_final)} | Latent: {len(anomalies_latent)}\n"
                f"Rules ({len(human_intent.get('rules', []) or [])}):\n" + "\n".join(rule_lines) + history_info
            )
            ax.text(
                0.98,
                0.98,
                info,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
        except Exception:
            pass

    #    legend      anomaly      
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_anomaly_patterns(pattern_samples: List[Dict[str, Any]], output_dir: str = "data/visualizations"):
    """
                        PDF  
    """
    print("🎨          ...")
    
    pattern_output_dir = Path(output_dir) / "anomaly_patterns"
    pattern_output_dir.mkdir(parents=True, exist_ok=True)
    
    #   matplotlib  
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    pattern_names = {
        "spike": "Spike",
        "dip": "Dip", 
        "multi_spike": "Multiple Spikes",
        "multi_dip": "Multiple Dips",
        "sudden_increase": "Sudden Increase",
        "sudden_decrease": "Sudden Decrease",
        "slow_increase": "Slow Increase",
        "slow_decrease": "Slow Decrease",
        "density_increase": "Density Increase",
        "hourly_pattern_break": "Hourly Pattern Break",
        "daily_pattern_break": "Daily Pattern Break",
        "weekly_pattern_break": "Weekly Pattern Break", 
        "seasonal_pattern_break": "Seasonal Pattern Break",
        "zero_values": "Zero Values",
        "saturation_values": "Saturation Values",
        "historical_pattern_mismatch": "Historical Pattern Mismatch"
    }
    
    #    pattern     PDF
    for sample in pattern_samples:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        timeseries = np.array(sample["timeseries_data"])
        anomalies = sample["anomalies_detected"]
        pattern_type = sample["pattern_type"]
        metric_type = sample["timeseries_info"]["metric_type"]
        metric_name = sample["timeseries_info"]["metric_name"]
        ts_info = TimeSeriesInfo.from_dict(sample["timeseries_info"])
        
        ax.plot(timeseries, 'b-', alpha=0.7, linewidth=1.5, label='Time Series')
        
        colors = ['red', 'orange', 'purple', 'green', 'brown']
        target_anomalies = [a for a in anomalies if a["pattern_type"] == pattern_type]
        
        for i, anomaly in enumerate(target_anomalies):
            start = anomaly["start_pos"]
            end = anomaly["end_pos"]
            color = colors[i % len(colors)]
            
            ax.axvspan(start, end, alpha=0.3, color=color, label=f'Anomaly {i+1}')
            
            duration = end - start + 1
            mid_point = (start + end) // 2
            ax.annotate(f'{duration}pts', 
                       xy=(mid_point, timeseries[mid_point]), 
                       xytext=(mid_point, min(timeseries)+(max(timeseries)-min(timeseries)) * 1.05),
                       fontsize=10, ha='center', alpha=0.7,
                       arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        en_name = pattern_names.get(pattern_type, pattern_type)
        target_count = len(target_anomalies)
        ax.set_title(f"{en_name} Anomaly Pattern\n({pattern_type}, {target_count} occurrences)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Point", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        char_str = f"{ts_info.characteristics['trend']}, {ts_info.characteristics['seasonal']}({ts_info.characteristics['frequency']}, {ts_info.magnitude}), {ts_info.characteristics['noise']}"
        for key, value in ts_info.additional_info.items():
            if "list" not in key:
                char_str += f", {key}: {value}"
        info_text = f"Sample Interval:{ts_info.sampling_interval}\nMetric: {metric_name}\nType: {metric_type}\nCharacteristics: {char_str}\nSequence Length: {len(timeseries)}"
        if target_anomalies:
            first_anomaly = target_anomalies[0]
            info_text += f"\nmagnitude: {first_anomaly.get('magnitude', 0):.3f}"
        
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        output_file = pattern_output_dir / f"{pattern_type}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved anomaly-pattern visualization for {pattern_type}: {output_file}")
    
    print("Finished generating anomaly-pattern visualizations.")


def visualize_complete_pipeline(samples: List[Dict[str, Any]], output_dir: str = "data/visualizations"):
    """Visualize the input, detected anomalies, and final intent-aware results."""
    print("Generating complete-pipeline visualizations...")

    intent_output_dir = Path(output_dir) / "human_intents"
    intent_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    intent_names = {
        "focus_on_increase_only": "Focus on Increase Only",
        "focus_on_decrease_only": "Focus on Decrease Only",
        "focus_on_zero_only": "Focus on Zero Only",
        "absolute_magnitude_threshold": "Absolute magnitude Threshold",
        "ignore_frequent_patterns": "Ignore Frequent Patterns",
        "truncate_overlong_duration": "Truncate Overlong Duration",
        "ignore_short_duration": "Ignore Short Duration",
        "merge_close_anomalies": "Merge Close Anomalies",
        "include_slow_trends": "Include Slow Trends",
        "include_saturation_values": "Include Saturation Values",
    }

    for sample in samples:
        intent_type = sample.get("intent_type") or sample.get("filter_type") or "unknown_intent"
        timeseries = np.array(sample["timeseries_data"])
        anomalies = sample["anomalies_detected"]
        anomalies_final = sample["anomalies_final"]
        metric_type = sample["timeseries_info"]["metric_type"]
        metric_name = sample["timeseries_info"]["metric_name"]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Step 1: input series with injected anomalies.
        ax1 = axes[0]
        ax1.plot(timeseries, 'b-', alpha=0.7, linewidth=1.5, label='Time Series')
        
        color_map = {
            "spike": "red", "dip": "orange", "multi_spike": "red", 
            "multi_dip": "orange", "sudden_increase": "red", 
            "sudden_decrease": "orange", "slow_increase": "red",
            "slow_decrease": "orange", "zero_values": "black",
            "saturation_values": "black", "density_increase": "lime",
            "daily_pattern_break": "purple",
            "hourly_pattern_break": "purple",
            "weekly_pattern_break": "purple",
            "multi_pattern_deduplication_filter": "yellow",
        }
        
        for i, anomaly in enumerate(anomalies):
            start = anomaly["start_pos"]
            end = anomaly["end_pos"]
            pattern_type = anomaly["pattern_type"]
            color = color_map.get(pattern_type, "red")
            
            ax1.axvspan(start, end, alpha=0.3, color=color)
            
            mid_point = (start + end) // 2
            pattern_abbr = ''.join([word[0] for word in pattern_type.split('_')])
            ax1.annotate(f'{pattern_abbr}\n({start}-{end})', 
                       xy=(mid_point, timeseries[mid_point]), 
                       xytext=(mid_point, min(timeseries)+(max(timeseries)-min(timeseries)) * 1.05),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5),
                       arrowprops=dict(arrowstyle='->', color=color))
        
        ax1.set_title(f"Step 1: Input Time Series with Injected Anomalies\nMetric: {metric_name} ({metric_type})", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time Point")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)
        
        # Step 2: detected anomalies.
        ax2 = axes[1]
        ax2.plot(timeseries, 'b-', alpha=0.7, linewidth=1.5, label='Time Series')
        
        for anomaly in anomalies:
            start = anomaly["start_pos"]
            end = anomaly["end_pos"]
            ax2.axvspan(start, end, alpha=0.4, color='orange', label='Detected Anomaly')
        
        original_count = len(anomalies)
        ax2.set_title(f"Step 2: Anomaly Detection Results ({original_count} anomalies detected)\nMetric: {metric_name} ({metric_type})", 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel("Time Point")
        ax2.set_ylabel("Value")
        ax2.grid(True, alpha=0.3)
        
        # Step 3: final results after applying human intent.
        ax3 = axes[2]
        ax3.plot(timeseries, 'b-', alpha=0.7, linewidth=1.5, label='Time Series')

        filtered_anomalies: List[AnomalyPattern] = []
        human_intent = sample.get("human_intent") or {}
        filter_info = f"Intent: {intent_names.get(intent_type, intent_type)}\n"
        for rule in human_intent.get("rules", []):
            filter_info += f"  - {rule.get('name')}\n"
            parameters = rule.get("parameters", {})
            rule_filtered_anomalies = [
                AnomalyPattern.from_dict(anomaly) for anomaly in rule.get("filtered_anomalies", [])
            ]
            filtered_anomalies.extend(rule_filtered_anomalies)
            for param, value in parameters.items():
                filter_info += f"      {param}: {value}\n"
            for anomaly in rule_filtered_anomalies:
                start = anomaly.start_pos
                end = anomaly.end_pos
                ax3.axvspan(start, end, alpha=0.3, color="red", label="Filtered Out")
                mid_point = (start + end) // 2
                ax3.annotate(
                    f"Filtered\n({intent_type})",
                    xy=(mid_point, timeseries[mid_point]),
                    xytext=(mid_point, min(timeseries) + (max(timeseries) - min(timeseries)) * 0.9),
                    fontsize=7,
                    ha="center",
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.3),
                )
        
        for anomaly in anomalies_final:
            start = anomaly["start_pos"]
            end = anomaly["end_pos"]
            ax3.axvspan(start, end, alpha=0.4, color='green', label='Kept')

        filter_info += f"Rules: {len(human_intent.get('rules', []))} applied"
        ax3.text(0.02, 0.98, filter_info, transform=ax3.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        kept_count = len(anomalies_final)
        filtered_count = len(filtered_anomalies)
        filter_name = intent_names.get(intent_type, intent_type)
        
        ax3.set_title(f"Step 3: {filter_name} Results\n({kept_count} kept, {filtered_count} filtered out) | Metric: {metric_name} ({metric_type})", 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel("Time Point")
        ax3.set_ylabel("Value")
        ax3.grid(True, alpha=0.3)
        
        handles, labels = ax3.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax3.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        output_file = intent_output_dir / f"{intent_type}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved intent visualization for {intent_type}: {output_file}")
    
    print("Finished generating intent visualizations.")



if __name__ == "__main__":
    # pattern_samples = generate_anomaly_pattern_samples()
    # visualize_anomaly_patterns(pattern_samples, output_dir="data/visualizations")
    
    intent_samples = generate_human_intent_samples()
    visualize_complete_pipeline(intent_samples, output_dir="data/visualizations")
