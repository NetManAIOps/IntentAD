"""Inject synthetic anomaly patterns into generated time series."""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from .util import TimeSeriesInfo, AnomalyPattern, generate_statistics

frequency2cycle = {
    "minutely": 60,
    "hourly": 3600,
    "daily": 3600*24,
    "weekly": 3600*24*7
}


class AnomalyInjector:
    """Inject anomaly patterns into time-series values."""
    
    def __init__(self):
        self.anomaly_patterns = {
            # Temporal patterns
            "spike": {"probability": 0.15, "duration_range": [1, 10], "magnitude_base": 0.2, "magnitude_std": 0.8},
            "dip": {"probability": 0.15, "duration_range": [1, 10], "magnitude_base": 0.2, "magnitude_std": 0.8},
            "multi_spike": {"probability": 0.1, "duration_range": [10, 50], "magnitude_base": 0.2, "magnitude_std": 0.8, "num_spikes": [2, 5]},
            "multi_dip": {"probability": 0.1, "duration_range": [10, 50], "magnitude_base": 0.2, "magnitude_std": 0.8, "num_dips": [2, 5]},
            "sudden_increase": {"probability": 0.12, "duration_range": [6, 50], "magnitude_base": 0.15, "magnitude_std": 0.8},
            "sudden_decrease": {"probability": 0.12, "duration_range": [6, 50], "magnitude_base": 0.15, "magnitude_std": 0.8},
            "slow_increase": {"probability": 0.08, "duration_range": [35, 100], "magnitude_base": 0.3, "magnitude_std": 0.2},
            "slow_decrease": {"probability": 0.08, "duration_range": [35, 100], "magnitude_base": 0.3, "magnitude_std": 0.2},

            # Change-shape patterns (default disabled; enabled by human intent scope)
            # Small but sustained baseline shift up/down
            "baseline_shift_up_small": {"probability": 0.02, "duration_range": [20, 120], "magnitude_base": 0.03, "magnitude_std": 0.05},
            "baseline_shift_down_small": {"probability": 0.02, "duration_range": [20, 120], "magnitude_base": 0.03, "magnitude_std": 0.05},

            # Volatility change: variance suddenly increases/decreases for a window
            # magnitude is interpreted as a ratio, not absolute magnitude
            "volatility_increase": {"probability": 0.02, "duration_range": [25, 120], "ratio_range": [1.1, 1.3, 1.6]},
            "volatility_decrease": {"probability": 0.02, "duration_range": [25, 120], "ratio_range": [1.1, 1.3, 1.6]},
            # "density_increase": {"probability": 0.05, "duration_range": [10, 30], "magnitude_base": 0.2, "magnitude_std": 1.0},
            
            # Periodic inconsistency patterns - unified
            # break_type: point | sequence | frequency
            "periodic_pattern_break_by_point": {"probability": 0.2, "duration_range": [1, 1], "magnitude_base": 0.1, "magnitude_std": 0.25},
            "periodic_pattern_break_by_sequence": {"probability": 0.2, "duration_range": [5, 50], "magnitude_base": 0.1, "magnitude_std": 0.25},
            "periodic_pattern_break_by_frequency": {"probability": 0.06, "duration_range": [5, 300], "frequency_range": [1/3, 3], "magnitude_base": 0.1, "magnitude_std": 0.25},

            # Special rules
            "zero_values": {"probability": 0.12, "duration_range": [1, 50], "magnitude_range": [0.0, 0.05]},
            "saturation_values": {"probability": 0.05, "duration_range": [1, 30], "magnitude_range": [0.95, 1.0]},
            # "historical_pattern_mismatch": {"probability": 0.04, "duration_range": [10, 40], "magnitude_base": 0.2, "magnitude_std": 1.0}
        }
        # set some pattern probabilities to 0 for testing
        # zero_probability_patterns = [ "slow_increase", "slow_decrease", "periodic_pattern_break", "saturation_values"]
        zero_probability_patterns = [
            "multi_spike",
            "multi_dip",
            "slow_increase",
            "slow_decrease",
            "saturation_values",
            # new change-shape patterns are opt-in via human intent
            "baseline_shift_up_small",
            "baseline_shift_down_small",
            "volatility_increase",
            "volatility_decrease",
            "periodic_pattern_break_by_point",
            "periodic_pattern_break_by_sequence",
            "periodic_pattern_break_by_frequency"
        ]
        self.all_anomalies_include = [
            "slow_increase",
            "slow_decrease",
            "volatility_increase",
            "volatility_decrease",
        ]
        for pattern in zero_probability_patterns:
            if pattern in self.anomaly_patterns:
                self.anomaly_patterns[pattern]["probability"] = 0.0

    def _build_history_anomalies_info(
        self,
        injected_anomalies: List[AnomalyPattern],
        max_history_points: int = 1000,
        ts_info: Optional[TimeSeriesInfo] = None
    ) -> List[Tuple[int, int, str, float]]:
        """Build independent history anomalies for intent filtering.

        Output tuple format: (start_idx, end_idx, pattern, magnitude), where
        start/end are negative offsets in [-max_history_points, -1].
        """
        max_history_points = min(max(1, int(max_history_points)), ts_info.seq_len*2 if ts_info else max_history_points)
        history: List[Tuple[int, int, str, float]] = []

        anchors = list(injected_anomalies or [])
        if anchors:
            #    0           anchors 
            max_count = min(8, max(2, len(anchors) * 1.5))
            target_count = random.randint(0, max_count)
        else:
            max_count = min(8, max(2, max_history_points // 50))
            target_count = random.randint(0, max_count)

        def _sample_duration(
            pattern_name: str,
            anchor_obj: Optional[AnomalyPattern],
        ) -> int:
            cfg = self.anomaly_patterns.get(pattern_name, {})
            cfg_min, cfg_max = 1, min(120, max_history_points)
            if isinstance(cfg.get("duration_range"), list) and len(cfg["duration_range"]) >= 2:
                cfg_min = max(1, int(cfg["duration_range"][0]))
                cfg_max = max(cfg_min, int(cfg["duration_range"][1]))
                cfg_max = min(cfg_max, max_history_points)

            if anchor_obj is None:
                #   anchor duration              
                return random.randint(cfg_min, cfg_max)

            anchor_duration = max(1, int(anchor_obj.end_pos) - int(anchor_obj.start_pos) + 1)
            jittered = int(round(anchor_duration * random.uniform(0.75, 1.25)))
            jittered = max(1, min(jittered, max_history_points))
            return max(cfg_min, min(cfg_max, jittered))

        def _is_overlapped(start_idx: int, end_idx: int) -> bool:
            for hs, he, _, _ in history:
                if not (end_idx < hs or start_idx > he):
                    return True
            return False

        attempts = 0
        valid_patterns = [key for key in self.anomaly_patterns.keys() if self.anomaly_patterns[key]["probability"] > 0.0]
        max_attempts = max(20, target_count * 40)
        while len(history) < target_count and attempts < max_attempts:
            attempts += 1
            anchor = random.choice(anchors) if anchors else None

            if anchor is not None and random.random() < 0.8:
                pattern = str(anchor.pattern_type)
            else:
                pattern = random.choice(list(valid_patterns))

            if anchor is not None:
                base_mag = max(1e-6, float(anchor.magnitude))
                magnitude = float(base_mag * random.uniform(0.75, 1.25))
            else:
                cfg = self.anomaly_patterns.get(pattern, {})
                if "magnitude_base" in cfg:
                    base_mag = max(1e-6, float(cfg.get("magnitude_base", 0.2)))
                    magnitude = float(base_mag * random.uniform(0.8, 1.4))
                elif "magnitude_range" in cfg and isinstance(cfg.get("magnitude_range"), list):
                    lo, hi = cfg["magnitude_range"][:2]
                    magnitude = float(random.uniform(float(lo), float(hi)))
                elif "ratio_range" in cfg and isinstance(cfg.get("ratio_range"), list):
                    lo, _, hi = cfg["ratio_range"]
                    magnitude = float(random.uniform(float(lo), float(hi)))
                else:
                    magnitude = float(random.uniform(0.1, 1.0))

            duration = _sample_duration(pattern, anchor)
            end_idx = -random.randint(1, max_history_points)
            start_idx = end_idx - duration + 1
            if start_idx < -max_history_points:
                start_idx = -max_history_points
            if end_idx < start_idx:
                end_idx = start_idx

            if _is_overlapped(start_idx, end_idx):
                continue

            history.append((int(start_idx), int(end_idx), str(pattern), float(magnitude)))

        history.sort(key=lambda x: (x[0], x[1], x[2]))
        return history

    def inject_anomalies(self, series: np.ndarray, ts_info: TimeSeriesInfo, anomaly_count_range: Tuple[int, int] = (0, 5), anomaly_type_range: Tuple[int, int] = (1, 3), inject_frequent_pattern: bool = False, inject_history_anomalies: List[Tuple[int, int, str, float]] = None) -> Tuple[np.ndarray, List[AnomalyPattern]]:
        """      """
        anomaly_series = series.copy()
        injected_anomalies = []
        existed_anomaly_type, existed_anomaly_count = 0, 0
        try:
            pre_frequency_prob = self.anomaly_patterns["periodic_pattern_break_by_frequency"]["probability"]
            if random.random() < pre_frequency_prob:
                #    TSInfo                     1/period_points
                sampling_interval = ts_info.sampling_interval
                freq_tag = ts_info.characteristics.get("frequency")
                if freq_tag == "minutely":
                    period_points = max(1, 60 // sampling_interval)
                elif freq_tag == "hourly":
                    period_points = max(1, 3600 // sampling_interval)
                elif freq_tag == "daily":
                    period_points = max(1, 24*3600 // sampling_interval)
                elif freq_tag == "weekly":
                    period_points = max(1, 7*24*3600 // sampling_interval)
                else:
                    period_points = max(5, ts_info.seq_len // 10)
                base_frequency = 1.0 / max(1, period_points)
                num_attempts = 10
                while num_attempts > 0:
                    freq_series, freq_anoms = self.generate_series_with_frequency_anomalies(
                        ts_info=ts_info,
                        frequency=base_frequency,
                        normal_duration_rate=max(30.0, ts_info.seq_len * 0.8),
                        anomaly_duration_rate=max(5.0, ts_info.seq_len * 0.05),
                        minimum_anomaly_duration=max(5, int(ts_info.seq_len * 0.02)),
                        minimum_normal_duration=max(15, int(ts_info.seq_len * 0.1)),
                        frequency_multiplier=3.0,
                        seed=None
                    )
                    if len(freq_anoms) > 0:
                        break
                    num_attempts -= 1
                anomaly_series = freq_series
                injected_anomalies.extend(freq_anoms)
                #               frequency     
                ts_info.additional_info = ts_info.additional_info or {}
                ts_info.additional_info["period_points"] = period_points
                existed_anomaly_type = 1
                existed_anomaly_count = len(freq_anoms)
        except Exception as e:
            print(f"Frequency pre-injection failed: {e}")
            pass
        
        min_seq_len_for_anomalies = 100  #       
        if ts_info.seq_len < min_seq_len_for_anomalies:
            max_anomalies = max(1, anomaly_count_range[0])
        else:
            max_anomalies = min(anomaly_count_range[1], ts_info.seq_len // 50)
            max_anomalies = max(max_anomalies, anomaly_count_range[0])  #          
        addi = {}
        inject_freq_cfg = None
        if inject_frequent_pattern and random.random() < 0.1:
            target_pattern = random.choice(["spike", "dip", "sudden_increase", "sudden_decrease", "zero_values"])
            addi["inject_frequent_pattern"] = {
                "pattern_type": target_pattern,
                "target_count": random.randint(3, 5),
                "magnitude_ref": None  #               
            }
            # ts_info.additional_info = addi
            inject_freq_cfg = addi["inject_frequent_pattern"]

        anomaly_type_count = random.randint(anomaly_type_range[0], anomaly_type_range[1])
        anomaly_count = random.randint(anomaly_count_range[0], max_anomalies)
        if inject_freq_cfg:
            base_count = inject_freq_cfg.get("target_count", 4)
            extra_count = random.randint(0, 1)
            anomaly_count = base_count + extra_count
            #      1                 2   
            anomaly_type_count = max(1, anomaly_type_count)
        # anomaly_type_count = anomaly_type_range[0] + int(np.abs(random.normalvariate(0, 1)) / 3 * (anomaly_type_range[1] - anomaly_type_range[0]))
        # anomaly_count = anomaly_count_range[0] + int(np.abs(random.normalvariate(0, 1)) / 3 * (max_anomalies - anomaly_count_range[0]))

        anomaly_type_count -= existed_anomaly_type
        anomaly_count -= existed_anomaly_count
        if anomaly_type_count <= 0 or anomaly_count <= 0:
            return anomaly_series, injected_anomalies

        tmp_pattern_types = {}
        if inject_freq_cfg:
            p = inject_freq_cfg["pattern_type"]
            tmp_pattern_types[p] = self.anomaly_patterns[p]
            attempts = 0
            while len(tmp_pattern_types) < anomaly_type_count and attempts < 20:
                attempts += 1
                other = self._select_anomaly_pattern(ts_info)
                if other and other not in tmp_pattern_types and other != p:
                    tmp_pattern_types[other] = self.anomaly_patterns[other]
        else:
            for _ in range(anomaly_type_count):
                pattern_type = self._select_anomaly_pattern(ts_info)
                if pattern_type and pattern_type not in tmp_pattern_types:
                    tmp_pattern_types[pattern_type] = self.anomaly_patterns[pattern_type]

        occupied_positions = set()
        for _ in range(anomaly_count):
            if inject_freq_cfg:
                if random.random() < 0.7:
                    pattern_type = inject_freq_cfg["pattern_type"]
                else:
                    others = [k for k in tmp_pattern_types.keys() if k != inject_freq_cfg["pattern_type"]]
                    pattern_type = random.choice(others) if others else inject_freq_cfg["pattern_type"]
            else:
                pattern_type = self._select_anomaly_pattern(ts_info, tmp_pattern_types)
            if not pattern_type:
                continue
            anomaly = self._generate_anomaly_pattern(pattern_type, ts_info, occupied_positions)
            if not anomaly:
                continue
            #   series anomaly.start_pos, anomaly.end_pos + 1      0       zero_values  
            if pattern_type == "zero_values" and np.any(anomaly_series[anomaly.start_pos:anomaly.end_pos + 1] == 0):
                continue
            if pattern_type == "saturation_values" and np.any(anomaly_series[anomaly.start_pos:anomaly.end_pos + 1] >= ts_info.normal_range[1]):
                continue
            if inject_freq_cfg and inject_freq_cfg["pattern_type"] == pattern_type:
                ref = inject_freq_cfg.get("magnitude_ref")
                if ref is None:
                    inject_freq_cfg["magnitude_ref"] = anomaly.magnitude
                else:
                    #    anomaly.magnitude       
                    jitter = random.uniform(0.9, 1.1)
                    anomaly.magnitude = max(0.0, ref * jitter)
            self._inject_single_anomaly(anomaly_series, anomaly, ts_info)
            injected_anomalies.append(anomaly)
            for pos in range(anomaly.start_pos, anomaly.end_pos + 1):
                occupied_positions.add(pos)
        # sort anomalies by start_pos
        injected_anomalies.sort(key=lambda x: x.start_pos)
        #       additional_info   ignore_recent_reoccurrence      
        try:
            if inject_history_anomalies:
                history_anomalies = self._build_history_anomalies_info(
                    injected_anomalies,
                    max_history_points=1000,
                    ts_info=ts_info
                )

            ts_info.additional_info = ts_info.additional_info or {}
            ts_info.additional_info["history_anomalies"] = history_anomalies
        except Exception:
            pass

        anomaly_series = np.clip(anomaly_series, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(anomaly_series)
        ts_info.statistics = statistics
        return anomaly_series, injected_anomalies

    def inject_single_pattern(self, series: np.ndarray, ts_info: TimeSeriesInfo, target_pattern: str) -> Tuple[np.ndarray, List[AnomalyPattern]]:
        """          """
        anomaly_series = series.copy()
        injected_anomalies = []
        
        if target_pattern not in self.anomaly_patterns:
            raise ValueError(f"Invalid pattern type: {target_pattern}")
        
        #   1-2        
        anomaly_count = random.randint(1, 2)
        occupied_positions = set()
        
        for _ in range(anomaly_count):
            anomaly = self._generate_anomaly_pattern(target_pattern, ts_info, occupied_positions)
            if not anomaly:
                continue
                
            self._inject_single_anomaly(anomaly_series, anomaly, ts_info)
            injected_anomalies.append(anomaly)
            for pos in range(anomaly.start_pos, anomaly.end_pos + 1):
                occupied_positions.add(pos)
        
        if not injected_anomalies:
            anomaly = self._generate_anomaly_pattern(target_pattern, ts_info, set())
            if anomaly:
                self._inject_single_anomaly(anomaly_series, anomaly, ts_info)
                injected_anomalies.append(anomaly)

        anomaly_series = np.clip(anomaly_series, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(anomaly_series)
        ts_info.statistics = statistics
        return anomaly_series, injected_anomalies

    def _select_anomaly_pattern(self, ts_info: TimeSeriesInfo, pattern_type_dict: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """               """
        weights = []
        patterns = []
        total_duration = ts_info.seq_len * ts_info.sampling_interval #  
        if pattern_type_dict is None:
            pattern_type_dict = self.anomaly_patterns

        for pattern, config in pattern_type_dict.items():
            weight = config["probability"]
            
            # Unified periodic pattern break conditional: require some periodicity and sufficient duration
            if "periodic_pattern_break" in pattern:
                has_periodic = ts_info.characteristics["seasonal"] in ["sin periodic fluctuation", "square periodic fluctuation", "triangle periodic fluctuation"]
                noise = ts_info.characteristics["noise"]
                if not has_periodic or noise != "almost no noise":
                    weight = 0.0
                else:
                    frequency = ts_info.characteristics["frequency"]    # minutely, hourly, daily, weekly
                    cycle_len = frequency2cycle[frequency]
                    if total_duration <= 2 * cycle_len:
                        weight = 0.0
                    if pattern == "periodic_pattern_break_by_frequency" and (ts_info.characteristics["seasonal"] != "sin periodic fluctuation" or ts_info.characteristics["trend"] != "keep steady"):
                        weight = 0.0
            elif pattern == "historical_pattern_mismatch":
                if ts_info.characteristics["seasonal"] != "irregular pattern":
                    weight = 0.0
                else:
                    weight = 0.8
            if pattern in ["slow_increase", "slow_decrease"]:
                if ts_info.characteristics["trend"] not in ["keep steady", "increase", "decrease"]:
                    weight = 0.0
            if pattern in ["baseline_shift_up_small", "baseline_shift_down_small"]:
                if ts_info.characteristics["trend"] not in ["keep steady"]:
                    weight = 0.0
                if ts_info.characteristics["seasonal"] not in ["no periodic fluctuation"]:
                    weight = 0.0
            if pattern in ["volatility_increase", "volatility_decrease"]:
                if ts_info.characteristics["noise"] != "noisy":
                    weight = 0.0
            if pattern == "volatility_decrease":
                if ts_info.characteristics["noise"] != "noisy":
                    weight = 0.0
            if pattern == "zero_values":
                if ts_info.baseline_value == 0:
                    weight = 0.0
            if ts_info.metric_type == "ErrorCount" and pattern == "zero_values":
                weight = 0  #     0    
            elif ts_info.metric_type == "SuccessRate" and pattern == "saturation_values":
                weight = 0  #      100%    
            elif pattern in ["slow_increase", "slow_decrease"] and ts_info.seq_len < 100:
                weight *= 0.1  #          
            elif pattern in ["multi_spike", "multi_dip"] and ts_info.seq_len < 50:
                weight *= 0.3  #             
                
            patterns.append(pattern)
            weights.append(weight)
        
        if sum(weights) == 0:
            return None
        return random.choices(patterns, weights=weights)[0]

    def _select_duration(self, pattern_type: str, ts_info: TimeSeriesInfo) -> int:
        """        """
        sampling_interval = ts_info.sampling_interval
        min_duration, max_duration = self.anomaly_patterns[pattern_type]["duration_range"]
        if pattern_type in ["multi_spike", "multi_dip"]:
            duration = random.randint(min_duration, max_duration)
        elif pattern_type in ["periodic_pattern_break_by_point", "periodic_pattern_break_by_sequence", "periodic_pattern_break_by_frequency"]:
            freq = ts_info.characteristics["frequency"]
            if freq == "minutely":
                period_points = max(1, 60 // sampling_interval)
            elif freq == "hourly":
                period_points = max(1, 3600 // sampling_interval)
            elif freq == "daily":
                period_points = max(1, 24*3600 // sampling_interval)
            elif freq == "weekly":
                period_points = max(1, 7*24*3600 // sampling_interval)
            else:
                raise NotImplementedError(f"Unknown frequency: {freq}")
            base_minutes = random.randint(min_duration, min(max_duration, max(1, ts_info.seq_len * sampling_interval // 60)))
            base_points = max(1, (base_minutes * 60) // sampling_interval)
            if pattern_type == "periodic_pattern_break_by_point":
                duration = 1
            elif pattern_type == "periodic_pattern_break_by_sequence":
                duration = random.randint(min_duration, min(max_duration, period_points//2))
            else:
                # frequency_range = self.anomaly_patterns[pattern_type]["frequency_range"]
                # m = random.choice(frequency_range)
                # duration = int(max(1, round(period_points * m)))
                duration = random.randint(min_duration, min(max_duration, period_points*2))
        elif pattern_type in ["historical_pattern_mismatch"]:
            duration = random.randint(min_duration, min(max_duration, ts_info.additional_info["irregular_pattern_length"]*2//3))
        else:
            duration = random.randint(min_duration, min(max_duration, ts_info.seq_len // 2))
        return duration

    def _generate_anomaly_pattern(
        self,
        pattern_type: str,
        ts_info: TimeSeriesInfo,
        occupied: set,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnomalyPattern]:
        """         """
        config = self.anomaly_patterns[pattern_type]
        duration = self._select_duration(pattern_type, ts_info)
        params = params or {}

        # allow human-intent overrides for some patterns
        if pattern_type in ["baseline_shift_up_small", "baseline_shift_down_small"]:
            min_duration = params.get("min_duration")
            if isinstance(min_duration, int) and min_duration > 0:
                duration = max(duration, min_duration)
            min_offset_ratio = params.get("min_offset_ratio")
            if isinstance(min_offset_ratio, (int, float)) and float(min_offset_ratio) > 0:
                # keep it as relative ratio (will be scaled in _validate_and_adjust_magnitude)
                config = dict(config)
                config["_min_offset_ratio"] = float(min_offset_ratio)
        if pattern_type in ["volatility_increase", "volatility_decrease"]:
            min_duration = params.get("min_duration")
            if isinstance(min_duration, int) and min_duration > 0:
                duration = max(duration, min_duration)
        
        max_attempts = 100
        max_available_duration = ts_info.seq_len - 20  #      10   
        # max_available_duration = ts_info.seq_len
        if duration > max_available_duration:
            duration = max(3, max_available_duration)  #   3  
        
        for _ in range(max_attempts):
            start_pos = random.randint(10, ts_info.seq_len - duration - 10)
            # start_pos = random.randint(0, ts_info.seq_len - duration)
            if ts_info.characteristics["seasonal"] == "irregular pattern":
                irregular_pattern_start_list = ts_info.additional_info.get("irregular_pattern_start_list", [])
                irregular_pattern_length = ts_info.additional_info.get("irregular_pattern_length", 0)
                if pattern_type != "historical_pattern_mismatch" and any(start_pos >= s and start_pos + duration < s + irregular_pattern_length for s in irregular_pattern_start_list): #             
                    continue
                elif pattern_type == "historical_pattern_mismatch" and not any(start_pos >= s and start_pos + duration < s + irregular_pattern_length for s in irregular_pattern_start_list):
                    continue #          
            
            end_pos = start_pos + duration - 1
            if any(pos in occupied for pos in range(start_pos, end_pos + 1)):
                continue
            if "magnitude_range" in config:
                min_mag, max_mag = config["magnitude_range"]
                magnitude = random.uniform(min_mag, max_mag)
            elif "magnitude_base" in config and "magnitude_std" in config:
                base = config["magnitude_base"]
                if ts_info.characteristics['noise'] == "noisy":
                    base *= 1.5
                if ts_info.characteristics["trend"] == "random":
                    base *= 1.2
                std = config["magnitude_std"]
                magnitude = base + np.abs(random.normalvariate(0, std))
                if "_min_offset_ratio" in config:
                    magnitude = max(float(magnitude), float(config["_min_offset_ratio"]))
            elif "ratio_range" in config:
                lo, mid, hi = config["ratio_range"]
                if ts_info.characteristics['noise'] == "noisy":
                    magnitude = random.uniform(float(mid), float(hi))
                else:
                    magnitude = random.uniform(float(lo), float(mid))
                # ratio override by intent
                # ratio = params.get("variance_ratio")
                # if isinstance(ratio, (int, float)) and ratio > 1:
                #     magnitude = float(ratio)

            additional_info: Dict[str, Any] = {}
            if pattern_type in ["baseline_shift_up_small", "baseline_shift_down_small"]:
                additional_info["direction"] = "up" if pattern_type.endswith("_up_small") else "down"
                additional_info["min_duration"] = params.get("min_duration")
                additional_info["min_offset_ratio"] = params.get("min_offset_ratio")
            if pattern_type in ["volatility_increase", "volatility_decrease"]:
                additional_info["direction"] = "increase" if pattern_type == "volatility_increase" else "decrease"
                additional_info["variance_ratio"] = params.get("variance_ratio")
                additional_info["window_size"] = params.get("window_size")

            description = AnomalyInjector._generate_anomaly_description(pattern_type, start_pos, end_pos, duration)
            
            return AnomalyPattern(
                name=f"{pattern_type}_{start_pos}_{end_pos}",
                start_pos=start_pos,
                end_pos=end_pos,
                magnitude=magnitude,
                pattern_type=pattern_type,
                description=description,
                additional_info=additional_info,
            )
        
        return None

    @classmethod
    def _generate_anomaly_description(cls, pattern_type: str, start_pos: int, end_pos: int, duration: int) -> str:
        """      """
        descriptions = {
            "spike": f"Spike anomaly: {duration} points forming spike shape from position {start_pos} to {end_pos}",
            "dip": f"Dip anomaly: {duration} points forming dip shape from position {start_pos} to {end_pos}",
            "multi_spike": f"Multiple spike anomaly: {duration} points containing multiple spike patterns from position {start_pos} to {end_pos}",
            "multi_dip": f"Multiple dip anomaly: {duration} points containing multiple dip patterns from position {start_pos} to {end_pos}",
            "sudden_increase": f"Sudden increase anomaly: rapid rise to new level then stabilize from position {start_pos} to {end_pos}",
            "sudden_decrease": f"Sudden decrease anomaly: rapid drop to new level then stabilize from position {start_pos} to {end_pos}",
            "slow_increase": f"Slow increase anomaly: gradual upward trend over {duration} points from position {start_pos} to {end_pos}",
            "slow_decrease": f"Slow decrease anomaly: gradual downward trend over {duration} points from position {start_pos} to {end_pos}",
            "baseline_shift_up_small": f"Small baseline shift (up): subtle but sustained upward level shift over {duration} points from position {start_pos} to {end_pos}",
            "baseline_shift_down_small": f"Small baseline shift (down): subtle but sustained downward level shift over {duration} points from position {start_pos} to {end_pos}",
            "volatility_increase": f"Volatility increase anomaly: variance increases over {duration} points from position {start_pos} to {end_pos}",
            "volatility_decrease": f"Volatility decrease anomaly: variance decreases (becomes smoother) over {duration} points from position {start_pos} to {end_pos}",
            "density_increase": f"Density increase anomaly: values clustering together over {duration} points from position {start_pos} to {end_pos}",
            "periodic_pattern_break_by_point": f"Periodic pattern break (point): disruption of periodic pattern at specific points over {duration} points from position {start_pos} to {end_pos}",
            "periodic_pattern_break_by_sequence": f"Periodic pattern break (sequence): disruption of periodic pattern over a sequence of {duration} points from position {start_pos} to {end_pos}",
            "periodic_pattern_break_by_frequency": f"Periodic pattern break (frequency): disruption of periodic pattern at specific frequencies over {duration} points from position {start_pos} to {end_pos}",
            "zero_values": f"Zero values anomaly: sustained near-zero values over {duration} points from position {start_pos} to {end_pos}",
            "saturation_values": f"Saturation anomaly: values reaching upper limits over {duration} points from position {start_pos} to {end_pos}",
            "historical_pattern_mismatch": f"Historical pattern mismatch: values differ from historical patterns over {duration} points from position {start_pos} to {end_pos}"
        }
        
        return descriptions.get(pattern_type, f"{pattern_type} anomaly from position {start_pos} to {end_pos}")

    def _validate_and_adjust_magnitude(self, magnitude: float, pattern_type: str, ts_info: TimeSeriesInfo, current_level: float) -> float:
        """                       """
        metric_type = ts_info.metric_type
        baseline_value = ts_info.baseline_value
        baseline_magnitude = ts_info.magnitude
        normal_range = ts_info.normal_range
        if pattern_type in ["spike", "multi_spike", "sudden_increase", "slow_increase", "baseline_shift_up_small"]:
            adjusted_magnitude = min(magnitude*baseline_magnitude, normal_range[1] - baseline_value)
        elif pattern_type in ["dip", "multi_dip", "sudden_decrease", "slow_decrease", "baseline_shift_down_small"]:
            adjusted_magnitude = min(magnitude*baseline_magnitude, baseline_value - normal_range[0])
        elif pattern_type in ["volatility_increase", "volatility_decrease"]:
            # ratio-based; do not scale by baseline magnitude
            adjusted_magnitude = float(magnitude)
        else:
            adjusted_magnitude = magnitude*baseline_magnitude
        
        if pattern_type in ["volatility_increase", "volatility_decrease"]:
            return adjusted_magnitude
        
        if ts_info.characteristics['noise'] == "noisy":
            adjusted_magnitude = adjusted_magnitude
        elif ts_info.characteristics["seasonal"] == "no periodic fluctuation" and ts_info.characteristics["trend"] == "keep steady" and pattern_type not in ["volatility_increase", "volatility_decrease"]:
            adjusted_magnitude = adjusted_magnitude * 0.4
        # else:
        #     adjusted_magnitude = magnitude * baseline_magnitude * 0.6
        if ts_info.value_type == "int":
            adjusted_magnitude = max(1, int(round(adjusted_magnitude)))
        return adjusted_magnitude

    def _inject_single_anomaly(self, series: np.ndarray, anomaly: AnomalyPattern, ts_info: TimeSeriesInfo):
        """      """
        self._current_metric_type = ts_info.metric_type
        
        start, end = anomaly.start_pos, anomaly.end_pos
        pattern_type = anomaly.pattern_type
        current_level = self._calculate_current_level(series, start, end)
        
        original_magnitude = anomaly.magnitude
        adjusted_magnitude = self._validate_and_adjust_magnitude(
            original_magnitude, pattern_type, ts_info, current_level
        )
        magnitude = adjusted_magnitude
        if pattern_type == "spike":
            abs_magnitude = self._inject_spike(series, start, end, magnitude, ts_info)
        elif pattern_type == "dip":
            abs_magnitude = self._inject_dip(series, start, end, magnitude, ts_info)
        elif pattern_type == "multi_spike":
            abs_magnitude, spike_positions = self._inject_multi_spike(series, start, end, magnitude, ts_info)
            anomaly.additional_info["spike_positions"] = spike_positions
        elif pattern_type == "multi_dip":
            abs_magnitude, dip_positions = self._inject_multi_dip(series, start, end, magnitude, ts_info)
            anomaly.additional_info["dip_positions"] = dip_positions
        elif pattern_type == "sudden_increase":
            abs_magnitude = self._inject_sudden_level_shift(series, start, end, magnitude, ts_info, direction="increase")
        elif pattern_type == "sudden_decrease":
            abs_magnitude = self._inject_sudden_level_shift(series, start, end, magnitude, ts_info, direction="decrease")
        elif pattern_type == "slow_increase":
            abs_magnitude = self._inject_slow_trend(series, start, end, magnitude, ts_info, direction="increase")
        elif pattern_type == "slow_decrease":
            abs_magnitude = self._inject_slow_trend(series, start, end, magnitude, ts_info, direction="decrease")
        elif pattern_type == "baseline_shift_up_small":
            abs_magnitude = self._inject_sudden_level_shift(series, start, end, magnitude, ts_info, direction="increase")
        elif pattern_type == "baseline_shift_down_small":
            abs_magnitude = self._inject_sudden_level_shift(series, start, end, magnitude, ts_info, direction="decrease")
        elif pattern_type == "volatility_increase":
            abs_magnitude = self._inject_volatility_change(series, start, end, float(magnitude), ts_info, direction="increase")
        elif pattern_type == "volatility_decrease":
            abs_magnitude = self._inject_volatility_change(series, start, end, float(magnitude), ts_info, direction="decrease")
        elif pattern_type == "density_increase":
            abs_magnitude = self._inject_density_increase(series, start, end, magnitude, ts_info)
        elif pattern_type == "periodic_pattern_break_by_point":
            abs_magnitude = self._inject_periodic_break_by_point(series, start, end, magnitude, ts_info)
        elif pattern_type == "periodic_pattern_break_by_sequence":
            abs_magnitude = self._inject_periodic_break_by_sequence(series, start, end, magnitude, ts_info)
        elif pattern_type == "periodic_pattern_break_by_frequency":
            abs_magnitude = self._inject_periodic_break_by_frequency(series, start, end, magnitude, ts_info)
        elif pattern_type == "zero_values":
            abs_magnitude = self._inject_zero_values(series, start, end, magnitude, ts_info)
        elif pattern_type == "saturation_values":
            abs_magnitude = self._inject_saturation_values(series, start, end, magnitude, ts_info)
        elif pattern_type == "historical_pattern_mismatch":
            abs_magnitude = self._inject_pattern_mismatch(series, start, end, magnitude, ts_info)
        anomaly.magnitude = abs_magnitude
        if ts_info.value_type == "int":
            series = np.round(series)

    def _inject_volatility_change(
        self,
        series: np.ndarray,
        start: int,
        end: int,
        ratio: float,
        ts_info: TimeSeriesInfo,
        direction: str,
    ) -> float:
        """Inject a volatility/variance change within [start, end].

        ratio > 1.0. For increase: add noise; for decrease: damp residual fluctuations.
        Returns an approximate "magnitude" in terms of robust scale (IQR-based) change.
        """
        seg = series[start : end + 1]
        if seg.size == 0:
            return 0.0

        def _robust_scale(values: np.ndarray) -> float:
            if values.size == 0:
                return 0.0
            q1, q3 = np.percentile(values, [25, 75])
            iqr = float(q3 - q1)
            #       sigma ≈ IQR / 1.349
            return iqr / 1.349 if iqr > 0 else 0.0

        base_scale = _robust_scale(seg)
        base_scale = max(base_scale, float(ts_info.magnitude) * 0.01)

        if direction == "increase":
            noise = np.random.normal(loc=0.0, scale=base_scale * (ratio - 1.0), size=seg.shape)
            series[start : end + 1] = seg + noise
            new_scale = _robust_scale(series[start : end + 1])
        elif direction == "decrease":
            window_size = max(5, len(seg) // 5)
            if window_size % 2 == 0:
                window_size += 1
            window_size = min(window_size, len(seg) if len(seg) % 2 == 1 else max(1, len(seg) - 1))

            if window_size >= 3:
                pad = window_size // 2
                padded = np.pad(seg, (pad, pad), mode="reflect")
                kernel = np.ones(window_size, dtype=float) / float(window_size)
                baseline = np.convolve(padded, kernel, mode="valid")
            else:
                baseline = np.full_like(seg, float(np.median(seg)))

            damped = baseline + (seg - baseline) / ratio
            series[start : end + 1] = damped
            new_scale = _robust_scale(series[start : end + 1])
        else:
            return 0.0

        return abs(new_scale - base_scale)

    def _calculate_current_level(self, series: np.ndarray, start: int, end: int) -> float:
        """                         """
        context_start = max(0, start - 60)
        context_end = min(len(series), end + 60)
        context_values = np.concatenate([
            series[context_start:start] if start > context_start else [],
            series[end + 1:context_end] if end + 1 < context_end else []
        ])
        
        if len(context_values) > 0:
            return np.mean(context_values)
        else:
            return np.mean(series[start:end + 1])

    def _inject_spike(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """  spike"""
        duration = end - start + 1

        if duration == 1:
            series[start] += magnitude
        else:
            mid_position = duration // 2
            for i in range(duration):
                if i <= mid_position:
                    factor = i / max(1, mid_position)
                    # factor = (i+1) / max(1, mid_position+1)
                else:
                    factor = (duration - 1 - i) / max(1, duration - 1 - mid_position)
                    # factor = (duration - i) / max(1, duration + 1 - mid_position)
                smooth_factor = 0.5 * (1 + np.cos(np.pi * (1 - factor)))
                series[start + i] += magnitude * smooth_factor
        
        return magnitude

    def _inject_dip(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """  dip"""
        duration = end - start + 1

        if duration == 1:
            series[start] -= magnitude
        else:
            mid_position = duration // 2
            for i in range(duration):
                if i <= mid_position:
                    factor = i / max(1, mid_position)
                    # factor = (i+1) / max(1, mid_position+1)
                else:
                    factor = (duration - 1 - i) / max(1, duration - 1 - mid_position)
                    # factor = (duration - i) / max(1, duration + 1 - mid_position)
                smooth_factor = 0.5 * (1 + np.cos(np.pi * (1 - factor)))
                series[start + i] -= magnitude * smooth_factor
        
        return magnitude

    def _inject_multi_spike(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """    spike"""
        duration = end - start + 1
        # num_spikes = random.randint(2, min(4, duration // 5))
        num_spikes = random.randint(max(2, duration // 11), min(5, duration // 5))
        
        spike_positions = []
        segment_length = duration // num_spikes
        
        for i in range(num_spikes):
            spike_start = start + i * segment_length
            spike_duration = random.randint(1, min(segment_length - 1, 8))
            spike_end = min(spike_start + spike_duration - 1, end)
            if i == num_spikes-1:
                spike_end = end
            spike_positions.append((spike_start, spike_end))
        
        magnitudes = []
        for spike_start, spike_end in spike_positions:
            mag = self._inject_spike(series, spike_start, spike_end, magnitude * random.uniform(0.8, 1.2), ts_info)
            magnitudes.append(mag)
        
        max_magnitude = max(magnitudes)
        return max_magnitude, spike_positions

    def _inject_multi_dip(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """    dip"""
        duration = end - start + 1
        # num_dips = random.randint(2, min(4, duration // 5))
        num_dips = random.randint(max(2, duration // 11), min(5, duration // 5))
        
        dip_positions = []
        segment_length = duration // num_dips
        
        for i in range(num_dips):
            dip_start = start + i * segment_length
            dip_duration = random.randint(1, min(segment_length - 1, 8))
            dip_end = min(dip_start + dip_duration - 1, end)
            if i == num_dips-1:
                dip_end = end
            dip_positions.append((dip_start, dip_end))
        
        magnitudes = []
        for dip_start, dip_end in dip_positions:
            mag = self._inject_dip(series, dip_start, dip_end, magnitude * random.uniform(0.8, 1.2), ts_info)
            magnitudes.append(mag)
        
        max_magnitude = max(magnitudes)
        return max_magnitude, dip_positions

    def _inject_sudden_level_shift(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo, direction: str):
        """        """
        duration = end - start + 1        
        level_change = magnitude * 0.85
        if direction == "increase":
            level_target = series[start:end+1].max() + level_change
            final_magnitude = level_target - series[start:end+1].mean()
        if direction == "decrease":
            level_target = series[start:end+1].min() - level_change
            final_magnitude = series[start:end+1].mean() - level_target
        
        noise_type = ts_info.characteristics["noise"]
        if noise_type == "almost no noise":
            if random.random() < 0.5:
                std = random.uniform(0.0, 0.003) * ts_info.magnitude
            else:
                std = 0.0
            noise = np.random.normal(0, std, duration)
        else:
            std = random.uniform(0.05, 0.1) * ts_info.magnitude
            noise = np.random.normal(0, std, duration)
        original_last_value = series[end]
        series[start:end+1] = level_target + noise

        # final_change = series[end] - original_last_value
        # #       end        final_change
        # if random.random() < 0.2 and end + 1 < len(series):
        #     series[end+1:] += final_change
        
        # change_duration = max(1, duration // 5)
        # for i in range(duration):
        #     if i < change_duration:
        #         progress = i+1 / max(1, change_duration)
        #         series[start + i] += level_change * progress
        #     else:
        #         series[start + i] += level_change
        
        return final_magnitude

    def _inject_slow_trend(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo, direction: str):
        """        """
        duration = end - start + 1
        total_change = magnitude * 0.85
        if direction == "decrease":
            total_change = -total_change
        
        for i in range(duration):
            progress = i / max(1, duration - 1)
            change = total_change * progress
            series[start + i] += change
        
        #       end        total_change
        if random.random() < 0.85 and end + 1 < len(series):
            series[end+1:] += total_change

        return magnitude * 0.85
    
    def _inject_periodic_break_by_point(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """           """
        series[start] += random.choice([-1, 1]) * magnitude
        return magnitude * 3
    
    def _inject_periodic_break_by_sequence(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        duration = end - start + 1
        mode = random.choice(["up", "down", "random"])  #     /  /    
        sign = 1 if mode == "up" else (-1 if mode == "down" else 0)
        max_magnitude = 0
        for i in range(duration):
            if mode == "random":
                rand_multi = random.uniform(-1, 1)
                series[start + i] += magnitude * rand_multi
                max_magnitude = max(max_magnitude, abs(magnitude * rand_multi))
            else:
                progress = i / max(1, duration - 1)
                series[start + i] += sign * magnitude * progress
        return magnitude * 3 if mode in ["up", "down"] else  max_magnitude * 3
    
    def _inject_periodic_break_by_frequency(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        sampling_interval = ts_info.sampling_interval
        freq = ts_info.characteristics["frequency"]
        if freq == "minutely":
            period_points = max(1, 60 // sampling_interval)
        elif freq == "hourly":
            period_points = max(1, 3600 // sampling_interval)
        elif freq == "daily":
            period_points = max(1, 24*3600 // sampling_interval)
        elif freq == "weekly":
            period_points = max(1, 7*24*3600 // sampling_interval)
        duration = end - start + 1
        tmp_ratio = duration / period_points
        new_period = int(duration / (tmp_ratio+1))
        break_magnitude = magnitude
        base_level = np.mean(series[max(0, start-10):min(len(series), end+10)])
        for i in range(duration):
            phase = (i / new_period) * 2 * math.pi
            series[start + i] = base_level + (series[start + i] - base_level) * 0.2 + math.sin(phase) * break_magnitude

        return abs(break_magnitude)

    def generate_series_with_frequency_anomalies(
        self,
        ts_info: TimeSeriesInfo,
        frequency: float = 0.03,
        normal_duration_rate: float = 450.0,
        anomaly_duration_rate: float = 15.0,
        minimum_anomaly_duration: int = 7,
        minimum_normal_duration: int = 20,
        frequency_multiplier: float = 3.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[AnomalyPattern]]:
        """                 periodic_pattern_break_by_frequency    

            series        TSInfo/AnomalyPattern               inject_anomalies        
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_samples = ts_info.seq_len
        t = np.arange(n_samples)
        #                characteristics    
        base_freq = frequency
        freq_function = np.full(n_samples, base_freq, dtype=float)

        anomaly_intervals: List[Tuple[int, int]] = []
        # current_time = 0
        # while current_time < n_samples:
        current_time = 10
        while current_time < n_samples-10:
            normal_duration = max(minimum_normal_duration, int(np.random.exponential(normal_duration_rate)))
            current_time += normal_duration
            if current_time >= n_samples-10:
                break
            anomaly_duration = max(minimum_anomaly_duration, int(np.random.exponential(anomaly_duration_rate)))
            anomaly_end = min(n_samples-10, current_time + anomaly_duration)

            # random frequency: (1, frequency_multiplier]
            tmp_frequency_multiplier = np.random.uniform(1.5, frequency_multiplier)
            if np.random.random() < 0.5:
                freq_function[current_time:anomaly_end] *= tmp_frequency_multiplier
            else:
                freq_function[current_time:anomaly_end] /= max(1e-6, tmp_frequency_multiplier)

            anomaly_intervals.append((current_time, anomaly_end, tmp_frequency_multiplier))
            current_time = anomaly_end

        dx = np.full_like(t, 1.0, dtype=float)
        x_plot = (freq_function * dx).cumsum()
        base_wave = np.sin(2 * np.pi * x_plot)

        #     TSInfo       
        series = ts_info.baseline_value + base_wave * ts_info.magnitude

        #    AnomalyPattern    frequency   
        anomalies: List[AnomalyPattern] = []
        for (s, e, freq_multi) in anomaly_intervals:
            duration = e - s
            if duration <= 0:
                continue
            mag = ts_info.magnitude * (freq_multi-1)
            desc = self._generate_anomaly_description("periodic_pattern_break_by_frequency", s, e-1, duration)
            anomalies.append(
                AnomalyPattern(
                    name=f"periodic_pattern_break_by_frequency_{freq_multi}",
                    start_pos=s,
                    end_pos=e-1,
                    magnitude=mag,
                    pattern_type="periodic_pattern_break_by_frequency",
                    description=desc
                )
            )
        ts_info.characteristics["trend"] = "keep steady"
        if random.random() < 0.5:
            std = random.uniform(0.0, 0.005) * ts_info.magnitude
        else:
            std = 0.0
        noise = np.random.normal(0, std, ts_info.seq_len)
        series += noise
        ts_info.characteristics["noise"] = "almost no noise"
        series = np.clip(series, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(series)
        ts_info.statistics = statistics
        return series, anomalies

    def _inject_density_increase(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """        """
        center_value = np.mean(series[max(0, start-10):min(len(series), end+10)])
        duration = end - start + 1
        
        for i in range(duration):
            series[start + i] = center_value + (series[start + i] - center_value) * 0.7
        
        return magnitude

    def _inject_zero_values(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """    """
        original_values = series[start:end+1].copy()
        
        # if ts_info.metric_type in ["SuccessRate", "ResourceUtilizationRate"]:
        #     zero_value = random.uniform(0.0, 0.01)
        #     series[start:end + 1] = zero_value
        # else:
        #     zero_value = random.uniform(0.0, ts_info.baseline_value * 0.01)
        #     series[start:end + 1] = zero_value if ts_info.value_type != "int" else 0
        zero_value = 0
        series[start:end + 1] = zero_value
        
        abs_magnitude = np.mean(np.abs(original_values - zero_value))
        return abs_magnitude

    def _inject_saturation_values(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """     """
        original_values = series[start:end+1].copy()
        
        # saturation_value = ts_info.normal_range[1] * random.uniform(min(0.95, ts_info.baseline_value), 1.0)
        saturation_value = ts_info.normal_range[1]
        series[start:end + 1] = saturation_value
        
        abs_magnitude = np.mean(np.abs(series[start:end+1] - original_values))
        return abs_magnitude

    def _inject_pattern_mismatch(self, series: np.ndarray, start: int, end: int, magnitude: float, ts_info: TimeSeriesInfo):
        """         """
        duration = end - start + 1
        
        mismatch_magnitude = magnitude * 0.5
        for i in range(duration):
            progress = i / max(1, duration - 1)
            pattern_value = mismatch_magnitude * math.sin(2 * math.pi * progress * 3 + math.pi)
            series[start + i] += pattern_value if ts_info.value_type != "int" else int(pattern_value) + 2*random.randint(-int(mismatch_magnitude), int(mismatch_magnitude))
        
        return mismatch_magnitude

    def get_available_patterns(self) -> List[str]:
        """             """
        non_zero_anomalies = [key for key in self.anomaly_patterns.keys() if self.anomaly_patterns[key]["probability"] > 0.0]
        available_patterns = non_zero_anomalies + self.all_anomalies_include
        available_patterns.sort()
        return available_patterns

    def inject_specific_pattern(self, series: np.ndarray, ts_info: TimeSeriesInfo, 
                              pattern_type: str, count: int = 1) -> Tuple[np.ndarray, List[AnomalyPattern]]:
        """           """
        if pattern_type not in self.anomaly_patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        series_copy = series.copy()
        anomaly_patterns = []
        occupied_positions = set()
        
        for _ in range(count):
            anomaly = self._generate_anomaly_pattern(pattern_type, ts_info, occupied_positions)
            if anomaly:
                self._inject_single_anomaly(series_copy, anomaly, ts_info)
                anomaly_patterns.append(anomaly)
                
                for pos in range(anomaly.start_pos, anomaly.end_pos + 1):
                    occupied_positions.add(pos)
        
        series_copy = np.clip(series_copy, ts_info.normal_range[0], ts_info.normal_range[1])
        statistics = generate_statistics(series_copy)
        ts_info.statistics = statistics
        return series_copy, anomaly_patterns



def all_anomaly_patterns() -> List[str]:
    """             """
    injector = AnomalyInjector()
    return injector.get_available_patterns()
