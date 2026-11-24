import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict


SAMPLING_INTERVAL_CHOICES = [60, 300, 345, 600, 864, 1200, 3600]
# 60, 300, 345, 600, 864, 900, 1200, 3600

SAMPLING_INTERVAL_STR_DICT = {
    60: "1 minute",
    300: "5 minutes",
    345: "5min 45s",
    600: "10 minutes",
    864: "14min 24s",
    900: "15 minutes",
    1200: "20 minutes",
    3600: "1 hour",
}


@dataclass
class TimeSeriesInfo:
    """Metadata for one time-series case."""
    metric_type: str
    metric_name: str
    seq_len: int
    sampling_interval: int
    value_type: str
    magnitude: float
    normal_range: Tuple[float, float]
    baseline_value: float
    characteristics: Dict[str, str]
    statistics: Dict[str, Any] = None
    additional_info: Dict[str, Any] = None
    
    def __init__(self, metric_type: str, metric_name: str, seq_len: int, 
                 sampling_interval: int, value_type: str, magnitude: float,
                 normal_range: Tuple[float, float], baseline_value: float,
                 characteristics: Dict[str, str] = None, statistics: Dict[str, Any] = None,
                 additional_info: Dict[str, Any] = None):
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.seq_len = seq_len
        self.sampling_interval = sampling_interval
        self.value_type = value_type
        self.magnitude = magnitude
        self.normal_range = normal_range
        self.baseline_value = baseline_value
        self.characteristics = characteristics or {}
        self.statistics = statistics or {}
        self.additional_info = additional_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesInfo':
        """Create a TimeSeriesInfo object from a dictionary."""
        return cls(
            metric_type=data['metric_type'],
            metric_name=data['metric_name'],
            seq_len=data['seq_len'],
            sampling_interval=data['sampling_interval'],
            value_type=data['value_type'],
            magnitude=data['magnitude'],
            normal_range=tuple(data['normal_range']),
            baseline_value=data['baseline_value'],
            characteristics=data.get('characteristics', {}),
            statistics=data.get('statistics', {}),
            additional_info=data.get('additional_info', {})
        )


@dataclass
class AnomalyPattern:
    """Metadata for one injected or detected anomaly segment."""
    name: str
    start_pos: int
    end_pos: int
    magnitude: float
    pattern_type: str
    description: str
    additional_info: Dict[str, Any] = None  #             None

    def __init__(self, name: str, start_pos: int, end_pos: int, magnitude: float, pattern_type: str, description: str, additional_info: Dict[str, Any] = None):
        self.name = name
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.magnitude = magnitude
        self.pattern_type = pattern_type
        self.description = description
        self.additional_info = additional_info if additional_info is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnomalyPattern':
        """     AnomalyPattern  """
        return cls(
            name=data['name'],
            start_pos=data['start_pos'],
            end_pos=data['end_pos'],
            magnitude=data['magnitude'],
            pattern_type=data['pattern_type'],
            description=data['description'],
            additional_info=data.get('additional_info', {})
        )

    def __hash__(self):
        return hash((self.name, self.start_pos, self.end_pos, self.magnitude, self.pattern_type))
    
    def __eq__(self, other):
        if not isinstance(other, AnomalyPattern):
            return False
        return (self.name == other.name and
                self.start_pos == other.start_pos and
                self.end_pos == other.end_pos and
                self.magnitude == other.magnitude and
                self.pattern_type == other.pattern_type)


def generate_statistics(series: np.ndarray) -> Dict[str, Any]:
    """      """
    seq_len = len(series)
    segment_len = 8
    segment_means = []
    for i in range(0, seq_len, segment_len):
        end_idx = min(i + segment_len, seq_len)
        segment_means.append(round(float(np.mean(series[i:end_idx])), 3))
    
    return {
        "mean": round(float(np.mean(series)), 3),
        "std": round(float(np.std(series)), 3),
        "max": round(float(np.max(series)), 3),
        "min": round(float(np.min(series)), 3),
        "max_pos": int(np.argmax(series)),
        "min_pos": int(np.argmin(series)),
        "start_value": round(float(series[0]), 3),
        "end_value": round(float(series[-1]), 3),
        # "segment_means": segment_means,
    }


def save_ts_json(dataset: Dict, output_path: str):
    """          JSON  """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, sample in enumerate(dataset):
            if i > 0:
                f.write(',\n')
            
            timeseries_data = sample.get('timeseries_data', [])
            sample_copy = sample.copy()
            sample_copy['timeseries_data'] = '__PLACEHOLDER__'
            sample_json = json.dumps(sample_copy, ensure_ascii=False, indent=2)
            timeseries_json = json.dumps(timeseries_data, ensure_ascii=False, separators=(',', ':'))
            sample_json = sample_json.replace('"__PLACEHOLDER__"', timeseries_json)
            f.write('  ' + sample_json.replace('\n', '\n  '))
        f.write('\n]')


def validate_and_fix_value_ranges(series: np.ndarray, ts_info: TimeSeriesInfo) -> np.ndarray:
        """                  """
        metric_type = ts_info.metric_type
        corrected_series = series.copy()
        
        if metric_type == "SuccessRate":
            corrected_series = np.clip(corrected_series, 0.0, 1.0)
            
        elif metric_type == "ResourceUtilizationRate":
            corrected_series = np.clip(corrected_series, 0.0, 1.0)
            
        elif metric_type == "Count":
            corrected_series = np.maximum(corrected_series, 0.0)
            reasonable_upper_limit = ts_info.baseline_value * 50  #   50 baseline
            corrected_series = np.minimum(corrected_series, reasonable_upper_limit)
            
        elif metric_type == "ErrorCount":
            corrected_series = np.maximum(corrected_series, 0.0)
            reasonable_upper_limit = max(ts_info.baseline_value * 20, 1000)  #   20  1000
            corrected_series = np.minimum(corrected_series, reasonable_upper_limit)
            
        elif metric_type == "LatencySecond":
            corrected_series = np.maximum(corrected_series, 0.0)
            reasonable_upper_limit = max(ts_info.baseline_value * 100, 300)  #   100  300 
            corrected_series = np.minimum(corrected_series, reasonable_upper_limit)
            
        elif metric_type == "QPS":
            corrected_series = np.maximum(corrected_series, 0.0)
            # QPS     
            reasonable_upper_limit = ts_info.baseline_value * 20  #   20 baseline
            corrected_series = np.minimum(corrected_series, reasonable_upper_limit)
        
        corrected_series = np.nan_to_num(corrected_series, nan=ts_info.baseline_value, 
                                       posinf=ts_info.normal_range[1], 
                                       neginf=ts_info.normal_range[0])
        
        return corrected_series
