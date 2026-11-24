from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .base_evaluator import BaseEvaluator
from .util import ts_precision_and_recall, improved_cardinality_fn


class PositionEvaluator(BaseEvaluator):
    """
    Evaluator for anomaly position detection using point-wise, delay, and event-wise F1 metrics.
    """
    def __init__(self, delay_threshold: int = 15, margin: int = 0):
        self.delay_threshold = delay_threshold
        self.margin = margin
        self.metric_types = ['range', 'standard', 'delay', 'point', 'event', 'temporal']
    
    def add_margin(self, segs: List[Tuple[int,int]], margin: int, seq_len: int) -> List[Tuple[int,int]]:
        return [(max(s - margin, 0), min(e + margin, seq_len-1)) for s, e in segs]

    # Helper to evaluate with a given threshold
    def eval_with_threshold(self,
                            th: float,
                            true_segments: List[Tuple[int, int]],
                            pred_segments: List[Tuple[int, int]],
                            pred_magnitudes: List[Tuple[int, int]] = None,
                            sequence_length: int = 0,
                            metric_types: List[str] = ['range', 'standard', 'delay', 'point', 'event', 'temporal']
    ) -> Dict[str, Dict[str, float]]:
        filtered = self._filter_segments_by_threshold(pred_segments, pred_magnitudes, th)
        ret = {}
        for metric_type in metric_types:
            ret[metric_type] = self.calculate_f1(true_segments, filtered, sequence_length, metric_type)
        return ret

    def calculate_f1(self,
                     true_segments: List[Tuple[int, int]],
                     pred_segments: List[Tuple[int, int]],
                     sequence_length: int,
                     metric_type: str
    ) -> Dict[str, float]:
        calculate_method = None
        if metric_type == 'standard': calculate_method = self.calculate_standard_f1
        elif metric_type == 'temporal': calculate_method = self.calculate_temporal_f1
        elif metric_type == 'point': calculate_method = self.calculate_point_f1
        elif metric_type == 'delay': calculate_method = self.calculate_delay_f1
        elif metric_type == 'event': calculate_method = self.calculate_event_f1
        elif metric_type == 'range': calculate_method = self.calculate_range_f1
        else: raise Exception("[ERROR] Unknown metric type: {metric_type}")
        return calculate_method(true_segments, pred_segments, sequence_length)

    def evaluate(self,
                 true_segments: List[Tuple[int, int]],
                 pred_segments: List[Tuple[int, int]],
                 pred_magnitudes: Optional[List[float]] = None,
                 sequence_length: int = 0,
                 metric_types: List[str] = ['range', 'standard', 'delay', 'point', 'event', 'temporal'],
                 best: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate position predictions directly on original predicted segments.

        This method no longer performs magnitude threshold sweeping. It computes
        metrics using the input ``pred_segments`` as-is, regardless of
        ``pred_magnitudes``.
        """
        # Backward compatibility: allow calling as evaluate(true, pred, seq_len)
        # If the third positional arg is actually an int (sequence_length), move it.
        if isinstance(pred_magnitudes, (int, np.integer)) and sequence_length == 0:
            sequence_length = int(pred_magnitudes)  # type: ignore[arg-type]
            pred_magnitudes = None

        if best:
            if pred_magnitudes is not None and pred_segments and (len(pred_segments) != len(pred_magnitudes)):
                raise ValueError(
                    f"Length mismatch: pred_segments({len(pred_segments)}) vs pred_magnitudes({len(pred_magnitudes)})"
                )
            thresholds: List[float]
            if pred_magnitudes:
                thresholds = sorted(set(float(m) for m in pred_magnitudes))
                thresholds.append(float('inf'))  # allow excluding all segments
            else:
                # No magnitudes provided; evaluate using current segments only
                thresholds = [float('-inf')]  # keep all segments as-is
            best_results = { metric_type:  {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': None} for metric_type in metric_types}
            for th in thresholds:
                res = self.eval_with_threshold(th, true_segments, pred_segments, pred_magnitudes, sequence_length, metric_types)
                for k in metric_types:
                    if res[k]['f1'] > best_results[k]['f1']:
                        best_results[k].update(res[k])
                        best_results[k]['threshold'] = th

            for k in best_results:
                if best_results[k]['threshold'] is None:
                    best_results[k]['threshold'] = float('-inf') if pred_magnitudes else None
            results = best_results
        else:
            results = self.eval_with_threshold(
                float('-inf'),
                true_segments,
                pred_segments,
                None,
                sequence_length,
                metric_types,
            )
            for metric_type in metric_types:
                results[metric_type]['threshold'] = None
        return results

    def _filter_segments_by_threshold(self,
                                      pred_segments: List[Tuple[int, int]],
                                      pred_magnitudes: Optional[List[float]],
                                      threshold: float
    ) -> List[Tuple[int, int]]:
        """
        Keep only segments whose magnitude >= threshold. If threshold is -inf,
        return all segments; if +inf, return empty.
        """
        if not pred_segments:
            return []
        if not pred_magnitudes:
            return pred_segments
        if threshold == float('-inf'):
            return pred_segments
        if threshold == float('inf'):
            return []
        filtered: List[Tuple[int, int]] = []
        for (seg, mag) in zip(pred_segments, pred_magnitudes):
            try:
                a = float(mag)
            except (TypeError, ValueError):
                # If magnitude is invalid, treat as -inf to keep conservative behavior
                a = float('-inf')
            if a >= threshold:
                filtered.append(seg)
        return filtered

    def anomaly_segments_to_binary_array(self, segments: List[Tuple[int, int]], length: int) -> np.ndarray:
        binary = np.zeros(length, dtype=int)
        for start, end in segments:
            s = max(0, min(start, length - 1))
            e = max(0, min(end, length - 1))
            if s <= e:
                binary[s:e+1] = 1
        return binary
    
    def calculate_standard_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int
    ) -> Dict[str, float]:
        y_true = self.anomaly_segments_to_binary_array(true_segments, sequence_length)
        y_pred = self.anomaly_segments_to_binary_array(pred_segments, sequence_length)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=1)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}
    
    def calculate_temporal_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int,
                           recall_cardinality_fn: Callable = improved_cardinality_fn,
                           weighted_precision: bool = True
    ) -> Dict[str, float]:
        y_true = self.anomaly_segments_to_binary_array(true_segments, sequence_length)
        y_pred = self.anomaly_segments_to_binary_array(pred_segments, sequence_length)

        label_ranges = [(start, end+1) for start, end in true_segments]
        prediction_ranges = [(start, end+1) for start, end in pred_segments]
        precision, recall = ts_precision_and_recall(y_true, y_pred, alpha=0, recall_cardinality_fn=recall_cardinality_fn, anomaly_ranges=label_ranges, prediction_ranges=prediction_ranges, weighted_precision=weighted_precision)
        if precision == recall == 0:
            # We simply set rec = 1 to avoid dividing by zero. The F-score will still be 0
            recall = 1

        f_score = 2 * precision * recall / (max(precision + recall, 1e-15))
        return {'precision': float(precision), 'recall': float(recall), 'f1': float(f_score)}

    def calculate_point_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int) -> Dict[str, float]:
        y_true = self.anomaly_segments_to_binary_array(true_segments, sequence_length)
        y_pred = self.anomaly_segments_to_binary_array(pred_segments, sequence_length)
        y_pred_adj = self._adjust_point_predictions(true_segments, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_adj, average='binary', zero_division=1)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    def _adjust_point_predictions(self, true_segments: List[Tuple[int, int]], y_pred: np.ndarray) -> np.ndarray:
        y_pred_adj = y_pred.copy()
        for s, e in true_segments:
            s2 = max(0, min(s, len(y_pred) - 1))
            e2 = max(0, min(e, len(y_pred) - 1))
            if np.any(y_pred[s2:e2+1] == 1):
                y_pred_adj[s2:e2+1] = 1
        return y_pred_adj

    def calculate_delay_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int
    ) -> Dict[str, float]:
        y_true = self.anomaly_segments_to_binary_array(true_segments, sequence_length)
        y_pred = self.anomaly_segments_to_binary_array(pred_segments, sequence_length)
        y_pred_adj = self._adjust_delay_predictions(true_segments, y_pred, self.delay_threshold)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_adj, average='binary', zero_division=1)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    def _adjust_delay_predictions(self, true_segments: List[Tuple[int, int]], y_pred: np.ndarray, delay: int) -> np.ndarray:
        y_pred_adj = np.zeros_like(y_pred)
        for s, e in true_segments:
            s2 = max(0, min(s, len(y_pred) - 1))
            e2 = max(0, min(e, len(y_pred) - 1))
            window_end = min(s2 + delay, len(y_pred) - 1)
            if np.any(y_pred[s2:window_end+1] == 1):
                y_pred_adj[s2:e2+1] = 1
        y_pred_adj = np.logical_or(y_pred_adj, y_pred).astype(int)
        return y_pred_adj

    def calculate_event_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int
    ) -> Dict[str, float]:
        if not true_segments and not pred_segments:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        total_len = sum(e - s + 1 for s, e in true_segments)
        comp_len = sequence_length - total_len + len(true_segments)
        if comp_len <= 0:
            comp_len = len(true_segments)
        y_true_c = np.zeros(comp_len, dtype=int)
        y_pred_c = np.zeros(comp_len, dtype=int)
        original_to_c = {}
        cp = 0
        sorted_true = sorted(true_segments, key=lambda x: x[0])
        curr = 0
        for s, e in sorted_true:
            for pos in range(curr, s):
                original_to_c[pos] = cp; cp += 1
            for pos in range(s, e+1):
                original_to_c[pos] = cp
            y_true_c[cp] = 1; cp += 1
            curr = e + 1
        for pos in range(curr, sequence_length):
            if cp < comp_len:
                original_to_c[pos] = cp; cp += 1
        for s, e in pred_segments:
            for pos in range(s, min(e+1, sequence_length)):
                if pos in original_to_c:
                    idx = original_to_c[pos]
                    if idx < comp_len:
                        y_pred_c[idx] = 1
        p, r, f, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average='binary', zero_division=1)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    def calculate_range_f1(self,
                           true_segments: List[Tuple[int, int]],
                           pred_segments: List[Tuple[int, int]],
                           sequence_length: int,
                           alpha_recall: float = 1.0
    ) -> Dict[str, float]:
        if not true_segments and not pred_segments:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

        # Range-based precision/recall/F1 following PRTS logic.
        # Inputs true_segments/pred_segments are already ranges, so no need to prepare from binary arrays.

        def _range_overlap(r1: Tuple[int, int], r2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            s = max(r1[0], r2[0])
            e = min(r1[1], r2[1])
            return (s, e) if s <= e else None

        def _omega_reward(rrange: Tuple[int, int], overlap: Optional[Tuple[int, int]]) -> float:
            # Flat positional bias, same as BaseTimeSeriesMetrics._omega_function with bias="flat"
            if overlap is None:
                return 0.0
            anomaly_length = rrange[1] - rrange[0] + 1
            if anomaly_length <= 0:
                return 0.0
            # With flat bias, reward is proportion of overlap length to anomaly length
            ov_len = overlap[1] - overlap[0] + 1
            if ov_len <= 0:
                return 0.0
            return ov_len / float(anomaly_length)

        def _precision(true_ranges: List[Tuple[int, int]], pred_ranges: List[Tuple[int, int]]) -> float:
            if not pred_ranges:
                return 0.0
            total = 0.0
            for pr in pred_ranges:
                omega_sum = 0.0
                overlap_count = 0
                for tr in true_ranges:
                    ov = _range_overlap(pr, tr)
                    if ov is not None:
                        overlap_count += 1
                        omega_sum += _omega_reward(pr, ov)
                # cardinality = 'one' => gamma = 1.0 regardless of overlap_count
                overlap_reward = 1.0 * omega_sum
                existence_reward = 1.0 if overlap_count > 0 else 0.0
                # alpha = 0.0 => use pure overlap_reward (align with PRTS default)
                total += overlap_reward
            return total / float(len(pred_ranges))

        def _recall(true_ranges: List[Tuple[int, int]], pred_ranges: List[Tuple[int, int]], alpha: float) -> float:
            if not true_ranges:
                return 0.0
            # clamp alpha into [0,1]
            alpha = max(0.0, min(1.0, alpha))
            total = 0.0
            for tr in true_ranges:
                omega_sum = 0.0
                overlap_count = 0
                for pr in pred_ranges:
                    ov = _range_overlap(tr, pr)
                    if ov is not None:
                        overlap_count += 1
                        omega_sum += _omega_reward(tr, ov)
                overlap_reward = 1.0 * omega_sum
                existence_reward = 1.0 if overlap_count > 0 else 0.0
                # weighted by alpha: alpha * existence + (1-alpha) * overlap
                total += alpha * existence_reward + (1.0 - alpha) * overlap_reward
            return total / float(len(true_ranges))

        p = _precision(true_segments, pred_segments)
        r = _recall(true_segments, pred_segments, alpha_recall)
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}