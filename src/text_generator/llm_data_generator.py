#!/usr/bin/env python3
"""Generate LLM QA JSONL data from IntentAD-style TSAD cases."""

import json
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path
import random
import re
from tqdm import tqdm
from .text_pool import TextPool
from ts_generator.util import TimeSeriesInfo, SAMPLING_INTERVAL_STR_DICT


class LLMSFTDataGenerator:
    """Generator for LLM supervised-finetuning and evaluation QA records."""
    
    def __init__(self):
        self.text_pool = TextPool()
        
    def load_tsad_dataset(self, filepath: str) -> List[Dict]:
        """Load a TSAD JSON dataset."""
        with open(filepath, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} cases from dataset")
        return samples

    # -------------------- Human Intent helpers (new schema + backward compat) --------------------
    def _get_human_intent(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return human_intent dict."""
        hi = sample.get("human_intent")
        return hi if isinstance(hi, dict) else {}

    def generate_human_intent_nl_fallback(self, human_intent: Dict[str, Any]) -> str:
        """Fallback natural-language intent when no LLM is available."""
        return self.text_pool.generate_human_intent_nl_fallback(human_intent or {})

    def _list_human_intent_rules(self, human_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        rules_any = (human_intent or {}).get("rules")
        if not isinstance(rules_any, list):
            return []
        return [r for r in rules_any if isinstance(r, dict)]

    def _contains_numeric_like(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            return re.fullmatch(r"[-+]?\d+(\.\d+)?", value.strip()) is not None
        if isinstance(value, (list, tuple, set)):
            return any(self._contains_numeric_like(v) for v in value)
        if isinstance(value, dict):
            return any(self._contains_numeric_like(v) for v in value.values())
        return False

    def _contains_relative_text(self, value: Any) -> bool:
        if isinstance(value, str):
            lowered = value.lower()
            keywords = [
                "small", "minor", "large", "major", "brief", "short", "long",
                "closely", "close", "far", "near", "rough", "roughly", "about",
                "around", "approx", "approximately", "at least", "at most", "more than",
                "less than", "higher", "lower", "slight", "significant",
            ]
            return any(k in lowered for k in keywords)
        if isinstance(value, (list, tuple, set)):
            return any(self._contains_relative_text(v) for v in value)
        if isinstance(value, dict):
            return any(self._contains_relative_text(v) for v in value.values())
        return False

    def _classify_constraint_count(self, human_intent: Dict[str, Any]) -> str:
        rules = self._list_human_intent_rules(human_intent)
        count = len(rules)
        if count == 0:
            return "none"
        elif count == 1:
            return "single"
        else:
            return "multiple"

    def _map_rule_to_intent_category(self, rule: Dict[str, Any]) -> str:
        rule_name = str(rule.get("name") or "").strip().lower()
        category = str(rule.get("category") or "").strip().lower()

        if rule_name == "merge_close_anomalies":
            return "others"

        mapping = {
            "focus": "type",
            "threshold": "magnitude",
            "ignore_duration": "duration",
            "ignore_frequency": "frequency",
            "truncate_duration": "duration",
            "others": "others",
            "merge": "others",
            "scope": "type",
        }
        return mapping.get(category, "others")

    def _classify_intent_category(self, human_intent: Dict[str, Any]) -> List[str]:
        rules = self._list_human_intent_rules(human_intent)
        categories: List[str] = []
        for rule in rules:
            mapped = self._map_rule_to_intent_category(rule)
            if mapped not in categories:
                categories.append(mapped)
        return categories

    def _classify_relative_expression(self, human_intent: Dict[str, Any], human_intent_nl: Optional[str] = None) -> str:
        rules = self._list_human_intent_rules(human_intent)
        has_relative = False
        if rules and human_intent_nl and self._contains_relative_text(human_intent_nl):
            has_relative = True

        return "with" if has_relative else "without"

    def _classify_history_dependency(self, human_intent: Dict[str, Any]) -> str:
        """Classify whether intent depends on history anomalies.

        Returns:
            "yes" if any rule name is one of:
            - ignore_frequent_patterns
            - ignore_rare_patterns
            - ignore_recent_reoccurrence
            otherwise "no".
        """
        target_rule_names = {
            "ignore_frequent_patterns",
            "ignore_rare_patterns",
            "ignore_recent_reoccurrence",
        }
        for rule in self._list_human_intent_rules(human_intent):
            rule_name = str(rule.get("name") or "").strip().lower()
            if rule_name in target_rule_names:
                return "yes"
        return "no"

    def _build_additional_info(self, sample: Dict[str, Any], human_intent_nl: Optional[str] = None) -> Dict[str, Any]:
        human_intent = self._get_human_intent(sample)
        intent_text = human_intent_nl if human_intent_nl is not None else self.generate_human_intent_nl_fallback(human_intent)
        return {
            "human_intent_structured": human_intent,
            "human_intent_text": intent_text,
            "intent_taxonomy": {
                "constraint_count": self._classify_constraint_count(human_intent),
                "intent_category": self._classify_intent_category(human_intent),
                "relative_expression": self._classify_relative_expression(human_intent, human_intent_nl=intent_text),
                "history_dependency": self._classify_history_dependency(human_intent),
            },
        }

    def _sanitize_human_intent_for_llm(self, human_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields we don't want to expose to LLM prompts.

        Currently drops per-rule keys: 'description' and 'filtered_anomalies'.
        """
        hi: Dict[str, Any] = dict(human_intent or {})
        rules_any = hi.get("rules")
        if isinstance(rules_any, list):
            sanitized_rules: List[Dict[str, Any]] = []
            for r in rules_any:
                if isinstance(r, dict):
                    rr = dict(r)
                    rr.pop("description", None)
                    rr.pop("filtered_anomalies", None)
                    sanitized_rules.append(rr)
            hi["rules"] = sanitized_rules
        return hi

    def _sanitize_human_intent_for_intent_nl_prompt(self, human_intent: Dict[str, Any], sampling_interval: Optional[int] = None) -> Dict[str, Any]:
        """Sanitize human_intent for NL intent rewrite prompts.

        Compared to _sanitize_human_intent_for_llm, also removes rule parameters so the
        prompt does not look like a 1-to-1 mapping from rules to a user sentence.
        """
        hi = self._sanitize_human_intent_for_llm(human_intent)
        rules_any = hi.get("rules")
        if not isinstance(rules_any, list):
            return hi

        def looks_like_param_key(key: str) -> bool:
            return False
            k = (key or "").lower()
            if k in {"name", "category", "type", "action", "enabled"}:
                return False
            # common parameter patterns
            if k.startswith(("min_", "max_", "avg_", "mean_", "std_", "top_", "bottom_")):
                return True
            if any(s in k for s in ("threshold", "ratio", "percent", "pct", "duration", "window", "freq", "period", "sigma")):
                return True
            return False

        def is_numeric_like(v: Any) -> bool:
            return False
            if isinstance(v, (int, float)):
                return True
            if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
                return True
            if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
                return True
            return False

        sanitized_rules: List[Dict[str, Any]] = []
        for r in rules_any:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            for k in list(rr.keys()):
                if looks_like_param_key(k) or is_numeric_like(rr.get(k)):
                    rr.pop(k, None)
            parameters = rr["parameters"]
            new_parameters = {}
            for parameter in parameters:
                if parameter in ["window_size", "max_gap", "min_duration", "max_duration"] and sampling_interval is not None:
                    new_parameters[f"{parameter}_format"] = self.text_pool._format_duration_compact(parameters[parameter] * sampling_interval)
                new_parameters[parameter] = parameters[parameter]
            rr["parameters"] = new_parameters
            sanitized_rules.append(rr)
        hi["rules"] = sanitized_rules
        return hi

    def _abstract_rule_descriptions(self, human_intent: Dict[str, Any]) -> List[str]:
        """High-level descriptions derived from rules.

        Note: keep parameter expressions (e.g., thresholds/durations) to provide the model
        enough specificity when desired.
        """
        rules_any = (human_intent or {}).get("rules")
        if not isinstance(rules_any, list):
            return []
        raw_descs: List[str] = []
        for r in rules_any:
            if not isinstance(r, dict):
                continue
            try:
                d = self.text_pool.generate_intent_description(r, include_window=False)
            except Exception:
                d = str(r.get("name") or "")
            d = re.sub(r"\s{2,}", " ", str(d)).strip(" \t\n\r.,:;")
            if d:
                raw_descs.append(d)
        # dedupe preserve order
        seen = set()
        descs: List[str] = []
        for d in raw_descs:
            if d not in seen:
                seen.add(d)
                descs.append(d)
        return descs

    def _anomaly_key(self, anomaly: Dict[str, Any]) -> tuple:
        try:
            return (
                int(anomaly.get("start_pos")),
                int(anomaly.get("end_pos")),
                str(anomaly.get("pattern_type") or anomaly.get("pattern") or "").lower(),
            )
        except Exception:
            return (
                anomaly.get("start_pos"),
                anomaly.get("end_pos"),
                str(anomaly.get("pattern_type") or anomaly.get("pattern") or "").lower(),
            )

    def _memory_window_candidates(self, human_intent: Dict[str, Any]) -> List[int]:
        windows: List[int] = []
        for rule in self._list_human_intent_rules(human_intent):
            category = str(rule.get("category") or "").strip().lower()
            name = str(rule.get("name") or "").strip().lower()
            if category == "ignore_frequency" or name in {"ignore_recent_reoccurrence"}:
                params = rule.get("parameters", {})
                if isinstance(params, dict):
                    try:
                        w = int(params.get("window_size", 0))
                    except Exception:
                        w = 0
                    if w > 0:
                        windows.append(w)
        return windows

    def _fmt_num(self, value: Any) -> str:
        try:
            return f"{float(value):.3g}"
        except Exception:
            return str(value)

    def _visible_mag_value(self, value: Any) -> Optional[float]:
        try:
            # Keep numeric precision aligned with what anomaly text exposes.
            return float(f"{float(value):.3g}")
        except Exception:
            return None

    def _visible_duration_value(self, anomaly_or_event: Any) -> Optional[int]:
        if not isinstance(anomaly_or_event, dict):
            return None
        try:
            duration_raw = anomaly_or_event.get("duration")
            if duration_raw is not None:
                duration = int(duration_raw)
                if duration > 0:
                    return duration
        except Exception:
            pass
        try:
            s = int(anomaly_or_event.get("start_pos"))
            e = int(anomaly_or_event.get("end_pos"))
            if e < s:
                s, e = e, s
            return e - s + 1
        except Exception:
            return None

    def _build_magnitude_comparison_trace(
        self,
        anomaly: Dict[str, Any],
        threshold_rule: Optional[Dict[str, Any]] = None,
        hit_rule: Optional[Dict[str, Any]] = None,
        intent_action_context: Optional[Dict[str, Any]] = None,
        include_history_summary: bool = False,
    ) -> Optional[str]:
        if not isinstance(anomaly, dict):
            return None
        # hit_rule     focus  ignore_duration,   return None
        if hit_rule and isinstance(hit_rule, dict):
            hit_category = hit_rule["category"]
            if hit_category in {"focus", "ignore_duration"}:
                return None
        cur_mag = self._visible_mag_value(anomaly.get("magnitude"))
        if cur_mag is None:
            return None

        if isinstance(threshold_rule, dict):
            rule_name = str(threshold_rule.get("name") or "").strip().lower()
            rule_category = str(threshold_rule.get("category") or "").strip().lower()
            if rule_category == "threshold" and rule_name == "absolute_magnitude_threshold":
                params = threshold_rule.get("parameters", {}) if isinstance(threshold_rule.get("parameters"), dict) else {}
                threshold_str = str(params.get("threshold_value_str", "")).strip()
                threshold_num = None
                if threshold_str:
                    try:
                        threshold_num = float(threshold_str)
                    except Exception:
                        threshold_num = None
                if threshold_num is None:
                    threshold_num = self._visible_mag_value(params.get("threshold_value"))
                if threshold_num is None:
                    return None
                #      True   “       should filter ” 
                should_filter = cur_mag < threshold_num
                threshold_show = threshold_str if threshold_str else self._fmt_num(threshold_num)
                return (
                    "Magnitude comparison: "
                    f"current={self._fmt_num(cur_mag)}, threshold={threshold_show}, "
                    f"check {self._fmt_num(cur_mag)} < {threshold_show} -> {should_filter}."
                )

        if not include_history_summary:
            return None

        if not isinstance(intent_action_context, dict):
            return None
        if str(intent_action_context.get("rule_name") or "").strip().lower() != "ignore_recent_reoccurrence":
            return None

        analysis = intent_action_context.get("analysis")
        if not isinstance(analysis, dict):
            return None
        mode = str(analysis.get("mode") or "both").strip().lower()
        if mode not in {"magnitude", "both"}:
            return None
        decisive_ref = analysis.get("decisive_reference") if isinstance(analysis.get("decisive_reference"), dict) else None
        if decisive_ref is not None:
            ref_mag = self._visible_mag_value(decisive_ref.get("magnitude"))
            if ref_mag is not None:
                #      True   “       should filter ” 
                should_filter = ref_mag >= cur_mag
                return (
                    "Historical magnitude-condition check: "
                    f"current={self._fmt_num(cur_mag)}, reference={self._fmt_num(ref_mag)}, "
                    f"check {self._fmt_num(ref_mag)} >= {self._fmt_num(cur_mag)} -> {should_filter}."
                )

        should_filter = bool(analysis.get("triggered", False))
        return f"Historical magnitude-condition check: current={self._fmt_num(cur_mag)}, triggered={should_filter}."

    def _build_duration_comparison_trace(
        self,
        anomaly: Dict[str, Any],
        hit_rule: Optional[Dict[str, Any]] = None,
        duration_rule: Optional[Dict[str, Any]] = None,
        intent_action_context: Optional[Dict[str, Any]] = None,
        include_history_summary: bool = False,
    ) -> Optional[str]:
        if not isinstance(anomaly, dict):
            return None
        # focus             return None
        if hit_rule and isinstance(hit_rule, dict):
            hit_category = hit_rule["category"]
            if hit_category in {"focus"}:
                return None
        cur_dur = self._visible_duration_value(anomaly)
        if cur_dur is None:
            return None

        #   magnitude trace           ignore_duration    
        #                         duration_rule      
        active_duration_rule: Optional[Dict[str, Any]] = None
        if isinstance(hit_rule, dict):
            hit_category = str(hit_rule.get("category") or "").strip().lower()
            if hit_category == "ignore_duration":
                active_duration_rule = hit_rule
        if active_duration_rule is None and isinstance(duration_rule, dict):
            duration_category = str(duration_rule.get("category") or "").strip().lower()
            if duration_category == "ignore_duration":
                active_duration_rule = duration_rule

        if isinstance(active_duration_rule, dict):
            active_name = str(active_duration_rule.get("name") or "").strip().lower()
            params = active_duration_rule.get("parameters", {}) if isinstance(active_duration_rule.get("parameters"), dict) else {}
            if active_name == "ignore_short_duration":
                try:
                    min_duration = int(params.get("min_duration"))
                except Exception:
                    min_duration = None
                if min_duration is None:
                    return None
                #      True   “       should filter ” 
                should_filter = cur_dur < min_duration
                return (
                    "Duration comparison: "
                    f"current={cur_dur}, minimum={min_duration}, "
                    f"check {cur_dur} < {min_duration} -> {should_filter}."
                )
            if active_name == "truncate_overlong_duration":
                try:
                    max_duration = int(params.get("max_duration"))
                except Exception:
                    max_duration = None
                if max_duration is None:
                    return None
                #      True   “       should filter ” 
                should_filter = cur_dur > max_duration
                return (
                    "Duration comparison: "
                    f"current={cur_dur}, maximum={max_duration}, "
                    f"check {cur_dur} > {max_duration} -> {should_filter}."
                )

        if not include_history_summary:
            return None
        if not isinstance(intent_action_context, dict):
            return None
        if str(intent_action_context.get("rule_name") or "").strip().lower() != "ignore_recent_reoccurrence":
            return None

        analysis = intent_action_context.get("analysis")
        if not isinstance(analysis, dict):
            return None
        mode = str(analysis.get("mode") or "both").strip().lower()
        if mode not in {"duration", "both"}:
            return None

        decisive_ref = analysis.get("decisive_reference") if isinstance(analysis.get("decisive_reference"), dict) else None
        if decisive_ref is not None:
            ref_dur = self._visible_duration_value(decisive_ref)
            if ref_dur is not None:
                #      True   “       should filter ” 
                should_filter = ref_dur >= cur_dur
                return (
                    "Historical duration-condition check: "
                    f"current={cur_dur}, reference={ref_dur}, "
                    f"check {ref_dur} >= {cur_dur} -> {should_filter}."
                )

        should_filter = bool(analysis.get("triggered", False))
        return f"Historical duration-condition check: current={cur_dur}, triggered={should_filter}."

    def _merge_comparison_traces(
        self,
        magnitude_trace: Optional[str],
        duration_trace: Optional[str],
        intent_action_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        mag = (magnitude_trace or "").strip()
        dur = (duration_trace or "").strip()
        if not mag and not dur:
            return None
        if not mag:
            return dur
        if not dur:
            return mag

        if isinstance(intent_action_context, dict):
            rule_name = str(intent_action_context.get("rule_name") or "").strip().lower()
            analysis = intent_action_context.get("analysis") if isinstance(intent_action_context.get("analysis"), dict) else {}
            mode = str(analysis.get("mode") or "both").strip().lower()
            if rule_name == "ignore_recent_reoccurrence" and mode == "both":
                mag_prefix = "Historical magnitude-condition check: "
                dur_prefix = "Historical duration-condition check: "
                if mag.startswith(mag_prefix) and dur.startswith(dur_prefix):
                    mag_body = mag[len(mag_prefix):].rstrip(".")
                    dur_body = dur[len(dur_prefix):]
                    return f"Historical magnitude/duration-condition check: {mag_body}; {dur_body}"

        return f"{mag} {dur}"

    def _query_history_anomalies(
        self,
        sample: Dict[str, Any],
        pattern: str,
        start: int,
        window_size: int,
    ) -> List[tuple]:
        history_hits: List[tuple] = []
        ts_info = sample.get("timeseries_info", {})
        addi = ts_info.get("additional_info", {}) if isinstance(ts_info, dict) else {}
        hist = addi.get("history_anomalies") if isinstance(addi, dict) else None
        if not isinstance(hist, list):
            return history_hits

        p_norm = str(pattern or "").lower()
        for item in hist:
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
            if str(p).lower() != p_norm:
                continue
            if e_i <= int(start) and (int(start) - e_i) <= int(window_size):
                history_hits.append((s_i, e_i, str(p), m_f))

        history_hits.sort(key=lambda x: (x[0], x[1], x[2]))
        return history_hits

    def _is_complex_intent_rule(self, rule: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(rule, dict):
            return False
        category = str(rule.get("category") or "").strip().lower()
        name = str(rule.get("name") or "").strip().lower()
        return category == "ignore_frequency" or name == "ignore_recent_reoccurrence"

    def _collect_history_events(self, sample: Dict[str, Any], pattern: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        ts_info = sample.get("timeseries_info", {})
        addi = ts_info.get("additional_info", {}) if isinstance(ts_info, dict) else {}
        hist = addi.get("history_anomalies") if isinstance(addi, dict) else None
        p_norm = str(pattern or "").lower()
        if not isinstance(hist, list):
            return events
        for item in hist:
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
            if str(p).lower() != p_norm:
                continue
            events.append(
                {
                    "start_pos": s_i,
                    "end_pos": e_i,
                    "pattern_type": str(p),
                    "magnitude": m_f,
                    "duration": e_i - s_i + 1,
                    "source": "history",
                }
            )
        events.sort(key=lambda x: (int(x["start_pos"]), int(x["end_pos"])))
        return events

    def _collect_current_events(self, anomaly_patterns: List[Dict[str, Any]], pattern: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        p_norm = str(pattern or "").lower()
        for a in anomaly_patterns:
            if not isinstance(a, dict):
                continue
            if str(a.get("pattern_type") or "").lower() != p_norm:
                continue
            try:
                s_i = int(a.get("start_pos"))
                e_i = int(a.get("end_pos"))
                m_f = float(a.get("magnitude"))
            except Exception:
                continue
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            events.append(
                {
                    "start_pos": s_i,
                    "end_pos": e_i,
                    "pattern_type": str(a.get("pattern_type") or ""),
                    "magnitude": m_f,
                    "duration": e_i - s_i + 1,
                    "source": "current",
                }
            )
        events.sort(key=lambda x: (int(x["start_pos"]), int(x["end_pos"])))
        return events

    def _build_ignore_frequency_analysis(
        self,
        anomaly: Dict[str, Any],
        rule: Dict[str, Any],
        sample: Dict[str, Any],
        anomaly_patterns: List[Dict[str, Any]],
        include_history: bool = True,
    ) -> Dict[str, Any]:
        params = rule.get("parameters", {}) if isinstance(rule, dict) else {}
        window_size = max(1, int(params.get("window_size", 100) or 100))
        min_occ = int(params.get("min_occurrences", 3) or 3)
        max_occ = int(params.get("max_occurrences", 1) or 1)
        rule_name = str(rule.get("name") or "").strip().lower()

        p_norm = str(anomaly.get("pattern_type") or "").lower()
        cur_start = int(anomaly.get("start_pos", 0) or 0)

        history_events = self._collect_history_events(sample=sample, pattern=p_norm) if include_history else []
        current_events = self._collect_current_events(anomaly_patterns=anomaly_patterns, pattern=p_norm)
        all_events = history_events + current_events
        all_events.sort(key=lambda x: (int(x["start_pos"]), int(x["end_pos"]), str(x.get("source") or "")))

        if not all_events:
            return {
                "analysis_kind": "ignore_frequency",
                "window_size": window_size,
                "max_count": 0,
                "best_window": None,
                "threshold": min_occ if rule_name == "ignore_frequent_patterns" else max_occ,
                "triggered": False,
                "support_anomalies": [],
                "analysis_note": "No same-pattern events found in current/history scope.",
            }

        pos_list = sorted(int(e["start_pos"]) for e in all_events)
        best_count = 0
        best_window = None
        best_indices = (0, -1)

        # Enumerate windows that contain current start and maximize same-pattern count.
        for left in pos_list:
            right = left + window_size - 1
            if not (left <= cur_start <= right):
                continue
            in_window = [p for p in pos_list if left <= p <= right]
            c = len(in_window)
            if c > best_count:
                best_count = c
                best_window = (left, right)

        if best_window is None:
            # Fallback to centered window when no candidate is found.
            left = cur_start - window_size + 1
            right = cur_start
            in_window = [p for p in pos_list if left <= p <= right]
            best_count = len(in_window)
            best_window = (left, right)

        left, right = best_window
        support_anomalies = [
            e for e in all_events if left <= int(e["start_pos"]) <= right
        ]
        support_anomalies.sort(key=lambda x: (int(x["start_pos"]), int(x["end_pos"])))

        if rule_name == "ignore_rare_patterns":
            threshold = max_occ
            triggered = 0 < best_count <= max_occ
            comparison = f"0 < count <= {max_occ}"
        else:
            threshold = min_occ
            triggered = best_count >= min_occ
            comparison = f"count >= {min_occ}"

        return {
            "analysis_kind": "ignore_frequency",
            "window_size": window_size,
            "max_count": best_count,
            "best_window": {"start": left, "end": right},
            "threshold": threshold,
            "triggered": triggered,
            "comparison": comparison,
            "support_anomalies": support_anomalies,
        }

    def _build_ignore_recent_reoccurrence_analysis(
        self,
        anomaly: Dict[str, Any],
        rule: Dict[str, Any],
        sample: Dict[str, Any],
        anomaly_patterns: List[Dict[str, Any]],
        include_history: bool = True,
    ) -> Dict[str, Any]:
        params = rule.get("parameters", {}) if isinstance(rule, dict) else {}
        window_size = max(1, int(params.get("window_size", 100) or 100))
        mode = str(params.get("mode", "both") or "both").strip().lower()
        if mode not in {"magnitude", "duration", "both"}:
            mode = "both"

        p_norm = str(anomaly.get("pattern_type") or "").lower()
        cur_start = int(anomaly.get("start_pos", 0) or 0)
        cur_end = int(anomaly.get("end_pos", cur_start) or cur_start)
        cur_dur = cur_end - cur_start + 1
        try:
            cur_mag = float(anomaly.get("magnitude", 0.0) or 0.0)
        except Exception:
            cur_mag = 0.0

        history_events = self._collect_history_events(sample=sample, pattern=p_norm) if include_history else []
        current_events = self._collect_current_events(anomaly_patterns=anomaly_patterns, pattern=p_norm)

        previous_events: List[Dict[str, Any]] = []
        for e in history_events + current_events:
            end_i = int(e.get("end_pos", e.get("start_pos", 0)) or 0)
            if end_i > cur_start:
                continue
            if cur_start - end_i > window_size:
                continue
            if int(e.get("start_pos", 0)) == cur_start and e.get("source") == "current":
                continue
            previous_events.append(e)

        previous_events.sort(key=lambda x: (int(x["end_pos"]), int(x["start_pos"])), reverse=True)

        # Decision policy for explanation:
        # scan from nearest history to older ones; once we find a stronger historical
        # reference (by configured mode), we can decide to filter and stop immediately.
        should_filter = False
        decisive_reference: Optional[Dict[str, Any]] = None
        checked_count = 0

        for ev in previous_events:
            checked_count += 1
            prev_mag = float(ev.get("magnitude", 0.0) or 0.0)
            prev_dur = int(ev.get("duration", int(ev.get("end_pos", 0)) - int(ev.get("start_pos", 0)) + 1) or 1)
            mag_ok = prev_mag >= cur_mag
            dur_ok = prev_dur >= cur_dur
            if mode == "magnitude":
                matched = mag_ok
            elif mode == "duration":
                matched = dur_ok
            else:
                matched = mag_ok and dur_ok

            if matched:
                should_filter = True
                decisive_reference = ev
                break

        if decisive_reference is not None:
            # Keep this key name as requested by downstream prompt style.
            stop_reason = "first_non_matching_reference"
        elif not previous_events:
            stop_reason = "no_previous_reference"
        else:
            stop_reason = "matched_all_in_window"

        return {
            "analysis_kind": "ignore_recent_reoccurrence",
            "window_size": window_size,
            "mode": mode,
            "triggered": should_filter,
            "matched_prefix_len": checked_count,
            "support_anomalies": [decisive_reference] if decisive_reference is not None else [],
            "decisive_reference": decisive_reference,
            "first_break": None,
            "stop_reason": stop_reason,
        }

    def _build_intent_action_context(
        self,
        sample: Dict[str, Any],
        anomaly: Dict[str, Any],
        hit_rule: Optional[Dict[str, Any]],
        rules_sorted: List[Dict[str, Any]],
        anomaly_patterns: List[Dict[str, Any]],
        include_history: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self._is_complex_intent_rule(hit_rule):
            return None
        complex_rules = [r for r in (rules_sorted or []) if self._is_complex_intent_rule(r)]
        if not complex_rules:
            return None

        selected_rule = hit_rule if self._is_complex_intent_rule(hit_rule) else complex_rules[0]
        if not isinstance(selected_rule, dict):
            return None

        rule_name = str(selected_rule.get("name") or "").strip().lower()
        rule_category = str(selected_rule.get("category") or "").strip().lower()
        context: Dict[str, Any] = {
            "rule_name": rule_name,
            "rule_category": rule_category,
            "current_anomaly": {
                "start_pos": int(anomaly.get("start_pos", 0) or 0),
                "end_pos": int(anomaly.get("end_pos", 0) or 0),
                "pattern_type": str(anomaly.get("pattern_type") or ""),
                "magnitude": anomaly.get("magnitude"),
            },
        }

        if rule_category == "ignore_frequency":
            context["analysis"] = self._build_ignore_frequency_analysis(
                anomaly=anomaly,
                rule=selected_rule,
                sample=sample,
                anomaly_patterns=anomaly_patterns,
                include_history=include_history,
            )
            return context

        if rule_name == "ignore_recent_reoccurrence":
            context["analysis"] = self._build_ignore_recent_reoccurrence_analysis(
                anomaly=anomaly,
                rule=selected_rule,
                sample=sample,
                anomaly_patterns=anomaly_patterns,
                include_history=include_history,
            )
            return context

        return None

    def build_human_intent_nl_prompts(
        self,
        human_intents: List[Dict[str, Any]],
        ts_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Build prompts for rewriting structured intent into natural language.

        Note: we provide both the original JSON intent and the formatted (TextPool) constraints
        so the model has enough context while still outputting only natural user text.
        """
        ts_infos = ts_infos or [{} for _ in human_intents]
        intro = """**Task:** Convert the structured anomaly detection rules in the `human_intent` field into a single, cohesive request in casual, natural English. **Strict Constraints:**
1. Output Format & Flow
* Output a single, flowing block of plain text. Absolutely no lists, bullet points, headers, or line breaks.
* Do not map rules 1-to-1 to sentences. Combine or expand rules to create a natural flow using the minimum necessary sentences.
* Skip polite intro phrases (e.g., "Hey," "Please check"). Start immediately with the core detection request.

2. Tone & Vocabulary
* Use highly varied, conversational wording. Vary sentence structures, verbs, and rhythm across outputs.
* Any wording examples in this instruction are style references only. Do NOT copy any example phrase verbatim or with near-verbatim paraphrases.
* Use broad terms ("issues," "anomalies," "unusual changes") for general detection. Only use specific directional terms if the rules explicitly specify an anomaly type.
* Treat inclusion-style rules (for example, include_slow_trends, include_volatility_changes) as additive preferences: phrase them as "also pay attention to" signals, not exclusive "only focus on" constraints, unless another rule explicitly restricts scope.
* Never mention technical details (JSON, rule names, underlying parameters) or inherent time-series characteristics (seasonality, frequency, trend, noise).
* Mention metric context (such as metric name/type) or sampling-interval context only when it directly helps express a specific rule in `human_intent.rules`; otherwise omit it to keep the request concise.
* Do not add unrelated background context, and do not reveal or infer background time-series characteristics.

3. Handling Numbers & Parameters
* Keep rule-critical numeric parameters exact when they are explicitly provided in `human_intent.rules`.
* **For window-size and duration constraints, do not use vague substitutes.** Keep them precise by converting points into the correct physical time based on sampling interval, or explicitly keep points when needed for clarity.
* **Only if a magnitude threshold is explicitly defined within `human_intent.rules`**, translate it into a natural instruction to disregard changes smaller than that specific amount (using [threshold_value_str]).
* You may optionally add a relative phrasing based on `timeseries_info.magnitude` (for example, an approximate multiple) only when the ratio is human-friendly and clean (e.g., about 0.5x, 1x, 2x, 3x), and only if it does not change the original threshold meaning.
* Do not invent magnitude references when the multiple is awkward or unclear.
* Frame handling actions (merging close anomalies, filtering brief ones, capping length) as natural user preferences (e.g., "group close issues together," "ignore overly brief changes").
* When both magnitude and duration are involved in a rule, frame the instruction as prioritizing more severe anomalies (e.g., "focus on more severe issues considering both their size and duration") rather than explicitly comparing magnitude or duration alone.

4. Handling Time & Intervals
* Strictly align all time references with the series' actual sampling interval. Convert point-based counts into correct physical time units (seconds, minutes, hours).
* Keep window-size related intervals precise; do not replace them with vague descriptors.
* If a converted duration is overly awkward (for example, includes odd seconds), prefer a nearby human-friendly rounded expression (e.g., 28 minutes 48 seconds -> 29 or 30 minutes; 583 hours 20 minutes -> 583 hours), as long as the rule intent is preserved.
* Use concise time expressions and avoid unnecessarily granular time strings.
"""
        prompts: List[str] = []
        for hi, ts in zip(human_intents, ts_infos):
            sampling_interval = ts.get("sampling_interval")
            hi_sanitized = self._sanitize_human_intent_for_intent_nl_prompt(hi, sampling_interval)
            sampling_interval_str = SAMPLING_INTERVAL_STR_DICT.get(sampling_interval, str(ts.get("sampling_interval", "Unknown")) + "seconds")
            ts_info_min = {
                "metric_type": (ts or {}).get("metric_type"),
                "metric_name": (ts or {}).get("metric_name"),
                "sampling_interval": sampling_interval_str,
                "seq_len": (ts or {}).get("seq_len"),
                "magnitude": (ts or {}).get("magnitude"),
                "characteristics": (ts or {}).get("characteristics"),
            }
            intent_hints = self._abstract_rule_descriptions(hi)
            ctx = {
                "timeseries_info": ts_info_min,
                "human_intent": hi_sanitized,
                "intent_hints": intent_hints,
            }
            prompts.append(
                f"Context JSON (for reference only):\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n{intro}\n"
            )
        return prompts

    def batch_generate_human_intent_nl(
        self,
        human_intents: List[Dict[str, Any]],
        llm_client: Any,
        max_retries: int = 3,
        ts_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Use LLM to rewrite structured human_intent into natural-language intent text for input."""
        prompts = self.build_human_intent_nl_prompts(human_intents, ts_infos=ts_infos)
        structured_list = [self.text_pool.generate_human_intent_structured(hi, ts_info) for hi, ts_info in zip(human_intents, ts_infos)]

        def _call_llm(batch_prompts: List[str]) -> List[str]:
            if getattr(llm_client, 'engine', None) in ('ark', 'gpt'):
                return llm_client.llm_batch_generate_parallel(batch_prompts)
            return llm_client.llm_batch_generate(batch_prompts, use_chat_template=True)

        def _is_valid_intent_nl(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            lowered = t.lower()
            # Avoid echoing prompt/meta formatting.
            forbidden_markers = [
                "formatted intent constraints",
                "structured",
                "json",
                "bullet",
                "header",
                "[human intent",
                "[user intent",
                "<think>",
            ]
            if any(m in lowered for m in forbidden_markers):
                return False
            # Avoid list-like formatting.
            if "\n-" in t or "\n•" in t:
                return False
            return True

        outputs: List[Optional[str]] = [None] * len(human_intents)
        invalid_indices: List[int] = list(range(len(human_intents)))
        attempt = 0

        while invalid_indices and attempt <= max_retries:
            suffix = "" if attempt == 0 else (
                "\n\nIMPORTANT REMINDER: Output must be ONLY 1-2 plain English sentences. "
                "No headers, no bullets, no JSON, no mention of 'structured', and do not copy the prompt. "
                "Do not reuse any example phrase verbatim. Keep all rule-critical numbers exact, especially window-size and explicit thresholds. "
                "Prefer concise, human-friendly rounded time expressions instead of awkward second-level precision when conversion looks mechanical."
            )
            retry_prompts = [prompts[i] + suffix for i in invalid_indices]
            outs = _call_llm(retry_prompts)

            still_invalid: List[int] = []
            for local_idx, global_idx in enumerate(invalid_indices):
                raw = outs[local_idx] if local_idx < len(outs) else ""
                txt = (raw or "").strip()
                if _is_valid_intent_nl(txt):
                    outputs[global_idx] = txt
                else:
                    still_invalid.append(global_idx)

            invalid_indices = still_invalid
            attempt += 1

        if invalid_indices:
            examples = []
            for i in invalid_indices[:3]:
                examples.append({
                    "index": i,
                    "llm_text": (outputs[i] or "")[:200],
                    "structured": structured_list[i][:200],
                })
            raise RuntimeError(
                f"LLM failed to generate valid natural-language user intent after {max_retries + 1} attempts. "
                f"Invalid indices: {invalid_indices}. Examples: {examples}"
            )

        return [o if o is not None else "" for o in outputs]
    
    def format_timeseries_for_llm(self, timeseries_data: List[float]) -> List[List[float]]:
        """          LLM    """
        return [timeseries_data]
    
    def generate_input_text(
        self,
        sample: Dict[str, Any],
        is_ts_modal: bool = True,
        is_cot: bool = True,
        timeseries_data: Optional[List[List[float]]] = None,
        is_dapo: bool = False,
        human_intent_nl: Optional[str] = None,
    ) -> str:
        """       

              use_tool                         
                 <|im_start|>     <tool_call>/<tool_response>   create_qa_pair     
        """
        
        # 1.         
        ts_info = sample["timeseries_info"]
        statistics = ts_info.get("statistics", {})
        sampling_interval_str = SAMPLING_INTERVAL_STR_DICT.get(ts_info.get("sampling_interval"), str(ts_info.get("sampling_interval", "Unknown")) + "seconds")
        
        ts_info_text = f"""[Timeseries Information]
Metric Type: {ts_info['metric_type']}
Metric Name: {ts_info.get('metric_name', 'Unknown')}
Sequence Length: {ts_info['seq_len']}
Sampling Interval: {sampling_interval_str}
"""
        
        # 2.     
        task_description = f"""[Task Description]
Please perform anomaly detection analysis on the following time series data:

Time Series Data: <ts>{timeseries_data if not is_ts_modal else ''}<ts/>
"""
        if is_cot or is_dapo:
            task_description += (
                "\nStep 1: please identify the characteristics of the input time series (trend, seasonal, frequency, noise).\n"
                "Step 2: please identify the anomaly patterns of the time series, providing the positions and specific anomaly types.\n"
            )
        if is_cot or is_dapo:
            task_description += "Step 3: follow the user's intent preference (if any) to decide which detected anomalies should be kept as final results.\n"

        #     “     human intent”              
        if human_intent_nl is None:
            human_intent_nl = self.generate_human_intent_nl_fallback(self._get_human_intent(sample))
        intent_text = f"[User Intent]\n{human_intent_nl}\n"
        full_input = f"{ts_info_text}\n\n{task_description}\n{intent_text}"
        return full_input
    

    def generate_output_text(
        self,
        sample: Dict[str, Any],
        is_cot: bool = False,
        is_reject: bool = False,
        is_ts_characteristic: bool = True,
        is_human_intent: bool = True,
        query_memory: bool = False,
        include_data_anomalies: bool = False,
    ) -> str:
        """                    """

        output_text = ""
        if is_cot and not query_memory:
            ts_info = sample["timeseries_info"]
            ts_characteristics = ts_info.get("characteristics", {})
            # 1.       
            if is_ts_characteristic:
                characteristic_desc = self.text_pool.generate_ts_characteristic_description(ts_characteristics, additional_info=ts_info)
                output_text += f"[Time Series Characteristics]\n{characteristic_desc}\n\n"
            
            # 2 human intent structured    think       input 
            if is_human_intent:
                human_intent = self._get_human_intent(sample)
                structured = self.text_pool.generate_human_intent_structured(human_intent, ts_info)
                output_text += f"[Human Intent]\n{structured}\n\n"
            
            # 3.       
            anomalies = sample.get("anomalies_detected", []) if not is_reject else sample.get("rejected", {}).get("anomalies_detected", [])
            anomalies = [a for a in anomalies if "fake" not in a["name"]]
            output_text += "[Anomaly Detection Results]\n"
            #      intent rules          “       ” 
            rules = self._get_human_intent(sample).get("rules", [])
            def anomaly_key(a: Dict[str, Any]):
                try:
                    return (a.get("start_pos"), a.get("end_pos"), (a.get("pattern_type") or a.get("pattern") or "").lower())
                except Exception:
                    return (a.get("start_pos"), a.get("end_pos"), str(a.get("pattern_type", "")).lower())

            #           category    name
            rules_sorted = sorted(rules or [], key=lambda r: (str(r.get("category", "")), str(r.get("name", ""))))
            filtered_maps = []
            if rules_sorted:
                for r in rules_sorted:
                    filt_set = set()
                    for fa in r.get("filtered_anomalies", []) or []:
                        filt_set.add(anomaly_key(fa))
                    filtered_maps.append((r, filt_set))

            # final anomalies: prefer anomalies_final, otherwise fall back to anomalies_detected
            final_anoms_any = sample.get("anomalies_final") if not is_reject else sample.get("rejected", {}).get("anomalies_final")
            final_anoms: List[Dict[str, Any]]
            if isinstance(final_anoms_any, list):
                final_anoms = final_anoms_any
            else:
                final_anoms = list(anomalies)
            
            threshold_rule = None
            for r in rules_sorted:
                if not isinstance(r, dict):
                    continue
                if str(r.get("category") or "").strip().lower() != "threshold":
                    continue
                if str(r.get("name") or "").strip().lower() == "absolute_magnitude_threshold":
                    threshold_rule = r
                    break
            duration_rule = None
            for r in rules_sorted:
                if not isinstance(r, dict):
                    continue
                if str(r.get("category") or "").strip().lower() != "ignore_duration":
                    continue
                if str(r.get("name") or "").strip().lower() in {"ignore_short_duration", "truncate_overlong_duration"}:
                    duration_rule = r
                    break

            if len(anomalies) == 0:
                output_text += f"{self.text_pool.generate_common_description('no-anomaly')}\n"
                final_anoms = []
            else:
                # summary: when intent exists, include final count
                remaining_cnt = len(final_anoms)
                rule_count = len(rules_sorted)
                # summary_line = self.text_pool.generate_detection_summary(
                #     total_detected=len(anomalies),
                #     final_remaining=remaining_cnt if rule_count > 0 else None,
                #     rule_count=rule_count,
                # output_text += summary_line + "\n"
                inline_filtered = 0
                inline_kept = 0

                final_set = set(anomaly_key(a) for a in (final_anoms or []))
                for i, anomaly in enumerate(anomalies, 1):
                    anomaly_desc = self.text_pool.generate_anomaly_description(anomaly)
                    output_text += f"{i}. {anomaly_desc}"
                    k = anomaly_key(anomaly)
                    hit_rule = None
                    for r, s in filtered_maps:
                        if k in s:
                            hit_rule = r
                            break
                    intent_action_context = self._build_intent_action_context(
                        sample=sample,
                        anomaly=anomaly,
                        hit_rule=hit_rule,
                        rules_sorted=rules_sorted,
                        anomaly_patterns=anomalies,
                        include_history=False,
                    )
                    magnitude_trace = self._build_magnitude_comparison_trace(
                        anomaly=anomaly,
                        threshold_rule=threshold_rule,
                        hit_rule=hit_rule,
                    )
                    duration_trace = self._build_duration_comparison_trace(
                        anomaly=anomaly,
                        duration_rule=duration_rule,
                        hit_rule=hit_rule,
                    )
                    comparison_trace = self._merge_comparison_traces(
                        magnitude_trace=magnitude_trace,
                        duration_trace=duration_trace,
                        intent_action_context=intent_action_context,
                    )
                    if comparison_trace:
                        output_text += f" {comparison_trace}"
                    if k in final_set:
                        inline_kept += 1
                        output_text += self.text_pool.generate_intent_action_line("kept", intent_action_context=intent_action_context) + "\n"
                    elif hit_rule is not None:
                        inline_filtered += 1
                        output_text += self.text_pool.generate_intent_action_line("excluded", rule=hit_rule, intent_action_context=intent_action_context) + "\n"
                    else:
                        # When intent exists but the rule attribution is missing, be explicit but neutral
                        if rules_sorted:
                            output_text += self.text_pool.generate_intent_action_line("not_kept", intent_action_context=intent_action_context) + "\n"
                        else:
                            output_text += "\n"
        else:
            sample = sample if not is_reject else sample.get("rejected", sample)
            anomalies = sample.get("anomalies_final")
            if not isinstance(anomalies, list):
                anomalies = sample.get("anomalies_detected", [])
        
        if is_cot: output_text += "</think>\n"
        # 4.       
        if include_data_anomalies and is_cot:
            output_text += "<data_anomaly>\n"
            anomalies = sample.get("anomalies_detected")
            if len(anomalies) == 0:
                output_text += "[]"
            else:
                output_text += str([(a["start_pos"], a["end_pos"], a["pattern_type"], float(f'{a["magnitude"]:.3g}')) for a in anomalies])
            if is_cot:
                output_text += "\n</data_anomaly>\n"
        # 5.      pattern          
        output_text += "<answer>\n" if is_cot else ""
        if is_cot:
            # In CoT mode, the <answer> reflects final anomalies after intent
            anomalies_any = sample.get("anomalies_final")
            if isinstance(anomalies_any, list):
                anomalies = anomalies_any
        if len(anomalies) == 0:
            output_text += "[]"
        else:
            positions = [(anomaly["start_pos"], anomaly["end_pos"], anomaly["pattern_type"], float(f'{anomaly["magnitude"]:.3g}')) for anomaly in anomalies]
            output_text += str(positions)
        if is_cot:
            output_text += "\n</answer>"
        return output_text
        
    def create_qa_pair(
        self,
        sample: Dict[str, Any],
        is_ts_modal: bool = True,
        is_cot: bool = True,
        reject=None,
        is_dapo: bool = False,
        use_tool: bool = False,
        query_memory: bool = False,
        include_data_anomalies: bool = False,
        human_intent_nl: Optional[str] = None,
    ) -> Dict[str, Any]:
        """    QA                 input    <|im_start|>     """
        timeseries_data = self.format_timeseries_for_llm(sample["timeseries_data"])
        if human_intent_nl is None:
            resolved_human_intent_nl = self.generate_human_intent_nl_fallback(self._get_human_intent(sample))
        else:
            resolved_human_intent_nl = human_intent_nl
        base_input = self.generate_input_text(
            sample,
            is_ts_modal=is_ts_modal,
            is_cot=is_cot,
            timeseries_data=timeseries_data,
            is_dapo=is_dapo,
            human_intent_nl=resolved_human_intent_nl,
        )
        output_text = self.generate_output_text(
            sample,
            is_cot=is_cot,
            is_ts_characteristic=False if is_cot and (use_tool or query_memory) else True,
            is_human_intent=False if is_cot and (use_tool or query_memory) else True,
            query_memory=query_memory,
            include_data_anomalies=include_data_anomalies,
        )
        anomalies = sample.get("anomalies_final")
        if not isinstance(anomalies, list):
            anomalies = sample.get("anomalies_detected", [])
        label = [(a["start_pos"], a["end_pos"], a["pattern_type"], float(f'{a["magnitude"]:.3g}')) for a in anomalies if "fake" not in a["name"]]

        input_text = base_input
        query_memory_count = 0
        raw_input_text = input_text
        raw_output_text = output_text

        if is_cot and use_tool:
            #   detection_chains     
            detection_chains: List[dict] = sample.get("detection_chains", [])
            ts_list_full: List[float] = sample.get("timeseries_data", [])
            anomaly_patterns: List[dict] = sample.get("anomalies_detected", [])

            max_steps = 1
            containing_max_tries = False
            im_parts: List[str] = []
            im_parts.append(f"{base_input}<|im_end|>\n")
            # im_parts.append(f"<|im_start|>user\n{base_input}<|im_end|>\n")

            characteristic_desc = self.text_pool.generate_ts_characteristic_description(
                sample["timeseries_info"].get("characteristics", {}),
                additional_info=sample.get("timeseries_info", {}),
            )
            characteristic_text = f"[Time Series Characteristics]\n{characteristic_desc}\n"

            human_intent = self._get_human_intent(sample)
            structured = self.text_pool.generate_human_intent_structured(human_intent, sample["timeseries_info"])
            intent_text = f"[Human Intent]\n{structured}\n"

            final_anoms_any = sample.get("anomalies_final")
            final_anoms = final_anoms_any if isinstance(final_anoms_any, list) else sample.get("anomalies_detected", [])
            final_keys = set(self._anomaly_key(a) for a in final_anoms if isinstance(a, dict))
            memory_windows = self._memory_window_candidates(human_intent)
            memory_window = max(memory_windows) if memory_windows else 0
            
            think_lines = [
                "<|im_start|>assistant\n<think>",
                characteristic_text,
                intent_text,
                self.text_pool.generate_common_description("first-inspect")
            ]
            if detection_chains:
                tool_call_blocks: List[str] = []
                tool_resp_blocks: List[str] = []

                for chain in detection_chains:
                    idx_i = chain.get("anomaly_index", 0)
                    steps = chain.get("steps_detail", [])
                    if not steps:
                        continue
                    s1 = steps[0]
                    pred = s1.get("prediction", [None, None])
                    slice_input = s1.get("slice_input")
                    consistent = s1.get("consistent", False)

                    pat = None
                    mag = None
                    if 0 <= idx_i < len(anomaly_patterns):
                        pat = anomaly_patterns[idx_i].get("pattern_type")
                        mag_val = anomaly_patterns[idx_i].get("magnitude")
                        try:
                            mag = float(f"{float(mag_val):.3g}") if mag_val is not None else None
                        except Exception:
                            mag = mag_val
                    step1_anom = {
                        "start_pos": pred[0] if isinstance(pred, (list, tuple)) and len(pred) == 2 else None,
                        "end_pos": pred[1] if isinstance(pred, (list, tuple)) and len(pred) == 2 else None,
                        "pattern_type": pat,
                        "magnitude": mag,
                    }
                    think_lines.append(f"Anomaly {idx_i}: {self.text_pool.generate_anomaly_description(step1_anom)}")

                    if use_tool and slice_input is not None:
                        st, en = int(slice_input[0]), int(slice_input[1])
                        think_lines.append(self.text_pool.generate_common_description("slice", idx_i=idx_i, st=st, en=en))
                        tool_call_blocks.append("<tool_call>\n" + json.dumps({
                            "name": "slice_timeseries",
                            "arguments": {"anomaly": idx_i, "start": st, "end": en}
                        }, ensure_ascii=False) + "\n</tool_call>")
                        #          tool_response    user     
                        tool_resp_text = f"For anomaly {idx_i}, the slice of original time series from {st} to {en} is: <ts><ts/>."
                        tool_resp_blocks.append("<tool_response>\n" + tool_resp_text + "\n</tool_response>")
                        timeseries_data.append(ts_list_full[st:en+1])
                    else:
                        think_lines.append(f"For anomaly {idx_i}, no slice window provided; skip slicing.")

                    if query_memory and memory_window > 0 and pat is not None and 0 <= idx_i < len(anomaly_patterns):
                        anom_item = anomaly_patterns[idx_i]
                        k = self._anomaly_key(anom_item) if isinstance(anom_item, dict) else None
                        try:
                            cur_start_idx = int(anom_item.get("start_pos")) if isinstance(anom_item, dict) else int(pred[0])
                        except Exception:
                            cur_start_idx = int(pred[0]) if isinstance(pred, (list, tuple)) and len(pred) == 2 else 0

                        #           memory   start      
                        if k in final_keys and cur_start_idx <= memory_window:
                            tool_call_blocks.append("<tool_call>\n" + json.dumps({
                                "name": "query_memory",
                                "arguments": {
                                    "pattern": str(pat),
                                    "start": int(cur_start_idx),
                                    "window_size": int(memory_window),
                                }
                            }, ensure_ascii=False) + "\n</tool_call>")
                            history_list = self._query_history_anomalies(
                                sample=sample,
                                pattern=str(pat),
                                start=int(cur_start_idx),
                                window_size=int(memory_window),
                            )
                            tool_resp_blocks.append("<tool_response>\n" + str(history_list) + "\n</tool_response>")
                
                max_steps = 0
                for chain in detection_chains:
                    steps = chain.get("steps_detail", [])
                    if steps and len(steps) > max_steps:
                        max_steps = len(steps)

                im_parts.append("\n".join(think_lines) + "\n" + "".join(tool_call_blocks) + "<|im_end|>\n")
                if tool_resp_blocks:
                    tool_resp = "<|im_start|>user\n" + "".join(tool_resp_blocks) + "<|im_end|>\n"
                    im_parts.append(tool_resp)

                #        step                  
                #             /           step   
                completed: set = set()
                for step_idx in range(2, max_steps+1):
                    #     step        
                    entries = []
                    for chain in detection_chains:
                        idx_i = chain.get("anomaly_index", 0)
                        steps = chain.get("steps_detail", [])
                        if idx_i in completed:
                            continue
                        if not steps:
                            continue
                        step = steps[step_idx-1]
                        entries.append((idx_i, step))
                    if not entries:
                        print(f"[DEBUG] Step {step_idx}: no valid entries, skip. {completed=}")
                        continue
                    
                    if step_idx == 1: step_str = "first"
                    elif step_idx == 2: step_str = "second"
                    elif step_idx == 3: step_str = "third"
                    else: step_str = f"{step_idx}th"
                    lines = [self.text_pool.generate_common_description("further-inspect", step_str=step_str)]
                    tool_call_blocks: List[str] = []
                    tool_resp_blocks: List[str] = []

                    for idx_i, step in entries:
                        pred = step.get("prediction", [None, None])
                        pred_rel = step.get("prediction_relative", pred)
                        slice_input = step.get("slice_input")
                        consistent = step.get("consistent", False)

                        pat = None
                        mag = None
                        if 0 <= idx_i < len(anomaly_patterns):
                            pat = anomaly_patterns[idx_i].get("pattern_type")
                            mag_val = anomaly_patterns[idx_i].get("magnitude")
                            try:
                                mag = float(f"{float(mag_val):.3g}") if mag_val is not None else None
                            except Exception:
                                mag = mag_val
                        step_desc_anom = {
                            "start_pos": pred_rel[0] if isinstance(pred_rel, (list, tuple)) and len(pred_rel) == 2 else None,
                            "end_pos": pred_rel[1] if isinstance(pred_rel, (list, tuple)) and len(pred_rel) == 2 else None,
                            "pattern_type": pat,
                            "magnitude": mag,
                        }
                        desc = self.text_pool.generate_anomaly_description(step_desc_anom)
                        relative_offset = pred[0] - pred_rel[0]

                        if consistent:
                            lines.append(
                                f"{self.text_pool.generate_common_description('anomaly-with-idx', idx_i=idx_i, pat=pat, desc=desc)} "
                                f"{self.text_pool.generate_common_description('get-raw-idx', relative_offset=relative_offset, st=pred[0], en=pred[1])} "
                                f"{self.text_pool.generate_common_description('consistent')}"
                            )
                            completed.add(idx_i)
                        else:
                            think_txt = (
                                f"{self.text_pool.generate_common_description('anomaly-with-idx', idx_i=idx_i, pat=pat, desc=desc)} "
                                f"{self.text_pool.generate_common_description('get-raw-idx', relative_offset=relative_offset, st=pred[0], en=pred[1])} "
                            )
                            if use_tool and slice_input is not None:
                                st, en = int(slice_input[0]), int(slice_input[1])
                                think_txt += self.text_pool.generate_common_description("inconsistent-then-slice", st=st, en=en)
                                lines.append(think_txt)
                                tool_call_blocks.append("<tool_call>\n" + json.dumps({
                                    "name": "slice_timeseries",
                                    "arguments": {"anomaly": idx_i, "start": st, "end": en}
                                }, ensure_ascii=False) + "\n</tool_call>")
                                tool_resp_text = f"For anomaly {idx_i}, the slice of original time series from {st} to {en} is: <ts><ts/>."
                                tool_resp_blocks.append("<tool_response>\n" + tool_resp_text + "\n</tool_response>")
                                timeseries_data.append(ts_list_full[st:en+1])
                            else:
                                containing_max_tries = True
                                think_txt += self.text_pool.generate_common_description("inconsistent-then-confirm", idx_i=idx_i)
                                lines.append(think_txt)
                                completed.add(idx_i)

                    assistant_text = "<|im_start|>assistant\n" + "\n".join(lines) + ("\n" + "".join(tool_call_blocks) if tool_call_blocks else "") + "<|im_end|>\n"
                    im_parts.append(assistant_text)
                    if tool_resp_blocks:
                        tool_resp = "<|im_start|>user\n" + "".join(tool_resp_blocks) + "<|im_end|>\n"
                        im_parts.append(tool_resp)
            else:
                im_parts.append("\n".join(think_lines) + f" {self.text_pool.generate_common_description('no-anomaly')}\n<|im_end|>\n")
            
            all_text = "<|im_end|>".join("".join(im_parts).split("<|im_end|>")[:-1]) + "\n" + output_text
            select_step = random.randint(max(1, max_steps-1), max_steps) if not containing_max_tries else max_steps
            # select_step = max_steps
            #   assistant     step   select_step  all_text    input_text output_text
            all_splits = all_text.split("<|im_start|>assistant\n")
            input_text = "<|im_start|>assistant\n".join(all_splits[:select_step])
            input_text = "\n<|im_end|>".join(input_text.split("<|im_end|>")[:-1])
            try:
                output_text = all_splits[select_step].split("<|im_end|>")[0]
            except IndexError:
                raise Exception(f"{all_text=}, {select_step=}, {max_steps=}, {len(all_splits)=}")
            
            # input_text = "".join(im_parts)
            # output_text = final_str + output_text

        if (is_cot or is_dapo) and query_memory and (not use_tool):
            anomaly_patterns: List[dict] = sample.get("anomalies_detected", [])
            anomaly_patterns = [a for a in anomaly_patterns if isinstance(a, dict) and "fake" not in str(a.get("name", ""))]
            human_intent = self._get_human_intent(sample)

            #   generate_output_text                          
            rules = self._get_human_intent(sample).get("rules", [])
            rules_sorted = sorted(rules or [], key=lambda r: (str(r.get("category", "")), str(r.get("name", ""))))
            
            filtered_maps = []
            for r in rules_sorted:
                filt_set = set()
                for fa in r.get("filtered_anomalies", []) or []:
                    if isinstance(fa, dict):
                        filt_set.add(self._anomaly_key(fa))
                filtered_maps.append((r, filt_set))

            final_anoms_any = sample.get("anomalies_final")
            final_anoms = final_anoms_any if isinstance(final_anoms_any, list) else list(anomaly_patterns)
            final_set = set(self._anomaly_key(a) for a in final_anoms if isinstance(a, dict))

            threshold_rule = None
            for r in rules_sorted:
                if not isinstance(r, dict):
                    continue
                if str(r.get("category") or "").strip().lower() != "threshold":
                    continue
                if str(r.get("name") or "").strip().lower() == "absolute_magnitude_threshold":
                    threshold_rule = r
                    break
            duration_rule = None
            for r in rules_sorted:
                if not isinstance(r, dict):
                    continue
                if str(r.get("category") or "").strip().lower() != "ignore_duration":
                    continue
                if str(r.get("name") or "").strip().lower() in {"ignore_short_duration", "truncate_overlong_duration"}:
                    duration_rule = r
                    break

            characteristic_desc = self.text_pool.generate_ts_characteristic_description(
                sample["timeseries_info"].get("characteristics", {}),
                additional_info=sample.get("timeseries_info", {}),
            )
            characteristic_text = f"[Time Series Characteristics]\n{characteristic_desc}\n"
            structured = self.text_pool.generate_human_intent_structured(human_intent, sample["timeseries_info"])
            intent_text = f"[Human Intent]\n{structured}\n"

            think_lines = [
                "<|im_start|>assistant\n<think>",
                characteristic_text,
                intent_text,
                "[Anomaly Detection Results]",
            ]

            containing_memory_related = any(
                (str(r.get("category", "")).strip().lower() == "ignore_frequency" or str(r.get("name", "")).strip().lower() in {"ignore_recent_reoccurrence"})                for r in rules_sorted
            )
            max_window_size = 0
            if containing_memory_related:
                for r in rules_sorted:
                    if str(r.get("category", "")).strip().lower() == "ignore_frequency" or str(r.get("name", "")).strip().lower() in {"ignore_recent_reoccurrence"}:
                        params = r.get("parameters", {}) if isinstance(r, dict) else {}
                        window_size = int(params.get("window_size", 0))
                        if window_size > max_window_size:
                            max_window_size = window_size

            tool_resp_blocks: List[str] = []
            pending_followup_lines: List[str] = []
            im_parts: List[str] = [f"{base_input}<|im_end|>\n"]
            if not anomaly_patterns:
                im_parts.append("\n".join(think_lines) + f" {self.text_pool.generate_common_description('no-anomaly')}\n<|im_end|>\n")
            
            for i, anomaly in enumerate(anomaly_patterns, 1):
                k = self._anomaly_key(anomaly)
                anomaly_desc = self.text_pool.generate_anomaly_description(anomaly)

                hit_rule = None
                for r, s in filtered_maps:
                    if k in s:
                        hit_rule = r
                        # break

                should_tool_call = False
                query_pattern = ""
                query_start = 0
                query_window_size = 0

                if hit_rule is None:
                    if containing_memory_related and max_window_size > 0 and int(anomaly.get("start_pos")) <= max_window_size:
                        should_tool_call = True
                else:
                    rule_name = str(hit_rule.get("name") or "").strip().lower()
                    rule_category = str(hit_rule.get("category") or "").strip().lower()
                    memory_related = rule_category == "ignore_frequency" or rule_name == "ignore_recent_reoccurrence"
                    if memory_related and max_window_size > 0 and int(anomaly.get("start_pos")) <= max_window_size:
                        should_tool_call = True

                line = f"{i}. {anomaly_desc}"
                intent_action_context = self._build_intent_action_context(
                    sample=sample,
                    anomaly=anomaly,
                    hit_rule=hit_rule,
                    rules_sorted=rules_sorted,
                    anomaly_patterns=anomaly_patterns,
                )
                magnitude_trace = self._build_magnitude_comparison_trace(
                    anomaly=anomaly,
                    threshold_rule=threshold_rule,
                    hit_rule=hit_rule,
                )
                duration_trace = self._build_duration_comparison_trace(
                    anomaly=anomaly,
                    hit_rule=hit_rule,
                    duration_rule=duration_rule,
                )
                comparison_trace = self._merge_comparison_traces(
                    magnitude_trace=magnitude_trace,
                    duration_trace=duration_trace,
                    intent_action_context=intent_action_context,
                )
                if should_tool_call:
                    query_pattern = str(anomaly.get("pattern_type") or "")
                    query_start = int(anomaly.get("start_pos"))
                    query_window_size = max_window_size
                    query_memory_count += 1
                    line += f" {self.text_pool.generate_common_description('check-anomaly-with-idx', idx_i=i, pat=str(anomaly.get('pattern_type') or ''))}"
                    #                          tool_response      
                    if comparison_trace:
                        line += f" {comparison_trace}"
                    think_lines.append(line)
                    # if magnitude_trace:
                    #     think_lines.append(magnitude_trace)
                    think_lines.append("<tool_call>\n" + json.dumps({
                        "name": "query_memory",
                        "arguments": {
                            "anomaly": i,
                            "pattern": query_pattern,
                            "start": query_start,
                            "window_size": query_window_size,
                        }
                    }, ensure_ascii=False) + "\n</tool_call>")
                    history_list = self._query_history_anomalies(
                        sample=sample,
                        pattern=query_pattern,
                        start=query_start,
                        window_size=query_window_size,
                    )
                    history_anomaly_desc = self.text_pool.generate_history_anomalies_description(i, history_list, query_pattern, query_start, query_window_size)
                    tool_resp_blocks.append("<tool_response>\n" + history_anomaly_desc + "\n</tool_response>")
                    line = f"{self.text_pool.generate_common_description('check-anomaly-with-idx-and-history' if containing_memory_related else 'check-anomaly-with-idx', idx_i=i, pat=str(anomaly.get('pattern_type') or ''))}"
                    history_mag_summary = self._build_magnitude_comparison_trace(
                        anomaly=anomaly,
                        hit_rule=hit_rule,
                        intent_action_context=intent_action_context,
                        include_history_summary=True,
                    )
                    history_dur_summary = self._build_duration_comparison_trace(
                        anomaly=anomaly,
                        hit_rule=hit_rule,
                        intent_action_context=intent_action_context,
                        include_history_summary=True,
                    )
                    history_comparison_trace = self._merge_comparison_traces(
                        magnitude_trace=history_mag_summary,
                        duration_trace=history_dur_summary,
                        intent_action_context=intent_action_context,
                    )
                    if history_comparison_trace:
                        line += f" {history_comparison_trace}"

                    if k in final_set:
                        line += self.text_pool.generate_intent_action_line("kept", intent_action_context=intent_action_context)
                    elif hit_rule is not None:
                        line += self.text_pool.generate_intent_action_line("excluded", rule=hit_rule, intent_action_context=intent_action_context)
                    else:
                        line += self.text_pool.generate_intent_action_line("not_kept", intent_action_context=intent_action_context) if rules_sorted else "\n"
                    pending_followup_lines.append(line)
                else:
                    # think_lines.append(line)
                    # if magnitude_trace:
                    #     think_lines.append(magnitude_trace)

                    line += f" {self.text_pool.generate_common_description('check-anomaly-with-idx-and-history' if containing_memory_related else 'check-anomaly-with-idx', idx_i=i, pat=str(anomaly.get('pattern_type') or ''))}"
                    if comparison_trace:
                        line += f" {comparison_trace}"
                    if k in final_set:
                        line += self.text_pool.generate_intent_action_line("kept", intent_action_context=intent_action_context)
                    elif hit_rule is not None:
                        line += self.text_pool.generate_intent_action_line("excluded", rule=hit_rule, intent_action_context=intent_action_context)
                    else:
                        line += self.text_pool.generate_intent_action_line("not_kept", intent_action_context=intent_action_context) if rules_sorted else "\n"
                    think_lines.append(line)

            if anomaly_patterns:
                im_parts.append("\n".join(think_lines) + "<|im_end|>\n")

            max_steps = 1
            containing_max_tries = False
            if tool_resp_blocks:
                im_parts.append("<|im_start|>user\n" + "".join(tool_resp_blocks) + "<|im_end|>\n")
                # memory      assistant              
                second_lines = [
                    "<|im_start|>assistant",
                ]
                if pending_followup_lines:
                    second_lines.extend(pending_followup_lines)
                else:
                    second_lines.append("Memory context has been retrieved for retained anomalies.")
                im_parts.append("\n".join(second_lines) + "<|im_end|>\n")
                max_steps = 2

            all_text = "<|im_end|>".join("".join(im_parts).split("<|im_end|>")[:-1]) + "\n" + output_text
            select_step = random.randint(max(1, max_steps - 1), max_steps) if not containing_max_tries else max_steps
            all_splits = all_text.split("<|im_start|>assistant\n")
            input_text = "<|im_start|>assistant\n".join(all_splits[:select_step])
            input_text = "\n<|im_end|>".join(input_text.split("<|im_end|>")[:-1])
            try:
                output_text = all_splits[select_step].split("<|im_end|>")[0]
            except IndexError:
                raise Exception(f"{all_text=}, {select_step=}, {max_steps=}, {len(all_splits)=}")
            if is_dapo and not is_cot:
                input_text = raw_input_text
                output_text = raw_output_text

        ret: Dict[str, Any] = {
            "seq_len": sample["timeseries_info"]["seq_len"],
            "input": input_text,
            "output": output_text,
            "label": json.dumps(label, ensure_ascii=False),
            "additional_info": self._build_additional_info(sample, human_intent_nl=resolved_human_intent_nl),
        }
        if use_tool or query_memory:
            tool_spec = []
            if use_tool:
                tool_spec.append({
                    "type": "function",
                    "function": {
                        "name": "slice_timeseries",
                        "description": "Slice the provided time series by [start, end] (inclusive). You should call this tool during thinking to further refine your anomaly detection.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "anomaly": {
                                    "type": "integer",
                                    "description": "The index of the anomaly being analyzed"
                                },
                                "start": {
                                    "type": "integer",
                                    "description": "The start index of the timeseries slice"
                                },
                                "end": {
                                    "type": "integer",
                                    "description": "The end index of the timeseries slice"
                                }
                            },
                            "required": ["start", "end"]
                        }
                    }
                })
            if query_memory:
                tool_spec.append({
                    "type": "function",
                    "function": {
                        "name": "query_memory",
                        "description": "Query history anomalies by anomaly type and recent window.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "anomaly": {
                                    "type": "integer",
                                    "description": "The index of the anomaly being analyzed"
                                },
                                "pattern": {
                                    "type": "string",
                                    "description": "The anomaly pattern type to query from history"
                                },
                                "start": {
                                    "type": "integer",
                                    "description": "Start index of current anomaly"
                                },
                                "window_size": {
                                    "type": "integer",
                                    "description": "Look-back window size in points"
                                }
                            },
                            "required": ["anomaly", "pattern", "start", "window_size"]
                        }
                    }
                })
            ret["tools"] = json.dumps(tool_spec, ensure_ascii=False)
        if reject:
            reject_text = self.generate_output_text(sample, is_cot=is_cot, is_reject=True, include_data_anomalies=include_data_anomalies)
            ret = {
                "input": input_text,
                "chosen": output_text,
                "rejected": reject_text,
                "label": json.dumps(label, ensure_ascii=False),
                "additional_info": self._build_additional_info(sample, human_intent_nl=resolved_human_intent_nl),
            }
        if is_dapo and query_memory:
            ts_info = sample.get("timeseries_info", {}) if isinstance(sample, dict) else {}
            addi = ts_info.get("additional_info", {}) if isinstance(ts_info, dict) else {}
            hist = addi.get("history_anomalies") if isinstance(addi, dict) else None
            ret["history_anomalies"] = json.dumps(hist, ensure_ascii=False) if isinstance(hist, list) else "[]"
            ret["query_memory_count"] = query_memory_count
        if is_ts_modal:
            num_ts = input_text.count("<ts><ts/>")
            ret["timeseries"] = timeseries_data[:num_ts]
        return ret
    

    def batch_generate_output_text(
        self,
        samples: List[Dict[str, Any]],
        llm_client: Any,
        is_cot: bool = True,
        is_reject: bool = False,
        use_tool: bool = False,
        query_memory: bool = False,
        include_data_anomalies: bool = False,
        max_retries: int = 2,
    ) -> List[str]:
        """                           

           
          1.    is_cot   True LLM                  
             [Time Series Characteristics]\n
             [Human Intent (Structured)]\n
             [Anomaly Detection Results]\n
             [Intent Application]\n
             [Final Results]\n
          2.         seasonality trend noise           frequency              
          3.             start_pos end_pos magnitude pattern_type        "No obvious anomalies detected."         
          4.              human intent rule          /      
          5.                                  
          6.            LLM                                  
        """
        if not is_cot:
            return [self.generate_output_text(s, is_cot=False, is_reject=is_reject, include_data_anomalies=include_data_anomalies) for s in samples]

        structure_lines = [
            "[Time Series Characteristics]",
            "[Human Intent (Structured)]",
            "[Anomaly Detection Results]",
            "[Intent Application]",
            "[Final Results]",
        ]

        prompts = self.build_output_text_prompts(samples=samples, is_reject=is_reject)

        def _call_llm(batch_prompts: List[str]) -> List[str]:
            if getattr(llm_client, 'engine', None) in ('ark', 'gpt'):
                return llm_client.llm_batch_generate_parallel(batch_prompts)
            return llm_client.llm_batch_generate(batch_prompts, use_chat_template=True)

        raw_answers = _call_llm(prompts)
        outputs: List[Optional[str]] = [None] * len(samples)

        def has_all_headers(text: str) -> bool:
            return all(h in text for h in structure_lines)

        def _normalize_verbatim(text: str) -> str:
            # Normalize only leading/trailing whitespace and per-line trailing spaces.
            return "\n".join([ln.rstrip() for ln in (text or "").strip().splitlines()]).strip()

        def _extract_section(text: str, header: str, next_headers: List[str]) -> str:
            if not text:
                return ""
            start = text.find(header)
            if start < 0:
                return ""
            after = start + len(header)
            end_candidates = []
            for nh in next_headers:
                p = text.find(nh, after)
                if p >= 0:
                    end_candidates.append(p)
            end = min(end_candidates) if end_candidates else len(text)
            return text[after:end].strip("\n")

        expected_structured_list: List[str] = []
        for sample in samples:
            working_sample = sample if not is_reject else sample.get("rejected", sample)
            human_intent = self._get_human_intent(working_sample)
            expected_structured_list.append(
                self.text_pool.generate_human_intent_structured(
                    human_intent,
                    working_sample.get("timeseries_info", {}),
                )
            )

        def _is_valid_output(idx: int, text: str) -> bool:
            if not has_all_headers(text):
                return False
            expected = expected_structured_list[idx]
            extracted = _extract_section(
                text,
                "[Human Intent (Structured)]",
                ["[Anomaly Detection Results]", "[Intent Application]", "[Final Results]"],
            )
            return _normalize_verbatim(extracted) == _normalize_verbatim(expected)

        invalid_indices: List[int] = []
        for idx in range(len(samples)):
            raw_text = raw_answers[idx].strip() if idx < len(raw_answers) else ""
            if not _is_valid_output(idx, raw_text):
                print(f"[WARNING] Invalid LLM output in first attempt for sample {idx} (missing headers or non-verbatim structured intent).")
                invalid_indices.append(idx)
                continue
            sample = samples[idx]
            working_sample = sample if not is_reject else sample.get("rejected", sample)
            detected_anomalies = working_sample.get("anomalies_detected", [])
            final_any = working_sample.get("anomalies_final")
            final_anomalies = final_any if isinstance(final_any, list) else detected_anomalies
            output_text = raw_text.strip()
            print(f"[INFO] LLM output accepted for sample {idx} on initial attempt.\n")
            anomalies_positions = final_anomalies
            output_text += "\n</think>\n<answer>\n"
            if len(anomalies_positions) == 0:
                output_text += "[]"
            else:
                positions = [(
                    a["start_pos"], a["end_pos"], a["pattern_type"], float(f"{a['magnitude']:.3g}")
                ) for a in anomalies_positions]
                output_text += str(positions)
            output_text += "\n</answer>"
            outputs[idx] = output_text

        attempt = 1
        while invalid_indices and attempt <= max_retries:
            print(f"\n[RETRY] Attempt {attempt} for {len(invalid_indices)} invalid samples: {invalid_indices}")
            retry_prompts = []
            for i in invalid_indices:
                retry_prompts.append(
                    prompts[i]
                    + "Reminder: Ensure ALL required headers appear exactly once and in order. "
                    + "The [Human Intent (Structured)] section MUST copy human_intent_structured verbatim (no paraphrase).\n"
                )
            retry_answers = _call_llm(retry_prompts)

            still_invalid: List[int] = []
            for local_idx, sample_global_idx in enumerate(invalid_indices):
                raw_text = retry_answers[local_idx].strip() if local_idx < len(retry_answers) else ""
                if not _is_valid_output(sample_global_idx, raw_text):
                    print(
                        f"[RETRY][WARNING] Sample {sample_global_idx} still invalid on attempt {attempt} "
                        f"(missing headers or non-verbatim structured intent)."
                    )
                    still_invalid.append(sample_global_idx)
                    continue
                sample = samples[sample_global_idx]
                working_sample = sample if not is_reject else sample.get("rejected", sample)
                detected_anomalies = working_sample.get("anomalies_detected", [])
                final_any = working_sample.get("anomalies_final")
                final_anomalies = final_any if isinstance(final_any, list) else detected_anomalies
                output_text = raw_text.strip()
                print(f"[RETRY][INFO] Sample {sample_global_idx} accepted on attempt {attempt}.")
                anomalies_positions = final_anomalies
                output_text += "\n</think>\n<answer>\n"
                if len(anomalies_positions) == 0:
                    output_text += "[]"
                else:
                    positions = [(
                        a["start_pos"], a["end_pos"], a["pattern_type"], float(f"{a['magnitude']:.3g}")
                    ) for a in anomalies_positions]
                    output_text += str(positions)
                output_text += "\n</answer>"
                outputs[sample_global_idx] = output_text
            invalid_indices = still_invalid
            attempt += 1

        if invalid_indices:
            raise RuntimeError(
                f"LLM failed to generate valid output text after {max_retries + 1} attempts. "
                f"Invalid indices: {invalid_indices}."
            )

        return [o if o is not None else "" for o in outputs]

    def build_output_text_prompts(
        self,
        samples: List[Dict[str, Any]],
        is_reject: bool = False,
    ) -> List[str]:
        """Build prompts for generating full reasoning output text (CoT sections)."""
        intro = (
            "You are a time series anomaly detection expert. Produce reasoning steps with diversified wording while keeping facts accurate. "
            "Do not hallucinate anomalies or rules."
        )
        char_req = "Include descriptions of trend, seasonality, noise, and frequency (if periodic; otherwise state no clear periodicity)."
        anomaly_req = "For each anomaly, describe start_pos, end_pos, magnitude, and pattern_type. If none, say something like 'No obvious anomalies detected.'"
        intent_req = "Describe each intent rule: its logic, rationale, and which anomalies it excludes or preserves."
        final_req = "Summarize final anomalies after applying the intent preference with their key info; if none remain, state that clearly."
        intent_structured_req = (
            "In the [Human Intent (Structured)] section, copy the provided human_intent_structured text verbatim (no paraphrase)."
        )

        structure_lines = [
            "[Time Series Characteristics]",
            "[Human Intent (Structured)]",
            "[Anomaly Detection Results]",
            "[Intent Application]",
            "[Final Results]",
        ]
        structure_preview = "\n".join(structure_lines)

        prompts: List[str] = []
        for sample in samples:
            working_sample = sample if not is_reject else sample.get("rejected", sample)
            characteristics = working_sample["timeseries_info"].get("characteristics", {})
            detected_anomalies = working_sample.get("anomalies_detected", [])
            human_intent = self._get_human_intent(working_sample)
            human_intent_sanitized = self._sanitize_human_intent_for_llm(human_intent)
            human_intent_structured = self.text_pool.generate_human_intent_structured(
                human_intent,
                working_sample.get("timeseries_info", {}),
            )
            final_anomalies_any = working_sample.get("anomalies_final")
            final_anomalies = final_anomalies_any if isinstance(final_anomalies_any, list) else detected_anomalies
            ctx = {
                "characteristics": characteristics,
                "human_intent": human_intent_sanitized,
                "human_intent_structured": human_intent_structured,
                "detected_anomalies": [
                    {
                        "start_pos": anomaly["start_pos"],
                        "end_pos": anomaly["end_pos"],
                        "magnitude": float(f"{anomaly['magnitude']:.3g}"),
                        "pattern_type": anomaly["pattern_type"],
                    }
                    for anomaly in detected_anomalies
                ],
                "final_anomalies": [
                    {
                        "start_pos": anomaly["start_pos"],
                        "end_pos": anomaly["end_pos"],
                        "magnitude": float(f"{anomaly['magnitude']:.3g}"),
                        "pattern_type": anomaly["pattern_type"],
                    }
                    for anomaly in final_anomalies
                ],
            }
            prompt = (
                f"{intro}\n\nContext JSON:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
                f"Output Format Requirements:\n"
                f"1. Use exactly these section headers in order (no extras):\n{structure_preview}\n"
                f"2. {char_req}\n"
                f"2.5. {intent_structured_req}\n"
                f"3. {anomaly_req}\n"
                f"4. {intent_req}\n"
                f"5. {final_req}\n"
                "Do NOT output JSON. Do NOT wrap with Markdown fences. Just plain text sections.\n"
                "Avoid verbatim repetition across anomalies; vary phrasing concisely.\n"
            )
            prompts.append(prompt)
        return prompts

    def batch_generate_qa_pairs_llm(
        self,
        samples: List[Dict[str, Any]],
        llm_client: Any,
        is_ts_modal: bool = True,
        is_cot: bool = True,
        reject: Optional[bool] = None,
        temperature: float = 0.7,
        is_dapo: bool = False,
        use_tool: bool = False,
        query_memory: bool = False,
        include_data_anomalies: bool = False,
        using_llm_input: bool = True,
        using_llm_output: bool = True,
    ) -> List[Dict[str, Any]]:
        timeseries_modal_list: List[List[List[float]]] = [self.format_timeseries_for_llm(s["timeseries_data"]) for s in samples]

        # 1) input side: choose whether to use LLM for natural-language intent
        human_intents = [self._get_human_intent(s) for s in samples]
        null_human_intent_idx_list = [i for i, hi in enumerate(human_intents) if len(hi["rules"]) == 0]
        valid_human_intents = [human_intents[idx] for idx in range(len(human_intents)) if idx not in null_human_intent_idx_list]
        if valid_human_intents:
            if using_llm_input:
                ts_infos = [s.get("timeseries_info", {}) for s in samples]
                valid_ts_infos = [ts_infos[idx] for idx in range(len(ts_infos)) if idx not in null_human_intent_idx_list]
                valid_intent_nl_list = self.batch_generate_human_intent_nl(valid_human_intents, llm_client=llm_client, ts_infos=valid_ts_infos)
            else:
                valid_intent_nl_list = [self.generate_human_intent_nl_fallback(hi) for hi in valid_human_intents]
        else:
            valid_intent_nl_list = []
        intent_nl_list = []
        vi_idx = 0
        for i in range(len(human_intents)):
            if i in null_human_intent_idx_list:
                intent_nl_list.append(None)
            else:
                intent_nl_list.append(valid_intent_nl_list[vi_idx])
                vi_idx += 1

        # If we are NOT using LLM for output and tool-mode is enabled, the easiest way to
        # keep the tool-dialogue input consistent is to generate full QA pairs locally.
        if (use_tool or query_memory) and (not using_llm_output):
            qa_pairs: List[Dict[str, Any]] = []
            for i, sample in enumerate(samples):
                qa = self.create_qa_pair(
                    sample,
                    is_ts_modal=is_ts_modal,
                    is_cot=is_cot,
                    reject=reject,
                    is_dapo=is_dapo,
                    use_tool=use_tool,
                    query_memory=query_memory,
                    include_data_anomalies=include_data_anomalies,
                    human_intent_nl=intent_nl_list[i],
                )
                qa_pairs.append(qa)
            return qa_pairs

        # 2)    input use_tool       create_qa_pair    tool        
        input_texts: List[str] = []
        for i, s in enumerate(samples):
            if use_tool or query_memory:
                input_texts.append(
                    self.create_qa_pair(
                        s,
                        is_ts_modal=is_ts_modal,
                        is_cot=is_cot,
                        reject=None,
                        is_dapo=is_dapo,
                        use_tool=use_tool,
                        query_memory=query_memory,
                        include_data_anomalies=include_data_anomalies,
                        human_intent_nl=intent_nl_list[i],
                    )["input"]
                )
            else:
                input_texts.append(
                    self.generate_input_text(
                        s,
                        is_ts_modal=is_ts_modal,
                        is_cot=is_cot,
                        timeseries_data=timeseries_modal_list[i],
                        is_dapo=is_dapo,
                        human_intent_nl=intent_nl_list[i],
                    )
                )

        # 3) output side: choose whether to use LLM to generate the reasoning text
        if using_llm_output:
            chosen_outputs = self.batch_generate_output_text(
                samples,
                llm_client=llm_client,
                is_cot=is_cot,
                is_reject=False,
                use_tool=use_tool,
                query_memory=query_memory,
                include_data_anomalies=include_data_anomalies,
            )
        else:
            chosen_outputs = [self.generate_output_text(s, is_cot=is_cot, is_reject=False, include_data_anomalies=include_data_anomalies) for s in samples]

        rejected_outputs: List[str] = []
        if reject:
            if using_llm_output:
                rejected_outputs = self.batch_generate_output_text(
                    samples,
                    llm_client=llm_client,
                    is_cot=is_cot,
                    is_reject=True,
                    include_data_anomalies=include_data_anomalies,
                )
            else:
                rejected_outputs = [self.generate_output_text(s, is_cot=is_cot, is_reject=True, include_data_anomalies=include_data_anomalies) for s in samples]

        qa_pairs: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            anomalies_any = sample.get("anomalies_final")
            anomalies = anomalies_any if isinstance(anomalies_any, list) else sample.get("anomalies_detected", [])
            label = [(
                a["start_pos"],
                a["end_pos"],
                a["pattern_type"],
                float(f"{a['magnitude']:.3g}")
            ) for a in anomalies]
            if reject:
                qa: Dict[str, Any] = {
                    "input": input_texts[i],
                    "chosen": chosen_outputs[i],
                    "rejected": rejected_outputs[i] if i < len(rejected_outputs) else "",
                    "label": json.dumps(label, ensure_ascii=False),
                    "additional_info": self._build_additional_info(sample, human_intent_nl=intent_nl_list[i]),
                }
            else:
                qa: Dict[str, Any] = {
                    "seq_len": sample["timeseries_info"]["seq_len"],
                    "input": input_texts[i],
                    "output": chosen_outputs[i],
                    "label": json.dumps(label, ensure_ascii=False),
                    "additional_info": self._build_additional_info(sample, human_intent_nl=intent_nl_list[i]),
                }
            if is_dapo and query_memory:
                ts_info = sample.get("timeseries_info", {}) if isinstance(sample, dict) else {}
                addi = ts_info.get("additional_info", {}) if isinstance(ts_info, dict) else {}
                hist = addi.get("history_anomalies") if isinstance(addi, dict) else None
                qa["history_anomalies"] = json.dumps(hist, ensure_ascii=False) if isinstance(hist, list) else "[]"
            if is_ts_modal:
                qa["timeseries"] = timeseries_modal_list[i]
            qa_pairs.append(qa)
        return qa_pairs
    
    def generate_dataset(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        num_samples: Optional[int] = None,
        is_ts_modal: bool = True,
        is_cot: bool = True,
        using_llm_input: Optional[bool] = None,
        using_llm_output: Optional[bool] = None,
        llm_client: Any = None,
        reject=None,
        temperature: float = 0.7,
        is_dapo: bool = False,
        use_tool: bool = False,
        query_memory: bool = False,
        include_data_anomalies: bool = False,
        dry_run_prompts: bool = False,
        dry_run_num_print: int = 3,
    ) -> str:
        """     SFT   """
        if output_file is None:
            raise ValueError("output_file must be provided")
        print(f"Loading original data...")
        samples = self.load_tsad_dataset(input_file)
        if num_samples:
            samples = samples[:num_samples]
        print(f"Starting to process {len(samples)} samples...")
        self.text_pool = TextPool()

        # Default per-side LLM usage flags when not provided
        if using_llm_input is None:
            using_llm_input = False
        if using_llm_output is None:
            using_llm_output = False
        using_llm_input = bool(using_llm_input)
        using_llm_output = bool(using_llm_output)

        if dry_run_prompts:
            preview_samples = samples[: max(1, int(dry_run_num_print))]

            if using_llm_input:
                human_intents = [self._get_human_intent(s) for s in preview_samples]
                ts_infos = [s.get("timeseries_info", {}) for s in preview_samples]
                intent_prompts = self.build_human_intent_nl_prompts(human_intents, ts_infos=ts_infos)
                print("\n========== DRY RUN: Human Intent NL Prompts ==========")
                for i, p in enumerate(intent_prompts, 1):
                    print(f"\n----- [Intent Prompt {i}/{len(intent_prompts)}] -----\n{p}")
            else:
                print("\n========== DRY RUN: Human Intent NL Prompts ==========")
                print("[SKIP] using_llm_input=False")

            if using_llm_output:
                output_prompts = self.build_output_text_prompts(preview_samples, is_reject=False)
                print("\n========== DRY RUN: Output Text Prompts (chosen) ==========")
                for i, p in enumerate(output_prompts, 1):
                    print(f"\n----- [Output Prompt {i}/{len(output_prompts)}] -----\n{p}")

                if reject:
                    output_prompts_rej = self.build_output_text_prompts(preview_samples, is_reject=True)
                    print("\n========== DRY RUN: Output Text Prompts (rejected) ==========")
                    for i, p in enumerate(output_prompts_rej, 1):
                        print(f"\n----- [Rejected Output Prompt {i}/{len(output_prompts_rej)}] -----\n{p}")
            else:
                print("\n========== DRY RUN: Output Text Prompts (chosen) ==========")
                print("[SKIP] using_llm_output=False")

            print("\n[DRY RUN] Prompts printed. Skipping LLM calls and dataset writing.")
            return str(output_file)

        if using_llm_input or using_llm_output:
            if llm_client is None:
                raise ValueError("llm_client must be provided when using_llm_input=True or using_llm_output=True")
            qa_pairs = self.batch_generate_qa_pairs_llm(
                samples,
                llm_client=llm_client,
                is_ts_modal=is_ts_modal,
                is_cot=is_cot,
                reject=reject,
                temperature=temperature,
                use_tool=use_tool,
                query_memory=query_memory,
                include_data_anomalies=include_data_anomalies,
                is_dapo=is_dapo,
                using_llm_input=using_llm_input,
                using_llm_output=using_llm_output,
            )
        else:
            qa_pairs = []
            for sample in tqdm(samples, desc="Generating QA pairs"):
                qa_pair = self.create_qa_pair(sample, is_ts_modal=is_ts_modal, is_cot=is_cot, reject=reject, is_dapo=is_dapo, use_tool=use_tool, query_memory=query_memory, include_data_anomalies=include_data_anomalies)
                qa_pairs.append(qa_pair)
                # try:
                #     qa_pair = self.create_qa_pair(sample, is_ts_modal=is_ts_modal, is_cot=is_cot, reject=reject, is_dapo=is_dapo, use_tool=use_tool)
                #     qa_pairs.append(qa_pair)
                # except Exception as e:
                #     print(f"Error processing sample: {e}")
                #     continue
        
        #    JSONL  
        print(f"Saving {len(qa_pairs)} QA pairs to {output_file}")
        output_path = Path(output_file)
        additional_info_dir = output_path.parent / "additional_info"
        additional_info_dir.mkdir(exist_ok=True)
        additional_info_file = additional_info_dir / f"{output_path.stem}{output_path.suffix or '.jsonl'}"

        with open(output_path, 'w', encoding='utf-8') as main_f, open(additional_info_file, 'w', encoding='utf-8') as add_f:
            for qa_pair in qa_pairs:
                record = dict(qa_pair)
                additional_info = record.pop("additional_info", None)
                main_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                add_f.write(json.dumps(additional_info if additional_info is not None else {}, ensure_ascii=False) + '\n')
        print(f"Saved additional_info to {additional_info_file}")
        return str(output_file)
