"""Generate synthetic base time series with configurable characteristics."""

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from .util import generate_statistics, save_ts_json, TimeSeriesInfo, SAMPLING_INTERVAL_CHOICES



class TimeSeriesGenerator:
    """Generate normal time-series values and metadata."""
    
    def __init__(self):
        self.instance_names = {
            "containers": [
                "container-001", "container-002", "container-003", "pod-web-001", "pod-api-002",
                "container-db-003", "frontend-pod-001", "backend-pod-002", "proxy-pod-001",
                "cache-pod-001", "db-pod-001", "search-node-001", "queue-node-001",
                "monitor-node-001", "dashboard-node-001", "ci-node-001", "registry-node-001",
                "config-node-001", "secret-node-001"
            ],
            "services": [
                "service-001", "service-002", "service-003", "service-004", "service-005",
                "service-006", "service-007", "service-008", "service-009", "service-010",
                "service-011", "service-012", "service-013", "service-014", "service-015",
                "service-016", "service-017", "service-018", "service-019"
            ],
            "hosts": [
                "host-001", "host-002", "host-003", "host-004", "host-005", "host-006",
                "host-007", "host-008", "host-009", "host-010", "host-011", "host-012",
                "host-013", "host-014", "host-015", "host-016", "host-017", "host-018",
                "host-019", "host-020", "host-021"
            ],
            "clusters": [
                "cluster-001", "cluster-002", "cluster-003", "cluster-004", "cluster-005",
                "cluster-006", "cluster-007", "cluster-008", "cluster-009", "cluster-010",
                "cluster-011", "cluster-012", "cluster-013", "cluster-014"
            ],
            "databases": [
                "database-001", "database-002", "database-003", "database-004", "database-005",
                "database-006", "database-007", "database-008", "database-009", "database-010",
                "database-011", "database-012", "database-013", "database-014", "database-015"
            ],
            "applications": [
                "application-001", "application-002", "application-003", "application-004",
                "application-005", "application-006", "application-007", "application-008",
                "application-009", "application-010", "application-011", "application-012",
                "application-013", "application-014", "application-015"
            ],
            "microservices": [
                "microservice-001", "microservice-002", "microservice-003", "microservice-004",
                "microservice-005", "microservice-006", "microservice-007", "microservice-008",
                "microservice-009", "microservice-010", "microservice-011", "microservice-012",
                "microservice-013", "microservice-014", "microservice-015"
            ],
            "infrastructure": [
                "infra-001", "infra-002", "infra-003", "infra-004", "infra-005", "infra-006",
                "infra-007", "infra-008", "infra-009", "infra-010", "infra-011", "infra-012",
                "infra-013", "infra-014", "infra-015"
            ]
        }
        
        self.metric_names = {
            "LatencySecond": [
                # API     
                "api_response_time", "http_response_latency", "grpc_call_duration", "rest_api_latency",
                "db_query_latency", "sql_execution_time", "transaction_commit_time", "connection_establish_time",
                "network_round_trip_time", "tcp_handshake_duration", "dns_lookup_time", "ssl_handshake_time",
                "cache_access_time", "disk_read_latency", "file_write_duration", "backup_restore_time",
                "auth_validation_time", "token_generation_time", "image_processing_duration", "video_encoding_time"
            ],
            "Count": [
                "http_requests_total", "api_calls_count", "page_views_total", "user_sessions_count",
                "orders_created_total", "payments_processed_total", "user_registrations_total", "login_attempts_total",
                "database_connections_total", "cache_hits_total", "file_uploads_total", "email_sent_total",
                "messages_published_total", "events_processed_total", "notifications_sent_total", "sms_delivered_total",
                "containers_started_total", "pods_created_total", "jobs_executed_total", "deployments_total"
            ],
            "ErrorCount": [
                # HTTP  
                "http_4xx_errors_total", "http_5xx_errors_total", "bad_request_errors_total", "internal_server_errors_total",
                "database_errors_total", "connection_timeout_errors_total", "memory_allocation_errors_total", "disk_full_errors_total",
                "authentication_failures_total", "authorization_failures_total", "validation_errors_total", "business_logic_errors_total",
                "network_timeout_errors_total", "dns_resolution_errors_total", "ssl_certificate_errors_total", "api_call_failures_total",
                "payment_gateway_errors_total", "third_party_api_errors_total", "message_delivery_failures_total", "backup_failures_total"
            ],
            "SuccessRate": [
                # API      
                "api_success_rate", "http_success_rate", "service_availability_rate", "endpoint_success_rate",
                "payment_success_rate", "order_completion_rate", "user_login_success_rate", "registration_success_rate",
                "database_availability_rate", "cache_hit_rate", "backup_success_rate", "deployment_success_rate",
                "message_delivery_rate", "email_delivery_success_rate", "sms_delivery_success_rate", "notification_success_rate",
                "server_uptime_rate", "service_health_score", "cluster_availability_rate", "node_ready_rate"
            ],
            "ResourceUtilizationRate": [
                "cpu_utilization_percent", "memory_usage_percent", "disk_usage_percent", "gpu_utilization_percent",
                "network_bandwidth_utilization", "connection_pool_usage_percent", "thread_pool_utilization", "worker_pool_usage_percent",
                "storage_capacity_usage", "inode_usage_percent", "swap_usage_percent", "cache_memory_usage_percent",
                "jvm_heap_usage_percent", "database_connection_usage", "queue_capacity_usage", "license_usage_percent",
                "container_cpu_usage_percent", "pod_memory_usage_percent", "cluster_resource_usage", "namespace_quota_usage"
            ],
            "QPS": [
                # API    
                "api_requests_per_second", "http_requests_per_second", "database_queries_per_second", "cache_operations_per_second",
                "transactions_per_second", "orders_per_second", "payments_per_second", "user_actions_per_second",
                "events_processed_per_second", "messages_consumed_per_second", "records_inserted_per_second", "files_processed_per_second",
                "login_attempts_per_second", "search_queries_per_second", "page_renders_per_second", "image_uploads_per_second",
                "container_starts_per_second", "pod_creations_per_second", "service_calls_per_second", "health_checks_per_second"
            ]
        }
        
        #        -     characteristics     
        self.metric_types = {
            "LatencySecond": {
                "range": [0, 10], 
                "baseline": 1.0,
                "value_type": "float",
                "baseline_magnitude": 0.95
            },
            "Count": {
                "range": [0, 1000], 
                "baseline": 100,
                "value_type": "float",
                # "value_type": "int",
                "baseline_magnitude": 95
            },
            "ErrorCount": {
                "range": [0, 100], 
                "baseline": 10,
                "value_type": "float",
                # "value_type": "int",
                "baseline_magnitude": 10
            },
            "SuccessRate": {
                "range": [0, 1], 
                "baseline": 0.95,
                "value_type": "float",
                "baseline_magnitude": 0.05
            },
            "ResourceUtilizationRate": {
                "range": [0, 1], 
                "baseline": 0.6,
                "value_type": "float",
                "baseline_magnitude": 0.3
            },
            "QPS": {
                "range": [0, 5000], 
                "baseline": 1000,
                "value_type": "float",
                # "value_type": "int",
                "baseline_magnitude": 900
            }
        }
        
        #        -   ChatTS           
        self.ts_characteristics = {
            "seasonal": {
                "no periodic fluctuation": 0.7,
                # "irregular pattern": 0.05,
                "sin periodic fluctuation": 0.15,
                "square periodic fluctuation": 0.05,
                "triangle periodic fluctuation": 0.05
            },
            "trend": {
                "decrease": 0.2,
                "increase": 0.2,
                "keep steady": 0.5,
                "random": 0.1,
            },
            "frequency": {
                # "minutely": 0.5,
                # "hourly": 0.5,
                # "daily": 0,
                "minutely": 0,
                "hourly": 0.2,
                "daily": 0.8,
                "weekly": 0,
            },
            "noise": {
                "noisy": 0.2,
                "almost no noise": 0.8
            },
        }

    def generate_base_ts(
        self,
        typical_seq_len: Optional[List[int]],
        typical_prob: float,
        seq_len_range: Tuple[int, int] = (100, 1000),
        sampling_interval: Union[int, str] = 60,
        is_train: bool = False,
        force_baseline_zero: bool = False,
    ) -> Tuple[TimeSeriesInfo, np.ndarray]:
        """          """
        if is_train:
            #         self.ts_characteristics            
            for key in self.ts_characteristics:
                if key in ["frequency"]:
                    continue
                types = list(self.ts_characteristics[key].keys())
                prob = 1.0 / len(types)
                for t in types:
                    self.ts_characteristics[key][t] = prob
            self.ts_characteristics["seasonal"] = {
                "no periodic fluctuation": 0.3,
                # "irregular pattern": 0.05,
                "sin periodic fluctuation": 0.5,
                "square periodic fluctuation": 0.1,
                "triangle periodic fluctuation": 0.1
            }
        # print(f"[DEBUG] {self.ts_characteristics=}")
        ts_info = self.generate_ts_info(
            typical_seq_len,
            typical_prob,
            seq_len_range,
            sampling_interval,
            force_baseline_zero=force_baseline_zero,
        )
        base_series = self.generate_ts_value(ts_info)
        # base_series = validate_and_fix_value_ranges(base_series, ts_info)
        statistics = generate_statistics(base_series)
        ts_info.statistics = statistics
        return ts_info, base_series

    def generate_ts_info(
        self,
        typical_seq_len: Optional[List[int]],
        typical_prob: float,
        seq_len_range: Tuple[int, int] = (100, 1000),
        sampling_interval: Union[int, str] = 60,
        force_baseline_zero: bool = False,
    ) -> TimeSeriesInfo:
        """          """
        metric_type = random.choice(list(self.metric_types.keys()))
        metric_config = self.metric_types[metric_type]
        # 80%     typical_seq_len        
        if typical_seq_len and random.random() < typical_prob:
            seq_len = random.choice(typical_seq_len)
        else:
            seq_len = random.randint(*seq_len_range)
        sampling_interval = random.choice(SAMPLING_INTERVAL_CHOICES) if sampling_interval == 'random' else sampling_interval #  
        characteristics = self._select_ts_characteristics(metric_type, seq_len, sampling_interval)
        
        instance_category = random.choice(list(self.instance_names.keys()))
        instance_name = random.choice(self.instance_names[instance_category])
        metric_name = random.choice(self.metric_names[metric_type])
        full_metric_name = f"{instance_name}##{metric_name}"
        
        if force_baseline_zero:
            baseline = 0 if metric_config["value_type"] == "int" else 0.0
            additional_info = {"force_baseline_zero": True}
        else:
            baseline = metric_config["baseline"] + random.uniform(-metric_config["baseline_magnitude"], metric_config["baseline_magnitude"])
            if metric_config["value_type"] == "int":
                baseline = int(baseline)
            additional_info = {}

        return TimeSeriesInfo(
            metric_type=metric_type,
            metric_name=full_metric_name,
            seq_len=seq_len,
            sampling_interval=sampling_interval,
            normal_range=tuple(metric_config["range"]),
            baseline_value=baseline,
            characteristics=characteristics,
            value_type=metric_config["value_type"],
            magnitude=0.0,
            additional_info=additional_info,
        )

    def generate_ts_value(self, ts_info: TimeSeriesInfo) -> np.ndarray:
        """               -   ChatTS     """
        seq_len = ts_info.seq_len
        baseline = ts_info.baseline_value
        value_type = ts_info.value_type
        characteristics = dict(ts_info.characteristics)
        r1, r2 = ts_info.normal_range
        existing_additional_info = dict(ts_info.additional_info or {})
        force_baseline_zero = bool(existing_additional_info.get("force_baseline_zero", False))
        use_sparse_zero_mode = force_baseline_zero and abs(float(baseline)) <= 1e-9 and abs(float(r1)) <= 1e-9

        if use_sparse_zero_mode:
            # baseline=0     “       0”               /    
            if characteristics.get("seasonal") in ["no periodic fluctuation", "irregular pattern"]:
                characteristics["seasonal"] = random.choice(["square periodic fluctuation", "triangle periodic fluctuation"])
            if characteristics.get("frequency") == "no periodicity":
                characteristics["frequency"] = random.choice(["hourly", "daily"])
            characteristics["trend"] = "keep steady"
            characteristics["noise"] = "almost no noise"
            ts_info.characteristics = characteristics
        

        #         ChatTS attribute_pool  
        attribute_pool = {
            "seasonal": {"type": characteristics.get("seasonal", "no periodic fluctuation")},
            "trend": {"type": characteristics.get("trend", "keep steady")},
            "frequency": {"type": characteristics.get("frequency", "no periodicity")},
            "noise": {"type": characteristics.get("noise", "almost no noise")},
            "overall_magnitude": None,
            "overall_bias": baseline,
            "sampling_interval": ts_info.sampling_interval,
        }
        
        #        -        normal_range
        range_span = r2 - r1
        if use_sparse_zero_mode:
            # baseline=0                 0
            attribute_pool["overall_magnitude"] = range_span * random.uniform(0.02, 0.12)
        elif ts_info.metric_type in ["SuccessRate", "ResourceUtilizationRate"]:
            attribute_pool["overall_magnitude"] = range_span * random.uniform(0.1, 0.2)
        elif ts_info.metric_type in ["LatencySecond"]:
            attribute_pool["overall_magnitude"] = range_span * random.uniform(0.1, 0.3)
        else:
            attribute_pool["overall_magnitude"] = range_span * random.uniform(0.3, 0.8)
        attribute_pool["overall_magnitude"] = min(attribute_pool["overall_magnitude"], (ts_info.normal_range[1]-baseline)*0.8)
        if not use_sparse_zero_mode:
            attribute_pool["overall_magnitude"] = min(attribute_pool["overall_magnitude"], (baseline-ts_info.normal_range[0])*0.8)
        if ts_info.value_type == "int":
            attribute_pool["overall_magnitude"] = max(1, int(attribute_pool["overall_magnitude"])+1)
        if use_sparse_zero_mode:
            series, magnitude, additional_info = self._generate_sparse_baseline_zero_timeseries(
                attribute_pool=attribute_pool,
                seq_len=seq_len,
                normal_range=ts_info.normal_range,
                value_type=ts_info.value_type,
            )
        else:
            series, magnitude, additional_info = self._generate_timeseries(attribute_pool, seq_len)

        merged_additional = dict(existing_additional_info)
        merged_additional.update(additional_info)
        ts_info.additional_info = merged_additional
        ts_info.magnitude = magnitude if magnitude>0 else attribute_pool["overall_magnitude"]*0.05
        if value_type == "int":
            series = np.round(series)
            ts_info.magnitude = max(ts_info.magnitude, max(1, int(ts_info.magnitude)))
        series = np.clip(series, ts_info.normal_range[0], ts_info.normal_range[1])
        
        return series

    def _select_ts_characteristics(self, metric_type: str, seq_len: int, sampling_interval: int) -> Dict[str, str]:
        """      """
        characteristics = {}
        total_duration = seq_len * sampling_interval #  
        
        seasonal_types = list(self.ts_characteristics["seasonal"].keys())
        seasonal_probs = list(self.ts_characteristics["seasonal"].values())
        seasonal_type = ""
        while True:
            seasonal_type = np.random.choice(seasonal_types, p=np.array(seasonal_probs)/sum(seasonal_probs))
            if self.metric_types[metric_type]["value_type"] == "int" and seasonal_type in ["sin periodic fluctuation", "triangle periodic fluctuation"]:
                continue
            if metric_type == "ErrorCount" and seasonal_type in ["irregular pattern"]:
                continue
            break
        characteristics["seasonal"] = seasonal_type
        
        if  characteristics["seasonal"] not in ["no periodic fluctuation", "irregular pattern"]:
            frequency = ""
            num_attempts = 0
            while num_attempts < 100:
                num_attempts += 1
                freq_types = list(self.ts_characteristics["frequency"].keys())
                freq_probs = list(self.ts_characteristics["frequency"].values())
                frequency = np.random.choice(freq_types, p=freq_probs)
                if frequency == "weekly":
                    if total_duration < 3600 * 24 * 14 or sampling_interval > 3600 * 24:
                        continue
                if frequency == "daily":
                    if total_duration < 3600 * 24 * 2 or sampling_interval > 1200:
                        continue
                if frequency == "hourly":
                    if total_duration < 3600 * 2 or sampling_interval > 60:
                        continue
                if frequency == "minutely":
                    if total_duration < 60 * 2 or sampling_interval > 15:
                        continue
                break
            if num_attempts >= 100:
                characteristics["frequency"] = "no periodicity"
                characteristics["seasonal"] = "no periodic fluctuation"
            else:
                characteristics["frequency"] = frequency
        else:
            characteristics["frequency"] = "no periodicity"

        if characteristics["seasonal"] == "irregular pattern":
            characteristics["trend"] = "keep steady"
            characteristics["noise"] = "almost no noise"
            return characteristics
        trend_types = list(self.ts_characteristics["trend"].keys())
        trend_probs = list(self.ts_characteristics["trend"].values())
        characteristics["trend"] = np.random.choice(trend_types, p=np.array(trend_probs)/sum(trend_probs))
        
        noise_types = list(self.ts_characteristics["noise"].keys())
        noise_probs = list(self.ts_characteristics["noise"].values())
        characteristics["noise"] = np.random.choice(noise_types, p=np.array(noise_probs)/sum(noise_probs))
        
        return characteristics

    def _generate_timeseries(self, attribute_pool: Dict[str, Any], seq_len: int) -> np.ndarray:
        """      """
        y = np.zeros(seq_len)
        overall_magnitude = attribute_pool["overall_magnitude"]
        overall_bias = attribute_pool["overall_bias"]
        additional_info = {}
        
        # 1.        
        magnitude_season, season = self._generate_seasonal_pattern(attribute_pool, overall_magnitude, seq_len, additional_info)
        y += season
        # 2.     
        magnitude_trend, y = self._generate_trend_pattern(attribute_pool, y, overall_magnitude, overall_bias, seq_len, additional_info)
        # 3.     
        magnitude_noise, noise = self._generate_noise_pattern(attribute_pool, y, max(magnitude_season, magnitude_trend / 2, attribute_pool["overall_magnitude"]*0.05), seq_len, additional_info)
        y += noise

        magnitude = max(magnitude_season, magnitude_trend / 2, magnitude_noise)
        return y, magnitude, additional_info

    def _generate_sparse_baseline_zero_timeseries(
        self,
        attribute_pool: Dict[str, Any],
        seq_len: int,
        normal_range: Tuple[float, float],
        value_type: str,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """  baseline=0 case   “     0 +       ”      """
        sampling_interval = max(1, int(attribute_pool.get("sampling_interval", 60) or 60))
        frequency_type = str((attribute_pool.get("frequency") or {}).get("type", "hourly"))
        seasonal_type = str((attribute_pool.get("seasonal") or {}).get("type", "square periodic fluctuation"))

        if frequency_type == "daily":
            period_raw = 24 * 3600 // sampling_interval
        elif frequency_type == "weekly":
            period_raw = 24 * 3600 * 7 // sampling_interval
        elif frequency_type == "hourly":
            period_raw = 3600 // sampling_interval
        else:
            period_raw = max(12, seq_len // 4)

        max_period = max(16, seq_len // 2)
        period = int(max(8, min(max(1, int(period_raw)), max_period)))

        pulse_type = "triangle" if "triangle" in seasonal_type else "square"
        upper = float(normal_range[1])
        amplitude = max(1e-6, upper * random.uniform(0.02, 0.12))
        if value_type == "int":
            amplitude = max(1.0, float(int(round(amplitude))))

        duty = random.uniform(0.05, 0.12)
        width = max(2, int(round(period * duty)))

        data = np.zeros(seq_len, dtype=float)
        for cycle_start in range(0, seq_len, period):
            local_max_start = min(seq_len - 1, cycle_start + max(0, period - width))
            if local_max_start <= cycle_start:
                start = cycle_start
            else:
                start = random.randint(cycle_start, local_max_start)
            end = min(seq_len, start + width)
            if end <= start:
                continue

            pulse_amp = amplitude * random.uniform(0.7, 1.0)
            if pulse_type == "square":
                data[start:end] = pulse_amp
            else:
                pulse_len = end - start
                if pulse_len <= 1:
                    data[start:end] = pulse_amp
                else:
                    up_len = max(1, pulse_len // 2)
                    down_len = pulse_len - up_len
                    up = np.linspace(0.0, pulse_amp, up_len, endpoint=False)
                    down = np.linspace(pulse_amp, 0.0, down_len, endpoint=True)
                    data[start:end] = np.concatenate([up, down])[:pulse_len]

        active_idx = np.where(data > 0)[0]
        if active_idx.size == 0:
            data[random.randint(0, seq_len - 1)] = amplitude
            active_idx = np.where(data > 0)[0]

        jitter_std = amplitude * random.uniform(0.01, 0.05)
        if active_idx.size > 0 and jitter_std > 0:
            data[active_idx] += np.random.normal(0.0, jitter_std, size=active_idx.size)

        data = np.clip(data, normal_range[0], normal_range[1])
        additional_info = {
            "baseline_zero_wave": pulse_type,
            "baseline_zero_period": int(period),
            "baseline_zero_duty": float(duty),
            "baseline_zero_zero_ratio": float(np.mean(data <= 1e-9)),
        }
        magnitude = float(np.max(data) - np.min(data))
        return data, magnitude, additional_info

    def _generate_seasonal_pattern(self, attribute_pool: Dict[str, Any], overall_magnitude: float, seq_len: int, additional_info: Dict[str, Any]) -> np.ndarray:
        """       """
        seasonal_type = attribute_pool["seasonal"]["type"]
        sampling_interval = attribute_pool["sampling_interval"]
        
        if "no periodic fluctuation" in seasonal_type:
            return 0, np.zeros(seq_len)
        if "irregular pattern" in seasonal_type:
            #   pattern                 pattern       
            pattern_length = random.randint(10, min(100, seq_len//5))
            additional_info["irregular_pattern_length"] = pattern_length
            additional_info["irregular_pattern_start_list"] = []
            pattern_magnitude = random.uniform(0.3, 0.7) * overall_magnitude
            # pattern = pattern_magnitude * np.sin(2 * np.pi * np.arange(pattern_length) / pattern_length + random.uniform(0, 2 * np.pi))
            # pattern = pattern / (np.max(pattern) - np.min(pattern)) * pattern_magnitude
            # pattern = pattern - np.mean(pattern)
            pattern = self._generate_sin_noise(pattern_magnitude, pattern_length*25, random.randint(2,7), is_zero_normalize=False)
            start_idx = random.randint(0, len(pattern)-pattern_length)
            pattern = pattern[start_idx:start_idx+pattern_length]
            data = np.zeros(seq_len)
            num_occurrences = random.randint(3, max(3, seq_len // (3 * pattern_length)))
            
            #   pattern     
            used_ranges = []  #         
            for _ in range(num_occurrences):
                attempts = 0
                max_attempts = 100  #       
                while attempts < max_attempts:
                    start_idx = random.randint(0, seq_len - pattern_length)
                    end_idx = start_idx + pattern_length
                    
                    #        pattern  
                    overlap = False
                    for used_start, used_end in used_ranges:
                        if not (end_idx <= used_start or start_idx >= used_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        data[start_idx:start_idx + pattern_length] += pattern
                        additional_info["irregular_pattern_start_list"].append(start_idx)
                        used_ranges.append((start_idx, end_idx))
                        break
                    
                    attempts += 1
                
                #                       pattern  
                if attempts >= max_attempts:
                    break
            additional_info["irregular_pattern_start_list"].sort()
            return np.max(pattern)-np.min(pattern), data
        
        if attribute_pool["frequency"]["type"] == "daily":
            period = 24 * 3600 // sampling_interval  # 24            
        elif attribute_pool["frequency"]["type"] == "weekly":
            period = 24 * 3600 * 7 // sampling_interval  # 7           
        elif attribute_pool["frequency"]["type"] == "hourly":
            period = 3600 // sampling_interval  #   1            
        else:
            period = 24 * 3600 * 30 // sampling_interval
        
        magnitude = random.uniform(0.5, 1) * overall_magnitude
        return magnitude, self._generate_seasonal_wave(period, magnitude, seq_len, seasonal_type)

    def _generate_seasonal_wave(self, period: float, magnitude: float, seq_len: int, wave_type: str) -> np.ndarray:
        """            """
        t = np.arange(seq_len)
        
        if "sin" in wave_type:
            num_harmonics = random.randint(1, max(2, min(int(period) // 6, 5)))
            data = np.zeros(seq_len)
            for n in range(1, num_harmonics + 1):
                phase = random.uniform(0, 2 * np.pi)
                harmonic_magnitude = magnitude / n
                data += harmonic_magnitude * np.sin(2 * np.pi * n * t / period + phase)
                
        elif "square" in wave_type:
            data = np.zeros(seq_len)
            start = random.uniform(0, 0.3)
            duration = random.uniform(0.1, 0.3)
            for i in range(seq_len):
                cycle_pos = (t[i] % period) / period
                if start <= cycle_pos < start + duration:
                    data[i] = magnitude
                    
        elif "triangle" in wave_type:
            data = np.zeros(seq_len)
            start = random.uniform(0, 0.3)
            duration = random.uniform(0.1, 0.6)
            end = start + duration
            for i in range(seq_len):
                cycle_pos = (t[i] % period) / period
                if start <= cycle_pos < end:
                    if cycle_pos < (start + end) / 2:
                        data[i] = magnitude * 2 * (cycle_pos - start) / duration
                    else:
                        data[i] = magnitude * 2 * (end - cycle_pos) / duration
        else:
            data = magnitude * np.sin(2 * np.pi * t / period + random.uniform(0, 2 * np.pi))
        
        if np.max(data) - np.min(data) > 1e-7:
            data = data / (np.max(data) - np.min(data)) * magnitude
        data -= np.mean(data)
        
        return data

    def _generate_trend_pattern(self, attribute_pool: Dict[str, Any], y: np.ndarray, overall_magnitude: float, overall_bias: float, seq_len: int, additional_info: Dict[str, Any]) -> np.ndarray:
        """      """
        trend_type = attribute_pool["trend"]["type"]
        seasonal_type = attribute_pool["seasonal"]["type"]
        fluctuation_mode = "default"
        if "square" in seasonal_type or "triangle" in seasonal_type:
            fluctuation_mode = "low"
        if seasonal_type == "no periodic fluctuation":
            fluctuation_mode = "zero"
        magnitude = random.uniform(0.5, 1) * overall_magnitude
        if trend_type == "increase":
            base_trend = np.linspace(0, magnitude, seq_len)
            fluctuation_magnitude = random.uniform(0, 0.3)* magnitude
            if fluctuation_mode == "low":
                fluctuation_magnitude *= random.uniform(0, 0.1)
            elif fluctuation_mode == "zero":
                fluctuation_magnitude = 0
            overall_pi_len = random.choice([2 * i for i in range(1, 10)])
            fluctuation = np.sin(np.linspace(0, overall_pi_len*np.pi, seq_len)) * fluctuation_magnitude
            trend = base_trend + fluctuation
            y += trend + overall_bias
        elif trend_type == "decrease":
            base_trend = np.linspace(magnitude, 0, seq_len)
            fluctuation_magnitude = random.uniform(0, 0.3)* magnitude
            if fluctuation_mode == "low":
                fluctuation_magnitude *= random.uniform(0, 0.1)
            elif fluctuation_mode == "zero":
                fluctuation_magnitude = 0
            overall_pi_len = random.choice([2 * i for i in range(1, 10)])
            fluctuation = np.sin(np.linspace(0, overall_pi_len*np.pi, seq_len)) * fluctuation_magnitude
            trend = base_trend + fluctuation
            y += trend + overall_bias
        elif trend_type == "keep steady":
            y += overall_bias
            trend = np.zeros(seq_len)
            magnitude = 0
        elif trend_type == "random":  #     
            trend = np.cumsum(np.random.normal(0, magnitude*0.1, seq_len))
            trend_range = np.max(trend) - np.min(trend)
            if trend_range > 0:
                trend = trend * (magnitude * 0.8 / trend_range)
            # smoothen trend
            window_size = max(5, seq_len // 20)
            trend = np.convolve(trend, np.ones(window_size)/window_size, mode='same')
            y += trend + overall_bias

        return np.max(trend)-np.min(trend), y

    def _generate_noise_pattern(self, attribute_pool: Dict[str, Any], y: np.ndarray, overall_magnitude: float, seq_len: int, additional_info: Dict[str, Any]) -> np.ndarray:
        """      """
        trend_type = attribute_pool["trend"]["type"]
        seasonal_type = attribute_pool["seasonal"]["type"]
        noise_type = attribute_pool["noise"]["type"]
        max_change = np.max(y) - np.min(y)
        
        if noise_type == "almost no noise":
            if seasonal_type not in ["square periodic fluctuation", "triangle periodic fluctuation"] and max_change > overall_magnitude * 0.5:
                std = random.uniform(0.0, 0.001) * overall_magnitude
            else:
                std = 0.0
            noise = np.random.normal(0, std, seq_len)
            # clean_indices = np.random.choice(seq_len, size=int(seq_len * 0.8), replace=False)
            # noise[clean_indices] = 0

        elif noise_type == "noisy":
            #             sin   
            if random.random() > 0.5:
                std = random.uniform(0.05, 0.15) * overall_magnitude
                if trend_type in ["random", "increase", "decrease"]:
                    std /= 2
                noise = np.random.normal(0, std, seq_len)
            else:
                # sin   
                std = 0.2 * overall_magnitude
                noise = self._generate_sin_noise(std, seq_len)
                std += 0.03 * overall_magnitude
                noise += np.random.normal(0, 0.03 * overall_magnitude, seq_len)
        else:
            std = 0.001 * overall_magnitude
            noise = np.random.normal(0, std, seq_len)
            # clean_indices = np.random.choice(seq_len, size=int(seq_len * 0.8), replace=False)
            # noise[clean_indices] = 0
        
        return np.max(noise)-np.min(noise), noise

    def _generate_sin_noise(self, magnitude: float, seq_len: int, num_harmonics: int=50, is_zero_normalize: bool=True) -> np.ndarray:
        """  sin   """
        t = np.arange(seq_len)
        data = np.zeros(seq_len)
        
        #           sin 
        for n in range(1, num_harmonics + 1):
            phase = random.uniform(0, 2 * np.pi)
            freq = random.uniform(10 / seq_len, 50 / seq_len)
            data += np.sin(freq * t + phase) * random.uniform(0.3, 1.0)
        
        if np.max(data) - np.min(data) > 1e-7:
            data = data / (np.max(data) - np.min(data)) * magnitude
        if is_zero_normalize:
            data -= np.mean(data)
        return data


def generate_ts_base_dataset(num_samples: int, output_path: str, is_train: bool = False, seq_len_range: Tuple[int, int] = (100, 1000), sampling_interval: Union[int, str] = 60, random_seed: int = None) -> Dict[str, Any]:
    """
                    
    
    Args:
        num_samples:          
        output_path:        .json   
        seq_len_range:            (100, 1000)
        sampling_interval:               'random'    
        random_seed:            
        
    Returns:
        Dict            
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    generator = TimeSeriesGenerator()
    dataset = []
    print(f"     {num_samples}        ...")
    
    for i in trange(num_samples, desc="Generating Base Time Series"):
        ts_info, ts_data = generator.generate_base_ts(seq_len_range, sampling_interval, is_train)
        
        sample = {
            "timeseries_info": ts_info.to_dict(),
            "timeseries_data": ts_data.tolist()
        }
        
        dataset.append(sample)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_ts_json(dataset, output_path)
    
    print(f"\n        ")
    print(f"    : {output_path}")
    print(f"    : {len(dataset)}")
    print(f"    : {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return dataset


def visualize_time_series(series: np.ndarray, ts_info: TimeSeriesInfo, save_dir: Optional[str] = None):
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Time Series', color='blue')
    
    #  characteristics           
    char_str = f"{ts_info.characteristics['trend']}, {ts_info.characteristics['seasonal']}({ts_info.characteristics['frequency']}, {ts_info.magnitude}), {ts_info.characteristics['noise']}"
    for key, value in ts_info.additional_info.items():
        if "list" not in key:
            char_str += f", {key}: {value}"
    plt.title(char_str)
    
    plt.xlabel(f"{ts_info.metric_name} ({ts_info.metric_type})")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        char_filename = "_".join([f"{k}_{v.replace(' ', '_')}" for k, v in ts_info.characteristics.items()])
        save_path = os.path.join(save_dir, f"type_{ts_info.metric_type}.characteristic_{char_filename}.name_{ts_info.metric_name.replace('##', '_')}.pdf")
        # tight layout to prevent clipping of ylabel
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    data_dir = "../data"
    visualize = True  #          
    output_file = os.path.join(data_dir, "dataset_base_timeseries.json")
    dataset = generate_ts_base_dataset(
        num_samples=30,
        output_path=output_file,
        is_train=True,
        seq_len_range=(480, 1440*3),
        sampling_interval=60,
        random_seed=2025
    )
    if visualize:
        vis_dir = os.path.join(data_dir, "visualizations", "base_timeseries")
        for sample in dataset:
            ts_info_dict = sample["timeseries_info"]
            ts_info = TimeSeriesInfo(**ts_info_dict)
            ts_data = np.array(sample["timeseries_data"])
            visualize_time_series(ts_data, ts_info, save_dir=vis_dir)
