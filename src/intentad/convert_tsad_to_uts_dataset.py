import ast
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from util import merge_sample

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
	from ts_generator.util import SAMPLING_INTERVAL_STR_DICT
except Exception:
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


SAMPLING_INTERVAL_PATTERN = re.compile(r"Sampling\s*Interval\s*:\s*([^\n\r]+)", re.IGNORECASE)
METRIC_TYPE_PATTERN = re.compile(r"Metric\s*Type\s*:\s*([^\n\r]+)", re.IGNORECASE)
METRIC_NAME_PATTERN = re.compile(r"Metric\s*Name\s*:\s*([^\n\r]+)", re.IGNORECASE)


def _normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip().lower())


def build_interval_reverse_map() -> Dict[str, int]:
	return {_normalize_text(v): int(k) for k, v in SAMPLING_INTERVAL_STR_DICT.items()}


def parse_interval_seconds(interval_text: str, reverse_map: Dict[str, int]) -> Optional[int]:
	raw = interval_text.strip()
	norm = _normalize_text(raw)
	if norm in reverse_map:
		return reverse_map[norm]

	unit_pattern = re.compile(
		r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b",
		re.IGNORECASE,
	)
	matches = unit_pattern.findall(raw)
	if matches:
		total_seconds = 0.0
		for val_str, unit in matches:
			value = float(val_str)
			u = unit.lower()
			if u.startswith("h"):
				total_seconds += value * 3600
			elif u.startswith("m") and u not in {"ms", "msec", "msecs"}:
				total_seconds += value * 60
			else:
				total_seconds += value
		seconds = int(round(total_seconds))
		return seconds if seconds > 0 else None

	if re.fullmatch(r"\d+", norm):
		value = int(norm)
		return value if value > 0 else None
	return None


def extract_sampling_interval_seconds(prompt_text: str, reverse_map: Dict[str, int]) -> int:
	match = SAMPLING_INTERVAL_PATTERN.search(prompt_text or "")
	if not match:
		return 60
	parsed = parse_interval_seconds(match.group(1), reverse_map)
	return parsed if parsed is not None else 60


def extract_optional(pattern: re.Pattern, text: str) -> Optional[str]:
	m = pattern.search(text or "")
	return m.group(1).strip() if m else None


def parse_label_field(label_field: Any) -> List[Sequence[Any]]:
	if label_field is None:
		return []
	if isinstance(label_field, list):
		return label_field
	if isinstance(label_field, str):
		s = label_field.strip()
		if not s:
			return []
		try:
			loaded = json.loads(s)
			if isinstance(loaded, list):
				return loaded
		except json.JSONDecodeError:
			pass
		try:
			loaded = ast.literal_eval(s)
			if isinstance(loaded, list):
				return loaded
		except Exception:
			return []
	return []


def to_interval_pairs(raw_intervals: Iterable[Any]) -> List[Tuple[int, int]]:
	pairs: List[Tuple[int, int]] = []
	for item in raw_intervals:
		start: Optional[int] = None
		end: Optional[int] = None
		if isinstance(item, (list, tuple)) and len(item) >= 2:
			try:
				start = int(float(item[0]))
				end = int(float(item[1]))
			except Exception:
				continue
		elif isinstance(item, dict):
			if "start" in item and "end" in item:
				try:
					start = int(float(item["start"]))
					end = int(float(item["end"]))
				except Exception:
					continue
		if start is None or end is None:
			continue
		if start > end:
			start, end = end, start
		pairs.append((start, end))
	return pairs


def intervals_to_binary_mask(intervals: List[Tuple[int, int]], length: int) -> np.ndarray:
	mask = np.zeros(length, dtype=np.int64)
	for start, end in intervals:
		if end < 0 or start >= length:
			continue
		s = max(0, start)
		e = min(length - 1, end)
		if s <= e:
			mask[s : e + 1] = 1
	return mask


def flatten_timeseries(timeseries_field: Any) -> np.ndarray:
	arr = np.asarray(timeseries_field, dtype=np.float64)
	if arr.ndim == 0:
		return np.asarray([float(arr)], dtype=np.float64)
	if arr.ndim >= 2:
		if arr.shape[0] == 1:
			arr = arr[0]
		elif arr.shape[1] == 1:
			arr = arr[:, 0]
	return np.asarray(arr, dtype=np.float64).reshape(-1)


def resolve_timeseries(record: Dict[str, Any]) -> np.ndarray:
	if "timeseries" in record:
		return flatten_timeseries(record.get("timeseries"))
	return flatten_timeseries(record.get("timeseries_data", []))


def resolve_case_id(record: Dict[str, Any], fallback_idx: int) -> int:
	try:
		if "case_id" in record:
			return int(record.get("case_id"))
	except Exception:
		pass
	return int(fallback_idx)


def resolve_interval_seconds(record: Dict[str, Any], reverse_map: Dict[str, int]) -> int:
	try:
		ts_info = record.get("timeseries_info") or {}
		if isinstance(ts_info, dict) and ts_info.get("sampling_interval") is not None:
			val = int(ts_info.get("sampling_interval"))
			if val > 0:
				return val
	except Exception:
		pass
	return extract_sampling_interval_seconds(str(record.get("input", "")), reverse_map)


def read_jsonl_records(input_jsonl: Path, max_cases: Optional[int]) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with input_jsonl.open("r", encoding="utf-8") as f:
		for line in f:
			if max_cases is not None and len(records) >= max_cases:
				break
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def build_info(
	record: Dict[str, Any],
	case_id: int,
	seq_len: int,
	train_seq_len: Optional[int],
	test_seq_len: Optional[int],
	interval_seconds: int,
	training_anomaly_ratio: float,
	test_anomaly_ratio: float,
) -> Dict[str, Any]:
	input_text = str(record.get("input", ""))
	ts_info = record.get("timeseries_info") if isinstance(record.get("timeseries_info"), dict) else {}
	metric_name = ts_info.get("metric_name") or extract_optional(METRIC_NAME_PATTERN, input_text)
	metric_type = ts_info.get("metric_type") or extract_optional(METRIC_TYPE_PATTERN, input_text)
	human_intent = record.get("human_intent")
	if human_intent is None:
		human_intent = input_text.split("[User Intent]\n")[-1].strip() if "[User Intent]" in input_text else None
	case_name = f"{case_id:03d}_{metric_name}" if metric_name else f"{case_id:03d}"
	info = {
		"case_id": case_id,
		"case_name": case_name,
		"seq_len": seq_len,
		"train_seq_len": train_seq_len,
		"test_seq_len": test_seq_len,
		"intervals": int(interval_seconds),
		"training set anomaly ratio": float(training_anomaly_ratio),
		"test set anomaly ratio": float(test_anomaly_ratio),
		"metric_type": metric_type,
		"metric_name": metric_name,
		# "human_intent": human_intent,
	}
	if "output" in record:
		info["raw_output"] = record.get("output")
	return info


def convert(input_jsonl: Path, output_dir: Path, base_ts: int, case_prefix: str, max_cases: Optional[int]) -> None:
	reverse_map = build_interval_reverse_map()

	processed = 0
	with input_jsonl.open("r", encoding="utf-8") as f:
		for idx, line in enumerate(f):
			if max_cases is not None and processed >= max_cases:
				break
			line = line.strip()
			if not line:
				continue
			record = json.loads(line)

			ts = resolve_timeseries(record)
			n = int(ts.shape[0])

			interval_seconds = resolve_interval_seconds(record, reverse_map)

			raw_intervals = parse_label_field(record.get("label"))
			interval_pairs = to_interval_pairs(raw_intervals)
			test_label = intervals_to_binary_mask(interval_pairs, n)

			test_timestamp = (base_ts + np.arange(n, dtype=np.int64) * int(interval_seconds)).astype(np.int64)

			info = build_info(
				record=record,
				case_id=idx,
				seq_len=n,
				train_seq_len=0,
				test_seq_len=n,
				interval_seconds=interval_seconds,
				training_anomaly_ratio=0.0,
				test_anomaly_ratio=float(test_label.mean()) if n > 0 else 0.0,
			)
			case_name = info["case_name"]
			case_dir = output_dir / case_name
			case_dir.mkdir(parents=True, exist_ok=True)
			
			np.save(case_dir / "test.npy", ts)
			np.save(case_dir / "test_label.npy", test_label)
			np.save(case_dir / "test_timestamp.npy", test_timestamp)
			with (case_dir / "info.json").open("w", encoding="utf-8") as fw:
				json.dump(info, fw, ensure_ascii=False, indent=2)

			processed += 1

	print(f"Converted {processed} cases to: {output_dir}")


def convert_pair(
	train_jsonl: Path,
	test_jsonl: Path,
	output_dir: Path,
	base_ts: int,
	max_cases: Optional[int],
) -> None:
	reverse_map = build_interval_reverse_map()

	train_records = read_jsonl_records(train_jsonl, max_cases)
	test_records = read_jsonl_records(test_jsonl, max_cases)

	train_by_case: Dict[int, Dict[str, Any]] = {}
	test_by_case: Dict[int, Dict[str, Any]] = {}
	for idx, rec in enumerate(train_records):
		train_by_case[resolve_case_id(rec, idx)] = rec
	for idx, rec in enumerate(test_records):
		test_by_case[resolve_case_id(rec, idx)] = rec

	case_ids = sorted(set(train_by_case.keys()) & set(test_by_case.keys()))
	processed = 0
	for case_id in case_ids:
		train_record = train_by_case[case_id]
		test_record = test_by_case[case_id]
		merged_record = merge_sample(train_record, test_record)

		train_ts = resolve_timeseries(train_record)
		test_ts = resolve_timeseries(test_record)
		train_n = int(train_ts.shape[0])
		test_n = int(test_ts.shape[0])

		interval_seconds = resolve_interval_seconds(merged_record, reverse_map)
		train_raw_intervals = parse_label_field(train_record.get("label"))
		train_interval_pairs = to_interval_pairs(train_raw_intervals)
		train_label = intervals_to_binary_mask(train_interval_pairs, train_n)
		raw_intervals = parse_label_field(test_record.get("label"))
		interval_pairs = to_interval_pairs(raw_intervals)
		test_label = intervals_to_binary_mask(interval_pairs, test_n)

		train_timestamp = (base_ts + np.arange(train_n, dtype=np.int64) * int(interval_seconds)).astype(np.int64)
		test_timestamp = (
			base_ts + np.arange(train_n, train_n + test_n, dtype=np.int64) * int(interval_seconds)
		).astype(np.int64)

		info = build_info(
			record=merged_record,
			case_id=case_id,
			seq_len=merged_record.get("seq_len", train_n + test_n),
			interval_seconds=interval_seconds,
			training_anomaly_ratio=float(train_label.mean()) if train_n > 0 else 0.0,
			test_anomaly_ratio=float(test_label.mean()) if test_n > 0 else 0.0,
			train_seq_len=train_n,
			test_seq_len=test_n,
		)

		case_name = info["case_name"]
		case_dir = output_dir / case_name
		case_dir.mkdir(parents=True, exist_ok=True)

		np.save(case_dir / "train.npy", train_ts)
		np.save(case_dir / "test.npy", test_ts)
		np.save(case_dir / "train_label.npy", train_label)
		np.save(case_dir / "test_label.npy", test_label)
		np.save(case_dir / "train_timestamp.npy", train_timestamp)
		np.save(case_dir / "test_timestamp.npy", test_timestamp)
		with (case_dir / "info.json").open("w", encoding="utf-8") as fw:
			json.dump(info, fw, ensure_ascii=False, indent=2)

		processed += 1

	print(f"Converted {processed} paired cases to: {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert IntentAD JSONL cases to EasyTSAD-style folders.")
	parser.add_argument("--mode", choices=["single", "pair"], default="pair")
	parser.add_argument("--dataset-name", default="example")
	parser.add_argument("--input-jsonl", type=Path, default=None)
	parser.add_argument("--train-jsonl", type=Path, default=None)
	parser.add_argument("--test-jsonl", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--base-timestamp", type=int, default=1700000000)
	parser.add_argument("--case-prefix", default="case")
	parser.add_argument("--max-cases", type=int, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	mode = args.mode
	dataset_name = args.dataset_name
	input_jsonl = args.input_jsonl
	train_jsonl = args.train_jsonl or PROJECT_ROOT / "data" / "intentad_bench" / dataset_name / "train.jsonl"
	test_jsonl = args.test_jsonl or PROJECT_ROOT / "data" / "intentad_bench" / dataset_name / "test.jsonl"
	output_dir = args.output_dir or PROJECT_ROOT / "data" / "intentad_bench" / "easytsad" / dataset_name
	base_timestamp = args.base_timestamp
	case_prefix = args.case_prefix
	max_cases = args.max_cases

	if mode == "single":
		if input_jsonl is None:
			raise ValueError("single mode requires input_jsonl")
		convert(
			input_jsonl=Path(input_jsonl),
			output_dir=output_dir,
			base_ts=base_timestamp,
			case_prefix=case_prefix,
			max_cases=max_cases,
		)
		return
	if mode == "pair":
		if train_jsonl is None or test_jsonl is None:
			raise ValueError("pair mode requires train_jsonl and test_jsonl")
		convert_pair(
			train_jsonl=Path(train_jsonl),
			test_jsonl=Path(test_jsonl),
			output_dir=output_dir,
			base_ts=base_timestamp,
			max_cases=max_cases,
		)
		return

	raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
	main()
