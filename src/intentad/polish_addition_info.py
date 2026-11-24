import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ADDITIONAL_INFO_PATH = Path(
	PROJECT_ROOT / "data" / "llm_sft" / "generated" / "full" / "additional_info" / "input.jsonl"
)
DEFAULT_BENCH_PATH = Path(
	PROJECT_ROOT / "data" / "intentad_bench" / "example" / "test.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
	PROJECT_ROOT / "data" / "llm_sft" / "generated" / "full" / "additional_info" / "updated.jsonl"
)

TIME_WORD_RE = re.compile(r"\b(second|minute|hour|day)s?\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Polish intent_taxonomy in additional_info JSONL by updating "
			"relative_expression and history_dependency."
		)
	)
	parser.add_argument(
		"--input-additional",
		type=Path,
		default=DEFAULT_ADDITIONAL_INFO_PATH,
		help="Path to additional_info jsonl file.",
	)
	parser.add_argument(
		"--input-bench",
		type=Path,
		default=DEFAULT_BENCH_PATH,
		help="Path to intentad benchmark test jsonl file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Path to output updated jsonl file.",
	)
	return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			raw = line.strip()
			if not raw:
				continue
			try:
				record = json.loads(raw)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
			if not isinstance(record, dict):
				raise ValueError(f"Non-object JSON in {path} line {line_no}")
			records.append(record)
	return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for record in records:
			f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _contains_time_word(text: str) -> bool:
	return bool(TIME_WORD_RE.search(text))


def _has_scope_category(structured: Dict[str, Any]) -> bool:
	categories = structured.get("categories", [])
	if not isinstance(categories, list):
		return False
	for category in categories:
		if str(category).strip().lower() == "scope":
			return True
	return False


def derive_relative_expression(additional_record: Dict[str, Any]) -> str:
	human_intent_text = str(additional_record.get("human_intent_text", ""))
	structured = additional_record.get("human_intent_structured", {})
	if not isinstance(structured, dict):
		structured = {}

	if _contains_time_word(human_intent_text) or _has_scope_category(structured):
		return "with"
	return "without"


def _to_number(value: Any) -> float:
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return 0.0
		try:
			return float(text)
		except ValueError:
			return 0.0
	return 0.0


def derive_history_dependency(bench_record: Dict[str, Any]) -> str:
	human_intent = bench_record.get("human_intent", {})
	if not isinstance(human_intent, dict):
		return "no"

	rules = human_intent.get("rules", [])
	if not isinstance(rules, list):
		return "no"

	for rule in rules:
		if not isinstance(rule, dict):
			continue
		history_query_analysis = rule.get("history_query_analysis")
		if not isinstance(history_query_analysis, dict):
			continue
		requires_history_count = _to_number(history_query_analysis.get("requires_history_count", 0))
		if requires_history_count > 0:
			return "yes"

	return "no"


def polish_records(
	additional_records: List[Dict[str, Any]],
	bench_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	if len(additional_records) != len(bench_records):
		raise ValueError(
			"Record count mismatch: "
			f"additional={len(additional_records)}, bench={len(bench_records)}"
		)

	updated_records: List[Dict[str, Any]] = []
	for idx, (additional_record, bench_record) in enumerate(zip(additional_records, bench_records)):
		if not isinstance(additional_record, dict):
			raise ValueError(f"additional record at index {idx} is not an object")
		if not isinstance(bench_record, dict):
			raise ValueError(f"bench record at index {idx} is not an object")

		taxonomy = additional_record.get("intent_taxonomy")
		if not isinstance(taxonomy, dict):
			taxonomy = {}
		additional_record["intent_taxonomy"] = taxonomy

		taxonomy["relative_expression"] = derive_relative_expression(additional_record)
		taxonomy["history_dependency"] = derive_history_dependency(bench_record)

		updated_records.append(additional_record)

	return updated_records


def main() -> None:
	args = parse_args()
	additional_records = read_jsonl(args.input_additional)
	bench_records = read_jsonl(args.input_bench)
	updated_records = polish_records(additional_records, bench_records)
	write_jsonl(args.output, updated_records)
	print(f"Updated {len(updated_records)} records -> {args.output}")


if __name__ == "__main__":
	main()
