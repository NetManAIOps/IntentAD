# Dataset Availability

This repository provides the synthetic IntentAD Bench data and utilities for
preparing evaluation inputs.

## Included

- `data/intentad_bench/intentad_bench.jsonl`: complete synthetic
  IntentAD Bench LLM-format data with 1,000 cases.
- `data/intentad_bench/additional_info/intentad_bench.jsonl`:
  per-case structured intent and taxonomy metadata. Each line includes
  `dataset_relative_path`, which points to `../intentad_bench.jsonl`
  instead of any machine-specific absolute path.
- `examples/intentad_bench_llm/`: a tiny sample extracted from the full
  IntentAD Bench data for quick inspection.
- `src/ts_generator/` and `src/intentad/generate_intentad_bench.py`: synthetic
  generation code for IntentAD-style time-series cases with human intent rules.
- `src/convert_uts_dataset_to_tsad.py`: conversion utilities for public UTS
  benchmark folders.
- `src/evaluate.py` and `scripts/evaluate_single_file.py`: metric code for
  interval-level anomaly detection evaluation.

## Not Included

- Private Service telemetry, private KPI names, deployment logs, alert payloads,
  expert annotations, and business-specific rules.
- Provider-specific LLM client code, API keys, internal model names, or internal
  endpoints.

## How To Rebuild Public/Shareable Inputs

1. Download the public datasets used by your experiment setup into `data/raw/`.
2. Convert them to the TSAD schema with `src/convert_uts_dataset_to_tsad.py`.
3. Generate or attach human-intent rules with the provided generator code.
4. Convert TSAD JSON cases to LLM QA JSONL with `src/generate_llm_dataset.py`.
5. Run inference with your own model checkpoint, then evaluate the prediction
   JSONL files with `scripts/evaluate_single_file.py` or `src/evaluate.py`.

The Service dataset remains private due to confidentiality constraints.
The same schema and evaluation code are provided here.
