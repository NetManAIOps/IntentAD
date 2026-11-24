# IntentAD Bench LLM Example

This directory contains a small sample extracted from the complete
IntentAD Bench file under `../../data/intentad_bench/`. It illustrates how
benchmark records are converted into LLM-facing inputs and outputs.

Each line contains:

- `input`: time-series metadata, task description, and user intent.
- `timeseries`: the numeric KPI sequence.
- `label`: final intent-aware anomaly intervals.
- `output`: model-style answer wrapped in `<answer>...</answer>`.
- `tools` and `history_anomalies`: optional retrieval-tool metadata for
  history-dependent intents.

The metric names are synthetic demo identifiers and do not correspond to private
services. The complete companion metadata is stored under
`../../data/intentad_bench/additional_info/`.
