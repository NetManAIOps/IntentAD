# Model Output Format

The evaluator expects one JSON object per line. Each record should contain:

- `input`: prompt text containing at least `Metric Name:` and `Sequence Length:`.
- `timeseries`: a list of numeric values, or a list containing one numeric list.
- `label`: the ground-truth anomaly list, e.g. `[(3, 4, 'spike', 4.0)]`.
- `output`: model answer containing an `<answer>...</answer>` block with a list
  of predicted intervals.

Use `demo_predictions.jsonl` to smoke-test the evaluator.
