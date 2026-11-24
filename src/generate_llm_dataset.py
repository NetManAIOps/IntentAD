import os
from text_generator import LLMSFTDataGenerator
from typing import Optional
import argparse


DRY_RUN_PROMPTS = False
DRY_RUN_NUM_PRINT = 5


def main(
    input_file,
    output_file,
    num_samples,
    is_ts_modal=True,
    is_cot=True,
    reject=None,
    using_llm_input: Optional[bool] = None,
    using_llm_output: Optional[bool] = None,
    llm_client=None,
    is_dapo=False,
    use_tool=True,
    query_memory=False,
    include_data_anomalies=False,
):
    generator = LLMSFTDataGenerator()
    generator.generate_dataset(
        input_file=input_file,
        num_samples=num_samples,
        output_file=output_file,
        is_ts_modal=is_ts_modal,
        is_cot=is_cot,
        reject=reject,
        using_llm_input=using_llm_input,
        using_llm_output=using_llm_output,
        llm_client=llm_client,
        is_dapo=is_dapo,
        use_tool=use_tool,
        query_memory=query_memory,
        include_data_anomalies=include_data_anomalies,
        dry_run_prompts=DRY_RUN_PROMPTS,
        dry_run_num_print=DRY_RUN_NUM_PRINT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TSAD JSON cases to LLM QA JSONL.")
    parser.add_argument("--input-file", required=True, help="Input TSAD JSON file.")
    parser.add_argument("--output-file", required=True, help="Output QA JSONL file.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--text-only", action="store_true", help="Do not include time-series values in the prompt.")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought style output.")
    parser.add_argument("--dapo", action="store_true", help="Generate the RL/DAPO data format.")
    parser.add_argument("--use-tool", action="store_true", help="Generate tool-call style samples.")
    parser.add_argument("--query-memory", action="store_true", help="Include retrieved history when present.")
    parser.add_argument("--include-data-anomalies", action="store_true")
    parser.add_argument("--dry-run-prompts", action="store_true", help="Print prompt examples without writing output.")
    parser.add_argument("--dry-run-num-print", type=int, default=5)
    args = parser.parse_args()

    DRY_RUN_PROMPTS = args.dry_run_prompts
    DRY_RUN_NUM_PRINT = args.dry_run_num_print
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    main(
        args.input_file,
        args.output_file,
        args.num_samples,
        is_ts_modal=not args.text_only,
        is_cot=not args.no_cot,
        reject=None,
        using_llm_input=False,
        using_llm_output=False,
        llm_client=None,
        is_dapo=args.dapo,
        use_tool=args.use_tool,
        query_memory=args.query_memory,
        include_data_anomalies=args.include_data_anomalies,
    )
