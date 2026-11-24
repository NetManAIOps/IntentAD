from pathlib import Path
from text_generator.llm_data_generator import LLMSFTDataGenerator
import sys
from typing import Optional
import json

SUPPORTED = ["WSD", "Yahoo", "AIOPS", "NAB", "TODS", "UCR", "Service", "Dodgers", "Genesis", "SensorScope", "MGAB", "NEK", "Dodgers", "Power", "TAO", "SMD", "Stock", "MSL", "SMAP", "OPPORTUNITY", "SED"]
SUPPORTED.extend([f"{dataset}_rules" for dataset in SUPPORTED])


def build_paths(data_dir: Path, dataset: str, is_train: int, split: str = "default", cot: int = 1, is_dapo: int = 0, is_filter: int = 0, is_merged: int = 0) -> tuple[Path, Path]:
    in_dir = data_dir / "tsad"
    out_dir = data_dir / "llm_sft"
    # if is_dapo:
    #     out_dir = out_dir / "dapo"
    if split != "default":
        in_dir = in_dir / split
        out_dir = out_dir / split
    if not cot or is_dapo:
        out_dir = out_dir / "no_cot"
    if is_filter:
        out_dir = out_dir / "filter"
    if is_train:
        in_dir = in_dir / "train"
        out_dir = out_dir / "train"
    if is_merged:
        in_dir = in_dir / "merged"
        out_dir = out_dir / "merged"
    in_file = in_dir / f"{dataset}{'_rules' if is_filter else ''}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    return in_file, out_dir / f"QA_{dataset}.jsonl"


def process_dataset(dataset: str, data_dir: Path, limit: int | None, is_train: int, split: str = "default", filter: int = 0, cot: int = 1, is_dapo: int = 0, use_tool: int = 0, query_memory: int = 1, include_data_anomalies: int = 1, using_llm_input: int = 0, using_llm_output: int = 0, is_merged: int = 0, llm_client: Optional[object] = None):
    gen = LLMSFTDataGenerator()
    in_file, out_file = build_paths(data_dir, dataset, is_train, split, cot, is_dapo, is_filter=filter, is_merged=is_merged)
    if not in_file.exists():
        print(f"[WARN] Input not found for {dataset}: {in_file}", file=sys.stderr)
        return
    print(f"[INFO] Generating LLM data for {dataset} from {in_file} -> {out_file}")
    gen.generate_dataset(str(in_file), str(out_file), num_samples=limit, is_cot=bool(cot), is_dapo=bool(is_dapo), use_tool=bool(use_tool), llm_client=llm_client, using_llm_input=using_llm_input, using_llm_output=using_llm_output, include_data_anomalies=include_data_anomalies, query_memory=query_memory)

    with open(in_file, 'r', encoding='utf-8') as f:
        data_source = json.load(f)
    with open(out_file, 'r', encoding='utf-8') as f:
        data_out = [json.loads(line) for line in f]
    for item_out, item_source in zip(data_out, data_source):
        item_out['source_split'] = item_source.get('source_split', 'unknown')
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_llm_client(enable_llm_client: bool, using_llm_input: int, using_llm_output: int):
    if not enable_llm_client:
        return None
    if not (using_llm_input or using_llm_output):
        return None

    raise RuntimeError(
        "The public artifact does not ship a provider-specific LLM backend. "
        "Keep using_llm_input/using_llm_output disabled, or pass your own client "
        "to process_dataset()."
    )


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'

    # Edit these defaults when running the converter directly.
    datasets_config: str | list[str] = ["AIOPS"]    # "Service"
    split = "long"               # default / short / long / season
    filter = 1                     # 1: read *_rules.json
    cot = 0                        # 1: enable CoT format
    is_train = 0                   # 1: use train split
    is_merged = 0                  # 1: use merged split
    is_dapo = 0                    # 1: DAPO/RL data format
    use_tool = 0                   # 1: generate tool-call format
    query_memory = 0               # 1: include query_memory
    include_data_anomalies = 1     # 1: include detected data anomalies
    using_llm_input = 0            # 1: rewrite inputs with an LLM
    using_llm_output = 0           # 1: rewrite outputs with an LLM
    limit_per_dataset = None

    # Disabled in the public artifact unless a custom provider client is added.
    enable_llm_client = False

    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir dose not exist: {data_dir}")

    if isinstance(datasets_config, str) and datasets_config.lower() == 'all':
        datasets = SUPPORTED
    else:
        if isinstance(datasets_config, str):
            datasets = [d.strip() for d in datasets_config.split(',') if d.strip()]
        else:
            datasets = [d.strip() for d in datasets_config if d.strip()]
        for d in datasets:
            if d not in SUPPORTED:
                raise ValueError(f"Unsupported dataset: {d}. Supported: {SUPPORTED}")

    llm_client = build_llm_client(enable_llm_client, using_llm_input, using_llm_output)

    for ds in datasets:
        process_dataset(
            ds,
            data_dir,
            limit_per_dataset,
            is_train,
            split,
            filter,
            cot,
            is_dapo,
            use_tool,
            query_memory,
            include_data_anomalies,
            using_llm_input,
            using_llm_output,
            is_merged,
            llm_client,
        )

    print("Finished converting all configured datasets.")


if __name__ == '__main__':
    main()
