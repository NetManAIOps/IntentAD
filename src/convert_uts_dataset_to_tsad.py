from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from ts_generator.util import save_ts_json

# Optional plotting support.
_MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

SUPPORTED_DATASETS = ["WSD", "Yahoo", "AIOPS", "NAB", "TODS", "UCR", "Service", "Dodgers", "Genesis", "SMD", "SensorScope", "MSL", "SMAP", "MGAB", "NEK", "Power", "SED", "Stock", "CreditCard", "LTDB", "TAO", "CATSv2", "OPPORTUNITY", "Exathlon", "SVDB", "Daphnet"]
DATASET_SHORT_LENGTHS = {
    "WSD": 480, "Yahoo": 480, "AIOPS": 480, "NAB": 200, "TODS": 100, "UCR": 200, "Service": 480, "Dodgers": 200, "Genesis": 200, "SMD": 200, "SensorScope": 200, "MSL": 200, "SMAP": 200, "MGAB": 200, "NEK": 200, "Power": 200, "SED": 200, "Stock": 200, "CreditCard": 200, "LTDB": 200, "TAO": 200, "CATSv2": 200, "OPPORTUNITY": 200, "Exathlon": 200, "SVDB": 200, "Daphnet": 200
}
DATASET_LONG_LENGTHS = {
    "WSD": 4320, "Yahoo": 480, "AIOPS": 720
}
DATASET_SEASON_LENGTHS = {
    "WSD": 1440, "Yahoo": 72, "AIOPS": 1440, "NAB": 288, "TODS": 100, "UCR": 250, "Service": 1440, "Dodgers": 288, "Genesis": 1440, "SMD": 1440, "SensorScope": 1440, "MSL": 1440, "SMAP": 1440, "MGAB": 96, "NEK": 1440, "Power": 672, "SED": 1440, "Stock": 24, "CreditCard": 1440, "LTDB": 1440, "TAO": 144, "CATSv2": 1440, "OPPORTUNITY": 1440, "Exathlon": 1440, "SVDB": 1440, "Daphnet": 1440
}
SEASON_DAYS = {
    "WSD": 3, "Yahoo": 4, "AIOPS": 3, "NAB": 4, "TODS": 4, "UCR": 4, "Service": 3, "Dodgers": 4, "Genesis": 3, "SMD": 3, "SensorScope": 3, "MSL": 3, "SMAP": 3, "MGAB": 7, "NEK": 3, "Power": 4, "SED": 3, "Stock": 7, "CreditCard": 3, "LTDB": 3, "TAO": 4, "CATSv2": 3, "OPPORTUNITY": 3, "Exathlon": 3, "SVDB": 3, "Daphnet": 3
}
RULE_ORDER = [
    "focus",
    "threshold",
    "ignore_frequency",
    "ignore_duration",
    "truncate_duration",
    "scope",
    "others",
    "merge",
]
DATASET_FILTER = {
    ds: {
        "rules": [],
        "categories": []
    } for ds in SUPPORTED_DATASETS
}

# ---------- case-by-case      ----------
def _compute_categories_from_rules(rules: List[Dict[str, Any]]) -> List[str]:
    cats = []
    seen = set()
    for cat in RULE_ORDER:
        for r in rules:
            c = r.get("category")
            if c == cat and c not in seen:
                cats.append(c)
                seen.add(c)
    for r in rules:
        c = r.get("category")
        if c and c not in seen:
            cats.append(c)
            seen.add(c)
    return cats

def _merge_human_intent(base_filter: Dict[str, Any], extra_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_rules = list((base_filter or {}).get("rules", []))
    idx_by_name = {r.get("name"): i for i, r in enumerate(merged_rules) if r.get("name")}
    for r in extra_rules or []:
        name = r.get("name")
        if name in idx_by_name:
            merged_rules[idx_by_name[name]] = r
        else:
            idx = len(merged_rules)
            merged_rules.append(r)
            if name:
                idx_by_name[name] = idx
    categories = _compute_categories_from_rules(merged_rules)
    return {"rules": merged_rules, "categories": categories}

def _load_dataset_case_rules(dataset: str) -> Dict[str, List[Dict[str, Any]]]:
    """   data/datasets_ad/rules/{dataset}.json ->   [{name, rules}]    {name: rules}   """
    try:
        script_dir = Path(__file__).resolve().parent.parent
        rules_path = script_dir / 'data' / 'datasets_ad' / 'rules' / f'{dataset}.json'
        if not rules_path.exists():
            return {}
        with open(rules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mapping: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get('name')
                rules = item.get('rules', []) if isinstance(item.get('rules'), list) else []
                if name:
                    mapping[name] = rules
        elif isinstance(data, dict):
            #            {name: rules}   
            for k, v in data.items():
                if isinstance(v, list):
                    mapping[str(k)] = v
        return mapping
    except Exception:
        return {}

def update_max_len_interval(max_len, dataset, sampling_interval):
    if dataset in ["TODS", "UCR", "Yahoo"]:
        return max_len, 24*3600*SEASON_DAYS.get(dataset, 4) // max_len
    elif dataset in ["Power"]:
        return 7*24*3600*SEASON_DAYS.get(dataset, 4) // sampling_interval, sampling_interval
    else:
        return 24*3600*SEASON_DAYS.get(dataset, 4) // sampling_interval, sampling_interval

def extract_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    in_flag = False
    start = 0
    for i, v in enumerate(labels):
        if v and not in_flag:
            in_flag = True
            start = i
        elif not v and in_flag:
            in_flag = False
            segs.append((start, i - 1))
    if in_flag:
        segs.append((start, len(labels) - 1))
    return segs


def slice_indices(length: int, max_len: int, overlap: bool) -> List[Tuple[int, int]]:
    if length <= max_len:
        return [(0, length - 1)]
    slices: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + max_len - 1, length - 1)
        slices.append((start, end))
        start = end + 1
    return slices


def build_entry(dataset: str, series: np.ndarray, global_slice: Tuple[int, int], segments: List[Tuple[int, int]], seq_id: str, part_idx: int, sampling_interval: int = 1) -> Dict[str, Any]:
    g_start, g_end = global_slice
    part = series[g_start:g_end + 1]
    arr = part
    stats = {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=0)),
        'max': float(arr.max()),
        'min': float(arr.min()),
        'max_pos': int(arr.argmax()),
        'min_pos': int(arr.argmin()),
        'start_value': float(arr[0]),
        'end_value': float(arr[-1]),
    }
    #    anomaly      
    clipped: List[Tuple[int, int]] = []
    for s, e in segments:
        if e < g_start or s > g_end:
            continue
        cs = max(s, g_start)
        ce = min(e, g_end)
        if cs <= ce:
            clipped.append((cs - g_start, ce - g_start))
    anomalies = []
    for idx, (s, e) in enumerate(clipped):
        anomalies.append({
            'name': f'anomaly_{dataset}_{seq_id}_{part_idx}_{idx}_{s}_{e}',
            'start_pos': int(s),
            'end_pos': int(e),
            'magnitude': max(float(np.abs(arr[s:e + 1].max() - arr.mean())), float(np.abs(arr[s:e + 1].min() - arr.mean()))),
            'pattern_type': 'unspecified',
            'description': f'Anomaly segment (dataset={dataset} seq={seq_id} part={part_idx}) from {s} to {e}',
            'additional_info': {}
        })
    entry = {
        'timeseries_info': {
            'metric_type': dataset.upper(),
            'metric_name': f'{dataset.upper()}##seq_{seq_id}##{part_idx}',
            'seq_len': len(part),
            'sampling_interval': int(sampling_interval),
            'value_type': 'float',
            'magnitude': float(arr.max() - arr.min()),
            'normal_range': [float(arr.min()), float(arr.max())],
            'baseline_value': float(arr.mean()),
            'characteristics': {},
            'statistics': stats,
            'additional_info': {
                'global_start': g_start,
                'global_end': g_end,
                'origin_seq_id': seq_id,
                'dataset': dataset
            }
        },
        'timeseries_data': arr.astype(float).tolist(),
        'human_intent': { 'rules': [], 'categories': [] },
        'anomalies_detected': anomalies,
        'anomalies_final': anomalies,
        'label': [[int(s), int(e)] for s, e in clipped]
    }
    return entry


def load_sequence(seq_dir: Path, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(seq_dir / f'{prefix}.npy')
    labels = np.load(seq_dir / f'{prefix}_label.npy')
    if labels.ndim > 1:
        labels = labels.squeeze()
    return data.astype(float), labels


def should_skip(seq_dir: Path, skip_normal: bool) -> bool:
    if not skip_normal:
        return False
    info_file = seq_dir / 'info.json'
    test_ratio = None
    if info_file.exists():
        try:
            info = json.load(open(info_file, 'r', encoding='utf-8'))
            test_ratio = info.get('testset anomaly ratio', None)
            if test_ratio is not None and test_ratio == 0:
                return True
        except Exception:
            pass
    #      :      0    
    try:
        labels = np.load(seq_dir / 'test_label.npy')
        if labels.ndim > 1:
            labels = labels.squeeze()
        if labels.sum() == 0:
            return True
    except Exception:
        return True
    return False


def _clip_segments_for_slice(segments: List[Tuple[int, int]], g_start: int, g_end: int) -> List[Tuple[int, int]]:
    clipped: List[Tuple[int, int]] = []
    for s, e in segments:
        if e < g_start or s > g_end:
            continue
        cs = max(s, g_start)
        ce = min(e, g_end)
        if cs <= ce:
            clipped.append((cs - g_start, ce - g_start))
    return clipped


def _save_part_plot(dataset: str, seq_id: str, part_idx: int, part: np.ndarray, clipped: List[Tuple[int, int]], vis_root: Path):
    if not _MATPLOTLIB_AVAILABLE:
        return
    ds_dir = vis_root / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = ds_dir / f"{seq_id}__part_{part_idx}.pdf"
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(part, color='#1f77b4', linewidth=0.8, label='value')
    ax.set_title(f"{dataset} seq={seq_id} part={part_idx} len={len(part)}")
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    has_anom = False
    for (s, e) in clipped:
        has_anom = True
        ax.axvspan(s, e, color='red', alpha=0.20)
        try:
            seg_idx = range(s, e + 1)
            ax.scatter(seg_idx, part[s:e+1], color='red', s=5, marker='o', edgecolors='black', linewidths=0.2, zorder=5)
        except Exception:
            pass
    if has_anom:
        from matplotlib.lines import Line2D
        custom_handles = [
            Line2D([0],[0], marker='o', color='red', markerfacecolor='red', markeredgecolor='black', markersize=5, linestyle='None', label='anomaly points'),
        ]
        ax.legend(handles=custom_handles, loc='upper right', fontsize=7, frameon=False)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.4)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    try:
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    finally:
        plt.close(fig)


def _collect_entries_for_prefix(
    dataset: str,
    dataset_dir: Path,
    prefix: str,
    max_len: int,
    skip_normal: bool,
    limit: int | None,
    filter: bool,
    overlap: int,
    split: str,
    vis_root: Path | None,
    enable_should_skip: bool,
    part_offset_by_seq: Dict[str, int] | None = None,
) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, int]]:
    seq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / f'{prefix}.npy').exists() and (d / f'{prefix}_label.npy').exists()]
    def sort_key(p: Path):
        name = p.name
        return (0, int(name)) if name.isdigit() else (1, name)

    seq_dirs.sort(key=sort_key)
    total = len(seq_dirs)
    entries: List[Dict[str, Any]] = []
    #          case-by-case      {seq_name: rules}
    case_rules_map = _load_dataset_case_rules(dataset)
    used_seq = 0
    slices_count_by_seq: Dict[str, int] = {}

    for seq_dir in seq_dirs:
        if limit and used_seq >= limit:
            break
        if prefix == 'test' and enable_should_skip and should_skip(seq_dir, skip_normal):
            continue
        try:
            data, labels = load_sequence(seq_dir, prefix)
        except Exception:
            continue

        segments = extract_segments(labels)
        if len(segments) == 0 and skip_normal:
            continue

        seq_id = seq_dir.name
        sampling_interval = 60
        info_path = seq_dir / 'info.json'
        if info_path.exists():
            try:
                _info = json.load(open(info_path, 'r', encoding='utf-8'))
                sampling_interval = _info.get('intervals', 60)
            except Exception:
                sampling_interval = 60

        if split == "season":
            max_len, sampling_interval = update_max_len_interval(max_len, dataset, sampling_interval)

        slices = slice_indices(len(data), max_len, overlap)
        part_offset = 0 if part_offset_by_seq is None else int(part_offset_by_seq.get(seq_id, 0))
        for part_idx, sl in enumerate(slices):
            merged_part_idx = part_idx + part_offset
            entry = build_entry(dataset, data, sl, segments, seq_id, merged_part_idx, sampling_interval=sampling_interval)
            entry['source_split'] = prefix

            if filter:
                global_filter = DATASET_FILTER[dataset] if dataset in DATASET_FILTER else {'rules': [], 'categories': []}
                extra_rules = case_rules_map.get(seq_id)
                if extra_rules:
                    entry['human_intent'] = _merge_human_intent(global_filter, extra_rules)
                else:
                    entry['human_intent'] = global_filter

            entries.append(entry)
            if vis_root is not None:
                g_start, g_end = sl
                part = data[g_start:g_end + 1]
                clipped = _clip_segments_for_slice(segments, g_start, g_end)
                _save_part_plot(dataset, seq_id, merged_part_idx, part, clipped, vis_root)

        used_seq += 1
        slices_count_by_seq[seq_id] = len(slices)

    return entries, total, used_seq, slices_count_by_seq


def process_dataset(dataset: str, base_dir: Path, output_dir: Path, max_len: int, skip_normal: bool, limit: int|None, is_train: bool, filter: bool, overlap: int = False, split: str = "default", vis_root: Path | None = None, enable_should_skip: bool = True, generate_merged: bool = True) -> Path:
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    prefix = 'train' if is_train else 'test'

    entries, total, used_seq, current_count_map = _collect_entries_for_prefix(
        dataset=dataset,
        dataset_dir=dataset_dir,
        prefix=prefix,
        max_len=max_len,
        skip_normal=skip_normal,
        limit=limit,
        filter=filter,
        overlap=overlap,
        split=split,
        vis_root=vis_root,
        enable_should_skip=enable_should_skip,
    )

    if split != "default":
        output_dir = output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_train:
        output_dir = output_dir / 'train'
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}.json" if not filter else output_dir / f"{dataset}_rules.json"
    save_ts_json(entries, output_path)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[{dataset}]\ttotal_seq={total} used_seq={used_seq} entries={len(entries)} -> {output_path}")

    if generate_merged:
        train_entries, train_total, train_used, train_count_map = (entries, total, used_seq, current_count_map)
        if prefix != 'train':
            train_entries, train_total, train_used, train_count_map = _collect_entries_for_prefix(
                dataset=dataset,
                dataset_dir=dataset_dir,
                prefix='train',
                max_len=max_len,
                skip_normal=skip_normal,
                limit=limit,
                filter=filter,
                overlap=overlap,
                split=split,
                vis_root=vis_root,
                enable_should_skip=enable_should_skip,
            )

        test_entries, test_total, test_used, _ = (entries, total, used_seq, current_count_map)
        if prefix != 'test':
            test_entries, test_total, test_used, _ = _collect_entries_for_prefix(
                dataset=dataset,
                dataset_dir=dataset_dir,
                prefix='test',
                max_len=max_len,
                skip_normal=skip_normal,
                limit=limit,
                filter=filter,
                overlap=overlap,
                split=split,
                vis_root=vis_root,
                enable_should_skip=enable_should_skip,
                part_offset_by_seq=train_count_map,
            )
        else:
            #     test      merged     test   part_idx      train
            test_entries, test_total, test_used, _ = _collect_entries_for_prefix(
                dataset=dataset,
                dataset_dir=dataset_dir,
                prefix='test',
                max_len=max_len,
                skip_normal=skip_normal,
                limit=limit,
                filter=filter,
                overlap=overlap,
                split=split,
                vis_root=vis_root,
                enable_should_skip=enable_should_skip,
                part_offset_by_seq=train_count_map,
            )

        merged_entries = train_entries + test_entries
        merged_dir = output_dir / 'merged'
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / (f"{dataset}.json" if not filter else f"{dataset}_rules.json")
        save_ts_json(merged_entries, merged_path)
        # with open(merged_path, 'w', encoding='utf-8') as f:
        #     json.dump(merged_entries, f, ensure_ascii=False, indent=2)
        print(
            f"[{dataset}][merged]\t"
            f"train_total_seq={train_total} train_used_seq={train_used} "
            f"test_total_seq={test_total} test_used_seq={test_used} "
            f"entries={len(merged_entries)} -> {merged_path}"
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert UTS univariate anomaly datasets to tsad format")
    parser.add_argument('--datasets', type=str, default='all', help='Comma-separated dataset names, or all.')
    parser.add_argument('--base-dir', type=str, help='UTS root directory containing dataset subdirectories.')
    parser.add_argument('--output-dir', type=str, help='Output directory. Defaults to data/tsad under the project root.')
    parser.add_argument('--max-len', type=int, default=1440*3)
    parser.add_argument('--overlap', type=int, default=0, help='Overlap time series slice; 1 to overlap, 0 to non overlap')
    parser.add_argument('--split', type=str, default="default", choices=["default", "short", "season", "long"], help='Dataset split/length mode.')
    parser.add_argument('--skip-normal', type=int, default=1, help='Skip normal samples; 1 to skip, 0 to keep')
    parser.add_argument('--enable-should-skip', type=int, default=1, help='Enable should_skip filter on test set; 1 to enable, 0 to disable')
    parser.add_argument('--generate-merged', type=int, default=1, help='Generate merged(train+test) output; 1 to generate, 0 to skip')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--is-train', type=int, default=0, help='process training set; 1 to process, 0 to skip')
    parser.add_argument('--filter', type=int, default=0, help='Add filter rules; 1 to add, 0 to skip')
    parser.add_argument('--plot', type=int, default=0, help='Generate plots; 1 to generate, 0 to skip')
    parser.add_argument('--vis-dir', type=str, help='Visualization root directory. Defaults to data/visualizations.')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    default_base = script_dir / 'data' / 'datasets_ad' / 'UTS'
    base_dir = Path(args.base_dir) if args.base_dir else default_base
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'data' / 'tsad'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets.lower() == 'all':
        datasets = SUPPORTED_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
        for d in datasets:
            if d not in SUPPORTED_DATASETS:
                raise ValueError(f"Unsupported dataset: {d}. Supported: {SUPPORTED_DATASETS}")

    vis_root = None
    if args.plot:
        vis_root = Path(args.vis_dir) if args.vis_dir else (script_dir / 'data' / 'visualizations' / 'datasets_ad')
        vis_root = vis_root / args.split / 'UTS' if args.split != "default" else vis_root / 'UTS'
        vis_root.mkdir(parents=True, exist_ok=True)
        if not _MATPLOTLIB_AVAILABLE:
            print('[WARN] matplotlib is unavailable. Skip visualization; install matplotlib and retry if plots are needed.')

    for ds in datasets:
        if args.split == "default":
            max_len = args.max_len
        elif args.split == "short":
            max_len = DATASET_SHORT_LENGTHS.get(ds, args.max_len)
        elif args.split == "season":
            max_len = DATASET_SEASON_LENGTHS.get(ds, args.max_len) * SEASON_DAYS.get(ds, 4)
        elif args.split == "long":
            max_len = DATASET_LONG_LENGTHS.get(ds, args.max_len)
        print(f"[INFO] default length of dataset [{ds}] is {max_len}")
        process_dataset(
            ds,
            base_dir,
            output_dir,
            max_len,
            args.skip_normal,
            args.limit,
            args.is_train,
            args.filter,
            args.overlap,
            args.split,
            vis_root,
            bool(args.enable_should_skip),
            bool(args.generate_merged),
        )


if __name__ == '__main__':
    main()
