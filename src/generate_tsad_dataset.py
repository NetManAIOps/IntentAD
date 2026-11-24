import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ts_generator import AnomalyDetectionDataGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), 'ts_generator'))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(generator_config, num_samples, reject, output_file=None, visualize_dir=None):
    generator = AnomalyDetectionDataGenerator(generator_config)
    generator.generate_dataset(num_samples=num_samples, reject=reject, output_file=output_file, visualize_dir=visualize_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IntentAD-style TSAD cases.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generated cases.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip PDF visualizations.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "tsad")
    parser.add_argument(
        "--visualize-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "visualizations" / "datasets_ad",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generator_config = {
        "sampling_interval": "random",  #         
        "typical_seq_len": [128, 256, 512],
        "typical_prob": 0.1,
        "seq_len_range": [128, 512],
        # "typical_seq_len": [512, 1024],
        # "typical_prob": 0.1,
        # "seq_len_range": [256, 768],
        "reject": None, # ["position", "pattern", "magnitude"], None
        "human_intent_count_range": [0, 4],
        "anomaly_count_range": [0, 3],
        "anomaly_type_range": [1, 3],
        "is_train": True,
        "is_filter": True,
        "is_short": False,
        "use_tool": False,
        "query_memory": True,
        "baseline_zero_min_zero_ratio": 0.55,
        #           5/6/7   
        # 5    (normal_case_ratio, has_final_anomaly_ratio, no_final_but_detected_ratio, has_filtered_anomaly_ratio, requires_history_case_ratio)
        # 6      5         baseline_zero_case_ratio
        # 7      6         no_intent_case_ratio human_intent.rules    
        # normal_case_ratio         simple     (1 - normal_case_ratio)
        #        normal/simple        requires_history_case_ratio   sum(rule.history_query_analysis.requires_history_count) > 0   case   
        "sample_type_ratio_tuple": (0.999, 0.75, 0.15, 0.5, 0.15, 0, 0.15),
        # "sample_type_ratio_tuple": (0.999, 0.75, 0.15, 0.5, 0, 0, 0.15),
        "enable_multiple_anomalies": True,
        "enable_complex_patterns": True,
        "visualize": not args.no_visualize
    }
    num_samples = args.num_samples
    reject = generator_config["reject"]
    output_root = str(args.output_root)
    visualize_root = str(args.visualize_root)
    if generator_config['is_short']:
        output_root = f"{output_root}/short"
        visualize_root = f"{visualize_root}/short"
    output_dir = f"{output_root}/generated/{'full' if generator_config['is_filter'] else 'ad_only'}"
    visualize_dir = f"{visualize_root}/generated/{'full' if generator_config['is_filter'] else 'ad_only'}"
    if reject:
        output_dir = f"{output_dir}/rl_" + "_".join(reject)
        visualize_dir = f"{visualize_dir}/rl_" + "_".join(reject)
    os.makedirs(output_dir, exist_ok=True)
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = dt_str if generator_config['is_train'] else f"{dt_str}_test_{num_samples}"
    output_file = os.path.join(output_dir, f"{file_name}.json")
    visualize_dir = os.path.join(visualize_dir, file_name)
    main(generator_config, num_samples, output_file=output_file, reject=reject, visualize_dir=visualize_dir if generator_config['visualize'] else None)
