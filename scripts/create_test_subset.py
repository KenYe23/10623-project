"""
Create a small test subset (first N samples) for sanity-checking the pipeline.
Outputs: data/PaperBananaBench/{task}/test_mini.json

Usage:
    python scripts/create_test_subset.py --task diagram --n 100
    python scripts/create_test_subset.py --task plot --n 10
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="diagram", choices=["diagram", "plot"])
    parser.add_argument("--n", type=int, default=10, help="Number of samples to include")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "PaperBananaBench" / args.task

    # Load full test split
    test_path = data_dir / "test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    subset = test_data[: args.n]

    # Verify GT images exist
    missing = []
    for item in subset:
        img_path = data_dir / item["path_to_gt_image"]
        if not img_path.exists():
            missing.append(item["path_to_gt_image"])

    if missing:
        print(f"WARNING: {len(missing)} GT images not found:")
        for m in missing:
            print(f"  {m}")

    # Write subset
    out_path = data_dir / "test_mini100.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Created {out_path} with {len(subset)} samples (from {len(test_data)} total)")


if __name__ == "__main__":
    main()
