"""
Create a test subset for sanity-checking the pipeline.
Outputs: data/PaperBananaBench/{task}/test_mini.json

Usage:
    python scripts/create_test_subset.py --task diagram --n 10
    python scripts/create_test_subset.py --task diagram --start 50 --n 50   # samples 51-100
    python scripts/create_test_subset.py --task diagram --start 50 --n 10   # samples 51-60
    python scripts/create_test_subset.py --task plot --start 0 --n 10
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="diagram", choices=["diagram", "plot"])
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to include")
    parser.add_argument("--name", type=str, default="", help="Custom output name (default: test_mini)")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "PaperBananaBench" / args.task

    # Load full test split
    test_path = data_dir / "test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    end = args.start + args.n
    subset = test_data[args.start:end]

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
    out_name = args.name if args.name else f"test_mini_{args.start + 1}_{end}"
    out_path = data_dir / f"{out_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Created {out_path} with {len(subset)} samples (indices {args.start}-{end - 1}, from {len(test_data)} total)")


if __name__ == "__main__":
    main()
