#!/usr/bin/env python3
"""
Merge multiple shard result JSON files into one deduplicated JSON.

Usage:
    python scripts/merge_shard_results.py \
        --inputs results/PaperBananaBench_diagram/*.json \
        --output results/PaperBananaBench_diagram/merged_parallel_test.json

Dedup key priority per sample:
    candidate_id -> paper_id -> id
"""

import argparse
import glob
import json
from pathlib import Path


def sample_key(item: dict) -> str:
    return str(item.get("candidate_id") or item.get("paper_id") or item.get("id") or "")


def main():
    parser = argparse.ArgumentParser(description="Merge shard result JSON files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input file patterns or explicit JSON files (supports globs).",
    )
    parser.add_argument("--output", required=True, help="Output merged JSON path.")
    args = parser.parse_args()

    # Expand globs
    input_files = []
    for p in args.inputs:
        matches = sorted(glob.glob(p))
        if matches:
            input_files.extend(matches)
        else:
            input_files.append(p)

    # Keep unique paths while preserving order
    seen_paths = set()
    unique_files = []
    for p in input_files:
        rp = str(Path(p).resolve())
        if rp not in seen_paths:
            seen_paths.add(rp)
            unique_files.append(Path(p))

    merged = []
    seen_ids = set()

    for path in unique_files:
        if not path.exists():
            print(f"[Skip] Missing file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[Skip] Not a JSON list: {path}")
            continue

        added = 0
        skipped_dup = 0
        for item in data:
            k = sample_key(item)
            if not k:
                # If no ID fields, include as-is (rare)
                merged.append(item)
                added += 1
                continue
            if k in seen_ids:
                skipped_dup += 1
                continue
            seen_ids.add(k)
            merged.append(item)
            added += 1

        print(f"[Read] {path} -> added {added}, skipped duplicates {skipped_dup}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[Done] Wrote {len(merged)} samples to {out_path}")


if __name__ == "__main__":
    main()
