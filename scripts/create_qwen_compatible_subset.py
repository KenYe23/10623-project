"""
Create a test subset filtered by content length for Qwen compatibility.
Outputs: data/PaperBananaBench/{task}/test_qwen_compatible.json

Qwen VL models have ~15KB text input limit. This script filters test cases
to only include those with methodology sections under the specified limit.

Usage:
    python scripts/create_qwen_compatible_subset.py --task diagram --max-kb 12
    python scripts/create_qwen_compatible_subset.py --task diagram --max-kb 10 --name test_qwen_10kb
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="diagram", choices=["diagram", "plot"])
    parser.add_argument("--max-kb", type=float, default=12.0, 
                        help="Maximum content size in KB (default: 12KB, leaving margin for Qwen's ~15KB limit)")
    parser.add_argument("--name", type=str, default="", 
                        help="Custom output name (default: test_qwen_compatible)")
    parser.add_argument("--source", type=str, default="test.json",
                        help="Source file to filter (default: test.json)")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "PaperBananaBench" / args.task

    # Load source split
    source_path = data_dir / args.source
    with open(source_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Filter by content length
    max_chars = int(args.max_kb * 1024)
    filtered = []
    excluded = []
    
    for item in test_data:
        content_len = len(item["content"])
        if content_len <= max_chars:
            filtered.append(item)
        else:
            excluded.append({
                "id": item["id"],
                "length_chars": content_len,
                "length_kb": content_len / 1024
            })

    # Sort excluded by length for better reporting
    excluded.sort(key=lambda x: x["length_kb"], reverse=True)

    # Print statistics
    print(f"Source: {source_path}")
    print(f"Max content size: {args.max_kb:.1f} KB ({max_chars:,} characters)")
    print()
    print(f"✓ Included: {len(filtered)}/{len(test_data)} test cases")
    print(f"✗ Excluded: {len(excluded)}/{len(test_data)} test cases")
    
    if filtered:
        print()
        print("Included tests:")
        for item in filtered:
            content_len = len(item["content"])
            print(f"  ✓ {item['id']}: {content_len:,} chars ({content_len/1024:.1f} KB)")
    
    if excluded:
        print()
        print("Excluded tests (too large for Qwen):")
        for exc in excluded:
            print(f"  ✗ {exc['id']}: {exc['length_chars']:,} chars ({exc['length_kb']:.1f} KB)")

    # Verify GT images exist
    missing = []
    for item in filtered:
        img_path = data_dir / item["path_to_gt_image"]
        if not img_path.exists():
            missing.append(item["path_to_gt_image"])

    if missing:
        print()
        print(f"WARNING: {len(missing)} GT images not found:")
        for m in missing:
            print(f"  {m}")

    # Write filtered subset
    out_name = args.name if args.name else "test_qwen_compatible"
    out_path = data_dir / f"{out_name}.json"
    
    if filtered:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print()
        print(f"✓ Created: {out_path}")
        print(f"  {len(filtered)} test cases under {args.max_kb:.1f} KB")
    else:
        print()
        print(f"✗ No test cases under {args.max_kb:.1f} KB limit - no file created")


if __name__ == "__main__":
    main()
