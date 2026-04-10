#!/usr/bin/env python3
"""
Aggregate evaluation JSONs into a mini-ablation table.

Usage:
    python scripts/ablation_table.py \
        --t0        results/.../baseline_t0_eval.json \
        --solo_t1   results/.../baseline_t1_eval.json \
        --solo_t3   results/.../baseline_dev_full_test.json \
        --debate_t1 results/.../parallel_t1_eval.json \
        --debate_t3 results/.../parallel_dev_parallel_debate_test.json

Outputs a formatted table (and CSV) with win rates per dimension per condition.
"""

import argparse
import json
import csv
import sys
from pathlib import Path


DIMS = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]


def compute_rates(data_list: list) -> dict:
    """Compute Model/Tie/Human win rates for each dimension."""
    rates = {}
    for dim in DIMS:
        outcomes = [d.get(f"{dim}_outcome", "N/A") for d in data_list]
        total = len([o for o in outcomes if o != "N/A"])
        if total == 0:
            rates[dim] = {"model": 0, "tie": 0, "human": 0, "n": 0}
            continue
        model = sum(1 for o in outcomes if o == "Model")
        human = sum(1 for o in outcomes if o == "Human")
        tie = total - model - human
        rates[dim] = {
            "model": round(100 * model / total, 1),
            "tie": round(100 * tie / total, 1),
            "human": round(100 * human / total, 1),
            "n": total,
        }
    return rates


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval JSONs into ablation table.")
    parser.add_argument("--t0", type=str, required=True,
                        help="t=0 eval JSON (shared initial image)")
    parser.add_argument("--solo_t1", type=str, required=True,
                        help="Solo Critic t=1 eval JSON")
    parser.add_argument("--solo_t3", type=str, required=True,
                        help="Solo Critic t=3 eval JSON (from main pipeline run)")
    parser.add_argument("--debate_t1", type=str, required=True,
                        help="Parallel Debate t=1 eval JSON")
    parser.add_argument("--debate_t3", type=str, required=True,
                        help="Parallel Debate t=3 eval JSON (from main pipeline run)")
    parser.add_argument("--csv", type=str, default="",
                        help="Optional: save table as CSV")
    args = parser.parse_args()

    conditions = [
        ("t=0 (shared)",      args.t0),
        ("Solo Critic t=1",   args.solo_t1),
        ("Solo Critic t=3",   args.solo_t3),
        ("Parallel Debate t=1", args.debate_t1),
        ("Parallel Debate t=3", args.debate_t3),
    ]

    all_rates = {}
    for label, path in conditions:
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {label}")
            continue
        data = load_json(path)
        all_rates[label] = compute_rates(data)
        print(f"Loaded {label}: {len(data)} samples from {path}")

    print()

    # ── Print table ──
    header = f"{'Condition':<25s}"
    for dim in DIMS:
        header += f" | {dim[:5].capitalize():>13s}"
    header += f" |   N"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label in [c[0] for c in conditions]:
        if label not in all_rates:
            continue
        rates = all_rates[label]
        row = f"{label:<25s}"
        for dim in DIMS:
            r = rates[dim]
            row += f" | {r['model']:4.1f}/{r['tie']:4.1f}/{r['human']:4.1f}"
        row += f" | {rates[DIMS[0]]['n']:3d}"
        print(row)

    print("=" * len(header))
    print("Format: Model% / Tie% / Human%")

    # ── Faithfulness delta ──
    if "Solo Critic t=3" in all_rates and "Parallel Debate t=3" in all_rates:
        solo = all_rates["Solo Critic t=3"]["faithfulness"]["model"]
        debate = all_rates["Parallel Debate t=3"]["faithfulness"]["model"]
        delta = debate - solo
        sign = "+" if delta >= 0 else ""
        print(f"\nFaithfulness Model Win Rate: Solo t=3 = {solo:.1f}%, "
              f"Parallel t=3 = {debate:.1f}%, Δ = {sign}{delta:.1f}pp")

    # ── CSV ──
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Condition"] + [d.capitalize() for d in DIMS] + ["N"])
            for label in [c[0] for c in conditions]:
                if label not in all_rates:
                    continue
                rates = all_rates[label]
                row = [label]
                for dim in DIMS:
                    r = rates[dim]
                    row.append(f"{r['model']}/{r['tie']}/{r['human']}")
                row.append(rates[DIMS[0]]["n"])
                writer.writerow(row)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
