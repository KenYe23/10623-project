#!/usr/bin/env python3
"""
Post-hoc evaluation of a specific critic round from a completed pipeline run.

The main pipeline only evaluates the *final* generated image.  This script lets
you re-evaluate any intermediate round so you can build the mini-ablation table
(t=0, t=1, t=3) without re-running the full generation pipeline.

Image key mapping (for task_name="diagram"):
  round=0  →  target_diagram_stylist_desc0_base64_jpg   (pre-critic, initial image)
  round=1  →  target_diagram_critic_desc0_base64_jpg    (after 1 critic iteration)
  round=2  →  target_diagram_critic_desc1_base64_jpg    (after 2 critic iterations)
  round=3  →  target_diagram_critic_desc2_base64_jpg    (after 3 critic iterations)

Usage:
    python scripts/eval_round.py \
        --input  results/.../timestamp_dev_full_test.json \
        --round  1 \
        --output results/.../baseline_t1_eval.json \
        --eval_model_name "bedrock/anthropic.claude-opus-4-6-v1"

Does NOT require a GPU — only makes Bedrock API calls for evaluation.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.eval_toolkits import get_score_for_image_referenced


def get_image_key(task_name: str, round_num: int) -> str:
    """Map ablation round number to the JSON key holding that round's image."""
    if round_num == 0:
        return f"target_{task_name}_stylist_desc0_base64_jpg"
    else:
        return f"target_{task_name}_critic_desc{round_num - 1}_base64_jpg"


async def evaluate_round(
    data_list: list,
    round_num: int,
    task_name: str,
    eval_model_name: str,
    work_dir: Path,
    max_concurrent: int = 10,
) -> list:
    """Re-evaluate all samples at a given round, returning updated data."""

    image_key = get_image_key(task_name, round_num)
    eval_dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]

    # Pre-flight: check how many samples have the target image
    available = sum(1 for d in data_list if image_key in d and d[image_key])
    print(f"Round {round_num} → image key: {image_key}")
    print(f"  {available}/{len(data_list)} samples have this image.")
    if available == 0:
        print("ERROR: No samples have this image. Did the pipeline run enough rounds?")
        return data_list

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_one(sample: dict) -> dict:
        async with semaphore:
            # Clear previous eval scores
            for dim in eval_dims:
                sample.pop(f"{dim}_reasoning", None)
                sample.pop(f"{dim}_outcome", None)

            # Point eval at the target round's image
            sample["eval_image_field"] = image_key

            sample = await get_score_for_image_referenced(
                sample,
                task_name=task_name,
                model_name=eval_model_name,
                work_dir=work_dir,
            )
            return sample

    tasks = [asyncio.create_task(eval_one(d)) for d in data_list]
    results = []

    with tqdm(total=len(tasks), desc=f"Evaluating t={round_num}", ascii=True) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            pbar.update(1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate a specific critic round from a completed pipeline run."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the completed pipeline result JSON.")
    parser.add_argument("--round", type=int, required=True,
                        help="Ablation round to evaluate: 0=initial, 1=after 1 critic round, etc.")
    parser.add_argument("--output", type=str, default="",
                        help="Output path (default: <input>_t<round>_eval.json)")
    parser.add_argument("--task_name", type=str, default="diagram",
                        help="Task name (default: diagram)")
    parser.add_argument("--eval_model_name", type=str,
                        default="bedrock/anthropic.claude-opus-4-6-v1",
                        help="Model to use for evaluation (default: Bedrock Claude Opus 4.6)")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent evaluation calls (default: 10)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_name(f"{input_path.stem}_t{args.round}_eval.json")

    work_dir = Path(__file__).resolve().parent.parent

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Round:  t={args.round}")
    print(f"Model:  {args.eval_model_name}")
    print()

    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    print(f"Loaded {len(data_list)} samples.")

    results = asyncio.run(evaluate_round(
        data_list=data_list,
        round_num=args.round,
        task_name=args.task_name,
        eval_model_name=args.eval_model_name,
        work_dir=work_dir,
        max_concurrent=args.max_concurrent,
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Print summary
    dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]
    print(f"\n{'='*60}")
    print(f"Evaluation Summary — t={args.round}")
    print(f"{'='*60}")
    for dim in dims:
        outcomes = [d.get(f"{dim}_outcome", "N/A") for d in results]
        total = len(outcomes)
        model = outcomes.count("Model")
        human = outcomes.count("Human")
        tie = total - model - human
        print(f"  {dim:15s}  Model: {model:3d} ({100*model/total:.1f}%)  "
              f"Tie: {tie:3d} ({100*tie/total:.1f}%)  "
              f"Human: {human:3d} ({100*human/total:.1f}%)")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
