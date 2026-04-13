# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main script to launch PaperVizAgent
"""

import asyncio
import json
import argparse
from pathlib import Path
import aiofiles
import numpy as np

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent
from agents.parallel_critic_agent import ParallelCriticAgent

from utils import config, paperviz_processor


async def main():
    """Main function"""
    # add command line args
    parser = argparse.ArgumentParser(description="PaperVizAgent processing script")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PaperBananaBench",
        help="name of the dataset to use (default: PaperBananaBench)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="diagram",
        choices=["diagram", "plot"],
        help="task type: diagram or plot (default: diagram)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="split of the dataset to use (default: test)",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="dev",
        help="name of the experiment to use (default: dev)",
    )
    parser.add_argument(
        "--retrieval_setting",
        type=str,
        default="auto",
        choices=["auto", "manual", "random", "none"],
        help="retrieval setting for planner agent (default: auto)",
    )
    parser.add_argument(
        "--max_critic_rounds",
        type=int,
        default=3,
        help="maximum number of critic rounds (default: 3)",
    )
    parser.add_argument(
        "--main_model_name",
        type=str,
        default="",
        help="main model name to use (default: " ")",
    )
    parser.add_argument(
        "--image_gen_model_name",
        type=str,
        default="",
        help="image generation model name to use (default: " ")",
    )
    parser.add_argument(
        "--critic_b_model_name",
        type=str,
        default="",
        help="second critic model for parallel debate mode (default: none)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from previous run, skipping already-processed samples",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="limit number of samples to process (0 = all)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index (inclusive) for sharding samples (default: 0)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="end index (exclusive) for sharding samples; -1 means until end",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="maximum number of samples processed concurrently",
    )
    args = parser.parse_args()

    exp_config = config.ExpConfig(
        dataset_name=args.dataset_name,
        task_name=args.task_name,
        split_name=args.split_name,
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        max_critic_rounds=args.max_critic_rounds,
        main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name,
        critic_b_model_name=args.critic_b_model_name,
        work_dir=Path(__file__).parent,
    )

    base_path = Path(__file__).parent / "data" / exp_config.dataset_name
    input_filename = base_path / exp_config.task_name / f"{exp_config.split_name}.json"
    output_filename = exp_config.result_dir / f"{exp_config.exp_name}.json"

    print(f"Input file: {input_filename}", f"Output file: {output_filename}")
    with open(input_filename, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Optional shard selection by index range
    total_before_slice = len(data_list)
    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx and args.end_idx > 0 else total_before_slice
    end_idx = min(end_idx, total_before_slice)
    if start_idx >= end_idx:
        print(
            f"Invalid shard range: start_idx={start_idx}, end_idx={end_idx}, "
            f"total={total_before_slice}. Nothing to process."
        )
        return
    if start_idx != 0 or end_idx != total_before_slice:
        data_list = data_list[start_idx:end_idx]
        print(
            f"[Shard] Selected index range [{start_idx}, {end_idx}) "
            f"=> {len(data_list)} samples (from {total_before_slice})."
        )

    # --- Resume logic: skip already-processed samples ---
    all_result_list = []
    if args.resume and output_filename.exists():
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            all_result_list = existing_results
            processed_ids = {
                d.get("candidate_id") or d.get("paper_id") or d.get("id", "")
                for d in existing_results
            }
            data_list = [
                d
                for d in data_list
                if (d.get("candidate_id") or d.get("paper_id") or d.get("id", ""))
                not in processed_ids
            ]
            print(
                f"[Resume] Loaded {len(existing_results)} existing results. {len(data_list)} samples remaining."
            )
        except (json.JSONDecodeError, Exception) as e:
            print(
                f"[Resume] Could not load existing results ({e}). Starting from scratch."
            )
            all_result_list = []

    if not data_list:
        print("All samples already processed. Nothing to do.")
        return

    # Optional smoke-test cap
    if args.max_samples > 0:
        original_len = len(data_list)
        data_list = data_list[: args.max_samples]
        print(
            f"[Sample Cap] Processing first {len(data_list)} samples "
            f"(from {original_len} remaining samples)."
        )

    # --- Create agents ---
    parallel_critic = None
    if exp_config.critic_b_model_name:
        parallel_critic = ParallelCriticAgent(
            critic_a_model=exp_config.main_model_name,
            critic_b_model=exp_config.critic_b_model_name,
            exp_config=exp_config,
        )

    # Create processor
    processor = paperviz_processor.PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
        parallel_critic_agent=parallel_critic,
    )

    # Batch process documents
    concurrent_num = max(1, args.max_concurrent)
    print(f"Using max concurrency: {concurrent_num}")

    async def save_results_and_scores(current_results):
        print(
            f"Incremental saving results (count: {len(current_results)}) to {output_filename}"
        )
        async with aiofiles.open(
            output_filename, "w", encoding="utf-8", errors="surrogateescape"
        ) as f:
            json_string = json.dumps(current_results, ensure_ascii=False, indent=4)
            json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
            await f.write(json_string)

    # Process samples incrementally
    idx = 0
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num
    ):
        all_result_list.append(result_data)
        idx += 1
        if idx % 10 == 0:
            await save_results_and_scores(all_result_list)

    # Final save
    await save_results_and_scores(all_result_list)
    print("Processing completed.")


if __name__ == "__main__":
    asyncio.run(main())
