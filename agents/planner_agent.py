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
Vanilla Agent - Directly rendering images based on the method section.
"""

import json
from typing import Dict, Any
from google.genai import types
import base64, io, asyncio
from PIL import Image

from utils import generation_utils
from utils.image_utils import resolve_image_path
from utils.model_utils import is_qwen_model
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Planner Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.main_model_name

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.system_prompt = PLOT_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
            }
        else:
            # Use Qwen-optimized prompt if Qwen model is detected
            if is_qwen_model(self.model_name):
                self.system_prompt = DIAGRAM_PLANNER_QWEN_SYSTEM_PROMPT
            else:
                self.system_prompt = DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "content_label": "Methodology Section",
                "visual_intent_label": "Diagram Caption",
            }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified processing method that works for both diagram and plot tasks.
        Uses task_config to determine task-specific parameters.
        Expects data['top10_references'] to be already populated by retriever.
        """
        cfg = self.task_config

        raw_content = data["content"]
        content = (
            json.dumps(raw_content)
            if isinstance(raw_content, (dict, list))
            else raw_content
        )
        description = data["visual_intent"]

        content_list = []

        # Check if retriever has already provided full examples (e.g., in manual mode)
        examples = data.get("retrieved_examples", [])
        if not examples:
            retrieved_ids = data.get("top10_references", [])
            if retrieved_ids:
                with open(
                    self.exp_config.work_dir
                    / f"data/PaperBananaBench/{cfg['task_name']}/ref.json",
                    "r",
                    encoding="utf-8",
                ) as f:
                    candidate_pool = json.load(f)
                id_to_item = {item["id"]: item for item in candidate_pool}
                examples = [
                    id_to_item[ref_id]
                    for ref_id in retrieved_ids
                    if ref_id in id_to_item
                ]

        user_prompt = ""
        for idx, item in enumerate(examples):
            user_prompt += f"Example {idx+1}:\n"

            item_content = item["content"]
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content)

            user_prompt += f"{cfg['content_label']}: {item_content}\n"
            user_prompt += f"{cfg['visual_intent_label']}: {item['visual_intent']}\nReference {cfg['task_name'].capitalize()}: "
            content_list.append({"type": "text", "text": user_prompt})

            # Resolve relative path using work_dir
            image_path = (
                self.exp_config.work_dir
                / f"data/PaperBananaBench/{cfg['task_name']}"
                / item["path_to_gt_image"]
            )
            image_path = resolve_image_path(image_path)
            with open(image_path, "rb") as f:
                ref_image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Log image size
            image_size_kb = len(base64.b64decode(ref_image_base64)) / 1024
            print(f"[PlannerAgent] Loading reference image {idx+1}: {image_path.name}, size={image_size_kb:.1f}KB", flush=True)
            
            content_list.append({"type": "image", "image_base64": ref_image_base64})
            user_prompt = ""

        user_prompt += f"Now, based on the following {cfg['content_label'].lower()} and {cfg['visual_intent_label'].lower()}, provide a detailed description for the figure to be generated.\n"
        user_prompt += f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {description}\n"
        user_prompt += "Detailed description of the target figure to be generated"
        if cfg["task_name"] == "diagram":
            user_prompt += " (do not include figure titles)"
        user_prompt += ":"

        content_list.append({"type": "text", "text": user_prompt})

        print(f"[PlannerAgent] Calling Bedrock with {len(content_list)} content blocks ({len([c for c in content_list if c.get('type')=='image'])} images)", flush=True)

        response_list = await generation_utils.call_model_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_count=1,
                max_output_tokens=8000,
            ),
            max_attempts=3,
            retry_delay=5,
            error_context="PlannerAgent",
        )

        for idx, response in enumerate(response_list):
            data[f"target_{cfg['task_name']}_desc{idx}"] = response.strip()

        return data


DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"""

DIAGRAM_PLANNER_QWEN_SYSTEM_PROMPT = """
Task: Generate detailed diagram descriptions for academic papers.

Input: Methodology section, figure caption, and reference examples.
Output: Comprehensive diagram description suitable for image generation.

Core Requirements:
1. Visual Structure: Describe all components (boxes, arrows, layers) and their spatial arrangement
2. Content: Label each element with precise text from the methodology
3. Connections: Specify all data/control flows between components
4. Style Details: Background (white/light), colors, line styles, font sizes, icon types

Key Principles:
- Focus on the CORE methodology flow and main contributions
- Learn layout patterns and design aesthetics from the provided examples
- Be explicit and detailed - vague descriptions produce poor images
- Extract the essential workflow even from lengthy methodology sections
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""
