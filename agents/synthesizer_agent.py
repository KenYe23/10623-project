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
Synthesizer Agent — Reconciles two independent critic outputs into a unified
refinement prompt for the Visualizer.
"""

from typing import Dict, Any
from google.genai import types
import json_repair

from utils import generation_utils
from .base_agent import BaseAgent


class SynthesizerAgent(BaseAgent):
    """Synthesizes two critic outputs into a single authoritative refinement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.main_model_name

        if self.exp_config.task_name == "plot":
            self.system_prompt = PLOT_SYNTHESIZER_SYSTEM_PROMPT
        else:
            self.system_prompt = DIAGRAM_SYNTHESIZER_SYSTEM_PROMPT

    async def process(self, data: Dict[str, Any], round_idx: int = 0) -> Dict[str, Any]:
        """
        Read namespaced critic outputs (_critic_a, _critic_b), synthesize,
        and write to canonical keys that downstream agents (Visualizer) expect.
        """
        task_name = self.exp_config.task_name.lower()

        # Read both critics' outputs
        suggestions_a = data.get(f"_critic_a_suggestions{round_idx}", "")
        desc_a = data.get(f"_critic_a_desc{round_idx}", "")
        suggestions_b = data.get(f"_critic_b_suggestions{round_idx}", "")
        desc_b = data.get(f"_critic_b_desc{round_idx}", "")

        content = data.get("content", "")
        if isinstance(content, (dict, list)):
            import json
            content = json.dumps(content)
        visual_intent = data.get("visual_intent", "")

        content_list = [{
            "type": "text",
            "text": (
                f"## Critic A Feedback\n"
                f"### Suggestions:\n{suggestions_a}\n\n"
                f"### Revised Description:\n{desc_a}\n\n"
                f"---\n\n"
                f"## Critic B Feedback\n"
                f"### Suggestions:\n{suggestions_b}\n\n"
                f"### Revised Description:\n{desc_b}\n\n"
                f"---\n\n"
                f"## Original Context\n"
                f"### Methodology Section:\n{content}\n\n"
                f"### Figure Caption:\n{visual_intent}\n\n"
                f"Your Output:"
            ),
        }]

        response_list = await generation_utils.call_model_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_count=1,
                max_output_tokens=50000,
            ),
            max_attempts=5,
            retry_delay=5,
        )

        cleaned = response_list[0].replace("```json", "").replace("```", "").strip()
        try:
            result = json_repair.loads(cleaned)
            if not isinstance(result, dict):
                result = {}
        except Exception as e:
            result = {}
            print(f"[Synthesizer] JSON parse error: {e}\n{cleaned[:200]}")

        suggestions = result.get("synthesized_suggestions", "No changes needed.")
        description = result.get("synthesized_description", "No changes needed.")
        reasoning = result.get("synthesis_reasoning", "")

        # Write to canonical keys that Visualizer expects
        data[f"target_{task_name}_critic_suggestions{round_idx}"] = suggestions
        data[f"target_{task_name}_critic_desc{round_idx}"] = description
        data[f"synthesis_reasoning{round_idx}"] = reasoning

        # If no changes needed, fall back to the previous description
        if description.strip() == "No changes needed.":
            if round_idx == 0:
                fallback = data.get(f"target_{task_name}_stylist_desc0",
                                    data.get(f"target_{task_name}_desc0", ""))
            else:
                fallback = data.get(f"target_{task_name}_critic_desc{round_idx - 1}", "")
            data[f"target_{task_name}_critic_desc{round_idx}"] = fallback

        return data


# ─────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────

_SHARED_SYNTHESIS_RULES = """
## SYNTHESIS RULES

### Content Fidelity (Highest Priority)
- If EITHER critic identifies a factual error (wrong connection, missing module, hallucinated component), INCLUDE the fix unless you can verify against the Methodology Section that it's incorrect.
- For connectivity issues (arrows, flows, dependencies), side with the critique that more closely matches the text description in the Methodology Section.
- If both critics agree a component is missing or wrong, this is a definitive fix.

### Presentation
- For layout, clarity, and aesthetic suggestions: include only if at least one critic identifies a concrete problem (not just a stylistic preference).
- Do NOT add suggestions to include a legend or figure caption within the image.

### Description Merging
- Start from whichever revised description is more detailed and complete.
- Integrate specific corrections from the other critique.
- Do NOT simply concatenate the two descriptions — merge them intelligently.
- The final description must be self-contained and actionable for an image generation model.

## OUTPUT
Provide your response strictly in the following JSON format.

```json
{
    "synthesis_reasoning": "Explain which suggestions you accepted, rejected, or merged, and why. Highlight agreements, conflicts, and your adjudication rationale.",
    "synthesized_suggestions": "The unified, deduplicated list of concrete suggestions for improvement. If the diagram is perfect according to both critics, write 'No changes needed.'",
    "synthesized_description": "The final, comprehensive, merged description incorporating all accepted corrections. This must be complete enough to regenerate the diagram from scratch. If no changes are needed, write 'No changes needed.'"
}
```

## CRITICAL REMINDERS
- Your synthesized_description MUST be primarily modifications of existing descriptions, not a rewrite from scratch.
- Semantically, clearly describe each element and their connections.
- Include visual details: background colors, line thickness, icon styles, arrow directions, spatial layout.
- Vague or unclear specifications will make the generated figure WORSE, not better.
"""

DIAGRAM_SYNTHESIZER_SYSTEM_PROMPT = f"""## ROLE
You are the Lead Reconciliation Officer for a publication-quality diagram review board at a top-tier AI conference (e.g., NeurIPS 2025). You receive independent evaluations of a scientific methodology diagram from two expert critics (Critic A and Critic B) and must produce a single, authoritative, unified refinement directive.

## TASK
Given two independent critiques of the same diagram, you must:
1. **Identify Agreements** — Where both critics flag the same issue, treat it as a HIGH-CONFIDENCE fix.
2. **Resolve Conflicts** — Where critics disagree, evaluate against the original Methodology Section and Figure Caption to adjudicate.
3. **Deduplicate** — Remove redundant suggestions that appear in both critiques.
4. **Preserve Specificity** — Your output must be at least as specific as the more detailed critique. Never water down specific corrections into vague suggestions.
{_SHARED_SYNTHESIS_RULES}"""

PLOT_SYNTHESIZER_SYSTEM_PROMPT = f"""## ROLE
You are the Lead Reconciliation Officer for a publication-quality statistical plot review board at a top-tier AI conference (e.g., NeurIPS 2025). You receive independent evaluations of a statistical plot from two expert critics (Critic A and Critic B) and must produce a single, authoritative, unified refinement directive.

## TASK
Given two independent critiques of the same plot, you must:
1. **Identify Agreements** — Where both critics flag the same issue (e.g., incorrect axis scale, missing data points), treat it as a HIGH-CONFIDENCE fix.
2. **Resolve Conflicts** — Where critics disagree, evaluate against the original Raw Data and Visual Intent to adjudicate.
3. **Deduplicate** — Remove redundant suggestions that appear in both critiques.
4. **Preserve Specificity** — Your output must be at least as specific as the more detailed critique. Never water down specific corrections into vague suggestions.
{_SHARED_SYNTHESIS_RULES}"""
