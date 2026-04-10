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
Parallel Critic Agent — Orchestrates two diverse VLM critics in parallel,
then passes both critiques to the Synthesizer for reconciliation.
"""

import asyncio
from typing import Dict, Any

from .base_agent import BaseAgent
from .critic_agent import CriticAgent
from .synthesizer_agent import SynthesizerAgent


class ParallelCriticAgent(BaseAgent):
    """Orchestrates two VLM critics in parallel, then synthesizes their outputs."""

    def __init__(self, critic_a_model: str, critic_b_model: str, **kwargs):
        super().__init__(**kwargs)
        # Create two independent CriticAgent instances with different models
        self.critic_a = CriticAgent(exp_config=self.exp_config)
        self.critic_a.model_name = critic_a_model
        self.critic_b = CriticAgent(exp_config=self.exp_config)
        self.critic_b.model_name = critic_b_model
        self.synthesizer = SynthesizerAgent(exp_config=self.exp_config)

    async def process(self, data: Dict[str, Any], source: str = "stylist") -> Dict[str, Any]:
        """
        Run both critics in parallel on the same data, then synthesize.

        Each critic writes to namespaced keys (_critic_a_*, _critic_b_*).
        The synthesizer reconciles them into canonical keys (target_{task}_critic_*).
        """
        round_idx = data.get("current_critic_round", 0)

        # Run critics in parallel — each writes to its own namespaced keys
        results = await asyncio.gather(
            self.critic_a.process(data, source=source, output_prefix="_critic_a"),
            self.critic_b.process(data, source=source, output_prefix="_critic_b"),
            return_exceptions=True,
        )

        # Log any individual critic failures but continue
        for i, res in enumerate(results):
            label = "A" if i == 0 else "B"
            if isinstance(res, Exception):
                print(f"⚠️ [ParallelCritic] Critic {label} failed in round {round_idx}: {res}")

        # Synthesize both critiques into canonical keys
        data = await self.synthesizer.process(data, round_idx=round_idx)
        return data
