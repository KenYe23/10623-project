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

"""Model-specific utilities for handling different VLM constraints."""


def is_qwen_model(model_name: str) -> bool:
    """Check if the model is a Qwen variant."""
    return "qwen" in model_name.lower()


def truncate_for_qwen(text: str, max_chars: int = 12000, strategy: str = "smart") -> str:
    """
    Truncate text for Qwen models which have ~15KB input text limit.
    
    Args:
        text: Input text to truncate
        max_chars: Maximum characters (default 12000 for ~13KB with safety margin)
        strategy: Truncation strategy
            - "start": Keep first max_chars
            - "end": Keep last max_chars
            - "smart": Keep first 60% + last 40% (preserves intro and conclusions)
    
    Returns:
        Truncated text with marker if truncation occurred
    """
    if len(text) <= max_chars:
        return text
    
    original_len = len(text)
    
    if strategy == "start":
        truncated = text[:max_chars]
        marker = f"\n\n[... Content truncated. Showing first {max_chars}/{original_len} characters ...]"
        return truncated + marker
    
    elif strategy == "end":
        truncated = text[-max_chars:]
        marker = f"[... Content truncated. Showing last {max_chars}/{original_len} characters ...]\n\n"
        return marker + truncated
    
    elif strategy == "smart":
        # Keep first 60% and last 40% to preserve intro + conclusion
        first_part_len = int(max_chars * 0.6)
        last_part_len = int(max_chars * 0.4)
        
        first_part = text[:first_part_len]
        last_part = text[-last_part_len:]
        
        marker = f"\n\n[... {original_len - max_chars} characters omitted from middle section ...]\n\n"
        return first_part + marker + last_part
    
    else:
        # Default to start strategy
        return truncate_for_qwen(text, max_chars, strategy="start")


def prepare_content_for_model(content: str, model_name: str, max_chars: int = 12000) -> str:
    """
    Prepare content for specific model constraints.
    
    Args:
        content: Raw content text
        model_name: Model identifier
        max_chars: Maximum characters for models with limits
    
    Returns:
        Content prepared for the specific model
    """
    if is_qwen_model(model_name):
        if len(content) > max_chars:
            print(f"[ModelUtils] Qwen detected - truncating content from {len(content)} to {max_chars} chars", flush=True)
            return truncate_for_qwen(content, max_chars=max_chars, strategy="smart")
    
    return content
