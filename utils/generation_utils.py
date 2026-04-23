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
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import httpx
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import replicate

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8-sig") as f:
        model_config = yaml.safe_load(f) or {}


def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


# Initialize clients lazily or with robust defaults
gemini_client = None
anthropic_client = None
openai_client = None
openrouter_client = None
openrouter_api_key = ""
replicate_api_token = ""


def reinitialize_clients():
    """(Re)build all API clients from current env vars / config file.

    Called once at module load and can be called again at runtime
    (e.g. after the user sets new API keys via the Gradio UI).

    Returns a list of client names that were successfully initialized.
    """
    global gemini_client, anthropic_client, openai_client
    global openrouter_client, openrouter_api_key, replicate_api_token

    initialized = []

    api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("Initialized Gemini Client with API Key")
        initialized.append("Gemini")
    else:
        gemini_client = None

    key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
    if key:
        anthropic_client = AsyncAnthropic(api_key=key)
        print("Initialized Anthropic Client with API Key")
        initialized.append("Anthropic")
    else:
        anthropic_client = None

    key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
    if key:
        openai_client = AsyncOpenAI(api_key=key)
        print("Initialized OpenAI Client with API Key")
        initialized.append("OpenAI")
    else:
        openai_client = None

    openrouter_api_key = get_config_val(
        "api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", ""
    )
    if openrouter_api_key:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        print("Initialized OpenRouter Client with API Key")
        initialized.append("OpenRouter")
    else:
        openrouter_client = None
    
    replicate_api_token = get_config_val(
        "api_keys", "replicate_api_token", "REPLICATE_API_TOKEN", ""
    )
    if replicate_api_token:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        print("Initialized Replicate API Token")
        initialized.append("Replicate")

    return initialized


# Run once at import time (preserves original behaviour)
reinitialize_clients()


# ---------------------------------------------------------------------------
# Image size guard for Bedrock (max ~5MB per image for some models)
# ---------------------------------------------------------------------------


def ensure_image_under_limit(b64_jpg: str, max_bytes: int = 4_500_000) -> str:
    """Re-encode at lower JPEG quality if base64 image exceeds max_bytes."""
    raw = base64.b64decode(b64_jpg)
    if len(raw) <= max_bytes:
        return b64_jpg
    img = Image.open(BytesIO(raw))
    for quality in (70, 50, 30, 15):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    # Last resort: resize
    img.thumbnail((768, 768), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if "nanoviz" in model_name or "image" in model_name:
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                    if part.text is not None
                ]
            result_list.extend([r for r in raw_response_list if r and r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""

            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2**attempt), 30)

            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
    return openai_contents


# ---------------------------------------------------------------------------
# Bedrock Converse API (bearer token auth via httpx)
# ---------------------------------------------------------------------------


def _convert_to_bedrock_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert generic content list to Bedrock Converse API content format."""
    bedrock_content = []
    for item in contents:
        if item.get("type") == "text":
            bedrock_content.append({"text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                fmt = media_type.split("/")[-1]  # "jpeg", "png", etc.
                if fmt == "jpg":
                    fmt = "jpeg"
                # Ensure image is under size limit (compress if needed)
                b64_data = ensure_image_under_limit(source["data"], max_bytes=3_500_000)
                
                # Log image size for debugging
                decoded_size = len(base64.b64decode(b64_data))
                print(f"[Bedrock] Adding image: format={fmt}, size={decoded_size/1024:.1f}KB", flush=True)
                
                bedrock_content.append(
                    {
                        "image": {
                            "format": fmt,
                            "source": {"bytes": b64_data},
                        }
                    }
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                b64_data = ensure_image_under_limit(item["image_base64"], max_bytes=3_500_000)
                decoded_size = len(base64.b64decode(b64_data))
                print(f"[Bedrock] Adding image: format=jpeg, size={decoded_size/1024:.1f}KB", flush=True)
                
                bedrock_content.append(
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": b64_data},
                        }
                    }
                )
    return bedrock_content


async def call_bedrock_converse_with_retry_async(
    model_id, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call AWS Bedrock Converse API with ABSK bearer-token auth.

    Args:
        model_id: Bedrock model ID (e.g. "global.anthropic.claude-sonnet-4-6")
        contents: Generic content list (same format used everywhere)
        config: Dict with keys: system_prompt, temperature, candidate_num, max_completion_tokens
        max_attempts: Number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)
    """
    bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    region = os.environ.get("AWS_BEDROCK_REGION", "us-east-1")

    if not bearer_token:
        raise RuntimeError(
            "Bedrock bearer token not set. Please set AWS_BEARER_TOKEN_BEDROCK environment variable."
        )

    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    candidate_num = config.get("candidate_num", 1)
    max_tokens = config.get("max_completion_tokens", 8000)

    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    bedrock_content = _convert_to_bedrock_format(contents)

    payload = {
        "messages": [{"role": "user", "content": bedrock_content}],
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["system"] = [{"text": system_prompt}]
        # Add cachePoint after system text for Anthropic models on Bedrock
        if config.get("use_prompt_caching", False):
            payload["system"].append({"cachePoint": {"type": "default"}})

    # Log payload size for debugging
    import json as json_module
    payload_json = json_module.dumps(payload)
    payload_size_kb = len(payload_json.encode('utf-8')) / 1024
    
    # Log more details about the request
    system_size_kb = len(system_prompt.encode('utf-8')) / 1024 if system_prompt else 0
    user_content_size_kb = sum(
        len(str(b.get('text', '')).encode('utf-8')) 
        for b in bedrock_content if 'text' in b
    ) / 1024
    
    print(f"[Bedrock] Request to {model_id}:", flush=True)
    print(f"  - Payload size: {payload_size_kb:.1f}KB", flush=True)
    print(f"  - System prompt: {system_size_kb:.1f}KB", flush=True)
    print(f"  - User content: {user_content_size_kb:.1f}KB", flush=True)
    print(f"  - Content blocks: {len(bedrock_content)}", flush=True)
    print(f"  - Max tokens: {max_tokens}, Temperature: {temperature}", flush=True)
    
    # For debugging 503 errors, save the problematic request
    if os.environ.get("BEDROCK_DEBUG_DUMP"):
        debug_dir = Path("/tmp/bedrock_debug")
        debug_dir.mkdir(exist_ok=True)
        debug_file = debug_dir / f"request_{int(time.time())}.json"
        with open(debug_file, "w") as f:
            json_module.dump({
                "url": url,
                "payload": payload,
                "system_size_kb": system_size_kb,
                "user_content_size_kb": user_content_size_kb,
            }, f, indent=2)
        print(f"  - Debug dump saved to: {debug_file}", flush=True)

    response_text_list = []

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Extract text from Bedrock Converse response
            output = data.get("output", {})
            message = output.get("message", {})
            content_blocks = message.get("content", [])

            text_parts = []
            for block in content_blocks:
                if "text" in block:
                    text_parts.append(block["text"])

            if text_parts:
                response_text_list.append("\n".join(text_parts))
                # Log cache stats if available
                usage = data.get("usage", {})
                cache_read = usage.get("cacheReadInputTokens", 0)
                cache_write = usage.get("cacheWriteInputTokens", 0)
                if cache_read or cache_write:
                    print(
                        f"[Bedrock cache] input={usage.get('inputTokens', 0)}, "
                        f"cache_write={cache_write}, cache_read={cache_read}, "
                        f"output={usage.get('outputTokens', 0)}"
                    )
                break
            else:
                print(f"[Bedrock] Empty response, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            resp_text = e.response.text[:500]  # Increased from 300 to see more error details
            
            # Log full error for 503s to help debug
            if e.response.status_code == 503:
                print(f"[Bedrock DEBUG] Full 503 response: {e.response.text}", flush=True)
                print(f"[Bedrock DEBUG] Request had {len(bedrock_content)} content blocks", flush=True)
                print(f"[Bedrock DEBUG] Model: {model_id}", flush=True)
                print(f"[Bedrock DEBUG] Payload size: {payload_size_kb:.1f}KB", flush=True)
                
                # Check if this might be a model-specific issue
                if "qwen" in model_id.lower():
                    print(f"[Bedrock WARNING] Qwen models have ~15KB text input limit.", flush=True)
                    if payload_size_kb > 15:
                        print(f"[Bedrock WARNING] Your payload ({payload_size_kb:.1f}KB) exceeds this limit.", flush=True)
                        print(f"[Bedrock WARNING] Consider using Claude: bedrock/global.anthropic.claude-sonnet-4-6", flush=True)
            
            # For 503 errors, use longer delays (AWS service temporarily unavailable)
            if e.response.status_code == 503:
                current_delay = min(retry_delay * (3**attempt), 180)  # More aggressive backoff
            else:
                current_delay = min(retry_delay * (2**attempt), 120)

            # Don't retry on unrecoverable quota/daily-limit 429s
            if e.response.status_code == 429 and "per day" in resp_text.lower():
                print(
                    f"Bedrock daily token quota exceeded{context_msg}: {resp_text}. "
                    f"Not retrying — this limit resets daily."
                )
                return ["Error"] * candidate_num

            print(
                f"Bedrock attempt {attempt + 1}/{max_attempts} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {resp_text}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} Bedrock attempts failed{context_msg}")
                print(f"Last error: HTTP {e.response.status_code} - {resp_text}")
                return ["Error"] * candidate_num
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 120)
            print(
                f"Bedrock attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} Bedrock attempts failed{context_msg}")
                return ["Error"] * candidate_num

    if not response_text_list:
        return ["Error"] * candidate_num

    # Generate remaining candidates if needed
    remaining = candidate_num - len(response_text_list)
    if remaining > 0:
        tasks = []
        for _ in range(remaining):
            tasks.append(_bedrock_single_call(url, headers, payload))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating subsequent Bedrock candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res)

    return response_text_list


async def _bedrock_single_call(url: str, headers: dict, payload: dict) -> str:
    """Single Bedrock Converse API call (helper for parallel candidate generation)."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content_blocks = data.get("output", {}).get("message", {}).get("content", [])
    text_parts = [b["text"] for b in content_blocks if "text" in b]
    return "\n".join(text_parts) if text_parts else "Error"


# ---------------------------------------------------------------------------
# FLUX.2-dev local HTTP server (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


async def call_flux2_image_with_retry_async(
    prompt: str,
    flux_base_url: str = "http://localhost:30000",
    width: int = 1024,
    height: int = 1024,
    max_attempts: int = 5,
    retry_delay: int = 30,
    error_context: str = "",
) -> List[str]:
    """
    Call self-hosted FLUX.2-dev via local OpenAI-compatible endpoint.
    Returns a list with one base64-encoded image string.
    """
    url = f"{flux_base_url}/v1/images/generations"
    payload = {
        "model": "black-forest-labs/FLUX.2-dev",
        "prompt": prompt,
        "n": 1,
        "response_format": "b64_json",
        "size": f"{width}x{height}",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get("data") and data["data"][0].get("b64_json"):
                return [data["data"][0]["b64_json"]]
            else:
                print(f"[FLUX2] No image data in response, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 120)
            print(
                f"FLUX2 attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} FLUX2 attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    
    When config["use_prompt_caching"] is True, the system prompt is sent as a
    content block with cache_control={"type": "ephemeral"} so Anthropic caches
    the (large) static prefix.  Subsequent calls with the same prefix pay only
    $0.30/1M instead of $3.00/1M for input tokens.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = (
        config["max_output_tokens"]
        if "max_output_tokens" in config
        else config.get("max_completion_tokens", 50000)
    )
    use_caching = config.get("use_prompt_caching", False)
    response_text_list = []

    # Build system parameter: either a plain string or a list of content blocks
    # with cache_control for prompt caching.
    if use_caching:
        system_param = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_param = system_prompt

    # --- Preparation Phase ---
    current_contents = contents

    # --- Validation and Remediation Phase ---
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_param,
            )
            # Log caching stats if available
            if use_caching and hasattr(first_response, "usage"):
                u = first_response.usage
                cache_write = getattr(u, "cache_creation_input_tokens", 0)
                cache_read = getattr(u, "cache_read_input_tokens", 0)
                print(
                    f"[Claude cache] input={u.input_tokens}, "
                    f"cache_write={cache_write}, cache_read={cache_read}, "
                    f"output={u.output_tokens}"
                )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_param,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenAI returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }

    gen_params.update(
        {
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
    )

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)

            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(
                    f"[Warning]: Failed to generate image via OpenAI, no data returned."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openrouter_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter API (OpenAI-compatible) with asynchronous retry logic.
    """
    if openrouter_client is None:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key. "
            "Please set OPENROUTER_API_KEY in environment, or configure "
            "api_keys.openrouter_api_key in configs/model_config.yaml."
        )

    model_name = _to_openrouter_model_id(model_name)
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenRouter returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter attempt {attempt + 1} failed{context_msg}: {error_str}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(f"Error: All {max_attempts} OpenRouter attempts failed{context_msg}.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent OpenRouter candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openrouter_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter image generation via direct httpx POST.
    """
    if not openrouter_api_key:
        raise RuntimeError("OpenRouter client was not initialized: missing API key.")

    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1k")

    model_name = _to_openrouter_model_id(model_name)
    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                print(
                    f"[Warning]: OpenRouter image generation returned no choices, retrying..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            message = choices[0].get("message", {})

            # Try extracting from inline_data in content (Gemini-style)
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "inline_data" in part:
                        b64_data = part["inline_data"].get("data", "")
                        if b64_data:
                            return [b64_data]

            # Try extracting from images field (OpenRouter standard)
            images = message.get("images")
            if images and len(images) > 0:
                img_item = images[0]
                if isinstance(img_item, dict):
                    data_url = img_item.get("image_url", {}).get("url", "")
                else:
                    data_url = str(img_item)
                if "," in data_url:
                    b64_data = data_url.split(",", 1)[1]
                else:
                    b64_data = data_url
                if b64_data:
                    return [b64_data]

            # Try extracting base64 from text content
            if isinstance(content, str) and content.startswith("data:image"):
                if "," in content:
                    b64_data = content.split(",", 1)[1]
                    if b64_data:
                        return [b64_data]

            print(
                f"[Warning]: OpenRouter image generation returned no images, retrying..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {e.response.text}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2**attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
 

async def call_replicate_flux_with_retry_async(
    prompt: str,
    aspect_ratio: str = "1:1",
    max_attempts: int = 5,
    retry_delay: int = 30,
    error_context: str = "",
) -> list:
    """
    ASYNC: Call Replicate API for flux-2-pro model with retry logic.
    Returns a list with base64-encoded image string.
    """
    if not replicate_api_token:
        print("[Warning]: Replicate API token not configured. Skipping.")
        return ["Error"]

    for attempt in range(max_attempts):
        try:
            output = await asyncio.to_thread(
                replicate.run,
                "black-forest-labs/flux-2-pro",
                input={
                    "prompt": prompt,
                    "resolution": "1 MP",
                    "aspect_ratio": aspect_ratio,
                    "input_images": [],
                    "output_format": "webp",
                    "output_quality": 90,
                    "safety_tolerance": 2,
                },
            )

            # Read the image data
            image_data = await asyncio.to_thread(output.read)

            # Convert to base64
            base64_str = base64.b64encode(image_data).decode("utf-8")

            return [base64_str]

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"[Replicate] Attempt {attempt + 1}/{max_attempts} failed{context_msg}: {e}"
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"[Replicate] All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


def _to_openrouter_model_id(model_name: str) -> str:
    """Convert a bare model name to OpenRouter format (provider/model)."""
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini"):
        return f"google/{model_name}"
    return model_name


async def call_model_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Unified router that dispatches to the correct provider based on model_name.

    Routing rules:
      1. Explicit prefix overrides: "bedrock/" -> Bedrock, "openrouter/" -> OpenRouter,
         "claude-" -> Anthropic, "gpt-"/"o1-"/"o3-"/"o4-" -> OpenAI
      2. No prefix: auto-detect based on which API key is configured.
         Priority: OpenRouter > Gemini > Anthropic > OpenAI
    """
    # Explicit provider prefix overrides auto-detection
    if model_name.startswith("bedrock/"):
        provider = "bedrock"
        actual_model = model_name[len("bedrock/") :]
    elif model_name.startswith("openrouter/"):
        provider = "openrouter"
        actual_model = model_name[len("openrouter/") :]
    elif model_name.startswith("claude-"):
        provider = "anthropic"
        actual_model = model_name
    elif any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        provider = "openai"
        actual_model = model_name
    else:
        # Auto-detect provider based on which API key is configured
        actual_model = model_name
        if openrouter_client is not None:
            provider = "openrouter"
            actual_model = _to_openrouter_model_id(model_name)
        elif gemini_client is not None:
            provider = "gemini"
        elif anthropic_client is not None:
            provider = "anthropic"
        elif openai_client is not None:
            provider = "openai"
        else:
            raise RuntimeError(
                "No API client available. Please configure at least one API key "
                "in configs/model_config.yaml or via environment variables."
            )

    if provider == "gemini":
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    # Convert Gemini GenerateContentConfig -> dict for all other providers
    cfg_dict = {
        "system_prompt": (
            config.system_instruction if hasattr(config, "system_instruction") else ""
        ),
        "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
        "candidate_num": (
            config.candidate_count if hasattr(config, "candidate_count") else 1
        ),
        "max_completion_tokens": (
            config.max_output_tokens if hasattr(config, "max_output_tokens") else 8192
        ),
    }

    if provider == "bedrock":
        return await call_bedrock_converse_with_retry_async(
            model_id=actual_model,
            contents=contents,
            config=cfg_dict,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    call_fn = {
        "openrouter": call_openrouter_with_retry_async,
        "anthropic": call_claude_with_retry_async,
        "openai": call_openai_with_retry_async,
    }[provider]

    return await call_fn(
        model_name=actual_model,
        contents=contents,
        config=cfg_dict,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
