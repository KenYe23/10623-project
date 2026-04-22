#!/usr/bin/env python3
"""Local HTTP server for FLUX.2-dev text-to-image generation (4-bit quantized).

Uses the diffusers library with bitsandbytes 4-bit quantization so the model
fits on a single H100 (or even an RTX 4090/5090 with ~20 GB VRAM).

Exposes an OpenAI-compatible endpoint used by PaperBanana:
- GET  /health
- GET  /v1/models
- POST /v1/images/generations
"""

from __future__ import annotations

import sys
import argparse
import base64
import io
import json
import random
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

# import requests
import torch
from diffusers import AutoModel, Flux2Pipeline
# from huggingface_hub import get_token
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
from gradio_client import Client


class Flux2Service:
    def __init__(
        self,
        repo_id: str,
        num_steps: int,
        guidance: float,
        cpu_offloading: bool,
        remote_text_encoder: bool,
    ):
        self.repo_id = repo_id
        self.num_steps = num_steps
        self.guidance = guidance
        self.device = "cuda:0"
        self.remote_text_encoder = remote_text_encoder
        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        print("=" * 80, flush=True)
        print("[INIT] Starting Flux2Service initialization", flush=True)
        print(f"[INIT] Repo: {repo_id}", flush=True)
        print(f"[INIT] Remote text encoder: {remote_text_encoder}", flush=True)
        print(f"[INIT] CPU offloading: {cpu_offloading}", flush=True)
        print("=" * 80, flush=True)

        self.tokenizer = None
        self.remote_encoder_client = None

        if remote_text_encoder:
            print("Using remote text encoder (Gradio Space) …", flush=True)
            self.remote_encoder_client = Client("multimodalart/mistral-text-encoder")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    subfolder="tokenizer",
                )
                print(f"✓ Loaded tokenizer from {repo_id}/tokenizer", flush=True)
            except Exception as e:
                print(f"⚠️ Could not load tokenizer: {e}", flush=True)
                print("⚠️ Will use character-based truncation fallback", flush=True)
                self.tokenizer = None
            text_encoder = None
        else:
            print(
                f"Loading 4-bit text encoder from {repo_id} (dtype={self.torch_dtype}) …",
                flush=True,
            )
            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                torch_dtype=self.torch_dtype,
                device_map="cpu",
            )

        print(f"Loading 4-bit transformer from {repo_id} …", flush=True)
        dit_device_map = self.device if remote_text_encoder else "cpu"
        dit = AutoModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
            device_map=dit_device_map,
        )

        print("Building Flux2Pipeline …", flush=True)
        self.pipe = Flux2Pipeline.from_pretrained(
            repo_id,
            text_encoder=text_encoder,
            transformer=dit,
            torch_dtype=self.torch_dtype,
        )

        if remote_text_encoder:
            print("Keeping transformer on GPU (text encoder is remote)", flush=True)
            self.pipe.to(self.device)
        elif cpu_offloading:
            print("Enabling CPU offloading for memory efficiency", flush=True)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self._lock = threading.Lock()
        print("=" * 80, flush=True)
        print("Flux2Service ready.", flush=True)
        print("=" * 80, flush=True)

    @staticmethod
    def _normalize_dims(width: int, height: int) -> tuple[int, int]:
        width = max(256, int(width))
        height = max(256, int(height))
        width = (width // 16) * 16
        height = (height // 16) * 16
        return max(width, 16), max(height, 16)

    @staticmethod
    def _pil_to_b64_png(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _truncate_prompt(self, prompt: str, max_tokens: int = 512) -> str:
        """Truncate prompt to fit within token limit for remote text encoder."""
        if not self.tokenizer:
            # Fallback: character-based truncation (roughly 4 chars per token)
            max_chars = max_tokens * 4
            if len(prompt) <= max_chars:
                print(f"✓ Prompt within limit: ~{len(prompt)//4} tokens (character-based estimate)", flush=True)
                return prompt
            print(f"⚠️  Prompt too long: ~{len(prompt)//4} tokens, truncating to {max_tokens}...", flush=True)
            truncated_prompt = prompt[:max_chars]
            print(f"✓ Truncated prompt: ~{max_tokens} tokens (character-based)", flush=True)
            return truncated_prompt
        
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) <= max_tokens:
            print(f"✓ Prompt within limit: {len(tokens)} tokens", flush=True)
            return prompt
        
        print(f"⚠️  Prompt too long: {len(tokens)} tokens, truncating to {max_tokens}...", flush=True)
        truncated_tokens = tokens[:max_tokens]
        truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        print(f"✓ Truncated prompt: {len(truncated_tokens)} tokens", flush=True)
        return truncated_prompt

    def _call_remote_text_encoder(self, prompt: str) -> torch.Tensor:
        prompt = self._truncate_prompt(prompt, max_tokens=512)
        print(f"📡 Calling remote text encoder for prompt ({len(prompt)} chars)...", flush=True)

        try:
            result = self.remote_encoder_client.predict(
                prompt=prompt,
                api_name="/encode_text",
            )

            # HF Space returns a path to a serialized tensor file
            prompt_embeds = torch.load(result[0], weights_only=False)
            print(
                f"✓ Prompt embeds loaded: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}",
                flush=True,
            )
            return prompt_embeds.to(self.device)

        except Exception as e:
            print(f"❌ Remote text encoder failed: {e}", flush=True)
            raise

    def generate(
        self, prompt: str, width: int, height: int, seed: int | None = None
    ) -> str:
        width, height = self._normalize_dims(width, height)
        if seed is None:
            seed = random.randrange(2**31)

        print("="*80, flush=True)
        print(f"🎨 Generating image: {width}x{height}, seed={seed}", flush=True)
        print(f"   Prompt: {prompt[:100]}...", flush=True)
        sys.stdout.flush()

        with self._lock:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            if self.remote_text_encoder:
                # Use remote text encoder
                prompt_embeds = self._call_remote_text_encoder(prompt)
                print(f"🔄 Running diffusion pipeline with remote embeds...", flush=True)
                sys.stdout.flush()
                result = self.pipe(
                    prompt_embeds=prompt_embeds,
                    width=width,
                    height=height,
                    generator=generator,
                    num_inference_steps=self.num_steps,
                    guidance_scale=self.guidance,
                )
            else:
                # Use local text encoder
                print(f"🔄 Running diffusion pipeline with local text encoder...", flush=True)
                sys.stdout.flush()
                result = self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    generator=generator,
                    num_inference_steps=self.num_steps,
                    guidance_scale=self.guidance,
                )
            
            print(f"✓ Diffusion complete, encoding to PNG...", flush=True)
            sys.stdout.flush()
            img = result.images[0]
            b64 = self._pil_to_b64_png(img)
            print(f"✓ Image generated successfully ({len(b64)} bytes base64)", flush=True)
            print("="*80, flush=True)
            sys.stdout.flush()
            return b64


class FluxHandler(BaseHTTPRequestHandler):
    service: Flux2Service | None = None

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            repo = self.service.repo_id if self.service else "flux.2-dev"
            self._write_json(200, {"object": "list", "data": [{"id": repo}]})
            return
        self._write_json(404, {"error": f"Not found: {self.path}"})

    def do_POST(self) -> None:
        print(f"\n[REQUEST] POST {self.path}", flush=True)
        sys.stdout.flush()
        
        if self.path != "/v1/images/generations":
            self._write_json(404, {"error": f"Not found: {self.path}"})
            return

        try:
            if self.service is None:
                raise RuntimeError("Service not initialized")

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            
            print(f"[REQUEST] Payload received: {len(raw)} bytes", flush=True)
            sys.stdout.flush()

            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                self._write_json(400, {"error": "Missing prompt"})
                return

            size = str(payload.get("size", "1024x1024")).lower()
            if "x" in size:
                w_str, h_str = size.split("x", 1)
                width, height = int(w_str), int(h_str)
            else:
                width = int(payload.get("width", 1024))
                height = int(payload.get("height", 1024))

            n = int(payload.get("n", 1))
            n = max(1, min(n, 4))

            seed = payload.get("seed")
            seed = int(seed) if seed is not None else None

            images = []
            for i in range(n):
                b64 = self.service.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    seed=None if seed is None else seed + i,
                )
                images.append({"b64_json": b64})

            self._write_json(
                200,
                {
                    "created": int(time.time()),
                    "data": images,
                },
            )

        except Exception as e:
            print("="*80, flush=True)
            print(f"❌ Error during generation:", flush=True)
            print(traceback.format_exc(), flush=True)
            print("="*80, flush=True)
            sys.stdout.flush()
            self._write_json(500, {"error": f"Generation failed: {e}"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLUX.2-dev 4-bit quantized image generation server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--repo_id",
        type=str,
        default="diffusers/FLUX.2-dev-bnb-4bit",
        help="HuggingFace repo with 4-bit quantized weights",
    )
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument(
        "--no_cpu_offloading",
        action="store_true",
        help="Disable CPU offloading (needs >80 GB VRAM)",
    )
    parser.add_argument(
        "--remote_text_encoder",
        action="store_true",
        help="Use HuggingFace's remote text encoder service (reduces VRAM, fits on V100)",
    )
    
    args = parser.parse_args()

    print("="*80, flush=True)
    print("Starting FLUX.2-dev 4-bit quantized server", flush=True)
    print(f"Host: {args.host}:{args.port}", flush=True)
    print(f"Repo: {args.repo_id}", flush=True)
    print(f"Remote text encoder: {args.remote_text_encoder}", flush=True)
    print("="*80, flush=True)
    sys.stdout.flush()

    FluxHandler.service = Flux2Service(
        repo_id=args.repo_id,
        num_steps=args.num_steps,
        guidance=args.guidance,
        cpu_offloading=not args.no_cpu_offloading,
        remote_text_encoder=args.remote_text_encoder,
    )

    server = ThreadingHTTPServer((args.host, args.port), FluxHandler)
    print(f"FLUX2 server running on http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
