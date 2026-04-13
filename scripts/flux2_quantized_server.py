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

import argparse
import base64
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

import torch
from diffusers import AutoModel, Flux2Pipeline
from PIL import Image
from transformers import Mistral3ForConditionalGeneration


class Flux2Service:
    def __init__(
        self,
        repo_id: str,
        num_steps: int,
        guidance: float,
        cpu_offloading: bool,
    ):
        self.repo_id = repo_id
        self.num_steps = num_steps
        self.guidance = guidance
        self.device = "cuda:0"
        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        print(f"Loading 4-bit text encoder from {repo_id} (dtype={self.torch_dtype}) …")
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
            device_map="cpu",
        )

        print(f"Loading 4-bit transformer from {repo_id} …")
        dit = AutoModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
            device_map="cpu",
        )

        print("Building Flux2Pipeline …")
        self.pipe = Flux2Pipeline.from_pretrained(
            repo_id,
            text_encoder=text_encoder,
            transformer=dit,
            torch_dtype=self.torch_dtype,
        )

        if cpu_offloading:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self._lock = threading.Lock()
        print("Flux2Service ready.")

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

    def generate(
        self, prompt: str, width: int, height: int, seed: int | None = None
    ) -> str:
        width, height = self._normalize_dims(width, height)
        if seed is None:
            seed = random.randrange(2**31)

        with self._lock:
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance,
            )
            img = result.images[0]
            return self._pil_to_b64_png(img)


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
        if self.path != "/v1/images/generations":
            self._write_json(404, {"error": f"Not found: {self.path}"})
            return

        try:
            if self.service is None:
                raise RuntimeError("Service not initialized")

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")

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
    args = parser.parse_args()

    FluxHandler.service = Flux2Service(
        repo_id=args.repo_id,
        num_steps=args.num_steps,
        guidance=args.guidance,
        cpu_offloading=not args.no_cpu_offloading,
    )

    server = ThreadingHTTPServer((args.host, args.port), FluxHandler)
    print(f"FLUX2 server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
