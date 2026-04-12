#!/usr/bin/env python3
"""Local HTTP server for FLUX.2-dev text-to-image generation.

Exposes an OpenAI-compatible endpoint used by PaperBanana:
- GET  /health
- GET  /v1/models
- POST /v1/images/generations
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from PIL import Image

# Allow importing the sibling flux2 repo without installation.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
_FLUX2_SRC = _PROJECT_ROOT / "flux2" / "src"
if str(_FLUX2_SRC) not in os.sys.path:
    os.sys.path.insert(0, str(_FLUX2_SRC))

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder


class Flux2Service:
    def __init__(
        self, model_name: str, cpu_offloading: bool, num_steps: int, guidance: float
    ):
        model_name = model_name.lower()
        if model_name not in FLUX2_MODEL_INFO:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.model_info = FLUX2_MODEL_INFO[model_name]
        self.cpu_offloading = cpu_offloading
        self.guidance = guidance
        self.num_steps = num_steps
        self.torch_device = torch.device("cuda")

        # Load modules similarly to flux2/scripts/cli.py
        self.text_encoder = load_text_encoder(model_name, device=self.torch_device)
        self.model = load_flow_model(
            model_name,
            debug_mode=False,
            device="cpu" if cpu_offloading else self.torch_device,
        )
        self.ae = load_ae(model_name, device=self.torch_device)

        self.text_encoder.eval()
        self.model.eval()
        self.ae.eval()

        # FLUX.2-dev defaults
        defaults = self.model_info.get("defaults", {})
        if self.num_steps <= 0:
            self.num_steps = int(defaults.get("num_steps", 50))
        if self.guidance <= 0:
            self.guidance = float(defaults.get("guidance", 4.0))

        self._lock = threading.Lock()

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
            with torch.no_grad():
                if self.model_info["guidance_distilled"]:
                    ctx = self.text_encoder([prompt]).to(torch.bfloat16)
                else:
                    ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
                    ctx_prompt = self.text_encoder([prompt]).to(torch.bfloat16)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

                if self.cpu_offloading:
                    self.text_encoder = self.text_encoder.cpu()
                    torch.cuda.empty_cache()
                    self.model = self.model.to(self.torch_device)

                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(
                    shape, generator=generator, dtype=torch.bfloat16, device="cuda"
                )
                x, x_ids = batched_prc_img(randn)
                timesteps = get_schedule(self.num_steps, x.shape[1])

                if self.model_info["guidance_distilled"]:
                    x = denoise(
                        self.model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=self.guidance,
                    )
                else:
                    x = denoise_cfg(
                        self.model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=self.guidance,
                    )

                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = self.ae.decode(x).float()

                if self.cpu_offloading:
                    self.model = self.model.cpu()
                    torch.cuda.empty_cache()
                    self.text_encoder = self.text_encoder.to(self.torch_device)

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
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
            model_name = self.service.model_name if self.service else "flux.2-dev"
            self._write_json(200, {"object": "list", "data": [{"id": model_name}]})
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
                    "created": int(torch.randint(0, 2**31 - 1, (1,)).item()),
                    "data": images,
                },
            )

        except Exception as e:
            self._write_json(500, {"error": f"Generation failed: {e}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX.2 local image generation server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model_name", type=str, default="flux.2-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--no_cpu_offloading", action="store_true")
    args = parser.parse_args()

    FluxHandler.service = Flux2Service(
        model_name=args.model_name,
        cpu_offloading=not args.no_cpu_offloading,
        num_steps=args.num_steps,
        guidance=args.guidance,
    )

    server = ThreadingHTTPServer((args.host, args.port), FluxHandler)
    print(f"FLUX2 server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
