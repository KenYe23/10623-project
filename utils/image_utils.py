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
Image utility functions for processing and converting images
"""

import base64
import io
import os
from pathlib import Path
from PIL import Image


def resolve_image_path(image_path):
    """
    Resolve an image path, handling Unicode encoding mismatches.

    Some filenames in the dataset JSON contain proper Unicode chars (e.g. \u2019
    RIGHT SINGLE QUOTATION MARK) while the files on disk were saved with
    double-encoded UTF-8 (UTF-8 bytes misinterpreted as CP1252 then re-encoded).
    When the direct path doesn't exist, this function searches the parent
    directory for a file whose ASCII-only characters match.
    """
    image_path = Path(image_path)
    if image_path.exists():
        return image_path

    parent = image_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {parent}")

    target_name = image_path.name
    target_ascii = ''.join(c for c in target_name if ord(c) < 128)

    for entry in os.listdir(parent):
        entry_ascii = ''.join(c for c in entry if ord(c) < 128)
        if entry_ascii == target_ascii:
            resolved = parent / entry
            print(f"\u26a0\ufe0f  Resolved Unicode filename mismatch: {target_name!r} -> {entry!r}")
            return resolved

    raise FileNotFoundError(f"No such file or directory: {image_path}")


def convert_png_b64_to_jpg_b64(png_b64_str: str) -> str:
    """
    Convert a PNG base64 string to a JPG base64 string.
    
    Args:
        png_b64_str: Base64 encoded PNG image string
        
    Returns:
        Base64 encoded JPG image string, or None if conversion fails
    """
    try:
        if not png_b64_str or len(png_b64_str) < 10:
            print(f"⚠️  Invalid base64 string (too short): {png_b64_str[:50] if png_b64_str else 'None'}")
            return None
            
        img = Image.open(io.BytesIO(base64.b64decode(png_b64_str))).convert("RGB")
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", quality=95)
        return base64.b64encode(out_io.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        print(f"   Input preview: {png_b64_str[:100] if png_b64_str else 'None'}")
        return None
