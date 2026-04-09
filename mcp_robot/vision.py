"""
Vision analysis using Google Gemini (google-genai SDK).

Supports:
  - Single-frame scene description
  - Multi-frame video clip analysis (action verification)
  - Before/after comparison to judge whether an action succeeded
"""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Sequence

from google import genai
from google.genai import types
from PIL import Image

from mcp_robot import config


def _client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY env var is not set. "
            "Export it before starting the MCP server."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


def _pil_to_part(img: Image.Image) -> types.Part:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# ── public API ────────────────────────────────────────────────────────────────

def analyze_frame(frame_b64: str, prompt: str) -> str:
    """
    Ask Gemini about a single camera frame.

    Args:
        frame_b64: Base64-encoded JPEG string.
        prompt:    Question or instruction for Gemini.

    Returns:
        Gemini's text response.
    """
    client = _client()
    part = _pil_to_part(_b64_to_pil(frame_b64))
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[part, prompt],
    )
    return response.text


def analyze_clip(frames_b64: Sequence[str], prompt: str) -> str:
    """
    Analyze a sequence of frames (short video clip) with Gemini.

    Frames are sent as individual image parts in a single prompt.
    Gemini 1.5+ can reason temporally across them.

    Args:
        frames_b64: Ordered list of base64-encoded JPEG strings.
        prompt:     Question or instruction for Gemini.

    Returns:
        Gemini's text response.
    """
    client = _client()
    parts: list = [_pil_to_part(_b64_to_pil(f)) for f in frames_b64]
    parts.append(prompt)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=parts,
    )
    return response.text


def verify_action(
    before_b64: str,
    after_b64: str,
    action_description: str,
) -> dict:
    """
    Ask Gemini whether an action succeeded by comparing before/after frames.

    Returns:
        {
            "success": bool,
            "confidence": "high" | "medium" | "low",
            "explanation": str,
        }
    """
    client = _client()
    prompt = (
        f"You are verifying the outcome of a robot action.\n"
        f"Action performed: {action_description}\n\n"
        f"The FIRST image is BEFORE the action.\n"
        f"The SECOND image is AFTER the action.\n\n"
        f"Answer in JSON with exactly these keys:\n"
        f'  "success": true or false\n'
        f'  "confidence": "high", "medium", or "low"\n'
        f'  "explanation": one or two sentences\n'
        f"Output only the JSON object, no markdown fences."
    )
    parts = [
        _pil_to_part(_b64_to_pil(before_b64)),
        _pil_to_part(_b64_to_pil(after_b64)),
        prompt,
    ]
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=parts,
    )
    text = response.text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"success": None, "confidence": "low", "explanation": text}
