"""
Gemini Robotics-ER 1.6 vision analysis.

Given a pair of before/after frame sets (Pi Camera + optional DroidCam),
asks Gemini to describe what changed in the scene as a result of a robot
action. The description is returned as text and embedded in motor-action
tool responses so the MCP client does not need to inspect raw images.
"""
from __future__ import annotations

import base64
import logging
import threading
from typing import Sequence

from google import genai
from google.genai import types

from mcp_robot import config

log = logging.getLogger(__name__)

_client: genai.Client | None = None
_client_lock = threading.Lock()


def is_available() -> bool:
    return bool(config.GEMINI_API_KEY)


def _get_client() -> genai.Client | None:
    global _client
    if _client is not None:
        return _client
    if not config.GEMINI_API_KEY:
        return None
    with _client_lock:
        if _client is None:
            _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _image_part(b64: str) -> types.Part:
    return types.Part.from_bytes(
        data=base64.b64decode(b64),
        mime_type="image/jpeg",
    )


_PROMPT = (
    "You are analysing a 4-motor Lego robot (left wheel, right wheel, arm, "
    "gripper).\n"
    "ACTION COMMANDED: {action}\n"
    "EXPECTED OUTCOME: {expected}\n\n"
    "Below are BEFORE images followed by AFTER images, each labelled by the "
    "camera they came from (pi_camera = robot's front-mounted camera, "
    "droidcam = third-person view).\n\n"
    "Reply in EXACTLY this format on two lines:\n"
    "Verdict: YES | NO | PARTIAL — <one short clause justifying the verdict>\n"
    "Changes: <1-2 short sentences describing what actually changed; if "
    "nothing visible changed, say so explicitly>"
)


def describe_change(
    action: str,
    expected: str,
    before: Sequence[tuple[str, str]],
    after: Sequence[tuple[str, str]],
    before_paths: Sequence[str | None] | None = None,
    after_paths: Sequence[str | None] | None = None,
) -> str:
    """
    Ask Gemini whether the *expected* outcome of an action was achieved, and
    describe what changed between before/after frames.

    Args:
        action:   Short description of the commanded action
                  (e.g. "drive forward 1.0s at speed 50").
        expected: What the action was *meant* to produce visually
                  (e.g. "the robot translates forward; pi_camera front view
                  shows new content; gripper unchanged").
        before:   Sequence of (camera_label, base64_jpeg) captured BEFORE.
        after:    Same, AFTER.

    Returns:
        Two-line "Verdict: …\\nChanges: …" string, or "" if vision is
        disabled / both frame sets are empty. Returns "(vision analysis
        failed: …)" on API errors.
    """
    client = _get_client()
    if client is None:
        return ""
    if not before and not after:
        return ""

    prompt = _PROMPT.format(action=action, expected=expected)
    parts: list[types.Part] = [types.Part.from_text(text=prompt)]
    parts.append(types.Part.from_text(text="=== BEFORE ==="))
    for label, b64 in before:
        parts.append(types.Part.from_text(text=f"[{label}]"))
        parts.append(_image_part(b64))
    parts.append(types.Part.from_text(text="=== AFTER ==="))
    for label, b64 in after:
        parts.append(types.Part.from_text(text=f"[{label}]"))
        parts.append(_image_part(b64))

    log.info(
        "Gemini query: action=%r expected=%r before_frames=%d after_frames=%d"
        "\n  before_paths=%s\n  after_paths=%s",
        action, expected, len(before), len(after),
        list(before_paths) if before_paths else [],
        list(after_paths) if after_paths else [],
    )
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[types.Content(role="user", parts=parts)],
        )
        text = (resp.text or "").strip()
        log.info("Gemini response: %s", text)
        return text
    except Exception as exc:
        log.warning("Gemini describe_change failed: %s", exc)
        return f"(vision analysis failed: {exc})"
