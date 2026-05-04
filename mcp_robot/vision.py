"""
Vision analysis for robot before/after frame pairs.

Supports two backends selected by config.VISION_BACKEND:
  "gemini"  — Gemini Robotics-ER via Google GenAI SDK
  "ollama"  — local Qwen2.5-VL (or any multimodal model) via Ollama
  "auto"    — try Gemini first; fall back to Ollama on quota/error
"""
from __future__ import annotations

import base64
import logging
import threading
from typing import Sequence

from mcp_robot import config

log = logging.getLogger(__name__)

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

_VIDEO_PROMPT = (
    "You are analysing {n_frames} sequential frames captured during a 4-motor "
    "Lego robot action (left wheel, right wheel, arm, gripper).\n"
    "ACTION COMMANDED: {action}\n"
    "EXPECTED OUTCOME: {expected}\n\n"
    "The frames below are in chronological order and show the complete motion. "
    "Camera labels: pi_camera = robot eye, droidcam = third-person view.\n\n"
    "Reply in EXACTLY this format on two lines:\n"
    "Verdict: YES | NO | PARTIAL — <one short clause justifying the verdict>\n"
    "Changes: <1-2 short sentences on what actually happened during the motion>"
)


# ── Gemini backend ─────────────────────────────────────────────────────────────

_gemini_client = None
_gemini_lock = threading.Lock()
_active_model: str | None = None
_model_lock = threading.Lock()


def is_available() -> bool:
    return bool(config.GEMINI_API_KEY) or config.VISION_BACKEND in ("ollama", "auto")


def _get_gemini_client():
    global _gemini_client, _active_model
    if _gemini_client is not None:
        return _gemini_client
    if not config.GEMINI_API_KEY:
        return None
    from google import genai
    with _gemini_lock:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
            _active_model = config.GEMINI_MODEL
    return _gemini_client


def _get_active_model() -> str:
    global _active_model
    if _active_model is None:
        _active_model = config.GEMINI_MODEL
    return _active_model


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("resource_exhausted", "quota", "429", "ratelimitexceeded", "requests per day"))


def _switch_gemini_to_fallback() -> str:
    global _active_model
    with _model_lock:
        if _active_model != config.GEMINI_FALLBACK_MODEL:
            _active_model = config.GEMINI_FALLBACK_MODEL
            log.warning(
                "Gemini quota exhausted — switching to fallback model: %s",
                config.GEMINI_FALLBACK_MODEL,
            )
    return _active_model


def _gemini_describe(
    action: str,
    expected: str,
    before: Sequence[tuple[str, str]],
    after: Sequence[tuple[str, str]],
) -> str:
    from google.genai import types

    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured (no GEMINI_API_KEY)")

    prompt = _PROMPT.format(action=action, expected=expected)
    parts: list = [types.Part.from_text(text=prompt)]
    parts.append(types.Part.from_text(text="=== BEFORE ==="))
    for label, b64 in before:
        parts.append(types.Part.from_text(text=f"[{label}]"))
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text="=== AFTER ==="))
    for label, b64 in after:
        parts.append(types.Part.from_text(text=f"[{label}]"))
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))

    model = _get_active_model()
    log.info("Gemini query model=%s action=%r", model, action)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
        )
        text = (resp.text or "").strip()
        log.info("Gemini response: %s", text)
        return text
    except Exception as exc:
        if _is_quota_error(exc) and model != config.GEMINI_FALLBACK_MODEL:
            fallback = _switch_gemini_to_fallback()
            log.info("Retrying with Gemini fallback model: %s", fallback)
            resp = client.models.generate_content(
                model=fallback,
                contents=[types.Content(role="user", parts=parts)],
            )
            text = (resp.text or "").strip()
            log.info("Gemini fallback response: %s", text)
            return text
        raise


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _ollama_describe(
    action: str,
    expected: str,
    before: Sequence[tuple[str, str]],
    after: Sequence[tuple[str, str]],
) -> str:
    import ollama

    prompt_text = _PROMPT.format(action=action, expected=expected)
    prompt_text += "\n\n=== BEFORE ===\n"
    for label, _ in before:
        prompt_text += f"[{label}]\n"
    prompt_text += "\n=== AFTER ===\n"
    for label, _ in after:
        prompt_text += f"[{label}]\n"

    # Ollama images= accepts raw bytes; we pass all frames in order (before then after)
    images = [base64.b64decode(b64) for _, b64 in list(before) + list(after)]

    log.info("Ollama query model=%s host=%s action=%r frames=%d",
             config.OLLAMA_MODEL, config.OLLAMA_HOST, action, len(images))

    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_text, "images": images}],
    )
    text = resp["message"]["content"].strip()
    log.info("Ollama response: %s", text)
    return text


# ── Ollama video backend ───────────────────────────────────────────────────────

_CLIP_PROMPT = (
    "You are analysing a video clip from a 4-motor Lego robot (left wheel, "
    "right wheel, arm, gripper).\n"
    "Camera: {camera}. The {n_frames} images below are sequential frames.\n\n"
    "Describe what you observe: robot position, any motion, visible objects, "
    "and the overall scene state. Be concise (2-4 sentences)."
)


def _ollama_video_describe(
    camera: str,
    frames: Sequence[str],
    paths: Sequence[str | None],
) -> str:
    import ollama

    valid_paths = [p for p in paths if p]
    if valid_paths:
        log.info(
            "Clip frames saved at:\n%s",
            "\n".join(f"  {p}" for p in valid_paths),
        )

    prompt = _CLIP_PROMPT.format(camera=camera, n_frames=len(frames))
    images = [base64.b64decode(b64) for b64 in frames]

    log.info("Ollama clip VQA model=%s host=%s camera=%r frames=%d",
             config.OLLAMA_MODEL, config.OLLAMA_HOST, camera, len(frames))

    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": images}],
    )
    text = resp["message"]["content"].strip()
    log.info("Ollama clip VQA response: %s", text)
    return text


# ── public API ─────────────────────────────────────────────────────────────────

def describe_change(
    action: str,
    expected: str,
    before: Sequence[tuple[str, str]],
    after: Sequence[tuple[str, str]],
    before_paths: Sequence[str | None] | None = None,
    after_paths: Sequence[str | None] | None = None,
) -> str:
    """
    Ask the configured vision backend whether the expected outcome was achieved.

    Returns a two-line "Verdict: …\\nChanges: …" string, or "" if no frames
    are available. Returns "(vision analysis failed: …)" on unrecoverable error.
    """
    if not before and not after:
        return ""

    log.info(
        "Vision query: backend=%s action=%r before=%d after=%d"
        "\n  before_paths=%s\n  after_paths=%s",
        config.VISION_BACKEND, action, len(before), len(after),
        list(before_paths) if before_paths else [],
        list(after_paths) if after_paths else [],
    )

    backend = config.VISION_BACKEND

    if backend == "gemini":
        try:
            return _gemini_describe(action, expected, before, after)
        except Exception as exc:
            log.warning("Gemini describe_change failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    if backend == "ollama":
        try:
            return _ollama_describe(action, expected, before, after)
        except Exception as exc:
            log.warning("Ollama describe_change failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    # "auto": Gemini first, Ollama as final fallback
    gemini_exc = None
    if config.GEMINI_API_KEY:
        try:
            return _gemini_describe(action, expected, before, after)
        except Exception as exc:
            log.warning("Gemini failed, trying Ollama fallback: %s", exc)
            gemini_exc = exc

    try:
        return _ollama_describe(action, expected, before, after)
    except Exception as exc:
        log.warning("Ollama fallback also failed: %s", exc)
        primary = str(gemini_exc) if gemini_exc else str(exc)
        return f"(vision analysis failed: {primary}; ollama: {exc})"


def _gemini_describe_video(
    action: str,
    expected: str,
    labeled_frames: Sequence[tuple[str, str]],
) -> str:
    from google.genai import types

    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured (no GEMINI_API_KEY)")

    prompt = _VIDEO_PROMPT.format(action=action, expected=expected, n_frames=len(labeled_frames))
    parts: list = [types.Part.from_text(text=prompt)]
    for label, b64 in labeled_frames:
        parts.append(types.Part.from_text(text=f"[{label}]"))
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))

    model = _get_active_model()
    log.info("Gemini video query model=%s action=%r frames=%d", model, action, len(labeled_frames))
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
        )
        text = (resp.text or "").strip()
        log.info("Gemini video response: %s", text)
        return text
    except Exception as exc:
        if _is_quota_error(exc) and model != config.GEMINI_FALLBACK_MODEL:
            fallback = _switch_gemini_to_fallback()
            resp = client.models.generate_content(
                model=fallback,
                contents=[types.Content(role="user", parts=parts)],
            )
            text = (resp.text or "").strip()
            log.info("Gemini video fallback response: %s", text)
            return text
        raise


def _ollama_describe_video(
    action: str,
    expected: str,
    labeled_frames: Sequence[tuple[str, str]],
) -> str:
    import ollama

    prompt_text = _VIDEO_PROMPT.format(
        action=action, expected=expected, n_frames=len(labeled_frames)
    )
    for label, _ in labeled_frames:
        prompt_text += f"\n[{label}]"

    images = [base64.b64decode(b64) for _, b64 in labeled_frames]

    log.info("Ollama video query model=%s host=%s action=%r frames=%d",
             config.OLLAMA_MODEL, config.OLLAMA_HOST, action, len(images))

    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_text, "images": images}],
    )
    text = resp["message"]["content"].strip()
    log.info("Ollama video response: %s", text)
    return text


def describe_action_video(
    action: str,
    expected: str,
    labeled_frames: Sequence[tuple[str, str]],
) -> str:
    """
    Ask the vision backend to assess whether *expected* was achieved, given a
    chronological sequence of (camera_label, base64_jpeg) frames captured
    during the action.

    Returns a two-line "Verdict: …\\nChanges: …" string, or "" if no frames.
    """
    if not labeled_frames:
        return ""

    log.info("Video vision query: backend=%s action=%r frames=%d",
             config.VISION_BACKEND, action, len(labeled_frames))

    backend = config.VISION_BACKEND

    if backend == "gemini":
        try:
            return _gemini_describe_video(action, expected, labeled_frames)
        except Exception as exc:
            log.warning("Gemini describe_action_video failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    if backend == "ollama":
        try:
            return _ollama_describe_video(action, expected, labeled_frames)
        except Exception as exc:
            log.warning("Ollama describe_action_video failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    # "auto": Gemini first, Ollama fallback
    if config.GEMINI_API_KEY:
        try:
            return _gemini_describe_video(action, expected, labeled_frames)
        except Exception as exc:
            log.warning("Gemini video failed, trying Ollama: %s", exc)

    try:
        return _ollama_describe_video(action, expected, labeled_frames)
    except Exception as exc:
        log.warning("Ollama video fallback failed: %s", exc)
        return f"(vision analysis failed: {exc})"


def describe_clip(
    camera: str,
    frames: Sequence[str],
    paths: Sequence[str | None] | None = None,
) -> str:
    """
    Ask the Ollama backend to describe a video clip (sequence of JPEG frames).

    Logs all saved frame file paths so results can be inspected manually.
    Returns a description string, or "" if no frames are available.
    """
    if not frames:
        return ""

    resolved = list(paths) if paths else []
    try:
        return _ollama_video_describe(camera, frames, resolved)
    except Exception as exc:
        log.warning("Clip VQA failed: %s", exc)
        return f"(clip VQA failed: {exc})"
