"""
Vision analysis for robot before/after frame pairs.

Supports two backends selected by config.VISION_BACKEND:
  "gemini"  — Gemini Robotics-ER via Google GenAI SDK
  "ollama"  — local Qwen3-VL (or any multimodal model) via Ollama
  "auto"    — try Gemini first; fall back to Ollama on quota/error
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import Sequence

import numpy as np

from mcp_robot import config

log = logging.getLogger(__name__)


_CAMERA_LABELS = {
    "pi_camera": "front cam — front view (robot's eye, mounted on the arm, looking forward from the gripper)",
    "droidcam":  "external cam — third-person view (overhead or side angle showing the whole robot)",
}

_VIDEO_PROMPT = (
    "You are analysing a 4-motor Lego robot (left wheel, right wheel, arm, "
    "gripper). The gripper defines the robot's front.\n"
    "ROBOT ORIENTATION: 'forward' always means the direction the gripper is currently "
    "pointing. After any turn the robot's heading changes — identify which way the "
    "gripper faces in the first frame before evaluating a drive action.\n"
    "GRIPPER STATE: only call the gripper 'open' when the jaws are FULLY spread apart — "
    "the angle between the two fingers must be approximately 180 degrees (fingers "
    "pointing in opposite directions). A partially-open gripper is NOT 'open' — describe "
    "it as 'partially open' or 'closed'. A non-wide-open gripper may fail to clear or "
    "grasp an object.\n"
    "ACTION COMMANDED: {action}\n"
    "EXPECTED OUTCOME: {expected}\n"
    "{context_section}"
    "Frames are grouped by camera below:\n"
    "  front cam: front view — mounted on the robot, looking forward from the gripper.\n"
    "  external cam: third-person view — overhead or side angle showing the whole robot.\n\n"
    "DIRECTIONAL LANGUAGE: When describing the robot's heading or turning direction, use "
    "clockwise/counter-clockwise (viewed from above) or compass directions "
    "(N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). Do NOT say the "
    "robot 'turned left' or 'turned "
    "right' — those terms are ambiguous across camera perspectives.\n\n"
    "FRAME ANNOTATIONS: GREEN arrow rooted at the gripper indicates the robot's current "
    "forward direction (heading). RED arrows are Optical Flow vectors, use them to identify "
    "which parts of the scene moved and how far. If the robot is covered with red arrows, it means "
    "the robot has moved.\n\n"
    "PLAN EVALUATION: If a CONTEXT was provided, assess whether the robot's final state "
    "in the last frame is compatible with the stated plan context — i.e. is the robot "
    "positioned/configured to successfully execute the next step? If no CONTEXT was "
    "provided, set Plan to N/A.\n\n"
    "Reply in EXACTLY this format on three lines:\n"
    "Verdict: YES | NO | PARTIAL — <one short clause justifying the verdict>\n"
    "Changes: <1-2 short sentences on what actually happened during the motion>\n"
    "Plan: OK | REPLAN | N/A — <one short clause: why the final state is or is not ready for the next step, or N/A if no context>"
)


# ── motion detection ──────────────────────────────────────────────────────────

_CAPTURE_MOTION_THRESHOLD = 2.5  # mean absolute pixel diff (0-255 scale) to stop action capture
_MOTION_PIXEL_THRESH = 20       # per-pixel diff considered "changed"
_MOTION_PIXEL_COUNT  = 500      # min changed pixels to declare motion (catches localised moves)


def _has_motion(frames_b64: Sequence[str]) -> bool:
    """
    Return True if consecutive frames show any pixel-level change above threshold.

    Groups frames by camera when labels are embedded in the sequence.
    Falls back to True (assume motion) on any decoding error so VQA is not skipped
    when the check itself fails.
    """
    try:
        import cv2

        if len(frames_b64) < 2:
            return True

        decoded = []
        for b64 in frames_b64:
            data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                decoded.append(img)

        if len(decoded) < 2:
            return True

        for a, b in zip(decoded[:-1], decoded[1:]):
            if a.shape == b.shape:
                abs_diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
                diff = float(np.mean(abs_diff))
                n_changed = int(np.sum(abs_diff > _MOTION_PIXEL_THRESH))
                if diff > _CAPTURE_MOTION_THRESHOLD or n_changed > _MOTION_PIXEL_COUNT:
                    return True
        return False
    except Exception:
        return True


def _has_motion_labeled(labeled: Sequence[tuple[str, str]]) -> bool:
    """Check per-camera motion from (camera_label, b64) pairs."""
    cameras: dict[str, list[str]] = {}
    for label, b64 in labeled:
        cameras.setdefault(label, []).append(b64)
    return any(_has_motion(frames) for frames in cameras.values())


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




# ── Ollama video backend ───────────────────────────────────────────────────────

_CLIP_PROMPT = (
    "You are analysing a video clip from a 4-motor Lego robot (left wheel, "
    "right wheel, arm, gripper). The gripper defines the robot's front.\n"
    "ROBOT ORIENTATION: 'forward' always means the direction the gripper is currently "
    "pointing — identify which way the gripper faces in the first frame before "
    "describing any movement.\n"
    "GRIPPER STATE: only call the gripper 'open' when the jaws are FULLY spread apart — "
    "the angle between the two fingers must be approximately 180 degrees (fingers "
    "pointing in opposite directions). A partially-open gripper is NOT 'open' — describe "
    "it as 'partially open' or 'closed'. A non-wide-open gripper may fail to clear or "
    "grasp an object.\n"
    "Camera: {camera}. The {n_frames} images below are sequential frames.\n\n"
    "Describe what you observe: robot position, any motion, visible objects, "
    "and the overall scene state. Be concise (2-4 sentences).\n"
    "DIRECTIONAL LANGUAGE: Describe turning direction as clockwise/counter-clockwise "
    "(viewed from above) or compass directions "
    "(N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW), "
    "not as 'left' or 'right'.\n"
    "FRAME ANNOTATIONS: GREEN arrow rooted at the gripper indicates the robot's current "
    "forward direction (heading). RED arrows are Optical Flow vectors, use them to identify "
    "which parts of the scene moved and how far. If the robot is covered with red arrows, it means "
    "the robot has moved.\n\n"
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
    t0 = time.monotonic()
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": images}],
    )
    elapsed = time.monotonic() - t0
    text = resp["message"]["content"].strip()
    log.info("Ollama clip VQA response (%.1fs): %s", elapsed, text)
    return text


# ── public API ─────────────────────────────────────────────────────────────────


def _gemini_describe_video(
    action: str,
    expected: str,
    labeled_frames: Sequence[tuple[str, str]],
    frame_paths: Sequence[str | None] | None = None,
    context: str = "",
) -> str:
    from google.genai import types

    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured (no GEMINI_API_KEY)")

    paths = list(frame_paths) if frame_paths else [None] * len(labeled_frames)
    image_log = "\n".join(
        f"  [{label}][{i:03d}] {path or '(no path)'}"
        for i, ((label, _), path) in enumerate(zip(labeled_frames, paths))
    )

    # Group frames by camera, preserving chronological order within each camera
    cameras: dict[str, list[str]] = {}
    for label, b64 in labeled_frames:
        cameras.setdefault(label, []).append(b64)

    context_section = f"CONTEXT: {context}\n" if context else ""
    prompt = _VIDEO_PROMPT.format(action=action, expected=expected, context_section=context_section)
    parts: list = [types.Part.from_text(text=prompt)]
    for camera, frames in cameras.items():
        desc = _CAMERA_LABELS.get(camera, camera)
        parts.append(types.Part.from_text(text=f"\n=== {desc} ({len(frames)} frames) ==="))
        for b64 in frames:
            parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))

    model = _get_active_model()
    log.info(
        "Gemini video query model=%s action=%r frames=%d"
        "\n--- PROMPT ---\n%s\n--- END PROMPT ---"
        "\n--- IMAGES ---\n%s\n--- END IMAGES ---",
        model, action, len(labeled_frames), prompt, image_log,
    )
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
    frame_paths: Sequence[str | None] | None = None,
    context: str = "",
) -> str:
    import ollama

    paths = list(frame_paths) if frame_paths else [None] * len(labeled_frames)

    # Group by camera, preserving chronological order within each camera
    cameras: dict[str, list[tuple[str, str | None]]] = {}
    for (label, b64), path in zip(labeled_frames, paths):
        cameras.setdefault(label, []).append((b64, path))

    context_section = f"CONTEXT: {context}\n" if context else ""
    prompt_text = _VIDEO_PROMPT.format(action=action, expected=expected, context_section=context_section)
    images: list[bytes] = []
    image_log_lines: list[str] = []
    for camera, frames in cameras.items():
        desc = _CAMERA_LABELS.get(camera, camera)
        prompt_text += f"\n\n=== {desc} ({len(frames)} frames) ==="
        for i, (b64, path) in enumerate(frames):
            images.append(base64.b64decode(b64))
            image_log_lines.append(f"  [{camera}][{i:03d}] {path or '(no path)'}")

    log.info(
        "Ollama video query model=%s host=%s action=%r cameras=%s total_frames=%d"
        "\n--- PROMPT ---\n%s\n--- END PROMPT ---"
        "\n--- IMAGES ---\n%s\n--- END IMAGES ---",
        config.OLLAMA_MODEL, config.OLLAMA_HOST, action,
        list(cameras.keys()), len(images),
        prompt_text, "\n".join(image_log_lines),
    )

    client = ollama.Client(host=config.OLLAMA_HOST)
    t0 = time.monotonic()
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_text, "images": images}],
    )
    elapsed = time.monotonic() - t0
    text = resp["message"]["content"].strip()
    log.info("Ollama video response (%.1fs):\n%s", elapsed, text)
    return text


def _subsample_frames(
    labeled_frames: Sequence[tuple[str, str]],
    frame_paths: Sequence[str | None] | None = None,
) -> tuple[list[tuple[str, str]], list[str | None]]:
    """Return 3 frames per camera (first, ~2/3, last), grouped by camera — 6 total for two cameras."""
    paths = list(frame_paths) if frame_paths else [None] * len(labeled_frames)

    # Group by camera, preserving original indices for path lookup
    camera_indices: dict[str, list[int]] = {}
    for i, (label, _) in enumerate(labeled_frames):
        camera_indices.setdefault(label, []).append(i)

    out_frames: list[tuple[str, str]] = []
    out_paths: list[str | None] = []
    for indices in camera_indices.values():
        n = len(indices)
        if n <= 3:
            picks = indices
        else:
            picks = sorted({indices[0], indices[int(n * 2 / 3)], indices[-1]})
        for i in picks:
            out_frames.append(labeled_frames[i])
            out_paths.append(paths[i])

    return out_frames, out_paths


def describe_action_video(
    action: str,
    expected: str,
    labeled_frames: Sequence[tuple[str, str]],
    frame_paths: Sequence[str | None] | None = None,
    context: str = "",
    raw_labeled_frames: Sequence[tuple[str, str]] | None = None,
) -> str:
    """
    Ask the vision backend to assess whether *expected* was achieved, given a
    chronological sequence of (camera_label, base64_jpeg) frames captured
    during the action.

    Args:
        labeled_frames:     Annotated frames (heading arrow) — sent to VQA.
        raw_labeled_frames: Unannotated frames — used only for the motion gate
                            so the arrow does not cause spurious motion hits.
                            Falls back to labeled_frames if not provided.

    Returns a two-line "Verdict: …\\nChanges: …" string, or "" if no frames.
    """
    if not labeled_frames:
        return ""

    motion_frames = raw_labeled_frames if raw_labeled_frames is not None else labeled_frames
    if not _has_motion_labeled(motion_frames):
        log.info("Video is static — skipping VQA for action %r", action)
        return "Verdict: NO — no motion detected in video\nChanges: No motion was detected in the captured video frames; the robot may not have moved."

    labeled_frames, frame_paths = _subsample_frames(labeled_frames, frame_paths)

    # Collapse each camera's subsampled frames into a single stacked composite.
    cameras_ordered: dict[str, list[tuple[int, str]]] = {}
    for i, (label, b64) in enumerate(labeled_frames):
        cameras_ordered.setdefault(label, []).append((i, b64))
    stacked_labeled: list[tuple[str, str]] = []
    stacked_paths: list[str | None] = []
    paths_list = list(frame_paths) if frame_paths else [None] * len(labeled_frames)
    for label, indexed in cameras_ordered.items():
        b64s = [b64 for _, b64 in indexed]
        stacked_b64 = stack_frames(b64s) if len(b64s) > 1 else b64s[0]
        stacked_labeled.append((label, stacked_b64))
        last_path = paths_list[indexed[-1][0]]
        stacked_path: str | None = None
        if last_path and len(b64s) > 1:
            stacked_path = os.path.join(os.path.dirname(last_path), f"{label}_stacked.jpg")
            with open(stacked_path, "wb") as fh:
                fh.write(base64.b64decode(stacked_b64))
        else:
            stacked_path = last_path
        stacked_paths.append(stacked_path)
    labeled_frames = stacked_labeled
    frame_paths = stacked_paths

    log.info("Video vision query: backend=%s action=%r frames=%d (stacked per camera)",
             config.VISION_BACKEND, action, len(labeled_frames))

    backend = config.VISION_BACKEND

    if backend == "gemini":
        try:
            return _gemini_describe_video(action, expected, labeled_frames, frame_paths, context=context)
        except Exception as exc:
            log.warning("Gemini describe_action_video failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    if backend == "ollama":
        try:
            return _ollama_describe_video(action, expected, labeled_frames, frame_paths, context=context)
        except Exception as exc:
            log.warning("Ollama describe_action_video failed: %s", exc)
            return f"(vision analysis failed: {exc})"

    # "auto": Gemini first, Ollama fallback
    if config.GEMINI_API_KEY:
        try:
            return _gemini_describe_video(action, expected, labeled_frames, frame_paths, context=context)
        except Exception as exc:
            log.warning("Gemini video failed, trying Ollama: %s", exc)

    try:
        return _ollama_describe_video(action, expected, labeled_frames, frame_paths, context=context)
    except Exception as exc:
        log.warning("Ollama video fallback failed: %s", exc)
        return f"(vision analysis failed: {exc})"


def stack_frames(frames_b64: Sequence[str], quality: int = 90) -> str:
    """
    Composite N frames into one image where later frames appear more opaque.

    Frame i gets weight (i+1) / sum(1..N), so the earliest frame is the most
    transparent ghost and the final frame is the most visible layer.

    Args:
        frames_b64: Base64-encoded JPEG frames, oldest first.
        quality:    JPEG quality for the output image (default 90).

    Returns a base64-encoded JPEG of the composited result.
    """
    import cv2

    if not frames_b64:
        raise ValueError("frames_b64 must not be empty")

    decoded = []
    for b64 in frames_b64:
        data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode a frame as JPEG")
        decoded.append(img.astype(np.float32))

    n = len(decoded)
    weights = [i + 1 for i in range(n)]  # [1, 2, 3, ...] — later = heavier
    total = sum(weights)

    h, w = decoded[0].shape[:2]
    composite = np.zeros((h, w, 3), dtype=np.float32)
    for frame, weight in zip(decoded, weights):
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        composite += frame * (weight / total)

    result = np.clip(composite, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode stacked frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode()


def describe_clip(
    camera: str,
    frames: Sequence[str],
    paths: Sequence[str | None] | None = None,
    raw_frames: Sequence[str] | None = None,
) -> str:
    """
    Ask the Ollama backend to describe a video clip (sequence of JPEG frames).

    Args:
        frames:     Annotated frames (heading arrow) — sent to VQA.
        raw_frames: Unannotated frames — used only for the motion gate so the
                    arrow does not cause spurious motion hits. Falls back to
                    frames if not provided.

    Returns a description string, or "" if no frames are available.
    """
    if not frames:
        return ""

    motion_frames = raw_frames if raw_frames is not None else frames
    if not _has_motion(motion_frames):
        log.info("Clip is static — skipping VQA for camera %r", camera)
        return "No motion detected in the captured video clip."

    resolved = list(paths) if paths else []
    try:
        return _ollama_video_describe(camera, frames, resolved)
    except Exception as exc:
        log.warning("Clip VQA failed: %s", exc)
        return f"(clip VQA failed: {exc})"
