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

# The consult_vqa_for_pddl_domain question prompt used to live here as
# CONSULT_DOMAIN_QUESTION; it's now neurosymbolic_counselor.counselor's
# DEFAULT_QUESTION (github.com/f1ren/NAPC) so there's
# one canonical copy instead of two that can drift. server.py and
# tests/vqa/pddl_consult_test.py both import it from there.

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
    "GRASP READINESS: An object is ready to be grasped (gripper may close) only when BOTH "
    "conditions are true: (1) the object is at least touching the robot's front body, AND "
    "(2) the green forward-arrow is well over the object — not merely touching its edge, but "
    "clearly passing through or covering it. If either condition is unmet, set the Plan "
    "verdict to REPLAN — the robot must navigate closer before closing the gripper.\n"
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
_active_locate_model: str | None = None
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


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "service_unavailable", "overloaded"))


_TRANSIENT_MAX_RETRIES = 3
_TRANSIENT_BASE_DELAY = 2.0


def _gemini_generate_with_retry(client, model: str, contents, config=None):
    """Call client.models.generate_content with exponential backoff on transient errors."""
    kwargs: dict = {"model": model, "contents": contents}
    if config is not None:
        kwargs["config"] = config
    last_exc: Exception | None = None
    for attempt in range(_TRANSIENT_MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            last_exc = exc
            delay = _TRANSIENT_BASE_DELAY * (2 ** attempt)
            log.warning("Gemini transient error (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, _TRANSIENT_MAX_RETRIES, delay, exc)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


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


def _get_active_locate_model() -> str:
    global _active_locate_model
    if _active_locate_model is None:
        _active_locate_model = config.LOCATE_OBJECT_MODEL
    return _active_locate_model


def _switch_locate_to_fallback() -> str:
    global _active_locate_model
    with _model_lock:
        if _active_locate_model != config.LOCATE_OBJECT_FALLBACK_MODEL:
            _active_locate_model = config.LOCATE_OBJECT_FALLBACK_MODEL
            log.warning(
                "locate_object_vlm quota exhausted — switching to fallback model: %s",
                config.LOCATE_OBJECT_FALLBACK_MODEL,
            )
    return _active_locate_model




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
    "GRASP READINESS: An object is ready to be grasped (gripper may close) only when BOTH "
    "conditions are true: (1) the object is at least touching the robot's front body, AND "
    "(2) the green forward-arrow is well over the object — not merely touching its edge, but "
    "clearly passing through or covering it. If either condition is unmet, do not close the "
    "gripper — report that the robot must navigate closer first.\n"
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
        resp = _gemini_generate_with_retry(
            client, model, [types.Content(role="user", parts=parts)],
        )
        text = (resp.text or "").strip()
        log.info("Gemini video response: %s", text)
        return text
    except Exception as exc:
        if _is_quota_error(exc) and model != config.GEMINI_FALLBACK_MODEL:
            fallback = _switch_gemini_to_fallback()
            resp = _gemini_generate_with_retry(
                client, fallback, [types.Content(role="user", parts=parts)],
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


def _peak_motion_index(frames_b64: Sequence[str]) -> int:
    """Return the index (1..n-1) of the frame with the highest optical-flow magnitude.

    Decodes frames to grayscale and delegates to heading.flow_magnitudes so the
    Farneback parameters stay in one place. Falls back to int(n*2/3) on any error.
    """
    import cv2
    from mcp_robot.heading import flow_magnitudes

    n = len(frames_b64)
    fallback = int(n * 2 / 3)
    if n < 3:
        return n // 2
    try:
        decoded: list[np.ndarray] = []
        for b64 in frames_b64:
            data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return fallback
            decoded.append(img)
        mags = flow_magnitudes(decoded)
        # mags[i] is the flow between frame i and frame i+1;
        # the frame that is the *destination* of peak flow is at index argmax+1.
        return int(np.argmax(mags)) + 1
    except Exception:
        return fallback


def _subsample_frames(
    labeled_frames: Sequence[tuple[str, str]],
    frame_paths: Sequence[str | None] | None = None,
    raw_labeled_frames: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[tuple[str, str]], list[str | None]]:
    """Return 3 frames per camera (first, peak-motion, last), grouped by camera.

    The inner frame is selected by optical flow: the frame with the highest mean
    flow magnitude from its predecessor, computed on raw_labeled_frames when
    provided so heading-arrow pixels don't produce spurious vectors.
    """
    paths = list(frame_paths) if frame_paths else [None] * len(labeled_frames)

    raw_camera_b64s: dict[str, list[str]] = {}
    if raw_labeled_frames is not None:
        for label, b64 in raw_labeled_frames:
            raw_camera_b64s.setdefault(label, []).append(b64)

    # Group by camera, preserving original indices for path lookup
    camera_indices: dict[str, list[int]] = {}
    for i, (label, _) in enumerate(labeled_frames):
        camera_indices.setdefault(label, []).append(i)

    out_frames: list[tuple[str, str]] = []
    out_paths: list[str | None] = []
    for label, indices in camera_indices.items():
        n = len(indices)
        if n <= 3:
            picks = indices
        else:
            raw_b64s = raw_camera_b64s.get(label)
            if raw_b64s and len(raw_b64s) == n:
                inner_local = _peak_motion_index(raw_b64s)
            else:
                inner_local = int(n * 2 / 3)
            picks = sorted({indices[0], indices[inner_local], indices[-1]})
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

    labeled_frames, frame_paths = _subsample_frames(labeled_frames, frame_paths, motion_frames)

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


def ask_with_images(
    prompt: str,
    labeled_images: Sequence[tuple[str, str]],
) -> str:
    """
    Send a free-form prompt and still images to the configured VQA backend.

    Args:
        prompt:         Full text prompt.
        labeled_images: List of (camera_label, base64_jpeg) pairs.

    Returns the raw text response from the VQA model.
    """
    if not labeled_images:
        raise ValueError("ask_with_images: no images provided")

    backend = config.VISION_BACKEND
    if backend == "gemini":
        return _gemini_ask_with_images(prompt, labeled_images)
    if backend == "ollama":
        return _ollama_ask_with_images(prompt, labeled_images)

    # "auto": Gemini first, Ollama fallback
    if config.GEMINI_API_KEY:
        return _gemini_ask_with_images(prompt, labeled_images)
    return _ollama_ask_with_images(prompt, labeled_images)


def _gemini_ask_with_images(
    prompt: str,
    labeled_images: Sequence[tuple[str, str]],
) -> str:
    from google.genai import types

    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured (no GEMINI_API_KEY)")

    parts: list = [types.Part.from_text(text=prompt)]
    for label, b64 in labeled_images:
        desc = _CAMERA_LABELS.get(label, label)
        parts.append(types.Part.from_text(text=f"\n=== {desc} ==="))
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))

    model = _get_active_model()
    log.info(
        "Gemini ask_with_images model=%s images=%d\n--- PROMPT ---\n%s\n--- END PROMPT ---",
        model, len(labeled_images), prompt,
    )
    resp = _gemini_generate_with_retry(
        client, model, [types.Content(role="user", parts=parts)],
    )
    text = (resp.text or "").strip()
    log.info("Gemini ask_with_images response:\n%s", text)
    return text


def _ollama_ask_with_images(
    prompt: str,
    labeled_images: Sequence[tuple[str, str]],
) -> str:
    import ollama

    full_prompt = prompt
    for label, _ in labeled_images:
        desc = _CAMERA_LABELS.get(label, label)
        full_prompt += f"\n\n=== {desc} ==="

    images = [base64.b64decode(b64) for _, b64 in labeled_images]
    log.info("Ollama ask_with_images model=%s host=%s images=%d",
             config.OLLAMA_MODEL, config.OLLAMA_HOST, len(images))

    client = ollama.Client(host=config.OLLAMA_HOST)
    t0 = time.monotonic()
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": full_prompt, "images": images}],
    )
    elapsed = time.monotonic() - t0
    text = resp["message"]["content"].strip()
    log.info("Ollama ask_with_images response (%.1fs):\n%s", elapsed, text)
    return text


# Minimum VLM confidence required before a locate_object_vlm detection is
# considered actionable. Below this, the caller must not act on the
# detection — but the achieved certainty is still reported (see
# LowConfidenceDetection) rather than being collapsed into a flat "not found".
_LOCATE_CONFIDENCE_THRESHOLD = 0.95


class LowConfidenceDetection(Exception):
    """A candidate object was found by locate_object_vlm, but its confidence
    was at or below _LOCATE_CONFIDENCE_THRESHOLD.

    Distinct from a plain "not found" (None) so callers can tell the MCP
    caller the actual certainty achieved instead of claiming nothing was seen.
    """

    def __init__(self, description: str, confidence: float, threshold: float = _LOCATE_CONFIDENCE_THRESHOLD):
        self.description = description
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Only {confidence:.0%} certainty was achieved for '{description}' — "
            f"need at least {threshold:.0%} certainty to act."
        )


def locate_object_vlm(
    bgr: np.ndarray,
    description: str,
) -> tuple[tuple[int, int, int, int], float, str, np.ndarray, np.ndarray, float] | None:
    """
    Ask Gemini Flash to locate an object and describe its color for CV refinement.

    Returns (rough_bbox, confidence, note, hsv_lo, hsv_hi, expected_area_frac) where:
      - rough_bbox: (x1,y1,x2,y2) pixel coords — intentionally coarse, used only to
        seed the classical CV search region (expanded 3x in cv_refine_location)
      - hsv_lo / hsv_hi: OpenCV HSV lower/upper bounds (H 0–179, S 0–255, V 0–255)
      - expected_area_frac: approximate fraction of image pixels the object occupies

    Returns None if the object is not found at all. Raises LowConfidenceDetection
    if a candidate was found but confidence is at or below
    _LOCATE_CONFIDENCE_THRESHOLD. Raises RuntimeError on API/parsing failures.
    """
    import json
    import re

    import cv2 as _cv2

    if not config.GEMINI_API_KEY:
        raise RuntimeError("locate_object_vlm: GEMINI_API_KEY not set — VLM localization disabled")

    from google.genai import types as _gtypes

    h, w = bgr.shape[:2]
    ok, buf = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("locate_object_vlm: failed to encode image as JPEG")

    prompt = (
        f"Find '{description}' in this image.\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "  \"found\": true if the object is visible, false otherwise\n"
        "  \"x1\": left edge of bounding box as a fraction of image width  (0.0–1.0)\n"
        "  \"y1\": top edge of bounding box as a fraction of image height (0.0–1.0)\n"
        "  \"x2\": right edge of bounding box as a fraction of image width  (0.0–1.0)\n"
        "  \"y2\": bottom edge of bounding box as a fraction of image height (0.0–1.0)\n"
        "  \"hsv_hue_lo\": lower hue bound in OpenCV range 0–179 (OpenCV halves standard 0–360°)\n"
        "  \"hsv_hue_hi\": upper hue bound in OpenCV range 0–179\n"
        "  \"hsv_sat_min\": minimum saturation 0–255 (0=grey, 255=fully saturated)\n"
        "  \"hsv_val_min\": minimum value/brightness 0–255 (0=black, 255=white)\n"
        "  \"approx_area_frac\": approximate fraction of the total image area the object occupies (0.0–1.0)\n"
        "  \"confidence\": how certain you are that you found the object (0.0–1.0)\n"
        "  \"note\": one brief sentence describing what you found and where\n\n"
        "OpenCV hue reference: red≈0–10 or 170–179, orange≈10–25, yellow≈25–35, "
        "green≈35–85, cyan≈85–100, blue≈100–130, purple≈130–160.\n\n"
        "Return only the JSON, no markdown, no other text."
    )

    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("locate_object_vlm: Gemini client unavailable")

    parts = [
        _gtypes.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
        _gtypes.Part.from_text(text=prompt),
    ]

    model = _get_active_locate_model()
    gen_config = _gtypes.GenerateContentConfig(
        thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
    )
    t0 = time.monotonic()
    try:
        resp = _gemini_generate_with_retry(
            client, model, [_gtypes.Content(role="user", parts=parts)],
            config=gen_config,
        )
    except Exception as exc:
        if _is_quota_error(exc) and model != config.LOCATE_OBJECT_FALLBACK_MODEL:
            model = _switch_locate_to_fallback()
            resp = _gemini_generate_with_retry(
                client, model, [_gtypes.Content(role="user", parts=parts)],
                config=gen_config,
            )
        else:
            raise
    text = (resp.text or "").strip()

    elapsed = time.monotonic() - t0
    log.info("VLM locate '%s' model=%s (%.1fs):\n%s",
             description, model, elapsed, text)

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        log.error("locate_object_vlm: no JSON found in response: %s", text)
        raise RuntimeError(f"locate_object_vlm: Gemini returned no JSON — raw: {text!r}")

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as exc:
        log.error("locate_object_vlm: JSON parse error: %s — raw: %s", exc, text)
        raise RuntimeError(f"locate_object_vlm: JSON parse error: {exc} — raw: {text!r}") from exc

    if not data.get("found", True):
        log.info("locate_object_vlm: '%s' not found in frame", description)
        return None

    try:
        # Gemini occasionally emits "x"/"y" instead of the requested "x1"/"y1"
        # for the top-left corner — fall back to those before giving up.
        raw_x1 = data.get("x1", data.get("x"))
        raw_y1 = data.get("y1", data.get("y"))
        raw_x2 = data["x2"]
        raw_y2 = data["y2"]
        # Gemini normally normalizes coords to [0,1], but occasionally emits
        # one axis already in pixel space (e.g. x1=0.49 alongside y1=194) — a
        # value > 1 can't be a normalized fraction, so treat it as already
        # being in pixel space instead of scaling it again.
        x1 = int(raw_x1) if raw_x1 > 1 else int(raw_x1 * w)
        y1 = int(raw_y1) if raw_y1 > 1 else int(raw_y1 * h)
        x2 = int(raw_x2) if raw_x2 > 1 else int(raw_x2 * w)
        y2 = int(raw_y2) if raw_y2 > 1 else int(raw_y2 * h)
    except (KeyError, TypeError) as exc:
        log.error("locate_object_vlm: missing bbox fields: %s — data: %s", exc, data)
        raise RuntimeError(f"locate_object_vlm: missing bbox fields in response: {data}") from exc

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        log.error("locate_object_vlm: degenerate bbox [%d,%d,%d,%d] — ignoring", x1, y1, x2, y2)
        raise RuntimeError(f"locate_object_vlm: degenerate bbox [{x1},{y1},{x2},{y2}] from response: {data}")

    hue_lo  = int(np.clip(data.get("hsv_hue_lo",   0), 0, 179))
    hue_hi  = int(np.clip(data.get("hsv_hue_hi", 179), 0, 179))
    sat_min = int(np.clip(data.get("hsv_sat_min",  40), 0, 255))
    val_min = int(np.clip(data.get("hsv_val_min",  40), 0, 255))
    hsv_lo  = np.array([hue_lo, sat_min, val_min], dtype=np.uint8)
    hsv_hi  = np.array([hue_hi,     255,     255], dtype=np.uint8)
    area_frac = float(np.clip(data.get("approx_area_frac", 0.01), 1e-5, 1.0))

    confidence = float(data.get("confidence", 0.7))
    note = str(data.get("note", ""))
    log.info(
        "locate_object_vlm: bbox=[%d,%d,%d,%d] hsv=[%d-%d,%d+,%d+] area_frac=%.4f conf=%.2f note=%r",
        x1, y1, x2, y2, hue_lo, hue_hi, sat_min, val_min, area_frac, confidence, note,
    )
    if confidence < _LOCATE_CONFIDENCE_THRESHOLD:
        log.info("locate_object_vlm: '%s' confidence %.2f at/below threshold %.2f — not acting",
                 description, confidence, _LOCATE_CONFIDENCE_THRESHOLD)
        raise LowConfidenceDetection(description, confidence)
    return (x1, y1, x2, y2), confidence, note, hsv_lo, hsv_hi, area_frac


# Max multiple of the rough bbox size to expand the search window to.
_CV_SEARCH_EXPANSION = 3.0

# Search radii (× rough-bbox size) to try, smallest first, capped at
# _CV_SEARCH_EXPANSION. The leading _CV_SEARCH_RADII_TRUSTED radii are all
# tried, keeping the largest in-range contour across them — the VLM's rough
# bbox can be anchored to just part of the object (a corner, an edge, a
# highlight), so the smallest radius often finds a plausible-sized but
# incomplete blob; only a larger radius reveals the object's true extent.
# The remaining (wider) radii are a last resort, tried only if nothing
# qualifies within the trusted radii, and stop at the first hit: a same-colour
# region further out (e.g. a sunlit wall behind a white switch) reliably
# merges with the target once the window is wide enough, and can still land
# inside the (generous) area range by sheer size — the area range alone can't
# be trusted to reject it, so once a trusted-radius match exists, wider radii
# are never even considered.
_CV_SEARCH_RADII = (0.0, 0.25, 0.5, 1.0, 2.0, _CV_SEARCH_EXPANSION)
_CV_SEARCH_RADII_TRUSTED = 4


def _contour_overlaps_bbox(
    contour: np.ndarray,
    origin: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> bool:
    """True if contour's bounding rect (ROI-local, translated by origin) overlaps bbox."""
    import cv2 as _cv2

    ox, oy = origin
    bx, by, bw, bh = _cv2.boundingRect(contour)
    cx1, cy1, cx2, cy2 = bx + ox, by + oy, bx + ox + bw, by + oy + bh
    bx1, by1, bx2, by2 = bbox
    return cx1 < bx2 and cx2 > bx1 and cy1 < by2 and cy2 > by1


def cv_refine_location(
    bgr: np.ndarray,
    rough_bbox: tuple[int, int, int, int],
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    expected_area_frac: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """
    Refine a VLM rough bounding box using HSV color segmentation.

    Searches progressively larger windows around the rough bbox (see
    _CV_SEARCH_RADII / _CV_SEARCH_RADII_TRUSTED). Within the trusted radii,
    tries all of them and keeps the largest color blob that falls within the
    expected area range AND overlaps the rough bbox, rather than stopping at
    the first radius with any in-range blob — the rough bbox can be anchored
    to only part of the object, so a small radius can find a plausible-sized
    but incomplete blob before a larger radius reveals the object's true,
    larger extent. The overlap requirement exists because area alone isn't a
    reliable filter: a same-colour background region just outside the rough
    bbox (e.g. sunlit floor below a white switch) can produce a contour that
    coincidentally falls within the generous area band once the window is
    wide enough, even though it shares no pixels with what the VLM actually
    pointed at. Only widens past the trusted radii if none of them land a
    match, and then stops at the first hit. Falls back to the largest raw
    blob at the widest radius if nothing lands in range anywhere.

    Returns (refined_bbox, centroid) in full-image pixel coords, or None if no
    matching blob is found at any radius.
    """
    import cv2 as _cv2

    img_h, img_w = bgr.shape[:2]
    rx1, ry1, rx2, ry2 = rough_bbox
    bbox_size = max(rx2 - rx1, ry2 - ry1)

    total_px = img_h * img_w
    exp_px   = expected_area_frac * total_px
    kernel   = np.ones((7, 7), np.uint8)

    best = None
    best_origin = (0, 0)
    last_contours: list = []
    last_origin = (0, 0)

    for i, radius in enumerate(_CV_SEARCH_RADII):
        if best is not None and i >= _CV_SEARCH_RADII_TRUSTED:
            break  # already have a trustworthy match nearby — don't risk widening into merged background

        pad = int(bbox_size * radius)
        sx1 = max(0, rx1 - pad)
        sy1 = max(0, ry1 - pad)
        sx2 = min(img_w, rx2 + pad)
        sy2 = min(img_h, ry2 + pad)

        roi = bgr[sy1:sy2, sx1:sx2]
        hsv_roi = _cv2.cvtColor(roi, _cv2.COLOR_BGR2HSV)

        # Handle hue wrap-around (e.g. red spans 170–179 and 0–10)
        if hsv_lo[0] <= hsv_hi[0]:
            mask = _cv2.inRange(hsv_roi, hsv_lo, hsv_hi)
        else:
            lo_a = hsv_lo.copy(); hi_a = np.array([179, hsv_hi[1], hsv_hi[2]], dtype=np.uint8)
            lo_b = np.array([0,   hsv_lo[1], hsv_lo[2]], dtype=np.uint8); hi_b = hsv_hi.copy()
            mask = _cv2.bitwise_or(_cv2.inRange(hsv_roi, lo_a, hi_a),
                                   _cv2.inRange(hsv_roi, lo_b, hi_b))

        mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, kernel)
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN,  kernel)

        contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        last_contours, last_origin = contours, (sx1, sy1)
        if not contours:
            continue

        valid = [c for c in contours
                 if exp_px * 0.05 <= _cv2.contourArea(c) <= exp_px * 20.0
                 and _contour_overlaps_bbox(c, (sx1, sy1), rough_bbox)]
        if not valid:
            continue

        candidate = max(valid, key=_cv2.contourArea)
        if best is None or _cv2.contourArea(candidate) > _cv2.contourArea(best):
            best = candidate
            best_origin = (sx1, sy1)
            log.info("cv_refine_location: new best contour at radius=%.2fx bbox_size (area=%.0f)",
                     radius, _cv2.contourArea(candidate))

    if best is None:
        if not last_contours:
            log.warning("cv_refine_location: no contours found in search region")
            return None
        # Nothing satisfied the area band at any radius. Prefer a contour that
        # at least overlaps the rough bbox over the single largest raw blob in
        # the widest window — the widest window can contain large, completely
        # unrelated regions (e.g. a sunlit floor patch), and picking by size
        # alone would pick one of those over a smaller, plausible, on-target
        # blob that merely missed the area band.
        overlapping = [c for c in last_contours
                       if _contour_overlaps_bbox(c, last_origin, rough_bbox)]
        if overlapping:
            log.warning("cv_refine_location: no contour in area range (exp=%.0fpx) at any radius; "
                        "using largest overlapping rough bbox", exp_px)
            best = max(overlapping, key=_cv2.contourArea)
        else:
            log.warning("cv_refine_location: no contour in area range or overlapping rough bbox "
                        "(exp=%.0fpx) at any radius; using largest raw blob", exp_px)
            best = max(last_contours, key=_cv2.contourArea)
        best_origin = last_origin

    sx1, sy1 = best_origin
    M = _cv2.moments(best)
    if M["m00"] == 0:
        log.warning("cv_refine_location: zero-area contour moment")
        return None
    cx = int(M["m10"] / M["m00"]) + sx1
    cy = int(M["m01"] / M["m00"]) + sy1

    bx, by, bw, bh = _cv2.boundingRect(best)
    refined_bbox = (bx + sx1, by + sy1, bx + sx1 + bw, by + sy1 + bh)

    log.info("cv_refine_location: centroid=(%d,%d) refined_bbox=%s area=%.0fpx",
             cx, cy, refined_bbox, _cv2.contourArea(best))
    return refined_bbox, (cx, cy)


def locate_object_hybrid(
    bgr: np.ndarray,
    description: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int], float, str, tuple[int, int, int, int]] | None:
    """
    Locate an object using a VLM→CV hybrid pipeline.

    Step 1 — Gemini: rough bbox + HSV color params.
    Step 2 — Classical CV: HSV segmentation within a 3× expanded search region
              around the rough bbox → precise contour centroid.

    Returns (bbox, centroid, confidence, note, rough_bbox) with pixel coords,
    where centroid is the CV-derived center (accurate) and bbox is the refined
    contour bbox. Falls back to the VLM rough bbox center if CV finds nothing.

    rough_bbox is the original (coarser, pre-refine) VLM box, always returned
    alongside the refined one: cv_refine_location's color segmentation can
    under-segment an object whose lighting isn't uniform (e.g. a cup with a
    shadowed interior vs. a bright rim), returning a bbox anchored well inside
    the object's true silhouette rather than covering it. That's harmless for
    navigation (centroid stays accurate), but callers that need the object's
    full extent — e.g. inpainting removal masks — should use the union of
    bbox and rough_bbox so they don't punch a hole entirely inside the object
    that a context-aware inpainter just reconstructs as more of that object.

    Raises LowConfidenceDetection (propagated from locate_object_vlm) if a
    candidate was found but below the confidence threshold required to act.
    """
    result = locate_object_vlm(bgr, description)
    if result is None:
        return None

    rough_bbox, confidence, note, hsv_lo, hsv_hi, area_frac = result

    refined = cv_refine_location(bgr, rough_bbox, hsv_lo, hsv_hi, area_frac)
    if refined is not None:
        bbox, centroid = refined
        log.info("locate_object_hybrid: CV succeeded — centroid=%s bbox=%s", centroid, bbox)
        return bbox, centroid, confidence, note, rough_bbox

    # CV found nothing — fall back to VLM rough center
    rx1, ry1, rx2, ry2 = rough_bbox
    centroid = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
    log.warning("locate_object_hybrid: CV failed — falling back to VLM rough bbox center %s", centroid)
    return rough_bbox, centroid, confidence, note, rough_bbox


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
