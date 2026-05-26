"""
CV-based grasp readiness check.

Uses YOLO to locate the target object and checks two conditions against the
green forward-arrow computed by heading.detect_heading():

  1. The object is touching the robot's front body (close to arrow_anchor).
  2. The green forward-arrow passes *well over* the object's center of mass
     (perpendicular distance from arrow ray < ARROW_OVER_FRAC * object radius).

Returns a GraspReadiness dataclass with ready flag + actionable text.
"""
from __future__ import annotations

import base64
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from mcp_robot import config
from mcp_robot.heading import (
    Heading,
    annotate_bgr,
    detect_heading,
    compute_heading_to_object_angle,
)

log = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

_YOLO_MODEL_NAME = "yolo11n.pt"
_YOLO_CONF       = 0.20
_YOLO_IOU        = 0.45

# Arrow is "well over" object when perpendicular distance < this fraction of
# the object's half-size (max(w,h)/2).
_ARROW_OVER_FRAC = 0.55

# Object nearest-bbox-point to arrow_anchor must be < this fraction of image
# diagonal to count as "touching robot body".
_BODY_TOUCH_FRAC = 0.08

# Maps a canonical target_class name to the set of COCO class names YOLO may
# use for that object.  Add rows here as new object types are introduced.
_CLASS_SYNONYMS: dict[str, frozenset[str]] = {
    "cup":    frozenset({"cup", "bowl", "bottle", "vase"}),
    "ball":   frozenset({"sports ball", "orange", "apple"}),
    "bottle": frozenset({"bottle", "cup", "vase"}),
}

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def radius(self) -> float:
        return max(self.width, self.height) / 2.0


@dataclass
class GraspReadiness:
    ready: bool
    reason: str
    action: str = ""
    object_detected: bool = False
    object_class: str = ""
    object_confidence: float = 0.0
    object_center: tuple[int, int] = field(default_factory=lambda: (0, 0))
    touches_body: bool = False
    arrow_well_over: bool = False
    perp_dist_px: float = 0.0
    dist_to_front_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "action": self.action,
            "object_detected": self.object_detected,
            "object_class": self.object_class,
            "object_confidence": round(self.object_confidence, 2),
            "object_center": list(self.object_center),
            "checks": {
                "touches_body": self.touches_body,
                "arrow_well_over": self.arrow_well_over,
            },
            "metrics": {
                "perp_dist_px": round(self.perp_dist_px, 1),
                "dist_to_front_px": round(self.dist_to_front_px, 1),
            },
        }

    def to_text(self) -> str:
        status = "READY" if self.ready else "NOT READY"
        lines = [f"Grasp readiness: {status}", f"Reason: {self.reason}"]
        if self.action:
            lines.append(f"Action needed: {self.action}")
        if self.object_detected:
            lines.append(
                f"Object: {self.object_class} (conf={self.object_confidence:.0%})"
                f" at {self.object_center}"
            )
        lines.append(
            f"Checks — touches_body={self.touches_body},"
            f" arrow_well_over={self.arrow_well_over}"
        )
        if self.object_detected:
            lines.append(
                f"Metrics — dist_to_front={self.dist_to_front_px:.0f}px,"
                f" perp_dist={self.perp_dist_px:.0f}px"
            )
        return "\n".join(lines)


# ── YOLO backend ──────────────────────────────────────────────────────────────

_yolo_model = None


def _load_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    log.info("Loading YOLO model: %s", _YOLO_MODEL_NAME)
    _yolo_model = YOLO(_YOLO_MODEL_NAME)
    return _yolo_model


def _yolo_detect(
    bgr: np.ndarray,
    target_class: str = "cup",
) -> list[DetectedObject]:
    """Run YOLO and return only detections matching *target_class* (or its synonyms)."""
    model = _load_model()
    results = model(bgr, conf=_YOLO_CONF, iou=_YOLO_IOU, verbose=False)
    allowed = _CLASS_SYNONYMS.get(target_class, frozenset({target_class}))
    objects: list[DetectedObject] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            name = model.names.get(cls_id, str(cls_id))
            if name in allowed:
                objects.append(DetectedObject(name, conf, x1, y1, x2, y2))
    log.info("YOLO detected %d object(s) matching '%s': %s",
             len(objects), target_class,
             [(o.class_name, f"{o.confidence:.2f}") for o in objects])
    return objects


# ── object selection ──────────────────────────────────────────────────────────

def _pick_target(objects: list[DetectedObject], h: Heading) -> DetectedObject | None:
    """Pick the object most likely to be the grasp target.

    Score = confidence / (1 + normalised_perp_dist) for objects in front of
    the robot (forward projection t > 0). Falls back to highest-confidence
    object if none are in front.
    """
    bx, by   = h.body_center
    fw       = h.forward

    scored: list[tuple[float, DetectedObject]] = []
    for obj in objects:
        ox, oy = obj.center
        dx, dy = ox - bx, oy - by
        t      = dx * fw[0] + dy * fw[1]
        if t <= 0:
            continue
        px = dx - t * fw[0]
        py = dy - t * fw[1]
        perp = math.hypot(px, py)
        score = obj.confidence / (1.0 + perp / max(obj.radius, 1.0))
        scored.append((score, obj))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # Fallback: object with highest confidence regardless of position
    if objects:
        return max(objects, key=lambda o: o.confidence)
    return None


# ── public API ─────────────────────────────────────────────────────────────────

def _save_debug_image(
    bgr: np.ndarray,
    heading: Heading | None,
    obj: DetectedObject | None,
    result: GraspReadiness,
) -> str | None:
    """Draw bbox + green arrow on a copy of bgr and save it to SNAPSHOT_DIR."""
    if not config.SNAPSHOT_DIR:
        return None
    try:
        annotated = annotate_bgr(bgr.copy()) if heading is not None else bgr.copy()

        if obj is not None:
            color = (0, 200, 0) if result.ready else (0, 0, 220)
            cv2.rectangle(annotated, (obj.x1, obj.y1), (obj.x2, obj.y2), color, 2)
            label = f"{obj.class_name} {obj.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ty = obj.y1 - 6 if obj.y1 > th + 6 else obj.y2 + th + 6
            cv2.rectangle(annotated, (obj.x1, ty - th - 2), (obj.x1 + tw + 2, ty + 2), color, cv2.FILLED)
            cv2.putText(annotated, label, (obj.x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        verdict = "READY" if result.ready else "NOT READY"
        cv2.putText(annotated, verdict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 0) if result.ready else (0, 0, 220), 2, cv2.LINE_AA)

        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        path = os.path.join(config.SNAPSHOT_DIR, f"grasp_readiness_{ts}_{ms:03d}.jpg")
        cv2.imwrite(path, annotated)
        log.info("Grasp readiness debug image saved: %s", path)
        return path
    except Exception as exc:
        log.warning("Failed to save grasp readiness debug image: %s", exc)
        return None


def _compute_readiness(
    bgr: np.ndarray,
    target_class: str = "cup",
) -> tuple[GraspReadiness, Heading | None, DetectedObject | None]:
    """Core logic — returns (result, heading, selected_object) for debug annotation."""
    h, w = bgr.shape[:2]
    diag = math.hypot(w, h)

    # ── 1. Detect heading ────────────────────────────────────────────────
    heading = detect_heading(bgr)
    if heading is None:
        return GraspReadiness(
            ready=False,
            reason="Robot not detected in frame — ensure the robot's yellow body is visible to the external camera.",
            action="Reposition the external camera or the robot so the yellow body is fully in view.",
        ), None, None

    # ── 2. Detect objects ────────────────────────────────────────────────
    try:
        objects = _yolo_detect(bgr, target_class=target_class)
    except Exception as exc:
        log.warning("YOLO detection failed: %s", exc)
        return GraspReadiness(
            ready=False,
            reason=f"Object detection failed: {exc}",
            action="Check that the YOLO model is installed and the image is valid.",
        ), heading, None

    if not objects:
        allowed = _CLASS_SYNONYMS.get(target_class, frozenset({target_class}))
        return GraspReadiness(
            ready=False,
            reason=f"No '{target_class}' detected in frame (looking for: {', '.join(sorted(allowed))}).",
            action="Ensure the target object is visible and not occluded. Try repositioning.",
        ), heading, None

    # ── 3. Pick the best candidate ───────────────────────────────────────
    obj = _pick_target(objects, heading)
    if obj is None:
        return GraspReadiness(
            ready=False,
            reason="No object found in front of the robot.",
            action="Turn the robot to face the target object, then reassess.",
            object_detected=True,
            object_class=objects[0].class_name,
            object_confidence=objects[0].confidence,
        ), heading, None

    ox, oy = obj.center
    bx, by = heading.body_center
    fw     = heading.forward
    ax, ay = heading.arrow_anchor

    # ── 4. Condition 2: arrow well over object ───────────────────────────
    dx, dy = ox - bx, oy - by
    t      = dx * fw[0] + dy * fw[1]
    perp_x = dx - t * fw[0]
    perp_y = dy - t * fw[1]
    perp_dist  = math.hypot(perp_x, perp_y)
    arrow_over = t > 0 and perp_dist < obj.radius * _ARROW_OVER_FRAC

    # ── 5. Condition 1: object touching robot body ───────────────────────
    near_x = float(max(obj.x1, min(ax, obj.x2)))
    near_y = float(max(obj.y1, min(ay, obj.y2)))
    dist_to_front     = math.hypot(ax - near_x, ay - near_y)
    body_touch_thresh = diag * _BODY_TOUCH_FRAC
    touches_body      = dist_to_front < body_touch_thresh

    common = dict(
        object_detected=True,
        object_class=obj.class_name,
        object_confidence=obj.confidence,
        object_center=obj.center,
        touches_body=touches_body,
        arrow_well_over=arrow_over,
        perp_dist_px=perp_dist,
        dist_to_front_px=dist_to_front,
    )

    if arrow_over and touches_body:
        return GraspReadiness(
            ready=True,
            reason=(
                f"Object ({obj.class_name}) is in grasp position: "
                f"touching robot body and arrow passes over its center "
                f"(perp={perp_dist:.0f}px < {obj.radius * _ARROW_OVER_FRAC:.0f}px threshold)."
            ),
            **common,
        ), heading, obj

    # Rotation hint: angle + CW/CCW needed to align arrow with object center.
    # Cross product (fw × to_object) in image coords (y-down):
    # positive → object is clockwise from current heading (viewed from above).
    cross = fw[0] * dy - fw[1] * dx
    rot_dir = "CW" if cross > 0 else "CCW"
    rot_deg = math.degrees(math.atan2(perp_dist, max(t, 1.0)))
    rot_hint = f"{rot_deg:.0f}° {rot_dir}"

    # Build actionable feedback
    if not touches_body and not arrow_over:
        reason = (
            f"Object is too far from the robot "
            f"(front-gap={dist_to_front:.0f}px, need <{body_touch_thresh:.0f}px) "
            f"and arrow misses its center (perp={perp_dist:.0f}px)."
        )
        action = f"Turn {rot_hint} to align the arrow, then drive forward to close the gap."
    elif not touches_body:
        reason = (
            f"Object is not close enough to the robot body "
            f"(front-gap={dist_to_front:.0f}px, need <{body_touch_thresh:.0f}px)."
        )
        action = "Drive forward to bring the object against the robot's front."
    else:
        if t <= 0:
            reason = "Object is behind the robot — arrow does not reach it."
            action = "Turn 180° to face the object."
        else:
            reason = (
                f"Arrow does not pass well over the object "
                f"(perp offset={perp_dist:.0f}px, need <{obj.radius * _ARROW_OVER_FRAC:.0f}px)."
            )
            action = f"Turn {rot_hint} so the green arrow passes through the object's center."

    return GraspReadiness(ready=False, reason=reason, action=action, **common), heading, obj


def check_grasp_readiness(bgr: np.ndarray, target_class: str = "cup") -> GraspReadiness:
    """Check whether the scene is ready for the gripper to close.

    Args:
        bgr: BGR image from the external (DroidCam) camera.
        target_class: canonical object type to look for (see _CLASS_SYNONYMS).

    Returns:
        GraspReadiness with ready flag + reason + actionable next step.
    """
    result, heading, obj = _compute_readiness(bgr, target_class=target_class)
    log.info(
        "Grasp readiness: ready=%s | %s%s",
        result.ready,
        result.reason,
        f" | action: {result.action}" if result.action else "",
    )
    _save_debug_image(bgr, heading, obj, result)
    return result


def check_grasp_readiness_jpeg_bytes(jpeg: bytes, target_class: str = "cup") -> GraspReadiness:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return GraspReadiness(ready=False, reason="Could not decode image.", action="Provide a valid JPEG frame.")
    return check_grasp_readiness(bgr, target_class=target_class)


def check_grasp_readiness_b64(b64: str, target_class: str = "cup") -> GraspReadiness:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return GraspReadiness(ready=False, reason="Could not decode base64 image.")
    return check_grasp_readiness_jpeg_bytes(raw, target_class=target_class)


# ── combined heading + object annotation ──────────────────────────────────────

def annotate_frame_with_object(
    bgr: np.ndarray,
    target_class: str = "cup",
) -> tuple[np.ndarray, float | None]:
    """Detect heading + YOLO object, annotate frame with arrow + bbox + angle line.

    Returns (annotated_bgr, angle_deg) where angle_deg is None when heading or
    object is not detected.  Positive angle = object is CW from forward.
    Falls back to heading-only annotation on YOLO failure.
    """
    heading = detect_heading(bgr)
    if heading is None:
        return bgr, None

    objects = _yolo_detect(bgr, target_class=target_class)
    obj = _pick_target(objects, heading) if objects else None
    if obj is None:
        return annotate_bgr(bgr), None

    angle_deg = compute_heading_to_object_angle(heading, obj.center)
    annotated = annotate_bgr(
        bgr,
        obj_center=obj.center,
        obj_bbox=(obj.x1, obj.y1, obj.x2, obj.y2),
    )
    return annotated, angle_deg


def annotate_frame_with_object_b64(
    b64: str,
    target_class: str = "cup",
) -> tuple[str, float | None]:
    """Base64 JPEG in/out version of annotate_frame_with_object."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode base64 JPEG for object annotation")
    annotated, angle = annotate_frame_with_object(bgr, target_class)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("cv2.imencode failed in annotate_frame_with_object_b64")
    return base64.b64encode(buf.tobytes()).decode(), angle
