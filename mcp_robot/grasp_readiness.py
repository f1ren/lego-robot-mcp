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
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from mcp_robot.heading import Heading, detect_heading

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

# Blob fallback: HSV ranges for light/white objects (paper ball on wood floor)
_BLOB_S_MAX     = 80    # low saturation → near-white/grey
_BLOB_V_MIN     = 150   # high brightness
_BLOB_AREA_MIN  = 0.0008  # fraction of frame area
_BLOB_AREA_MAX  = 0.06
_BLOB_CIRC_MIN  = 0.25    # 4π·area/perimeter² — filters cables and noise
_BLOB_ASPECT_MAX = 2.5    # filters elongated objects (cables)

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


def _yolo_detect(bgr: np.ndarray) -> list[DetectedObject]:
    model = _load_model()
    results = model(bgr, conf=_YOLO_CONF, iou=_YOLO_IOU, verbose=False)
    objects: list[DetectedObject] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            name = model.names.get(cls_id, str(cls_id))
            objects.append(DetectedObject(name, conf, x1, y1, x2, y2))
    log.info("YOLO detected %d objects: %s",
             len(objects), [(o.class_name, f"{o.confidence:.2f}") for o in objects])
    return objects


# ── white-blob fallback detector ─────────────────────────────────────────────

# Yellow HSV range (robot body) — reuse same ranges as heading.py so we can
# exclude the body pixels from white-blob candidates.
_YELLOW_HSV_LO = np.array([15, 130, 100], dtype=np.uint8)
_YELLOW_HSV_HI = np.array([35, 255, 255], dtype=np.uint8)


def _blob_detect(bgr: np.ndarray) -> list[DetectedObject]:
    """Fallback: find light-coloured, roughly-circular blobs that are not the robot body.

    Designed for a white crumpled paper ball on a wooden floor, but works for
    any light-coloured object of similar shape/size.
    """
    h, w = bgr.shape[:2]
    frame_area = float(h * w)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Mask for light/white objects
    white_mask = cv2.inRange(
        hsv,
        np.array([0,   0,   _BLOB_V_MIN], dtype=np.uint8),
        np.array([179, _BLOB_S_MAX, 255], dtype=np.uint8),
    )

    # Exclude the yellow robot body (dilated generously so cables near it are suppressed too)
    yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LO, _YELLOW_HSV_HI)
    yellow_mask = cv2.dilate(yellow_mask, np.ones((25, 25), np.uint8))
    white_mask  = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))

    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[DetectedObject] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (_BLOB_AREA_MIN * frame_area < area < _BLOB_AREA_MAX * frame_area):
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < _BLOB_CIRC_MIN:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > _BLOB_ASPECT_MAX:
            continue
        objects.append(DetectedObject("object", 0.5, x, y, x + bw, y + bh))

    log.info("Blob fallback detected %d candidate(s)", len(objects))
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

def check_grasp_readiness(bgr: np.ndarray) -> GraspReadiness:
    """Check whether the scene is ready for the gripper to close.

    Args:
        bgr: BGR image from the external (DroidCam) camera.

    Returns:
        GraspReadiness with ready flag + reason + actionable next step.
    """
    h, w = bgr.shape[:2]
    diag = math.hypot(w, h)

    # ── 1. Detect heading ────────────────────────────────────────────────
    heading = detect_heading(bgr)
    if heading is None:
        return GraspReadiness(
            ready=False,
            reason="Robot not detected in frame — ensure the robot's yellow body is visible to the external camera.",
            action="Reposition the external camera or the robot so the yellow body is fully in view.",
        )

    # ── 2. Detect objects ────────────────────────────────────────────────
    try:
        objects = _yolo_detect(bgr)
    except Exception as exc:
        log.warning("YOLO detection failed: %s", exc)
        return GraspReadiness(
            ready=False,
            reason=f"Object detection failed: {exc}",
            action="Check that the YOLO model is installed and the image is valid.",
        )

    if not objects:
        log.info("YOLO found nothing — trying white-blob fallback")
        objects = _blob_detect(bgr)

    if not objects:
        return GraspReadiness(
            ready=False,
            reason="No object detected in frame.",
            action="Ensure the target object is visible and not occluded. Try repositioning.",
        )

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
        )

    ox, oy   = obj.center
    bx, by   = heading.body_center
    fw       = heading.forward
    ax, ay   = heading.arrow_anchor

    # ── 4. Condition 2: arrow well over object ───────────────────────────
    dx, dy = ox - bx, oy - by
    t      = dx * fw[0] + dy * fw[1]
    perp_x = dx - t * fw[0]
    perp_y = dy - t * fw[1]
    perp_dist  = math.hypot(perp_x, perp_y)
    arrow_over = t > 0 and perp_dist < obj.radius * _ARROW_OVER_FRAC

    # ── 5. Condition 1: object touching robot body ───────────────────────
    # Distance from arrow_anchor (body front edge) to the nearest point on
    # the object bounding box.
    near_x = float(max(obj.x1, min(ax, obj.x2)))
    near_y = float(max(obj.y1, min(ay, obj.y2)))
    dist_to_front = math.hypot(ax - near_x, ay - near_y)
    body_touch_thresh = diag * _BODY_TOUCH_FRAC
    touches_body = dist_to_front < body_touch_thresh

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
        )

    # Build actionable feedback
    if not touches_body and not arrow_over:
        reason = (
            f"Object is too far from the robot "
            f"(front-gap={dist_to_front:.0f}px, need <{body_touch_thresh:.0f}px) "
            f"and arrow misses its center (perp={perp_dist:.0f}px)."
        )
        action = "Drive forward toward the object to close the gap."
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
            action = "Adjust heading so the green arrow passes through the object's center."

    return GraspReadiness(ready=False, reason=reason, action=action, **common)


def check_grasp_readiness_jpeg_bytes(jpeg: bytes) -> GraspReadiness:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return GraspReadiness(ready=False, reason="Could not decode image.", action="Provide a valid JPEG frame.")
    return check_grasp_readiness(bgr)


def check_grasp_readiness_b64(b64: str) -> GraspReadiness:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return GraspReadiness(ready=False, reason="Could not decode base64 image.")
    return check_grasp_readiness_jpeg_bytes(raw)
