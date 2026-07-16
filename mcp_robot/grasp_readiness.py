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

from mcp_robot import config, vision
from mcp_robot.heading import (
    Heading,
    annotate_bgr,
    detect_heading,
    compute_heading_to_object_angle,
)

log = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

_YOLO_MODEL_NAME = "yolo11n.pt"
_YOLO_CONF       = 0.35
_YOLO_IOU        = 0.45

# Arrow is "well over" object when perpendicular distance < this fraction of
# the object's half-size (max(w,h)/2).
_ARROW_OVER_FRAC = 0.55

# Object nearest-bbox-point to arrow_anchor must be < this fraction of image
# diagonal to count as "touching robot body".
_BODY_TOUCH_FRAC = 0.08

# Maps a canonical target_class name to the set of COCO class names YOLO may
# use for that object.  Add rows here as new object types are introduced.
# Use None to accept ALL YOLO detections (e.g. when the target is not a COCO
# class and no synonym is a good proxy — annotation falls back to the
# highest-confidence object in front of the robot).
_CLASS_SYNONYMS: dict[str, frozenset[str] | None] = {
    "cup":    frozenset({"cup", "bowl", "bottle", "vase"}),
    "ball":   frozenset({"sports ball", "orange", "apple"}),
    "bottle": frozenset({"bottle", "cup", "vase"}),
    # "button" is not a COCO class; best-effort proxies that may visually
    # overlap with physical push-buttons or button panels.
    "button": frozenset({"remote", "mouse", "keyboard", "cell phone"}),
    # "any" accepts every YOLO detection — picks the most forward object.
    "any":    None,
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
    note: str = ""
    contact_px: tuple[int, int] | None = None  # floor-contact point from VLM; None → use center
    outer_bbox: tuple[int, int, int, int] | None = None  # coarser VLM rough bbox; None → x1..y2 is already the full extent

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def nav_point(self) -> tuple[int, int]:
        """Best pixel to navigate toward: VLM floor-contact if available, else bbox center."""
        return self.contact_px if self.contact_px is not None else self.center

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
    missing_distance_mm: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "action": self.action,
            "object_detected": self.object_detected,
            "object_class": self.object_class,
            "object_confidence": round(self.object_confidence, 2),
            "object_center": list(self.object_center),
            "note": self.note,
            "checks": {
                "touches_body": self.touches_body,
                "arrow_well_over": self.arrow_well_over,
            },
            "metrics": {
                "perp_dist_px": round(self.perp_dist_px, 1),
                "dist_to_front_px": round(self.dist_to_front_px, 1),
                "missing_distance_mm": (
                    round(self.missing_distance_mm, 1)
                    if self.missing_distance_mm is not None else None
                ),
            },
        }

    def to_text(self) -> str:
        status = "READY" if self.ready else "NOT READY"
        lines = [f"Grasp readiness: {status}", f"Reason: {self.reason}"]
        if self.action:
            lines.append(f"Action needed: {self.action}")
        if self.missing_distance_mm is not None and self.missing_distance_mm > 0:
            lines.append(f"Missing drive distance: ~{self.missing_distance_mm:.0f}mm")
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
        if self.note:
            lines.append(f"VLM note: {self.note}")
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
    target_class: str,
) -> list[DetectedObject]:
    """Run YOLO and return only detections matching *target_class* (or its synonyms).

    If *target_class* maps to None in _CLASS_SYNONYMS (the "any" sentinel), or
    if the key is absent AND the literal class name does not appear in the YOLO
    vocabulary, all detections are returned so the caller can pick the best one.
    """
    model = _load_model()
    results = model(bgr, conf=_YOLO_CONF, iou=_YOLO_IOU, verbose=False)

    # Resolve allowed set.  None → accept everything (e.g. "any" or unknown class).
    raw_synonyms = _CLASS_SYNONYMS.get(target_class, frozenset({target_class}))
    accept_all = raw_synonyms is None
    allowed: frozenset[str] = frozenset() if accept_all else raw_synonyms  # type: ignore[assignment]

    objects: list[DetectedObject] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            name = model.names.get(cls_id, str(cls_id))
            if accept_all or name in allowed:
                objects.append(DetectedObject(name, conf, x1, y1, x2, y2))
    log.info("YOLO detected %d object(s) matching '%s' (accept_all=%s): %s",
             len(objects), target_class, accept_all,
             [(o.class_name, f"{o.confidence:.2f}") for o in objects])
    return objects


# ── VLM fallback detector ─────────────────────────────────────────────────────

def _vlm_detect(bgr: np.ndarray, description: str) -> DetectedObject | vision.LowConfidenceDetection | None:
    """
    Call Claude Sonnet to locate *description* in *bgr*.

    Used when YOLO finds no matching objects for non-COCO classes (e.g. "light
    switch", "door handle"). Returns a DetectedObject so the rest of the
    heading-angle pipeline can operate unchanged.

    Returns a vision.LowConfidenceDetection instance (not raised) if a
    candidate was found but below the confidence threshold required to act
    — an expected outcome, the same as a plain miss (None). Callers should
    check with isinstance() and report the achieved certainty rather than
    treating it like a plain miss.

    Raises vision.VQAResponseParseError (uncaught) if Gemini responded
    but its reply couldn't be parsed — callers must catch this specifically
    and report it as a VQA failure, not fold it into a "not found"
    result: unlike a genuine miss, this is not evidence the object is absent.
    Any other failure (network, quota, config) is swallowed to None, same as
    a plain miss, since those are not informative about object presence.
    """
    try:
        result = vision.locate_object_hybrid(bgr, description)
    except vision.VQAResponseParseError:
        raise
    except Exception as exc:
        log.warning("VLM detect fallback failed for '%s': %s", description, exc)
        return None
    if result is None or isinstance(result, vision.LowConfidenceDetection):
        return result
    (x1, y1, x2, y2), centroid, confidence, note, rough_bbox = result
    return DetectedObject(
        class_name=description,
        confidence=confidence,
        x1=x1, y1=y1, x2=x2, y2=y2,
        note=note,
        contact_px=centroid,
        outer_bbox=rough_bbox,
    )


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
    target_class_yolo: str,
    target_class_free_text: str = "",
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
    objects: list[DetectedObject] = []
    if target_class_yolo:
        try:
            objects = _yolo_detect(bgr, target_class=target_class_yolo)
        except Exception as exc:
            log.warning("YOLO detection failed: %s", exc)
            return GraspReadiness(
                ready=False,
                reason=f"Object detection failed: {exc}",
                action="Check that the YOLO model is installed and the image is valid.",
            ), heading, None

    if not objects:
        raw_allowed = _CLASS_SYNONYMS.get(target_class_yolo, frozenset({target_class_yolo})) if target_class_yolo else None
        looking_for = (
            "any object" if raw_allowed is None
            else ", ".join(sorted(raw_allowed))
        )
        # VLM path: try Gemini Flash if a free-text description was provided
        if target_class_free_text:
            try:
                vlm_obj = _vlm_detect(bgr, target_class_free_text)
            except vision.VQAResponseParseError as exc:
                return GraspReadiness(
                    ready=False,
                    reason=(
                        f"No '{target_class_yolo}' detected by YOLO (looking for: {looking_for}). "
                        f"{exc}"
                    ),
                    action=(
                        "This is a VQA parsing failure, not a confirmed absence — recall "
                        "this tool to retry, or inspect vision.py:locate_object_vlm if it recurs."
                    ),
                ), heading, None
            if isinstance(vlm_obj, vision.LowConfidenceDetection):
                return GraspReadiness(
                    ready=False,
                    reason=(
                        f"No '{target_class_yolo}' detected by YOLO (looking for: {looking_for}). "
                        f"{vlm_obj}"
                    ),
                    action="Reposition for a clearer view of the target, then retry.",
                ), heading, None
            if vlm_obj is not None:
                log.info("_compute_readiness: YOLO found nothing for %r; VLM found '%s' at %s (conf=%.2f)",
                         target_class_yolo or "(skipped)", target_class_free_text,
                         vlm_obj.center, vlm_obj.confidence)
                objects = [vlm_obj]
            else:
                return GraspReadiness(
                    ready=False,
                    reason=(
                        f"No '{target_class_yolo}' detected by YOLO (looking for: {looking_for})"
                        f" and Gemini Flash also did not find '{target_class_free_text}'."
                    ),
                    action="Ensure the target object is visible and not occluded. Try repositioning.",
                ), heading, None
        else:
            return GraspReadiness(
                ready=False,
                reason=f"No '{target_class_yolo}' detected in frame (looking for: {looking_for}).",
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
            note=objects[0].note,
        ), heading, None

    # Union with outer_bbox (the VLM's coarser pre-refine box) when set, for
    # the geometry below only. cv_refine_location's colour segmentation — or
    # the robot's own arm/gripper occluding part of the object from the
    # external camera, as happens whenever the target sits right at the front
    # body — can anchor obj.x1..y2 well inside the object's true silhouette,
    # undershooting both its radius (arrow_well_over) and its near-robot edge
    # (touches_body). Same union inpainting.py already uses for its removal
    # mask, for the same reason (mcp_robot/inpainting.py:221-236). obj.center/
    # contact_px/nav_point are left untouched — still the more accurate point
    # for navigation aim, and YOLO detections never set outer_bbox anyway.
    gx1, gy1, gx2, gy2 = obj.x1, obj.y1, obj.x2, obj.y2
    if obj.outer_bbox is not None:
        ox1, oy1, ox2, oy2 = obj.outer_bbox
        gx1, gy1 = min(gx1, ox1), min(gy1, oy1)
        gx2, gy2 = max(gx2, ox2), max(gy2, oy2)
    g_radius = max(gx2 - gx1, gy2 - gy1) / 2.0

    ox, oy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
    bx, by = heading.body_center
    fw     = heading.forward
    ax, ay = heading.arrow_anchor

    # ── 4. Condition 2: arrow well over object ───────────────────────────
    dx, dy = ox - bx, oy - by
    t      = dx * fw[0] + dy * fw[1]
    perp_x = dx - t * fw[0]
    perp_y = dy - t * fw[1]
    perp_dist  = math.hypot(perp_x, perp_y)
    arrow_over = t > 0 and perp_dist < g_radius * _ARROW_OVER_FRAC

    # ── 5. Condition 1: object touching robot body ───────────────────────
    near_x = float(max(gx1, min(ax, gx2)))
    near_y = float(max(gy1, min(ay, gy2)))
    dist_to_front = math.hypot(ax - near_x, ay - near_y)

    # Real-world gap in mm, via the same body-plate px->mm calibration
    # drive_to()/click_button() use (navigation.mm_per_px) — see
    # config.GRASP_TOUCH_THRESHOLD_MM for why this replaced a fixed
    # image-diagonal pixel fraction (it wasn't perspective-invariant and
    # under-detected real gaps for objects higher in frame). Imported
    # lazily: navigation imports DetectedObject from this module at top
    # level, so a module-level import here would be circular.
    from mcp_robot import navigation as nav_mod
    mm_scale = nav_mod.mm_per_px(heading.body_area)
    dist_to_front_mm = dist_to_front * mm_scale if mm_scale is not None else None
    if dist_to_front_mm is not None:
        touches_body = dist_to_front_mm < config.GRASP_TOUCH_THRESHOLD_MM
        missing_distance_mm = max(0.0, dist_to_front_mm - config.GRASP_TOUCH_THRESHOLD_MM)
    else:
        # Body plate not measurable (shouldn't happen once heading is
        # detected — body_area backs body_center) — fall back to the old
        # pixel-diagonal heuristic rather than failing closed.
        touches_body = dist_to_front < diag * _BODY_TOUCH_FRAC
        missing_distance_mm = None

    common = dict(
        object_detected=True,
        object_class=obj.class_name,
        object_confidence=obj.confidence,
        object_center=(ox, oy),
        touches_body=touches_body,
        arrow_well_over=arrow_over,
        perp_dist_px=perp_dist,
        dist_to_front_px=dist_to_front,
        missing_distance_mm=missing_distance_mm,
        note=obj.note,
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
    rot_hint = f"{rot_deg:.0f}{rot_dir}"
    gap_desc = f"{dist_to_front_mm:.0f}mm" if dist_to_front_mm is not None else f"{dist_to_front:.0f}px"

    # Build actionable feedback
    if not touches_body and not arrow_over:
        reason = (
            f"Object is too far from the robot (front-gap={gap_desc}) "
            f"and arrow misses its center (perp={perp_dist:.0f}px)."
        )
        if missing_distance_mm is not None:
            action = (
                f"Turn {rot_hint} to align the arrow, then drive forward "
                f"~{missing_distance_mm:.0f}mm to close the gap."
            )
        else:
            action = f"Turn {rot_hint} to align the arrow, then drive forward to close the gap."
    elif not touches_body:
        reason = f"Object is not close enough to the robot body (front-gap={gap_desc})."
        if missing_distance_mm is not None:
            action = f"Drive forward ~{missing_distance_mm:.0f}mm to bring the object against the robot's front."
        else:
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


def check_grasp_readiness(
    bgr: np.ndarray,
    target_class_yolo: str,
    target_class_free_text: str = "",
) -> GraspReadiness:
    """Check whether the scene is ready for the gripper to close.

    Args:
        bgr: BGR image from the external (SimpleIPCamera) camera.
        target_class_yolo: canonical YOLO class to look for (see _CLASS_SYNONYMS).
            No default — every caller must state what it is looking for.
        target_class_free_text: free-text description for Gemini Flash fallback
            (e.g. "light switch"). Used only when YOLO finds nothing.

    Returns:
        GraspReadiness with ready flag + reason + actionable next step.
    """
    result, heading, obj = _compute_readiness(
        bgr,
        target_class_yolo=target_class_yolo,
        target_class_free_text=target_class_free_text,
    )
    log.info(
        "Grasp readiness: ready=%s | %s%s",
        result.ready,
        result.reason,
        f" | action: {result.action}" if result.action else "",
    )
    _save_debug_image(bgr, heading, obj, result)
    return result


def check_grasp_readiness_jpeg_bytes(
    jpeg: bytes,
    target_class_yolo: str,
    target_class_free_text: str = "",
) -> GraspReadiness:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return GraspReadiness(ready=False, reason="Could not decode image.", action="Provide a valid JPEG frame.")
    return check_grasp_readiness(bgr, target_class_yolo=target_class_yolo, target_class_free_text=target_class_free_text)


def check_grasp_readiness_b64(
    b64: str,
    target_class_yolo: str,
    target_class_free_text: str = "",
) -> GraspReadiness:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return GraspReadiness(ready=False, reason="Could not decode base64 image.")
    return check_grasp_readiness_jpeg_bytes(raw, target_class_yolo=target_class_yolo, target_class_free_text=target_class_free_text)


# ── combined heading + object annotation ──────────────────────────────────────

def annotate_frame_with_object(
    bgr: np.ndarray,
    target_class_yolo: str,
    target_class_free_text: str = "",
) -> tuple[np.ndarray, float | None, str, float | None, float | None, int | None]:
    """Detect heading + object, annotate frame with arrow + bbox + angle line.

    Detection strategy:
      1. YOLO — if *target_class_yolo* is non-empty, run YOLO filtered to that
         canonical class (see _CLASS_SYNONYMS).
      2. Gemini Flash VLM — if YOLO finds nothing and *target_class_free_text*
         is non-empty, query Gemini Flash with that free-text description.
         Use this for objects outside COCO-80, e.g. "light switch".

    Returns (annotated_bgr, angle_deg, note, object_distance_px, robot_radius_px,
    body_area_px). angle_deg / object_distance_px are None when heading or object
    is not detected. robot_radius_px / body_area_px are None when heading
    detection fails. body_area_px is the robot's own visible yellow-body pixel
    count — pass it to navigation.mm_per_px() to convert object_distance_px to
    real-world mm, the same calibration navigate_to uses for drive distances.
    Positive angle = object is CW from forward.
    Falls back to heading-only annotation on YOLO/VLM failure.
    """
    heading = detect_heading(bgr)
    if heading is None:
        return bgr, None, "", None, None, None

    objects = _yolo_detect(bgr, target_class=target_class_yolo) if target_class_yolo else []
    obj = _pick_target(objects, heading) if objects else None

    # VLM path: YOLO found nothing (or was skipped) → ask Gemini Flash
    if obj is None and target_class_free_text:
        try:
            obj = _vlm_detect(bgr, target_class_free_text)
        except vision.VQAResponseParseError as exc:
            return annotate_bgr(bgr), None, str(exc), None, heading.body_radius_px, heading.body_area

    # _vlm_detect may return a LowConfidenceDetection (candidate seen but
    # below threshold) instead of None — that's not a plain miss, but it
    # also has no .center, so it must be treated like one here.
    if obj is None or isinstance(obj, vision.LowConfidenceDetection):
        note = str(obj) if isinstance(obj, vision.LowConfidenceDetection) else ""
        return annotate_bgr(bgr), None, note, None, heading.body_radius_px, heading.body_area

    angle_deg = compute_heading_to_object_angle(heading, obj.center)
    dist_px = math.hypot(
        heading.body_center[0] - obj.center[0],
        heading.body_center[1] - obj.center[1],
    )
    annotated = annotate_bgr(
        bgr,
        obj_center=obj.center,
        obj_bbox=(obj.x1, obj.y1, obj.x2, obj.y2),
    )
    return annotated, angle_deg, obj.note, dist_px, heading.body_radius_px, heading.body_area


def annotate_frame_with_object_b64(
    b64: str,
    target_class_yolo: str,
    target_class_free_text: str = "",
) -> tuple[str, float | None, str, float | None, float | None, int | None]:
    """Base64 JPEG in/out version of annotate_frame_with_object."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode base64 JPEG for object annotation")
    annotated, angle, note, dist_px, robot_radius_px, body_area_px = annotate_frame_with_object(
        bgr,
        target_class_yolo=target_class_yolo,
        target_class_free_text=target_class_free_text,
    )
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("cv2.imencode failed in annotate_frame_with_object_b64")
    return base64.b64encode(buf.tobytes()).decode(), angle, note, dist_px, robot_radius_px, body_area_px
