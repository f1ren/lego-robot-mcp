"""
Remove the robot and target object from a DroidCam frame using LaMa inpainting.

The robot is identified by its yellow LEGO body (HSV color mask + rotated-rectangle
footprint that covers the full chassis including the gripper arm).  The arm/gripper
chain is caught by a separate dark-pixel pass: any dark blob (V<70, S<90) that
overlaps the dilated footprint zone is included, regardless of the robot's heading.
Blue wheel hubs are caught by a separate blue-pixel proximity pass so they don't
bleed into LaMa's output at the mask boundary.
The target is identified by its YOLO bounding box.  All regions are combined into a
single binary mask handed to the LaMa deep-inpainting model.

Public API:
    build_removal_mask(bgr, nav_heading, target) -> np.ndarray (uint8, 255 = remove)
    remove_robot_and_target(bgr, nav_heading, target) -> tuple[np.ndarray, np.ndarray]
        Returns (inpainted_bgr, mask_uint8)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from mcp_robot.heading import Heading, YELLOW_HSV_LO, YELLOW_HSV_HI
from mcp_robot.grasp_readiness import DetectedObject
from mcp_robot.navigation import _robot_footprint_mask

log = logging.getLogger(__name__)

# Dark-pixel thresholds for the arm/gripper chain (same as heading.py internals).
_ARM_V_MAX = 70
_ARM_S_MAX = 90
# Maximum blob area (fraction of frame) to include as arm/gripper.
# Excludes large dark regions like furniture, cabinet edges, or floor shadows.
_ARM_MAX_BLOB_FRAC = 0.08

# HSV range for the robot's blue LEGO wheel hubs.
_BLUE_WHEEL_HSV_LO = np.array([100, 100,  60], dtype=np.uint8)
_BLUE_WHEEL_HSV_HI = np.array([130, 255, 220], dtype=np.uint8)
_BLUE_WHEEL_MAX_BLOB_FRAC = 0.003  # wheel hubs ~200px; cup ~6500px — excludes cup

_lama_model = None


def _load_lama():
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama
        log.info("Loading LaMa inpainting model…")
        _lama_model = SimpleLama()
        log.info("LaMa model ready")
    return _lama_model


def _arm_gripper_mask(
    hsv: np.ndarray,
    robot_footprint: np.ndarray,
    proximity_dilation_px: int = 150,
    nav_heading: Heading | None = None,
) -> np.ndarray:
    """Return a mask of arm/gripper dark pixels connected to the robot body.

    The arm chain can extend in any direction depending on heading and arm height,
    so this is direction-agnostic: find all dark blobs, keep only those that
    overlap the footprint zone dilated by proximity_dilation_px, and exclude
    blobs large enough to be background elements (furniture, shadows).

    When nav_heading is provided a forward cap is applied: only dark blobs whose
    centroid is no more forward than the footprint's front edge are included.
    This prevents masking vertical obstacles (wall switch, door) that the arm
    is pressed against — those regions must remain visible so the depth model
    classifies them as obstacles.  White/grey obstacles are safe regardless
    (never dark enough to pass the HSV threshold).
    """
    h, w = hsv.shape[:2]

    dark_raw = cv2.inRange(
        hsv,
        np.array([0,   0,   0], dtype=np.uint8),
        np.array([179, _ARM_S_MAX, _ARM_V_MAX], dtype=np.uint8),
    )
    # Close small gaps in the chain links so each arm segment is one blob.
    dark = cv2.morphologyEx(dark_raw, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    # Proximity zone: footprint grown outward to catch arm segments that sit
    # just outside the body rectangle regardless of heading.
    k = 2 * proximity_dilation_px + 1
    proximity = cv2.dilate(robot_footprint, np.ones((k, k), np.uint8))

    # Forward cap: exclude pixels more forward than the footprint's front edge.
    # Derived from the maximum forward projection of any footprint pixel along
    # the heading vector, so blobs at the obstacle level are never included.
    if nav_heading is not None:
        fp_rows, fp_cols = np.where(robot_footprint > 0)
        if len(fp_rows):
            cx, cy = nav_heading.body_center
            fw = nav_heading.forward
            fwd_fp = fw[0] * (fp_cols - cx) + fw[1] * (fp_rows - cy)
            max_fwd = float(fwd_fp.max())
            yy, xx = np.mgrid[0:h, 0:w]
            fwd_all = fw[0] * (xx - cx) + fw[1] * (yy - cy)
            fwd_cap = (fwd_all <= max_fwd).astype(np.uint8) * 255
            proximity = cv2.bitwise_and(proximity, fwd_cap)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    result = np.zeros((h, w), dtype=np.uint8)
    max_blob_px = int(h * w * _ARM_MAX_BLOB_FRAC)

    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > max_blob_px:
            continue  # too large to be part of the robot arm
        blob = (labels == lbl).astype(np.uint8) * 255
        clipped = cv2.bitwise_and(blob, proximity)
        if clipped.any():
            # Add only the portion within the (forward-capped) proximity zone,
            # not the full blob — prevents a chain blob that straddles the cap
            # from pulling in pixels at the obstacle level.
            result = cv2.bitwise_or(result, clipped)

    return result


def _blue_wheel_mask(
    hsv: np.ndarray,
    robot_footprint: np.ndarray,
    proximity_dilation_px: int = 80,
) -> np.ndarray:
    """Return a mask of blue wheel-hub pixels adjacent to the robot body.

    Blue wheel hubs sit at the sides and rear of the chassis, just outside or
    at the edge of the yellow footprint rectangle.  Without explicit masking
    they bleed into LaMa's output as blue artifacts at the mask boundary.
    """
    h, w = hsv.shape[:2]
    blue_raw = cv2.inRange(hsv, _BLUE_WHEEL_HSV_LO, _BLUE_WHEEL_HSV_HI)
    blue = cv2.morphologyEx(blue_raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    k = 2 * proximity_dilation_px + 1
    proximity = cv2.dilate(robot_footprint, np.ones((k, k), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    result = np.zeros((h, w), dtype=np.uint8)
    max_blob_px = int(h * w * _BLUE_WHEEL_MAX_BLOB_FRAC)

    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > max_blob_px:
            continue  # too large to be a wheel (e.g. the blue cup target)
        blob = (labels == lbl).astype(np.uint8) * 255
        if cv2.bitwise_and(blob, proximity).any():
            result = cv2.bitwise_or(result, blob)

    return result


def build_removal_mask(
    bgr: np.ndarray,
    nav_heading: Heading | None,
    target: DetectedObject | None,
    *,
    robot_dilation_px: int = 15,
    target_padding_px: int = 10,
) -> np.ndarray:
    """Return a uint8 mask (same H×W as bgr) where 255 = pixels to remove.

    Robot region: rotated-rectangle footprint (yellow chassis) + dark arm/gripper
    blobs connected to it, all dilated by robot_dilation_px pixels for safety.

    Target region: YOLO bounding box padded by target_padding_px on each side.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # ── Robot mask ────────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)

    robot_footprint = _robot_footprint_mask(yellow_raw, 0.0, nav_heading)

    # Arm/gripper: dark blobs adjacent to the footprint.  No forward cap here —
    # when the gripper presses against a switch/wall the arm chain is dark and
    # should be masked so LaMa fills it from the surrounding obstacle context
    # (e.g. the white switch behind it).  The switch itself is white/grey and
    # never caught by the dark-blob threshold.
    arm_mask = _arm_gripper_mask(hsv, robot_footprint)
    # Blue wheel hubs: sit at the chassis sides/rear, excluded from yellow mask.
    wheel_mask = _blue_wheel_mask(hsv, robot_footprint)

    combined = cv2.bitwise_or(robot_footprint, cv2.bitwise_or(arm_mask, wheel_mask))

    if robot_dilation_px > 0:
        k = 2 * robot_dilation_px + 1
        combined = cv2.dilate(combined, np.ones((k, k), np.uint8))

    mask = cv2.bitwise_or(mask, combined)

    # ── Target mask ───────────────────────────────────────────────────────────
    if target is not None:
        p = target_padding_px
        x1 = max(0, target.x1 - p)
        y1 = max(0, target.y1 - p)
        x2 = min(w - 1, target.x2 + p)
        y2 = min(h - 1, target.y2 + p)
        mask[y1:y2, x1:x2] = 255

    return mask


def remove_robot_and_target(
    bgr: np.ndarray,
    nav_heading: Heading | None = None,
    target: DetectedObject | None = None,
    *,
    robot_dilation_px: int = 15,
    target_padding_px: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint the robot and target out of bgr using LaMa.

    Returns (inpainted_bgr, mask) where:
      - inpainted_bgr: same shape as bgr, masked regions replaced with floor texture
      - mask: uint8 H×W array, 255 = pixels that were removed/inpainted
    """
    from PIL import Image as _PIL

    mask = build_removal_mask(
        bgr, nav_heading, target,
        robot_dilation_px=robot_dilation_px,
        target_padding_px=target_padding_px,
    )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_image = _PIL.fromarray(rgb)
    pil_mask  = _PIL.fromarray(mask)

    lama = _load_lama()
    result_pil = lama(pil_image, pil_mask)

    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # LaMa may pad the output to a multiple of 8 — crop back to original size.
    result_bgr = result_bgr[:bgr.shape[0], :bgr.shape[1]]

    return result_bgr, mask
