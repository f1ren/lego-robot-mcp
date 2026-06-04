"""
Remove the robot and target object from a DroidCam frame using LaMa inpainting.

The robot is identified by its yellow LEGO body (HSV color mask + rotated-rectangle
footprint that covers the full chassis including the gripper arm).  The target is
identified by its YOLO bounding box.  Both regions are combined into a single binary
mask and handed to the LaMa deep-inpainting model, which fills the holes with
plausible floor texture.

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

_lama_model = None


def _load_lama():
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama
        log.info("Loading LaMa inpainting model…")
        _lama_model = SimpleLama()
        log.info("LaMa model ready")
    return _lama_model


def build_removal_mask(
    bgr: np.ndarray,
    nav_heading: Heading | None,
    target: DetectedObject | None,
    *,
    robot_dilation_px: int = 15,
    target_padding_px: int = 10,
) -> np.ndarray:
    """Return a uint8 mask (same H×W as bgr) where 255 = pixels to remove.

    Robot region: full rotated-rectangle footprint (yellow chassis + gripper arm),
    dilated by robot_dilation_px pixels for safety.

    Target region: YOLO bounding box padded by target_padding_px on each side.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # ── Robot mask ────────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)

    robot_footprint = _robot_footprint_mask(yellow_raw, 0.0, nav_heading)

    if robot_dilation_px > 0:
        k = 2 * robot_dilation_px + 1
        robot_footprint = cv2.dilate(robot_footprint, np.ones((k, k), np.uint8))

    mask = cv2.bitwise_or(mask, robot_footprint)

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
