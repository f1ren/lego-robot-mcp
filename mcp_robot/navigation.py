"""
CV-based navigation: obstacle detection, A* path planning, overlay drawing,
and per-step command synthesis.

Public API:
    detect_obstacles(bgr, heading, target) -> ObstacleMap
    plan_path(obs_map) -> NavPlan
    draw_nav_overlay(bgr, obs_map, plan, step) -> np.ndarray
    commands_for_step(obs_map, plan, heading) -> tuple[float, float]
    at_target(obs_map) -> bool
    save_debug_images(bgr, obs_map, plan, outdir, step) -> dict[str, str]
"""
from __future__ import annotations

import heapq
import logging
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from mcp_robot.heading import (
    Heading,
    YELLOW_HSV_LO,
    YELLOW_HSV_HI,
)
from mcp_robot.grasp_readiness import DetectedObject

log = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

_GRID_COLS = 80
_GRID_ROWS = 60

# Corner sample size for floor color detection (fraction of image edge).
_FLOOR_SAMPLE_FRAC = 0.08
# LAB color-distance threshold: pixels within this distance of the floor
# sample color are considered navigable floor.
_FLOOR_LAB_TOL = 45

# Obstacle inflation in grid cells (robot half-width clearance).
_ROBOT_RADIUS_CELLS = 3

# "At target" threshold: stop navigating when robot centroid is within this
# fraction of the image diagonal from the target centroid.
_AT_GOAL_FRAC = 0.10

# Fixed drive duration per navigation step (seconds).
_DRIVE_STEP_S = 1.2

# How far along the planned path to look for the turn-to direction.
_LOOKAHEAD_CELLS = 5

# Overlay drawing colours (BGR)
_PATH_COLOR    = (255, 100,   0)  # blue
_ROBOT_COLOR   = (  0, 220, 220)  # yellow-green
_TARGET_COLOR  = (  0, 165, 255)  # orange


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ObstacleMap:
    """All obstacle-detection outputs for one frame."""
    free_mask: np.ndarray           # HxW uint8 binary: 255 = navigable
    grid: np.ndarray                # (_GRID_ROWS, _GRID_COLS) bool: True = navigable
    grid_scale_x: float             # pixels per grid column
    grid_scale_y: float             # pixels per grid row
    h: int
    w: int
    robot_px: tuple[int, int]       # image pixel coords of robot centroid
    target_px: tuple[int, int] | None
    robot_grid: tuple[int, int]     # (row, col) in grid
    target_grid: tuple[int, int] | None


@dataclass
class NavPlan:
    """Output of plan_path: an A* path from robot to target."""
    path_grid: list[tuple[int, int]]   # grid (row, col) coords, robot→target
    path_px: list[tuple[int, int]]     # image pixel coords, robot→target
    reachable: bool
    reason: str = ""


# ── internal helpers ──────────────────────────────────────────────────────────

def _px_to_grid(px: tuple[int, int], w: int, h: int) -> tuple[int, int]:
    col = max(0, min(_GRID_COLS - 1, int(px[0] / w * _GRID_COLS)))
    row = max(0, min(_GRID_ROWS - 1, int(px[1] / h * _GRID_ROWS)))
    return (row, col)


def _grid_to_px(rc: tuple[int, int], w: int, h: int) -> tuple[int, int]:
    x = int((rc[1] + 0.5) / _GRID_COLS * w)
    y = int((rc[0] + 0.5) / _GRID_ROWS * h)
    return (x, y)


def _astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Standard A* on a boolean grid (True=navigable). Returns path or None."""
    rows, cols = grid.shape
    gr, gc = goal

    def h(r: int, c: int) -> float:
        return abs(r - gr) + abs(c - gc)

    # (f, g, row, col) — path reconstructed via came_from
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (h(*start), 0.0, start[0], start[1]))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, cost, r, c = heapq.heappop(open_heap)
        if (r, c) == goal:
            # Reconstruct path
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not grid[nr, nc]:
                    continue
                step_cost = 1.414 if (dr and dc) else 1.0
                new_g = cost + step_cost
                nb = (nr, nc)
                if new_g < g_score.get(nb, float("inf")):
                    g_score[nb] = new_g
                    came_from[nb] = (r, c)
                    heapq.heappush(open_heap, (new_g + h(nr, nc), new_g, nr, nc))

    return None


# ── public API ────────────────────────────────────────────────────────────────

def detect_obstacles(
    bgr: np.ndarray,
    nav_heading: Heading | None,
    target: DetectedObject | None,
    floor_lab_tol: float = _FLOOR_LAB_TOL,
) -> ObstacleMap:
    """
    Build a pixel-resolution obstacle mask and a coarse navigable grid from
    an external (DroidCam) BGR frame.

    Floor is detected by sampling the image corners and thresholding by LAB
    color distance. The robot's yellow body is explicitly added to free space
    so it does not block its own path. Obstacles are then inflated by
    _ROBOT_RADIUS_CELLS grid cells so the planned path has clearance.
    """
    h, w = bgr.shape[:2]
    lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    # ── 1. Yellow body mask (needed for both floor sampling and free space) ──
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)

    # ── 2. Floor colour: anchor on pixels just outside the robot body ─────
    # The ring of pixels immediately surrounding the yellow body is
    # guaranteed to be floor (the robot rests on it). This is far more
    # reliable than sampling corners, which can be walls or shadows.
    floor_lab: np.ndarray | None = None
    if yellow_raw.sum() > 200 * 255:   # robot body is visible
        yellow_big = cv2.dilate(yellow_raw, np.ones((43, 43), np.uint8))
        ring_mask = cv2.bitwise_and(yellow_big, cv2.bitwise_not(yellow_raw))
        ring_pixels = lab_img[ring_mask > 0]
        if len(ring_pixels) >= 50:
            floor_lab = np.median(ring_pixels.astype(np.float32), axis=0)
            log.debug("Floor colour sampled from %d ring pixels around robot body", len(ring_pixels))

    if floor_lab is None:
        # Fallback: combine image corners with centre-bottom strip.
        mh = max(1, int(h * _FLOOR_SAMPLE_FRAC))
        mw = max(1, int(w * _FLOOR_SAMPLE_FRAC))
        centre_bot = lab_img[int(h * 0.70):, int(w * 0.25):int(w * 0.75)].reshape(-1, 3)
        corner_px = np.vstack([
            lab_img[:mh,   :mw  ].reshape(-1, 3),
            lab_img[:mh,   w-mw:].reshape(-1, 3),
            lab_img[h-mh:, :mw  ].reshape(-1, 3),
            lab_img[h-mh:, w-mw:].reshape(-1, 3),
            centre_bot,
        ])
        floor_lab = np.median(corner_px.astype(np.float32), axis=0)
        log.debug("Floor colour sampled from corners+centre-bottom (robot not visible)")

    lab_f = lab_img.astype(np.float32)
    dist = np.sqrt(
        (lab_f[:, :, 0] - floor_lab[0]) ** 2 +
        (lab_f[:, :, 1] - floor_lab[1]) ** 2 +
        (lab_f[:, :, 2] - floor_lab[2]) ** 2
    )
    floor_mask = (dist < floor_lab_tol).astype(np.uint8) * 255

    # ── 3. Robot yellow body → free (robot can't block its own path) ──────
    yellow = yellow_raw.copy()
    # Dilate the body mask slightly to cover any adjacent shadows.
    yellow = cv2.dilate(yellow, np.ones((17, 17), np.uint8))

    free_mask = cv2.bitwise_or(floor_mask, yellow)

    # Morphological cleanup: close small holes in the floor, remove tiny noise.
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    # ── 3. Build coarse grid (majority vote per cell) ─────────────────────
    cell_h = h / _GRID_ROWS
    cell_w = w / _GRID_COLS
    raw_grid = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.uint8)
    for gr in range(_GRID_ROWS):
        y0, y1 = int(gr * cell_h), max(int(gr * cell_h) + 1, int((gr + 1) * cell_h))
        for gc in range(_GRID_COLS):
            x0, x1 = int(gc * cell_w), max(int(gc * cell_w) + 1, int((gc + 1) * cell_w))
            raw_grid[gr, gc] = 255 if free_mask[y0:y1, x0:x1].mean() > 100 else 0

    # ── 4. Inflate obstacles by robot clearance ──────────────────────────
    r = _ROBOT_RADIUS_CELLS
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inflated = cv2.erode(raw_grid, struct)
    grid = (inflated > 0)

    # ── 5. Locate robot and target; ensure their cells are navigable ──────
    robot_px: tuple[int, int] = nav_heading.body_center if nav_heading is not None else (w // 2, h // 2)
    target_px: tuple[int, int] | None = target.center if target is not None else None

    robot_grid = _px_to_grid(robot_px, w, h)
    target_grid = _px_to_grid(target_px, w, h) if target_px is not None else None

    # Always make the robot's current cell passable (override inflation).
    grid[robot_grid[0], robot_grid[1]] = True
    # Open a small 3×3 patch around the target so A* can reach it.
    if target_grid is not None:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                gr2 = max(0, min(_GRID_ROWS - 1, target_grid[0] + dr))
                gc2 = max(0, min(_GRID_COLS - 1, target_grid[1] + dc))
                grid[gr2, gc2] = True

    return ObstacleMap(
        free_mask=free_mask,
        grid=grid,
        grid_scale_x=cell_w,
        grid_scale_y=cell_h,
        h=h, w=w,
        robot_px=robot_px,
        target_px=target_px,
        robot_grid=robot_grid,
        target_grid=target_grid,
    )


def plan_path(obs_map: ObstacleMap) -> NavPlan:
    """Run A* from robot grid cell to target grid cell."""
    if obs_map.target_grid is None:
        return NavPlan([], [], reachable=False, reason="No target detected")

    start = obs_map.robot_grid
    goal  = obs_map.target_grid

    if start == goal:
        return NavPlan([start], [obs_map.robot_px], reachable=True, reason="Already at target")

    path_grid = _astar(obs_map.grid, start, goal)
    if path_grid is None:
        log.warning("navigate_to: A* found no path from %s to %s", start, goal)
        return NavPlan([], [], reachable=False, reason=f"No path from grid{start}→grid{goal}")

    path_px = [_grid_to_px(rc, obs_map.w, obs_map.h) for rc in path_grid]
    return NavPlan(
        path_grid=path_grid,
        path_px=path_px,
        reachable=True,
        reason=f"Path: {len(path_grid)} cells",
    )


def draw_nav_overlay(
    bgr: np.ndarray,
    obs_map: ObstacleMap,
    plan: NavPlan,
    step: int = 0,
) -> np.ndarray:
    """Annotate bgr with obstacle tint, A* path, robot, and target markers."""
    out = bgr.copy()

    # Semi-transparent red tint over obstacle pixels.
    obstacle_layer = out.copy()
    obstacle_layer[obs_map.free_mask == 0] = (0, 0, 160)
    cv2.addWeighted(obstacle_layer, 0.40, out, 0.60, 0, out)

    # Draw the planned path.
    if plan.reachable and len(plan.path_px) > 1:
        pts = np.array(plan.path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, _PATH_COLOR, 2, cv2.LINE_AA)
        for pt in plan.path_px[::6]:
            cv2.circle(out, pt, 3, _PATH_COLOR, -1)

    # Robot marker.
    rx, ry = obs_map.robot_px
    cv2.circle(out, (rx, ry), 10, _ROBOT_COLOR, 2)
    cv2.putText(out, "R", (rx - 5, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ROBOT_COLOR, 1, cv2.LINE_AA)

    # Target marker.
    if obs_map.target_px is not None:
        tx, ty = obs_map.target_px
        cv2.circle(out, (tx, ty), 10, _TARGET_COLOR, 2)
        cv2.putText(out, "T", (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TARGET_COLOR, 1, cv2.LINE_AA)

    # Step label (ASCII only — cv2 cannot render Unicode).
    label = f"Step {step} - {plan.reason}"
    cv2.rectangle(out, (4, 4), (len(label) * 9 + 8, 26), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def commands_for_step(
    obs_map: ObstacleMap,
    plan: NavPlan,
    nav_heading: Heading | None,
) -> tuple[float, float]:
    """Return (turn_deg, drive_s) for the next navigation micro-step.

    turn_deg: signed body-degrees to rotate (positive = CW viewed from above).
    drive_s:  seconds to drive straight forward afterward.
    """
    if not plan.reachable or len(plan.path_px) < 2:
        return 0.0, 0.0

    idx = min(_LOOKAHEAD_CELLS, len(plan.path_px) - 1)
    next_px = plan.path_px[idx]
    robot_px = obs_map.robot_px

    dx = next_px[0] - robot_px[0]
    dy = next_px[1] - robot_px[1]

    if math.hypot(dx, dy) < 1.0:
        return 0.0, 0.0

    # Signed turn angle (CW positive, matching turn() and heading conventions).
    if nav_heading is not None:
        fw = nav_heading.forward
        # cross product in image y-down coords: positive → target is CW from forward
        cross = fw[0] * dy - fw[1] * dx
        dot   = fw[0] * dx + fw[1] * dy
        turn_deg = math.degrees(math.atan2(cross, dot))
    else:
        turn_deg = 0.0

    # Drive duration: reduce when already close to target.
    drive_s = _DRIVE_STEP_S
    if obs_map.target_px is not None:
        tdist = math.hypot(robot_px[0] - obs_map.target_px[0],
                           robot_px[1] - obs_map.target_px[1])
        if tdist < math.hypot(obs_map.w, obs_map.h) * 0.18:
            drive_s = min(drive_s, 0.8)

    return turn_deg, drive_s


def at_target(obs_map: ObstacleMap) -> bool:
    """True when robot centroid is within _AT_GOAL_FRAC × diagonal of target."""
    if obs_map.target_px is None:
        return False
    dist = math.hypot(
        obs_map.robot_px[0] - obs_map.target_px[0],
        obs_map.robot_px[1] - obs_map.target_px[1],
    )
    return dist < math.hypot(obs_map.w, obs_map.h) * _AT_GOAL_FRAC


def save_debug_images(
    bgr: np.ndarray,
    obs_map: ObstacleMap,
    plan: NavPlan,
    outdir: str,
    step: int,
) -> dict[str, str]:
    """Write raw frame, free-space mask, and nav overlay to outdir.

    Returns a dict mapping name → saved path. The raw frame is intentionally
    saved so unit tests can load it as a reproducible fixture.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: dict[str, str] = {}

    raw_path = os.path.join(outdir, f"step_{step:02d}_raw.jpg")
    cv2.imwrite(raw_path, bgr)
    saved["raw"] = raw_path

    # Obstacle mask visualisation: green = navigable, red = blocked.
    mask_vis = np.zeros_like(bgr)
    mask_vis[obs_map.free_mask >  0] = (0, 160, 0)
    mask_vis[obs_map.free_mask == 0] = (0, 0, 180)
    mask_path = os.path.join(outdir, f"step_{step:02d}_obstacle_mask.jpg")
    cv2.imwrite(mask_path, mask_vis)
    saved["obstacle_mask"] = mask_path

    overlay_bgr = draw_nav_overlay(bgr, obs_map, plan, step)
    overlay_path = os.path.join(outdir, f"step_{step:02d}_nav_overlay.jpg")
    cv2.imwrite(overlay_path, overlay_bgr)
    saved["nav_overlay"] = overlay_path

    log.info("Navigation debug images (step %d) saved to: %s", step, outdir)
    return saved
