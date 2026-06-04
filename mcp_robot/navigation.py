"""
CV-based navigation: obstacle detection, A* path planning, overlay drawing,
and per-step command synthesis.

Public API:
    estimate_depth(bgr) -> np.ndarray          (float32 depth map, larger = farther)
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
from collections import deque
from dataclasses import dataclass, field

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

# Mahalanobis-distance threshold for depth-gradient floor classification.
# Pixels within this many IQR-derived sigmas of the robot-ring floor gradient
# are classified as navigable floor.
_DEPTH_GRAD_N_SIGMA = 2.5

# Minimum obstacle inflation in grid cells — used when the robot body cannot
# be detected.  Normally the inflation radius is derived from the yellow body
# bounding box so the path stays at least one robot-width from obstacles.
_MIN_ROBOT_RADIUS_CELLS = 3

# Safety margin applied to the robot-disk radius before Minkowski-sum erosion.
# 1.0 = exact robot radius; 1.2 = 20% extra clearance on all sides.
_CSPACE_BUFFER_SCALE = 1.7

# Fixed drive duration per navigation step (seconds).
_DRIVE_STEP_S = 1.2

# How far along the planned path to look for the turn-to direction.
_LOOKAHEAD_CELLS = 5

# Overlay drawing colours (BGR)
_PATH_COLOR    = (255, 100,   0)  # blue
_ROBOT_COLOR   = (  0, 220, 220)  # yellow-green
_TARGET_COLOR  = (  0, 165, 255)  # orange

# ── Depth Anything model ──────────────────────────────────────────────────────

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_depth_pipeline = None


def _load_depth_model():
    global _depth_pipeline
    if _depth_pipeline is not None:
        return _depth_pipeline
    from transformers import pipeline as _hf_pipeline
    _depth_pipeline = _hf_pipeline(
        "depth-estimation",
        model=_DEPTH_MODEL_ID,
        device="cpu",
    )
    log.info("Depth Anything V2 Small loaded (%s)", _DEPTH_MODEL_ID)
    return _depth_pipeline


def estimate_depth(bgr: np.ndarray) -> np.ndarray:
    """Run Depth Anything V2 Small on a BGR frame.

    Returns a float32 depth map with the same (H, W) as the input where
    larger values indicate greater distance from the camera.
    """
    from PIL import Image as _PIL
    pipe = _load_depth_model()
    rgb = _PIL.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = pipe(rgb)
    return np.array(result["depth"], dtype=np.float32)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ObstacleMap:
    """All obstacle-detection outputs for one frame."""
    free_mask: np.ndarray            # HxW uint8 binary: 255 = navigable
    grid: np.ndarray                 # (_GRID_ROWS, _GRID_COLS) bool: True = navigable (inflated)
    raw_grid: np.ndarray             # (_GRID_ROWS, _GRID_COLS) bool: navigable before inflation
    grid_scale_x: float              # pixels per grid column
    grid_scale_y: float              # pixels per grid row
    h: int
    w: int
    robot_px: tuple[int, int]        # image pixel coords of robot centroid
    target_px: tuple[int, int] | None
    robot_grid: tuple[int, int]      # (row, col) in grid
    target_grid: tuple[int, int] | None
    robot_radius_px: float           # half robot body size (pixels); used for clearance
    target_radius_px: float          # half target bounding box (pixels); 0 if no target
    depth_map: np.ndarray | None = field(default=None, repr=False)   # raw depth, for debug
    cleaned_bgr: np.ndarray | None = field(default=None, repr=False)  # inpainted frame used for depth


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


def _robot_ring(yellow_raw: np.ndarray) -> np.ndarray | None:
    """Return the ring of pixels just outside the yellow robot body, or None."""
    if yellow_raw.sum() <= 200 * 255:
        return None
    yellow_big = cv2.dilate(yellow_raw, np.ones((43, 43), np.uint8))
    ring = cv2.bitwise_and(yellow_big, cv2.bitwise_not(yellow_raw))
    return ring if (ring > 0).sum() >= 50 else None


def _robot_footprint_mask(
    yellow_raw: np.ndarray,
    robot_radius_px: float,
    nav_heading: Heading | None,
) -> np.ndarray:
    """Rotated-rectangle mask sized to the full physical robot, not just the yellow board.

    The yellow LEGO chassis is the reference rectangle.  Multipliers extend it
    to cover the rest of the robot:
      forward  ×2.25 — gripper arm + fingers protrude well in front of the board
      backward ×1.25 — small rear overhang (wheels, battery cable)
      sides    ×1.15 — wheels sit slightly outside the board on both sides

    Falls back to a small dilation of the raw yellow pixels when heading is
    unavailable.
    """
    h, w = yellow_raw.shape[:2]

    contours, _ = cv2.findContours(yellow_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or nav_heading is None:
        return cv2.dilate(yellow_raw, np.ones((25, 25), np.uint8))

    main = max(contours, key=cv2.contourArea)
    pts  = main.reshape(-1, 2).astype(np.float64)

    cx = float(nav_heading.body_center[0])
    cy = float(nav_heading.body_center[1])
    fw   = np.array(nav_heading.forward, dtype=np.float64)
    side = np.array([-fw[1], fw[0]],    dtype=np.float64)

    rel       = pts - np.array([cx, cy])
    fwd_proj  = rel @ fw
    side_proj = rel @ side

    front  = float(fwd_proj.max())                               * 4.00
    back   = float(fwd_proj.min())                               * 1.35
    half_w = float(max(abs(side_proj.max()), abs(side_proj.min()))) * 1.35

    corners = np.array([
        [cx + fw[0] * front + side[0] * half_w,  cy + fw[1] * front + side[1] * half_w],
        [cx + fw[0] * front - side[0] * half_w,  cy + fw[1] * front - side[1] * half_w],
        [cx + fw[0] * back  - side[0] * half_w,  cy + fw[1] * back  - side[1] * half_w],
        [cx + fw[0] * back  + side[0] * half_w,  cy + fw[1] * back  + side[1] * half_w],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    return mask


def _depth_gradient_floor_mask(
    depth: np.ndarray,
    ring: np.ndarray,
    n_sigma: float = _DEPTH_GRAD_N_SIGMA,
) -> np.ndarray | None:
    """Floor mask from depth-gradient direction similarity to the robot-ring reference.

    The floor is a planar surface with a characteristic depth gradient (surface
    orientation).  Any pixel whose Sobel gradient vector falls within n_sigma
    IQR-derived standard deviations of the floor-gradient sample is classified
    as navigable.

    Key advantage over depth-value thresholding: shadow areas have the same
    surface orientation as the lit floor (same physical plane), so they produce
    the same gradient even though their absolute depth values are wrong.  Depth
    models reliably get the gradient right even when absolute depth is off.

    Returns None when there are not enough ring samples.
    """
    if ring is None or (ring > 0).sum() < 50:
        return None

    # Smooth depth before gradient to reduce sensor noise.
    depth_smooth = cv2.GaussianBlur(depth, (9, 9), 0)
    gx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=5)

    # Sample floor gradient from the robot ring (guaranteed floor pixels).
    ring_gx = gx[ring > 0].astype(np.float64)
    ring_gy = gy[ring > 0].astype(np.float64)

    floor_gx = float(np.median(ring_gx))
    floor_gy = float(np.median(ring_gy))

    # Robust σ from IQR (÷1.35 converts IQR → σ for a normal distribution).
    q1x, q3x = np.percentile(ring_gx, [25, 75])
    q1y, q3y = np.percentile(ring_gy, [25, 75])
    sigma_gx = max(float((q3x - q1x) / 1.35), 1.0)
    sigma_gy = max(float((q3y - q1y) / 1.35), 1.0)

    log.debug("Depth gradient floor: gx=%.1f±%.1f  gy=%.1f±%.1f",
              floor_gx, sigma_gx, floor_gy, sigma_gy)

    # Mahalanobis-like distance in gradient space.
    z_gx = ((gx - floor_gx) / sigma_gx).astype(np.float32)
    z_gy = ((gy - floor_gy) / sigma_gy).astype(np.float32)
    dist = np.sqrt(z_gx ** 2 + z_gy ** 2)

    return (dist <= n_sigma).astype(np.uint8) * 255



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

    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (h(*start), 0.0, start[0], start[1]))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, cost, r, c = heapq.heappop(open_heap)
        if (r, c) == goal:
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


def _nearest_free_cell(
    grid: np.ndarray,
    goal: tuple[int, int],
) -> tuple[int, int] | None:
    """BFS outward from goal; return the nearest cell where grid is True, or None.

    Used to snap an ideal approach point that lands inside the Minkowski-sum
    inflation buffer (yellow zone) to the boundary of the navigable C-space.
    """
    rows, cols = grid.shape
    if grid[goal[0], goal[1]]:
        return goal
    visited: set[tuple[int, int]] = {goal}
    q: deque[tuple[int, int]] = deque([goal])
    while q:
        r, c = q.popleft()
        if grid[r, c]:
            return (r, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return None


def _bresenham_cells(
    r0: int, c0: int, r1: int, c1: int
):
    """Yield every grid cell (row, col) on the straight line from (r0,c0) to (r1,c1)."""
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _has_line_of_sight(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """True if every cell on the Bresenham line from a to b is navigable."""
    for r, c in _bresenham_cells(a[0], a[1], b[0], b[1]):
        if not grid[r, c]:
            return False
    return True


def _simplify_path(
    path: list[tuple[int, int]],
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    """Reduce an A* path to the fewest straight segments via greedy line-of-sight.

    Scans forward from each waypoint to find the farthest successor reachable
    by a straight Bresenham line that stays entirely inside navigable cells.
    """
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _has_line_of_sight(grid, path[i], path[j]):
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


# ── public API ────────────────────────────────────────────────────────────────

def detect_obstacles(
    bgr: np.ndarray,
    nav_heading: Heading | None,
    target: DetectedObject | None,
    use_depth: bool = True,
) -> ObstacleMap:
    """Build a pixel-resolution obstacle mask and coarse navigable grid.

    Floor is detected by comparing the depth-map gradient at every pixel to the
    floor-gradient reference sampled from the ring immediately outside the robot's
    yellow body.  Pixels whose Sobel gradient vector falls within _DEPTH_GRAD_N_SIGMA
    IQR-derived standard deviations of the floor reference are navigable; all others
    are obstacles.

    This gradient approach is colour-independent and shadow-robust: shadow areas
    share the same surface orientation as the lit floor, so their gradient matches
    the reference even though their absolute depth values differ.

    Obstacles are inflated by _ROBOT_RADIUS_CELLS grid cells to give the robot
    clearance.  When depth estimation is unavailable the function raises RuntimeError.
    """
    h, w = bgr.shape[:2]

    # ── 1. Yellow body mask, robot radius, and robot ring ────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)

    # Robot radius from the largest yellow contour's bounding box.
    # Using all-yellow-pixel span would include stray floor/furniture pixels
    # with similar hue, inflating the radius to frame-spanning values.
    _y_contours, _ = cv2.findContours(yellow_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if _y_contours:
        _main = max(_y_contours, key=cv2.contourArea)
        _, _, _cw, _ch = cv2.boundingRect(_main)
        robot_radius_px = float(max(_cw, _ch)) / 2.0
    else:
        robot_radius_px = 40.0  # fallback when body not visible

    # Build a rotated-rectangle footprint mask that covers the full robot body
    # (yellow chassis + wheels + gripper arm).  A rectangle oriented along the
    # heading fits the robot much better than a circle, and extending further
    # forward ensures the gripper fingers are also marked as obstacle-free.
    yellow_dilated = _robot_footprint_mask(yellow_raw, robot_radius_px, nav_heading)

    ring = _robot_ring(yellow_raw)

    # ── 1b. Remove robot and target before depth estimation ──────────────
    # Local import avoids circular dependency (inpainting imports _robot_footprint_mask
    # from this module).  The cleaned frame gives the depth model a clear view of the
    # floor where the robot and target stood, so their presence no longer creates false
    # obstacle regions in the depth gradient mask.
    from mcp_robot.inpainting import remove_robot_and_target as _inpaint
    bgr_for_depth, _ = _inpaint(bgr, nav_heading, target)
    log.debug("Robot/target inpainting for depth estimation succeeded")

    # ── 2. Depth gradient floor mask ─────────────────────────────────────
    depth_map: np.ndarray | None = None
    grad_free: np.ndarray | None = None
    if use_depth:
        depth_map = estimate_depth(bgr_for_depth)
        if ring is not None:
            grad_free = _depth_gradient_floor_mask(depth_map, ring)
            if grad_free is not None:
                log.debug("Depth gradient floor mask: %.1f%% navigable",
                          (grad_free > 0).mean() * 100)

    if grad_free is None:
        raise RuntimeError(
            "Depth gradient floor mask unavailable — "
            "either depth estimation failed or robot body not visible."
        )

    # ── 3. Free mask: gradient floor + yellow body ────────────────────────
    free_mask = cv2.bitwise_or(grad_free, yellow_dilated)
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    # ── 4. Build coarse grid (majority vote per cell) ─────────────────────
    cell_h = h / _GRID_ROWS
    cell_w = w / _GRID_COLS
    base_grid = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.uint8)
    for gr in range(_GRID_ROWS):
        y0, y1 = int(gr * cell_h), max(int(gr * cell_h) + 1, int((gr + 1) * cell_h))
        for gc in range(_GRID_COLS):
            x0, x1 = int(gc * cell_w), max(int(gc * cell_w) + 1, int((gc + 1) * cell_w))
            base_grid[gr, gc] = 255 if free_mask[y0:y1, x0:x1].mean() > 100 else 0

    # ── 5. C-space: inflate obstacles by the robot disk radius ───────────
    # Treat the robot as a disk of radius robot_radius_px.  Eroding the
    # navigable grid by that disk is the Minkowski-sum expansion that converts
    # the workspace into configuration space (C-space): A* then operates on
    # robot-center points only, with all collision geometry baked in.
    #
    # Cells are not square, so we derive separate x/y radii in grid units to
    # keep the clearance circular in pixel space (not stretched by cell aspect).
    rx_cells = max(_MIN_ROBOT_RADIUS_CELLS, round(robot_radius_px * _CSPACE_BUFFER_SCALE / cell_w))
    ry_cells = max(_MIN_ROBOT_RADIUS_CELLS, round(robot_radius_px * _CSPACE_BUFFER_SCALE / cell_h))
    log.debug("Robot disk: %.1f px → (%d col, %d row) cells", robot_radius_px, rx_cells, ry_cells)

    # Compute target position and radius early so they can be pre-marked.
    target_px: tuple[int, int] | None = target.center if target is not None else None
    if target is not None:
        target_radius_px = float(max(target.x2 - target.x1,
                                     target.y2 - target.y1)) / 2.0
    else:
        target_radius_px = 0.0

    # Pre-mark both the robot disk and the target disk in base_grid so they
    # survive C-space erosion.  Neither is an obstacle to itself: the
    # depth-gradient mask classifies them as non-floor (which is correct), but
    # that creates obstacle cells that the Minkowski-sum expansion would enlarge,
    # trapping the robot or burying the target inside its own exclusion zone.
    robot_px: tuple[int, int] = nav_heading.body_center if nav_heading is not None else (w // 2, h // 2)
    robot_grid = _px_to_grid(robot_px, w, h)
    for _dr in range(-ry_cells, ry_cells + 1):
        for _dc in range(-rx_cells, rx_cells + 1):
            if (_dr / max(ry_cells, 1)) ** 2 + (_dc / max(rx_cells, 1)) ** 2 <= 1.0:
                _gr = max(0, min(_GRID_ROWS - 1, robot_grid[0] + _dr))
                _gc = max(0, min(_GRID_COLS - 1, robot_grid[1] + _dc))
                base_grid[_gr, _gc] = 255

    if target_px is not None and target_radius_px > 0:
        t_grid = _px_to_grid(target_px, w, h)
        t_rx = max(1, round(target_radius_px / cell_w))
        t_ry = max(1, round(target_radius_px / cell_h))
        log.debug("Target disk: %.1f px → (%d col, %d row) cells", target_radius_px, t_rx, t_ry)
        for _dr in range(-t_ry, t_ry + 1):
            for _dc in range(-t_rx, t_rx + 1):
                if (_dr / max(t_ry, 1)) ** 2 + (_dc / max(t_rx, 1)) ** 2 <= 1.0:
                    _gr = max(0, min(_GRID_ROWS - 1, t_grid[0] + _dr))
                    _gc = max(0, min(_GRID_COLS - 1, t_grid[1] + _dc))
                    base_grid[_gr, _gc] = 255

    struct = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * rx_cells + 1, 2 * ry_cells + 1),
    )
    inflated = cv2.erode(base_grid, struct)
    grid = (inflated > 0)

    # ── 6. Finalise grid ─────────────────────────────────────────────────
    target_grid = _px_to_grid(target_px, w, h) if target_px is not None else None

    # Belt-and-suspenders: force the robot's own cell navigable after erosion.
    # The target is NOT forced — its cells survive or not based on surrounding
    # floor, so the approach goal naturally snaps to connected green cells.
    grid[robot_grid[0], robot_grid[1]] = True

    return ObstacleMap(
        free_mask=free_mask,
        grid=grid,
        raw_grid=(base_grid > 0),
        grid_scale_x=cell_w,
        grid_scale_y=cell_h,
        h=h, w=w,
        robot_px=robot_px,
        target_px=target_px,
        robot_grid=robot_grid,
        target_grid=target_grid,
        robot_radius_px=robot_radius_px,
        target_radius_px=target_radius_px,
        depth_map=depth_map,
        cleaned_bgr=bgr_for_depth,
    )


def _approach_goal(obs_map: ObstacleMap) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return (approach_grid, approach_px) — the A* goal for the current step.

    The goal is the point at distance (robot_radius + target_radius) from the
    target center, along the current robot→target line.  This positions the robot
    at grasp distance without overshooting into the target.

    Returns None when no target is set or the robot is already within approach distance.
    """
    if obs_map.target_px is None:
        return None
    dx = obs_map.robot_px[0] - obs_map.target_px[0]
    dy = obs_map.robot_px[1] - obs_map.target_px[1]
    dist = math.hypot(dx, dy)
    approach_r = obs_map.robot_radius_px + obs_map.target_radius_px
    if dist <= approach_r:
        return None  # already at grasp distance
    # Unit vector from target toward robot; scale to approach_r
    ux, uy = dx / dist, dy / dist
    apx = int(obs_map.target_px[0] + ux * approach_r)
    apy = int(obs_map.target_px[1] + uy * approach_r)
    return _px_to_grid((apx, apy), obs_map.w, obs_map.h), (apx, apy)


def plan_path(obs_map: ObstacleMap) -> NavPlan:
    """Run A* from the robot's grid cell to the nearest C-space-free approach point.

    The ideal approach goal is the point at (robot_radius + target_radius) from the
    target along the robot→target line.  That point often falls inside the Minkowski-sum
    inflation buffer (yellow zone).  We snap it to the nearest navigable (green) cell
    so A* always operates entirely within the collision-free C-space, never cutting
    through the inflation buffer.
    """
    if obs_map.target_grid is None:
        return NavPlan([], [], reachable=False, reason="No target detected")

    approach = _approach_goal(obs_map)
    if approach is None:
        return NavPlan([obs_map.robot_grid], [obs_map.robot_px],
                       reachable=True, reason="At grasp distance")

    ideal_goal, goal_px = approach
    start = obs_map.robot_grid

    # Always guarantee the robot's own cell is navigable.
    g = obs_map.grid.copy()
    g[start[0], start[1]] = True

    # Snap the approach goal to the nearest green C-space cell so the path
    # never enters the Minkowski-sum inflation buffer (yellow zone).
    goal = _nearest_free_cell(g, ideal_goal)
    if goal is None:
        return NavPlan([], [], reachable=False, reason="No free cell near approach")
    if goal != ideal_goal:
        goal_px = _grid_to_px(goal, obs_map.w, obs_map.h)
        log.debug("Approach goal snapped from %s to nearest free cell %s", ideal_goal, goal)

    path_grid = _astar(g, start, goal)

    if path_grid is None:
        # Obstacle inflation may have sealed thin corridors — retry on the
        # pre-inflation grid at reduced clearance.
        g_raw = obs_map.raw_grid.copy()
        g_raw[start[0], start[1]] = True
        goal_raw = _nearest_free_cell(g_raw, ideal_goal)
        if goal_raw is not None:
            path_grid = _astar(g_raw, start, goal_raw)
            if path_grid is not None:
                goal = goal_raw
                goal_px = _grid_to_px(goal_raw, obs_map.w, obs_map.h)
                log.info("navigate_to: using raw-grid path (reduced clearance)")
                path_grid = _simplify_path(path_grid, g_raw)
        if path_grid is None:
            log.warning("navigate_to: no path from %s to approach %s", start, goal)
            return NavPlan([], [], reachable=False,
                           reason=f"No path from grid{start}→approach{goal}")
    else:
        path_grid = _simplify_path(path_grid, g)

    path_px = [_grid_to_px(rc, obs_map.w, obs_map.h) for rc in path_grid]
    # Replace the last waypoint with the exact sub-cell approach pixel.
    if path_px:
        path_px[-1] = goal_px
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

    obstacle_layer = out.copy()
    obstacle_layer[obs_map.free_mask == 0] = (0, 0, 160)
    cv2.addWeighted(obstacle_layer, 0.40, out, 0.60, 0, out)

    if plan.reachable and len(plan.path_px) > 1:
        pts = np.array(plan.path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, _PATH_COLOR, 2, cv2.LINE_AA)
        for pt in plan.path_px[::6]:
            cv2.circle(out, pt, 3, _PATH_COLOR, -1)

    rx, ry = obs_map.robot_px
    cv2.circle(out, (rx, ry), 10, _ROBOT_COLOR, 2)
    cv2.putText(out, "R", (rx - 5, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ROBOT_COLOR, 1, cv2.LINE_AA)

    if obs_map.target_px is not None:
        tx, ty = obs_map.target_px
        cv2.circle(out, (tx, ty), 10, _TARGET_COLOR, 2)
        cv2.putText(out, "T", (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TARGET_COLOR, 1, cv2.LINE_AA)
        # Grasp approach circle: robot should stop when its centre reaches this ring.
        approach_r = int(obs_map.robot_radius_px + obs_map.target_radius_px)
        cv2.circle(out, (tx, ty), approach_r, _TARGET_COLOR, 1, cv2.LINE_AA)

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

    if nav_heading is not None:
        fw = nav_heading.forward
        cross = fw[0] * dy - fw[1] * dx
        dot   = fw[0] * dx + fw[1] * dy
        turn_deg = math.degrees(math.atan2(cross, dot))
    else:
        turn_deg = 0.0

    drive_s = _DRIVE_STEP_S
    if obs_map.target_px is not None:
        tdist = math.hypot(robot_px[0] - obs_map.target_px[0],
                           robot_px[1] - obs_map.target_px[1])
        if tdist < math.hypot(obs_map.w, obs_map.h) * 0.18:
            drive_s = min(drive_s, 0.8)

    return turn_deg, drive_s


def at_target(obs_map: ObstacleMap) -> bool:
    """True when robot is within grasp distance of the target.

    Grasp distance = robot_radius + target_radius, i.e. the two objects are
    just touching — the robot is ready to close the gripper.
    """
    if obs_map.target_px is None:
        return False
    dist = math.hypot(
        obs_map.robot_px[0] - obs_map.target_px[0],
        obs_map.robot_px[1] - obs_map.target_px[1],
    )
    return dist <= obs_map.robot_radius_px + obs_map.target_radius_px


def detect_robot_px(bgr: np.ndarray) -> tuple[int, int] | None:
    """Return the robot's yellow-body centroid as (x, y) pixel coords, or None.

    Cheap enough to call on every DroidCam frame during motor execution.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)
    ys, xs = np.where(yellow > 0)
    if len(xs) < 200:
        return None
    return (int(xs.mean()), int(ys.mean()))


def draw_tracking_overlay(
    base_overlay: np.ndarray,
    trail: list[tuple[int, int]],
) -> np.ndarray:
    """Draw live robot positions (trail) onto a pre-planned nav overlay image.

    Older positions are rendered dimmer; the newest is a bright cyan dot + ring.
    base_overlay must already contain the A* path and obstacle tint.
    """
    out = base_overlay.copy()
    n = len(trail)
    for i, px in enumerate(trail):
        alpha = 0.3 + 0.7 * (i + 1) / max(n, 1)
        c = int(alpha * 255)
        cv2.circle(out, px, 4, (0, c, c), -1)
    if trail:
        rx, ry = trail[-1]
        cv2.circle(out, (rx, ry), 12, (0, 255, 255), 2)
        cv2.circle(out, (rx, ry), 5,  (0, 255, 255), -1)
    return out


def build_cspace_bgr(
    bgr: np.ndarray,
    obs_map: ObstacleMap,
    plan: NavPlan,
) -> np.ndarray:
    """Return a BGR image of the C-space planning grid with path overlay."""
    h_img, w_img = bgr.shape[:2]
    raw_up  = cv2.resize(obs_map.raw_grid.astype(np.uint8) * 255,
                         (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    grid_up = cv2.resize(obs_map.grid.astype(np.uint8) * 255,
                         (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    cspace = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    cspace[grid_up  > 0]                     = (0, 160,   0)
    cspace[(raw_up > 0) & (grid_up == 0)]    = (0, 200, 220)
    cspace[raw_up == 0]                      = (0,   0, 180)
    rx, ry = obs_map.robot_px
    cv2.circle(cspace, (rx, ry), int(obs_map.robot_radius_px), _ROBOT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(cspace, (rx, ry), 4, _ROBOT_COLOR, -1)
    cv2.putText(cspace, "R", (rx + int(obs_map.robot_radius_px) + 4, ry + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ROBOT_COLOR, 1, cv2.LINE_AA)
    if obs_map.target_px is not None:
        tx, ty = obs_map.target_px
        cv2.circle(cspace, (tx, ty), int(obs_map.target_radius_px), _TARGET_COLOR, 2, cv2.LINE_AA)
        cv2.circle(cspace, (tx, ty), 4, _TARGET_COLOR, -1)
        cv2.putText(cspace, "T", (tx + int(obs_map.target_radius_px) + 4, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TARGET_COLOR, 1, cv2.LINE_AA)
        approach_r = int(obs_map.robot_radius_px + obs_map.target_radius_px)
        cv2.circle(cspace, (tx, ty), approach_r, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(cspace, "approach", (tx + approach_r + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1, cv2.LINE_AA)
    if plan.reachable and len(plan.path_px) > 1:
        pts = np.array(plan.path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(cspace, [pts], False, _PATH_COLOR, 2, cv2.LINE_AA)
        cv2.circle(cspace, plan.path_px[-1], 5, _PATH_COLOR, -1)
    return cspace


def save_debug_images(
    bgr: np.ndarray,
    obs_map: ObstacleMap,
    plan: NavPlan,
    outdir: str,
    step: int,
) -> dict[str, str]:
    """Write pipeline debug images to outdir, prefixed a–f for alphabetical sort order.

    a_raw              — unmodified input frame
    a2_cleaned         — inpainted frame (robot + target removed) used for depth estimation
    b_depth            — Depth Anything V2 heat-map (INFERNO colormap)
    c_depth_gradient   — floor classification from depth gradients alone (no robot body)
    d_free_mask        — navigable floor after OR-ing in the robot body footprint
    e_cspace           — C-space grid: green=navigable, yellow=inflation buffer, red=obstacle
    f_nav_overlay      — final overlay: obstacle tint + A* path + R/T markers
    """
    os.makedirs(outdir, exist_ok=True)
    saved: dict[str, str] = {}
    p = os.path.join  # shorthand

    # a — raw input
    raw_path = p(outdir, f"step_{step:02d}_a_raw.jpg")
    cv2.imwrite(raw_path, bgr)
    saved["raw"] = raw_path

    # a2 — inpainted frame used for depth estimation (robot + target removed)
    if obs_map.cleaned_bgr is not None:
        cleaned_path = p(outdir, f"step_{step:02d}_a2_cleaned.jpg")
        cv2.imwrite(cleaned_path, obs_map.cleaned_bgr)
        saved["cleaned"] = cleaned_path

    if obs_map.depth_map is not None:
        d = obs_map.depth_map

        # b — depth heat-map
        d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        depth_path = p(outdir, f"step_{step:02d}_b_depth.jpg")
        cv2.imwrite(depth_path, depth_vis)
        saved["depth"] = depth_path

        # c — depth-gradient floor mask (no robot body carve-out)
        hsv_dbg = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow_dbg = cv2.inRange(hsv_dbg, YELLOW_HSV_LO, YELLOW_HSV_HI)
        ring_dbg = _robot_ring(yellow_dbg)
        grad_mask = _depth_gradient_floor_mask(d, ring_dbg) if ring_dbg is not None else None
        if grad_mask is not None:
            grad_vis = np.zeros_like(bgr)
            grad_vis[grad_mask >  0] = (0, 160, 0)
            grad_vis[grad_mask == 0] = (0, 0, 180)
            grad_path = p(outdir, f"step_{step:02d}_c_depth_gradient.jpg")
            cv2.imwrite(grad_path, grad_vis)
            saved["gradient_mask"] = grad_path

    # d — free mask (gradient floor + robot footprint combined)
    mask_vis = np.zeros_like(bgr)
    mask_vis[obs_map.free_mask >  0] = (0, 160, 0)
    mask_vis[obs_map.free_mask == 0] = (0, 0, 180)
    mask_path = p(outdir, f"step_{step:02d}_d_free_mask.jpg")
    cv2.imwrite(mask_path, mask_vis)
    saved["obstacle_mask"] = mask_path

    # e — C-space planning grid
    cspace = build_cspace_bgr(bgr, obs_map, plan)
    cspace_path = p(outdir, f"step_{step:02d}_e_cspace.jpg")
    cv2.imwrite(cspace_path, cspace)
    saved["cspace"] = cspace_path

    # f — nav overlay
    overlay_bgr = draw_nav_overlay(bgr, obs_map, plan, step)
    overlay_path = p(outdir, f"step_{step:02d}_f_nav_overlay.jpg")
    cv2.imwrite(overlay_path, overlay_bgr)
    saved["nav_overlay"] = overlay_path

    log.info("Navigation debug images (step %d) saved to: %s", step, outdir)
    return saved
