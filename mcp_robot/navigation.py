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

# Obstacle inflation in grid cells (robot half-width clearance).
_ROBOT_RADIUS_CELLS = 2

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
    depth_map: np.ndarray | None = field(default=None, repr=False)  # raw depth, for debug


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

    # ── 1. Yellow body mask and robot ring ────────────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)
    yellow_dilated = cv2.dilate(yellow_raw, np.ones((17, 17), np.uint8))
    ring = _robot_ring(yellow_raw)

    # ── 2. Depth gradient floor mask ─────────────────────────────────────
    depth_map: np.ndarray | None = None
    grad_free: np.ndarray | None = None
    if use_depth:
        depth_map = estimate_depth(bgr)
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

    # ── 5. Inflate obstacles by robot clearance ───────────────────────────
    r = _ROBOT_RADIUS_CELLS
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inflated = cv2.erode(base_grid, struct)
    grid = (inflated > 0)

    # ── 7. Locate robot and target; ensure their cells are navigable ──────
    robot_px: tuple[int, int] = nav_heading.body_center if nav_heading is not None else (w // 2, h // 2)
    target_px: tuple[int, int] | None = target.center if target is not None else None

    robot_grid = _px_to_grid(robot_px, w, h)
    target_grid = _px_to_grid(target_px, w, h) if target_px is not None else None

    grid[robot_grid[0], robot_grid[1]] = True
    if target_grid is not None:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                gr2 = max(0, min(_GRID_ROWS - 1, target_grid[0] + dr))
                gc2 = max(0, min(_GRID_COLS - 1, target_grid[1] + dc))
                grid[gr2, gc2] = True

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
        depth_map=depth_map,
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
        # Thin corridors (e.g. floor in shadow under cabinet) can be erased by
        # obstacle inflation. Retry on the raw grid which preserves them at the
        # cost of less clearance. Apply the same forced-navigable patches so
        # robot and target cells are reachable.
        fallback = obs_map.raw_grid.copy()
        fallback[start[0], start[1]] = True
        if obs_map.target_grid is not None:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    gr2 = max(0, min(_GRID_ROWS - 1, goal[0] + dr))
                    gc2 = max(0, min(_GRID_COLS - 1, goal[1] + dc))
                    fallback[gr2, gc2] = True
        path_grid = _astar(fallback, start, goal)
        if path_grid is not None:
            log.info("navigate_to: inflated path blocked; using raw-grid path (reduced clearance)")
        else:
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
    """Write raw frame, free-space mask, depth map, and nav overlay to outdir."""
    os.makedirs(outdir, exist_ok=True)
    saved: dict[str, str] = {}

    raw_path = os.path.join(outdir, f"step_{step:02d}_raw.jpg")
    cv2.imwrite(raw_path, bgr)
    saved["raw"] = raw_path

    mask_vis = np.zeros_like(bgr)
    mask_vis[obs_map.free_mask >  0] = (0, 160, 0)
    mask_vis[obs_map.free_mask == 0] = (0, 0, 180)
    mask_path = os.path.join(outdir, f"step_{step:02d}_obstacle_mask.jpg")
    cv2.imwrite(mask_path, mask_vis)
    saved["obstacle_mask"] = mask_path

    if obs_map.depth_map is not None:
        d = obs_map.depth_map
        d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        depth_path = os.path.join(outdir, f"step_{step:02d}_depth.jpg")
        cv2.imwrite(depth_path, depth_vis)
        saved["depth"] = depth_path

        # Gradient mask: green=floor (gradient matches robot ring), red=obstacle.
        depth_smooth = cv2.GaussianBlur(d, (9, 9), 0)
        gx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=5)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow_raw = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)
        ring = _robot_ring(yellow_raw)
        grad_mask = _depth_gradient_floor_mask(d, ring) if ring is not None else None
        if grad_mask is not None:
            grad_vis = np.zeros_like(bgr)
            grad_vis[grad_mask >  0] = (0, 160, 0)
            grad_vis[grad_mask == 0] = (0, 0, 180)
            grad_path = os.path.join(outdir, f"step_{step:02d}_gradient_mask.jpg")
            cv2.imwrite(grad_path, grad_vis)
            saved["gradient_mask"] = grad_path

    overlay_bgr = draw_nav_overlay(bgr, obs_map, plan, step)
    overlay_path = os.path.join(outdir, f"step_{step:02d}_nav_overlay.jpg")
    cv2.imwrite(overlay_path, overlay_bgr)
    saved["nav_overlay"] = overlay_path

    log.info("Navigation debug images (step %d) saved to: %s", step, outdir)
    return saved
