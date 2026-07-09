"""
CV-based navigation: obstacle detection, A* path planning, overlay drawing,
and per-step command synthesis.

Public API:
    estimate_depth(bgr) -> np.ndarray          (float32 depth map, larger = nearer)
    detect_obstacles(bgr, heading, target) -> ObstacleMap
    plan_path(obs_map, heading=None) -> NavPlan
    draw_nav_overlay(bgr, obs_map, plan, step) -> np.ndarray
    commands_for_step(obs_map, plan, heading) -> tuple[float, float, bool]
    near_target(obs_map) -> bool
    save_debug_images(bgr, obs_map, plan, outdir, step, suffix="") -> dict[str, str]
    mm_per_px(body_area_px) -> float | None    (shared px->mm distance calibration)
    mm_to_wheel_degrees(distance_mm) -> float  (shared mm->encoder-degrees conversion)
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

from mcp_robot import config
from mcp_robot.heading import (
    Heading,
    YELLOW_HSV_LO,
    YELLOW_HSV_HI,
    body_hull,
)
from mcp_robot.grasp_readiness import DetectedObject

log = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

_GRID_COLS = 80
_GRID_ROWS = 60

# Mahalanobis-distance threshold for depth-gradient floor classification.
# Pixels within this many IQR-derived sigmas of the robot-ring floor gradient
# are classified as navigable floor.
_DEPTH_GRAD_N_SIGMA = 4

# Minimum obstacle inflation in grid cells — used when the robot body cannot
# be detected.  Normally the inflation radius is derived from the yellow body
# bounding box so the path stays at least one robot-width from obstacles.
_MIN_ROBOT_RADIUS_CELLS = 3

# Safety margin applied to the robot-disk radius before Minkowski-sum erosion.
# 1.0 = exact robot radius; 1.2 = 20% extra clearance on all sides.
_CSPACE_BUFFER_SCALE = 4

# Grace factor on near_target's distance threshold. The A* approach goal is
# snapped to the boundary of the C-space inflation buffer (robot_radius *
# _CSPACE_BUFFER_SCALE from the target), so the path's last waypoint sits
# right at that boundary — drive/grid imprecision can leave the robot a
# little outside it even though it effectively arrived. 1.5 = 50% slack.
_NEAR_TARGET_GRACE = 1.5

# How far ahead of the robot (in robot radii) to check the raw grid for an
# obstacle before deciding an in-place turn is unsafe. The chassis/gripper
# extends roughly this far forward of the body centroid (see
# _robot_footprint_mask's forward multiplier), so this is the zone a pivot
# turn would sweep through first.
_AHEAD_CLEARANCE_SCALE = 4.0

# Minimum |turn_deg| to the next waypoint for it to count as "behind" the
# robot. When the waypoint is this close to 180° away AND something blocks
# the raw grid directly ahead, commands_for_step reverses straight toward it
# instead of pivoting in place — pivoting would swing the chassis/gripper
# into the obstacle ahead before completing the turn.
_REVERSE_TURN_THRESHOLD_DEG = 150.0

# Cap on how far to drive in a single navigation step, expressed in body
# lengths. Keeps the closed loop responsive — re-checking heading and position
# often — even when the next waypoint is many step-lengths away. Shorter
# waypoint hops are driven their exact distance (no capping needed).
_MAX_STEP_BODY_LENGTHS = 3.0
_MAX_STEP_MM = _MAX_STEP_BODY_LENGTHS * config.ROBOT_BODY_LENGTH_MM

# Blind-driving distance used only when the yellow body wasn't detected, i.e.
# no pixel->mm calibration is possible (see mm_per_px).
_FALLBACK_STEP_MM = 100.0

_WHEEL_CIRCUMFERENCE_MM = math.pi * config.WHEEL_DIAMETER_MM


def mm_to_wheel_degrees(distance_mm: float) -> float:
    """Straight-line travel distance (mm) -> wheel-encoder degrees, for
    passing directly to robot.drive_degrees()."""
    return (distance_mm / _WHEEL_CIRCUMFERENCE_MM) * 360.0


def mm_per_px(body_area_px: int) -> float | None:
    """Pixel -> millimetre scale derived from the robot's visible yellow-body
    area.  Area is rotation-invariant — unlike an axis-aligned bounding-box
    measurement such as robot_radius_px, which grows and shrinks as the robot
    turns — so this calibration stays valid no matter which way the robot is
    currently facing.  Returns None when the body wasn't detected.

    Shared distance-estimation primitive: used by commands_for_step() below
    for navigate_to's per-step drive distance, and by server.py's click_button
    to convert the measured distance to a switch into press degrees — both
    go through this same calibration so "how far to drive" is computed one way
    across the codebase.
    """
    if body_area_px <= 0:
        return None
    return math.sqrt(config.ROBOT_BODY_AREA_MM2 / body_area_px)


# ── Floor-plane homography (perspective correction) ───────────────────────────
# Re-derived every frame from the robot's own 152x88mm yellow top-plate — no
# external calibration markers needed. The plate's 4 visible corners (from its
# convex hull) are mapped to their known mm offsets from the body center,
# giving a full perspective transform that corrects pixel-space angles and
# distances for the external camera's tilt.

# Robot's visible yellow top-plate, half-extents in mm.
_PLATE_HALF_LEN_MM = config.ROBOT_BODY_LENGTH_MM / 2.0   # along `forward`
_PLATE_HALF_WID_MM = config.ROBOT_BODY_WIDTH_MM / 2.0    # perpendicular to `forward`

# approxPolyDP epsilon (fraction of hull perimeter) for reducing the plate's
# convex hull down to its 4 corners.
_HOMOGRAPHY_CORNER_EPS_FRAC = 0.02

# Reject a fitted homography if any image-frame corner projects beyond this
# many mm from the robot. A bad corner correspondence (e.g. one corner
# misidentified) produces a near-degenerate transform that extrapolates the
# image edges to absurd distances (tens of metres); a good fit keeps them
# within a few metres for a tabletop/floor-level view.
_HOMOGRAPHY_MAX_FLOOR_MM = 4000.0


def _project_to_floor_mm(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Apply floor homography `H` to one pixel coordinate -> (x_mm, y_mm)."""
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def _floor_homography(bgr: np.ndarray | None, nav_heading: Heading) -> np.ndarray | None:
    """Fit a pixel -> floor-plane-mm homography from the robot's own body.

    The robot's 152x88mm yellow top-plate is visible in every frame, so its 4
    corners double as floor-plane calibration points with a known real-world
    size — no external markers needed. Corners are extracted from the body's
    convex hull via approxPolyDP, classified into the plate's 4 quadrants using
    `nav_heading` (along/perpendicular to `forward`), and paired with their
    known mm offsets from the body center. `cv2.getPerspectiveTransform` then
    fits the transform.

    The resulting (x_mm, y_mm) frame has +x along `forward` and +y perpendicular
    (rotated from forward in the same sense as cross(forward, v)) — i.e. the
    same orientation as the pixel-space (dot, cross) decomposition used
    elsewhere, so atan2(y_mm, x_mm) preserves the "positive = CW from above"
    turn-angle convention while correcting for perspective.

    Returns None if the plate's hull doesn't reduce to a clean quadrilateral
    spanning all 4 quadrants, or if the fit projects any image corner beyond
    _HOMOGRAPHY_MAX_FLOOR_MM (a degenerate/extrapolating fit).
    """
    if bgr is None:
        return None
    body = body_hull(bgr)
    if body is None:
        return None
    peri = cv2.arcLength(body, True)
    approx = cv2.approxPolyDP(body, _HOMOGRAPHY_CORNER_EPS_FRAC * peri, True).reshape(-1, 2)
    if len(approx) != 4:
        return None

    bx, by = nav_heading.body_center
    fx, fy = nav_heading.forward
    px_axis, py_axis = -fy, fx  # perpendicular axis (90° from forward, cross-product sense)

    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    quadrants = set()
    for x, y in approx:
        dx, dy = float(x) - bx, float(y) - by
        along = dx * fx + dy * fy
        perp = dx * px_axis + dy * py_axis
        quadrants.add((along >= 0, perp >= 0))
        mm_x = _PLATE_HALF_LEN_MM if along >= 0 else -_PLATE_HALF_LEN_MM
        mm_y = _PLATE_HALF_WID_MM if perp >= 0 else -_PLATE_HALF_WID_MM
        src_pts.append((float(x), float(y)))
        dst_pts.append((mm_x, mm_y))

    if len(quadrants) != 4:
        return None

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    h, w = bgr.shape[:2]
    for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
        fmx, fmy = _project_to_floor_mm(H, cx, cy)
        if math.hypot(fmx, fmy) > _HOMOGRAPHY_MAX_FLOOR_MM:
            return None

    return H


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

    Returns a float32 depth map, scaled to [0, 255], with the same (H, W) as
    the input, where larger values indicate a nearer point (this model
    predicts inverse depth / disparity, not metric distance).

    Starts from the pipeline's "predicted_depth" tensor (the model's raw
    float output), NOT "depth" (a uint8 PIL image the pipeline derives from
    it by normalising to 0-1 and quantising to 256 levels purely for
    human-viewable heat-maps). On a floor plane spanning much of the frame,
    that 256-level quantisation turns the smooth depth gradient into a
    staircase — long flat "plateaus" (zero local gradient) separated by
    sharp one-step "cliffs" (huge local gradient) — and both deviate from
    the true smooth floor gradient enough to trip
    _depth_gradient_floor_mask's n_sigma gate, fooling it into flagging
    plain floor as an obstacle in periodic bands.

    Still rescaled to [0, 255] (min-max, like the discarded "depth" image),
    NOT left in predicted_depth's native units: _depth_gradient_floor_mask's
    minimum-sigma floor (see the `max(..., 1.0)` calls there) is an absolute
    constant tuned against that 0-255 range. predicted_depth's native range
    is roughly 25x smaller (~0-10 for this model/checkpoint) and not fixed
    across scenes, so leaving it unscaled shrinks real gradients well below
    that floor — the floor then dominates every z-score, and vertical
    surfaces (walls, cabinet edges) that should read as obstacles instead
    fall "within n_sigma" of the floor-gradient reference and get
    misclassified as navigable. Rescaling keeps every tunable that assumes
    this range (the sigma floor, _DEPTH_GRAD_N_SIGMA, the blur/Sobel kernel
    sizes) correctly calibrated while still being continuous float32, not
    quantised to 256 discrete integer levels.
    """
    from PIL import Image as _PIL
    pipe = _load_depth_model()
    rgb = _PIL.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = pipe(rgb)
    depth = result["predicted_depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin) * 255.0
    return depth


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
    buffer_radius_px: float          # Minkowski-sum buffer radius (robot_radius_px * _CSPACE_BUFFER_SCALE)
    target_radius_px: float          # half target bounding box (pixels); 0 if no target
    depth_map: np.ndarray | None = field(default=None, repr=False)   # raw depth, for debug
    cleaned_bgr: np.ndarray | None = field(default=None, repr=False)  # inpainted frame used for depth
    bgr: np.ndarray | None = field(default=None, repr=False)  # original frame, for floor homography


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
    """Rotated-rectangle mask covering the yellow chassis body only.

    Multipliers are kept tight so the rectangle does NOT extend into vertical
    obstacles (switch, wall) that may be directly in front of the gripper.
    The arm/gripper chain is handled separately by _arm_gripper_mask, which
    catches dark blobs near this rectangle while being blind to white/grey
    obstacles.

      forward  ×1.5 — just past the chassis front edge; arm chain caught by proximity
      backward ×1.35 — small rear overhang (wheels, battery cable)
      sides    ×1.35 — wheels sit slightly outside the board on both sides

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

    front  = float(fwd_proj.max())                               * 1.50
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
    floor_sample: np.ndarray,
    n_sigma: float = _DEPTH_GRAD_N_SIGMA,
) -> np.ndarray | None:
    """Floor mask from depth-gradient direction similarity to a floor reference.

    The floor is a planar surface with a characteristic depth gradient (surface
    orientation).  Any pixel whose Sobel gradient vector falls within n_sigma
    IQR-derived standard deviations of the floor-gradient sample is classified
    as navigable.

    *floor_sample* is a mask of pixels guaranteed to be floor — typically the
    robot body footprint on the inpainted depth map (compact, consistent depth)
    or, as a fallback, the ring just outside the yellow body.

    Key advantage over depth-value thresholding: shadow areas have the same
    surface orientation as the lit floor (same physical plane), so they produce
    the same gradient even though their absolute depth values are wrong.  Depth
    models reliably get the gradient right even when absolute depth is off.

    Returns None when there are not enough samples.
    """
    if floor_sample is None or (floor_sample > 0).sum() < 50:
        return None

    # Smooth depth before gradient to suppress wood-plank texture noise.
    # k=31 eliminates local texture while preserving the macro surface
    # orientation that separates floor from vertical surfaces like cabinets.
    depth_smooth = cv2.GaussianBlur(depth, (31, 31), 0)
    gx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=5)

    # Sample floor gradient from the reference region (guaranteed floor pixels).
    ref_gx = gx[floor_sample > 0].astype(np.float64)
    ref_gy = gy[floor_sample > 0].astype(np.float64)

    floor_gx = float(np.median(ref_gx))
    floor_gy = float(np.median(ref_gy))

    # Robust σ from IQR (÷1.35 converts IQR → σ for a normal distribution).
    q1x, q3x = np.percentile(ref_gx, [25, 75])
    q1y, q3y = np.percentile(ref_gy, [25, 75])
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


def _obstacle_ahead(obs_map: ObstacleMap, nav_heading: Heading | None) -> bool:
    """True if the raw (uninflated) grid is blocked within
    _AHEAD_CLEARANCE_SCALE robot radii directly ahead of the robot, along its
    current heading.

    Used to decide whether an in-place turn is safe: if the chassis/gripper
    is already nose-to-obstacle, pivoting toward any other waypoint would
    swing the front of the robot into that obstacle before completing the
    turn — reversing straight back first is safer.
    """
    if nav_heading is None:
        return False
    fw = nav_heading.forward
    reach_px = obs_map.robot_radius_px * _AHEAD_CLEARANCE_SCALE
    ahead_px = (
        obs_map.robot_px[0] + fw[0] * reach_px,
        obs_map.robot_px[1] + fw[1] * reach_px,
    )
    ahead_grid = _px_to_grid(ahead_px, obs_map.w, obs_map.h)
    cells = list(_bresenham_cells(obs_map.robot_grid[0], obs_map.robot_grid[1],
                                   ahead_grid[0], ahead_grid[1]))
    ray_cells = cells[1:]
    blocked = any(not obs_map.raw_grid[r, c] for r, c in ray_cells)
    log.info(
        "[_obstacle_ahead] robot_grid=%s -> ahead_grid=%s (reach=%.0fpx along fw=(%.2f,%.2f))  "
        "raw_grid along ray=%s -> blocked=%s",
        obs_map.robot_grid, ahead_grid, reach_px, fw[0], fw[1],
        [int(obs_map.raw_grid[r, c]) for r, c in ray_cells], blocked,
    )
    return blocked


def _reverse_exit_cell(
    obs_map: ObstacleMap,
    nav_heading: Heading | None,
) -> tuple[int, int] | None:
    """Farthest cell directly behind the robot (along -forward) that is both
    raw-navigable (safe for the chassis to back through) and C-space-free (a
    legitimate place for the robot's center to stop).

    Walks backward up to robot_radius * _CSPACE_BUFFER_SCALE pixels — the
    same radius used to inflate obstacles into the C-space buffer — so the
    search never proposes a point the robot's center couldn't actually reach.
    Stops at the first raw-grid obstacle behind the robot, since the chassis
    cannot back through it. Returns None if reversing wouldn't clear the
    buffer (no C-space-free cell found on the way).
    """
    if nav_heading is None:
        return None
    fw = nav_heading.forward
    reach_px = obs_map.robot_radius_px * _CSPACE_BUFFER_SCALE
    back_px = (
        obs_map.robot_px[0] - fw[0] * reach_px,
        obs_map.robot_px[1] - fw[1] * reach_px,
    )
    back_grid = _px_to_grid(back_px, obs_map.w, obs_map.h)
    best: tuple[int, int] | None = None
    for r, c in _bresenham_cells(obs_map.robot_grid[0], obs_map.robot_grid[1],
                                  back_grid[0], back_grid[1]):
        if (r, c) == obs_map.robot_grid:
            continue
        if not obs_map.raw_grid[r, c]:
            break  # chassis can't back through an obstacle
        if obs_map.grid[r, c]:
            best = (r, c)
    return best


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

    ring = _robot_ring(yellow_raw)

    # ── 2. Remove robot and target before depth estimation ───────────────
    # Local import avoids circular dependency (inpainting imports _robot_footprint_mask
    # from this module).  The cleaned frame gives the depth model a clear view of the
    # floor where the robot and target stood, so their presence no longer creates false
    # obstacle regions in the depth gradient mask.
    from mcp_robot.inpainting import remove_robot_and_target as _inpaint
    bgr_for_depth, _ = _inpaint(bgr, nav_heading, target)
    log.debug("Robot/target inpainting for depth estimation succeeded")

    # ── 3. Depth gradient floor mask ─────────────────────────────────────
    # Floor reference: the robot body footprint on the inpainted depth map.
    # The inpainted region is guaranteed floor with a compact y-range, giving
    # tight σ values.  Erode to stay away from inpainting edge artifacts.
    # Fall back to the ring if the yellow body is too small.
    robot_floor_ref = cv2.erode(yellow_raw, np.ones((15, 15), np.uint8))
    if (robot_floor_ref > 0).sum() < 50:
        robot_floor_ref = ring  # fallback

    depth_map: np.ndarray | None = None
    grad_free: np.ndarray | None = None
    if use_depth:
        depth_map = estimate_depth(bgr_for_depth)
        if robot_floor_ref is not None:
            grad_free = _depth_gradient_floor_mask(depth_map, robot_floor_ref)
            if grad_free is not None:
                log.debug("Depth gradient floor mask: %.1f%% navigable",
                          (grad_free > 0).mean() * 100)

    if grad_free is None:
        raise RuntimeError(
            "Depth gradient floor mask unavailable — "
            "either depth estimation failed or robot body not visible."
        )

    # ── 4. Free mask: gradient floor + yellow chassis ────────────────────
    # OR in only the yellow chassis pixels (small dilation) — NOT the full
    # forward-extended footprint.  The footprint's 4× forward reach covers
    # vertical obstacles (switch, wall) directly in front of the gripper,
    # which must stay classified as obstacles.  The depth model already sees
    # floor under the robot because we inpainted it out; the chassis OR is
    # only a fallback for any residual depth-model misclassification of the
    # chassis itself.  A* start-cell navigability is guaranteed separately by
    # the belt-and-suspenders force at step 7.
    yellow_chassis = cv2.dilate(yellow_raw, np.ones((35, 35), np.uint8))
    free_mask = cv2.bitwise_or(grad_free, yellow_chassis)
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    # ── 5. Build coarse grid (majority vote per cell) ─────────────────────
    cell_h = h / _GRID_ROWS
    cell_w = w / _GRID_COLS
    base_grid = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.uint8)
    for gr in range(_GRID_ROWS):
        y0, y1 = int(gr * cell_h), max(int(gr * cell_h) + 1, int((gr + 1) * cell_h))
        for gc in range(_GRID_COLS):
            x0, x1 = int(gc * cell_w), max(int(gc * cell_w) + 1, int((gc + 1) * cell_w))
            base_grid[gr, gc] = 255 if free_mask[y0:y1, x0:x1].mean() > 100 else 0

    # ── 6. C-space: inflate obstacles by the robot disk radius ───────────
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

    target_px: tuple[int, int] | None = target.nav_point if target is not None else None
    if target is not None:
        target_radius_px = float(max(target.x2 - target.x1,
                                     target.y2 - target.y1)) / 2.0
    else:
        target_radius_px = 0.0

    robot_px: tuple[int, int] = nav_heading.body_center if nav_heading is not None else (w // 2, h // 2)
    robot_grid = _px_to_grid(robot_px, w, h)

    struct = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * rx_cells + 1, 2 * ry_cells + 1),
    )
    inflated = cv2.erode(base_grid, struct)
    grid = (inflated > 0)

    # ── 7. Finalise grid ─────────────────────────────────────────────────
    target_grid = _px_to_grid(target_px, w, h) if target_px is not None else None

    # The robot's own cell is left as-is (not forced navigable) so plan_path
    # can detect whether the robot's current position falls inside the
    # Minkowski-sum buffer; plan_path makes its own forced-navigable copy for
    # A*. The target is NOT forced either — its cells survive or not based on
    # surrounding floor, so the approach goal naturally snaps to connected
    # green cells.

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
        buffer_radius_px=robot_radius_px * _CSPACE_BUFFER_SCALE,
        target_radius_px=target_radius_px,
        depth_map=depth_map,
        cleaned_bgr=bgr_for_depth,
        bgr=bgr,
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


def plan_path(obs_map: ObstacleMap, nav_heading: Heading | None = None) -> NavPlan:
    """Run A* from the robot's grid cell to the nearest C-space-free approach point.

    The ideal approach goal is the point at (robot_radius + target_radius) from the
    target along the robot→target line.  That point often falls inside the Minkowski-sum
    inflation buffer (yellow zone).  We snap it to the nearest navigable (green) cell
    so A* always operates entirely within the collision-free C-space, never cutting
    through the inflation buffer.

    nav_heading, if provided, is used only by the buffer-exit branch below to
    check whether the robot should back straight away from an obstacle ahead
    before A* takes over (see _obstacle_ahead / _reverse_exit_cell).
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

    # ── Buffer-zone exit ──────────────────────────────────────────────────────
    # If the robot's current cell falls inside the Minkowski-sum inflation
    # buffer (yellow zone) of this C-space snapshot, never plan through the
    # buffer to the target. Instead, find the nearest green (C-space free)
    # cell and plan a two-phase path: phase 1 exits the buffer on the raw grid
    # (buffer cells are traversable); phase 2 then runs A* entirely within the
    # green C-space from the exit cell to the goal.
    # Checked fresh against obs_map.grid on every call (not a cached flag), so
    # a robot that drifts into the buffer mid-navigation is caught on its next
    # replan too, not just the initial plan.
    if not obs_map.grid[start[0], start[1]]:
        # ── Reverse from obstacle ahead ────────────────────────────────────────
        # If the buffer-triggering obstacle is directly ahead of the robot, an
        # in-place turn toward any other waypoint would swing the chassis/gripper
        # into it before completing the turn. Back straight away first: walk the
        # raw grid backward from the robot's cell to the farthest C-space-free
        # cell within the buffer radius, and route through that before A*.
        obstacle_ahead = _obstacle_ahead(obs_map, nav_heading)
        if obstacle_ahead:
            reverse_cell = _reverse_exit_cell(obs_map, nav_heading)
            if reverse_cell is not None:
                log.info("navigate_to: obstacle ahead — reversing to %s before planning",
                         reverse_cell)
                path1 = [start, reverse_cell]
                path2 = _astar(g, reverse_cell, goal)
                if path2 is not None:
                    path_grid = path1[:-1] + _simplify_path(path2, g)
                else:
                    path_grid = path1
                path_px = [_grid_to_px(rc, obs_map.w, obs_map.h) for rc in path_grid]
                if path_px:
                    path_px[-1] = goal_px
                return NavPlan(
                    path_grid=path_grid,
                    path_px=path_px,
                    reachable=path2 is not None,
                    reason=f"Reverse from obstacle ahead → path: {len(path_grid)} cells",
                )
            # Reversing wouldn't clear the buffer either — fall through to the
            # nearest-free-cell BFS below.
            log.info("navigate_to: obstacle ahead but no C-space-free cell behind "
                     "robot — falling back to buffer-exit BFS")

        log.info("navigate_to: robot in buffer zone (obstacle_ahead=%s) — "
                 "routing to nearest green cell first", obstacle_ahead)
        g_no_start = g.copy()
        g_no_start[start[0], start[1]] = False
        exit_cell = _nearest_free_cell(g_no_start, start)
        if exit_cell is not None:
            g_raw = obs_map.raw_grid.copy()
            g_raw[start[0], start[1]] = True
            path1 = _astar(g_raw, start, exit_cell) or [start, exit_cell]
            path2 = _astar(g, exit_cell, goal)
            if path2 is not None:
                path_grid = path1[:-1] + _simplify_path(path2, g)
            else:
                path_grid = _simplify_path(path1, g_raw)
            path_px = [_grid_to_px(rc, obs_map.w, obs_map.h) for rc in path_grid]
            if path_px:
                path_px[-1] = goal_px
            return NavPlan(
                path_grid=path_grid,
                path_px=path_px,
                reachable=path2 is not None,
                reason=f"Buffer exit → path: {len(path_grid)} cells",
            )
        # No green cell found at all — fall through to normal planning

    # ── Normal planning: robot starts in green C-space ────────────────────────
    path_grid = _astar(g, start, goal)

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


def _lookahead_waypoint(
    path_px: list[tuple[int, int]],
    robot_px: tuple[int, int],
    lookahead_px: float,
) -> tuple[int, int]:
    """Pure-pursuit waypoint: first path point at least `lookahead_px` ahead
    of the robot's *closest point on the route* (not from the route start).

    Re-scanning from path_px[1] every step breaks down once the robot is
    roughly mid-route: nearby waypoints near the *start* of the path (which
    the robot has already passed) can be farther than `lookahead_px` from the
    robot's current position than waypoints further along, so the scan would
    latch onto a stale, already-passed point and send the robot looping back
    on itself. Anchoring the scan at the closest-approach index guarantees it
    only ever looks forward along the route.

    Falls back to the route's last point — the approach goal nearest the
    target — once every remaining waypoint is already within reach.
    """
    closest_i = min(
        range(len(path_px)),
        key=lambda i: math.hypot(path_px[i][0] - robot_px[0], path_px[i][1] - robot_px[1]),
    )
    n_waypoints = len(path_px)
    for idx, px in enumerate(path_px[closest_i + 1:], start=closest_i + 1):
        if math.hypot(px[0] - robot_px[0], px[1] - robot_px[1]) >= lookahead_px:
            log.info(
                "[_lookahead_waypoint] closest=%d, targeting=%d of %d at %s "
                "(lookahead=%.0fpx)",
                closest_i, idx, n_waypoints - 1, px, lookahead_px,
            )
            return px
    log.info(
        "[_lookahead_waypoint] closest=%d, fell back to last waypoint %d at %s "
        "(all remaining within lookahead=%.0fpx)",
        closest_i, n_waypoints - 1, path_px[-1], lookahead_px,
    )
    return path_px[-1]


def commands_for_step(
    obs_map: ObstacleMap,
    plan: NavPlan,
    nav_heading: Heading | None,
) -> tuple[float, float, bool]:
    """Return (turn_deg, drive_deg, reverse) for the next navigation micro-step.

    turn_deg:  signed body-degrees to rotate (positive = CW viewed from above).
               Always 0.0 when reverse is True (see below). When a floor
               homography can be derived from the robot's own body (see
               _floor_homography), turn_deg is computed from the waypoint's
               floor-plane position rather than its raw pixel direction —
               correcting for the external camera's tilt/perspective.
    drive_deg: wheel-encoder degrees to drive afterward — forward when reverse
               is False, straight backward when reverse is True. Pass straight
               to robot.drive_degrees(), which (unlike a duration-based drive)
               covers a precise, repeatable distance regardless of speed,
               battery voltage, or load. The pixel distance to the next
               waypoint is converted to mm using the robot's apparent body
               size as a ruler (see mm_per_px), then to wheel-encoder degrees
               via the wheel circumference — so the robot drives toward the
               waypoint by roughly the right amount instead of for a fixed,
               waypoint-distance-agnostic duration (which previously caused it
               to overshoot close waypoints and then thrash ~180° trying to
               turn back onto them). When a floor homography is available, the
               final mm conversion uses the homography's local mm-per-pixel
               scale along this specific direction instead of the global
               body-area-derived scalar, correcting for perspective
               foreshortening (far waypoints cover more real mm per pixel).
    reverse:   True when the next waypoint is roughly behind the robot (within
               _REVERSE_TURN_THRESHOLD_DEG of 180°) AND the raw grid is blocked
               directly ahead (_obstacle_ahead). Pivoting in place toward a
               waypoint behind the robot would swing the chassis/gripper into
               that obstacle first, so the robot backs straight toward the
               waypoint instead — see plan_path's reverse-exit branch, which
               places such a waypoint at path_px[1] when needed.
    """
    if not plan.reachable or len(plan.path_px) < 2:
        return 0.0, 0.0, False

    robot_px = obs_map.robot_px
    next_px = _lookahead_waypoint(plan.path_px, robot_px, obs_map.robot_radius_px)

    dx = next_px[0] - robot_px[0]
    dy = next_px[1] - robot_px[1]
    dist_px = math.hypot(dx, dy)

    if dist_px < 1.0:
        return 0.0, 0.0, False

    reverse = False
    mm_per_px_dir: float | None = None
    if nav_heading is not None:
        fw = nav_heading.forward
        cross = fw[0] * dy - fw[1] * dx
        dot   = fw[0] * dx + fw[1] * dy
        turn_deg = math.degrees(math.atan2(cross, dot))
        target_angle_deg = math.degrees(math.atan2(dy, dx))
        heading_angle_deg = math.degrees(math.atan2(fw[1], fw[0]))
        log.info(
            "[commands_for_step] heading=(%.2f, %.2f) heading_deg=%.1f°  "
            "target_dir=(%.0f, %.0f) target_deg=%.1f°  turn=%.1f°",
            fw[0], fw[1], heading_angle_deg,
            dx, dy, target_angle_deg,
            turn_deg,
        )
        px_scale = mm_per_px(nav_heading.body_area)

        # Floor-plane correction: re-derive a perspective homography from the
        # robot's own body each frame and use it to correct both the turn
        # angle and the distance-per-pixel along this specific direction,
        # accounting for the external camera's tilt. Falls back to the
        # pixel-space turn_deg / scalar px_scale above when unavailable.
        H = _floor_homography(obs_map.bgr, nav_heading)
        if H is not None:
            rmx, rmy = _project_to_floor_mm(H, robot_px[0], robot_px[1])
            nmx, nmy = _project_to_floor_mm(H, next_px[0], next_px[1])
            fmx, fmy = nmx - rmx, nmy - rmy
            dist_mm = math.hypot(fmx, fmy)
            floor_turn_deg = math.degrees(math.atan2(fmy, fmx))
            log.info(
                "[commands_for_step] floor homography: robot=(%.0f,%.0f)mm "
                "waypoint=(%.0f,%.0f)mm  dist=%.0fmm  turn=%.1f° (pixel turn=%.1f°)",
                rmx, rmy, nmx, nmy, dist_mm, floor_turn_deg, turn_deg,
            )
            turn_deg = floor_turn_deg
            mm_per_px_dir = dist_mm / dist_px

        min_reliable_dist = obs_map.robot_radius_px * 0.5
        if dist_px < min_reliable_dist and abs(turn_deg) >= 20.0:
            log.info(
                "[commands_for_step] waypoint only %.1fpx away (< %.1f) — "
                "suppressing %.1f° turn (direction unreliable at this range)",
                dist_px, min_reliable_dist, turn_deg,
            )
            turn_deg = 0.0
        elif abs(turn_deg) >= _REVERSE_TURN_THRESHOLD_DEG:
            if _obstacle_ahead(obs_map, nav_heading):
                log.info(
                    "[commands_for_step] waypoint behind robot (turn=%.1f°) and "
                    "obstacle ahead — reversing instead of pivoting",
                    turn_deg,
                )
                turn_deg = 0.0
                reverse = True
            else:
                log.info(
                    "[commands_for_step] waypoint behind robot (turn=%.1f°, >= %.0f° "
                    "reverse threshold) but raw grid clear ahead — pivoting in place",
                    turn_deg, _REVERSE_TURN_THRESHOLD_DEG,
                )
    else:
        turn_deg = 0.0
        px_scale = None
        log.info("[commands_for_step] no heading available — turn_deg=0")

    if px_scale is not None:
        step_px = min(dist_px, _MAX_STEP_MM / px_scale)

        # Don't drive past grasp distance: stop with the robot and target
        # rings just touching, not stacked on top of each other.
        if obs_map.target_px is not None:
            tdist_px = math.hypot(robot_px[0] - obs_map.target_px[0],
                                  robot_px[1] - obs_map.target_px[1])
            remaining_px = tdist_px - (obs_map.robot_radius_px + obs_map.target_radius_px)
            if remaining_px > 0:
                step_px = min(step_px, remaining_px)

        step_mm_per_px = mm_per_px_dir if mm_per_px_dir is not None else px_scale
        drive_deg = mm_to_wheel_degrees(step_px * step_mm_per_px)
        log.info(
            "[commands_for_step] mm_per_px=%.3f%s (body_area=%dpx²)  "
            "step=%.0fpx -> %.0fmm -> %.0f wheel-deg",
            px_scale,
            f" (dir={mm_per_px_dir:.3f})" if mm_per_px_dir is not None else "",
            nav_heading.body_area, step_px, step_px * step_mm_per_px, drive_deg,
        )
    else:
        drive_deg = mm_to_wheel_degrees(min(_FALLBACK_STEP_MM, _MAX_STEP_MM))
        log.info("[commands_for_step] no body-area calibration — "
                 "falling back to %.0fmm blind step", _FALLBACK_STEP_MM)

    return turn_deg, drive_deg, reverse


def near_target(obs_map: ObstacleMap) -> bool:
    """True when the robot's center is within C-space-buffer distance of the target.

    Threshold = robot_radius * _CSPACE_BUFFER_SCALE * _NEAR_TARGET_GRACE — loose
    enough that navigation can stop early and report success before the robot
    gets close enough to collide with the target, with a grace margin since the
    A* path's last waypoint sits right at the (un-graced) buffer boundary.
    """
    if obs_map.target_px is None:
        return False
    dist = math.hypot(
        obs_map.robot_px[0] - obs_map.target_px[0],
        obs_map.robot_px[1] - obs_map.target_px[1],
    )
    threshold = obs_map.robot_radius_px * _CSPACE_BUFFER_SCALE * _NEAR_TARGET_GRACE
    ratio = dist / obs_map.robot_radius_px if obs_map.robot_radius_px else float("inf")
    result = dist <= threshold
    log.info(
        "[near_target] dist=%.1fpx threshold=%.1fpx (robot_radius=%.1f * %.1f * %.1f) "
        "dist/robot_radius=%.2f -> %s",
        dist, threshold, obs_map.robot_radius_px, _CSPACE_BUFFER_SCALE, _NEAR_TARGET_GRACE, ratio, result,
    )
    return result


def turn_to_face_target(obs_map: ObstacleMap, nav_heading: Heading) -> float | None:
    """Signed body-degrees to rotate so the robot faces the target.

    Returns None when no target is set or the robot is already facing it
    (within 8°). Positive = CW viewed from above.
    """
    if obs_map.target_px is None:
        return None
    dx = obs_map.target_px[0] - obs_map.robot_px[0]
    dy = obs_map.target_px[1] - obs_map.robot_px[1]
    if math.hypot(dx, dy) < 1.0:
        return None
    fw = nav_heading.forward
    cross = fw[0] * dy - fw[1] * dx
    dot = fw[0] * dx + fw[1] * dy
    turn_deg = math.degrees(math.atan2(cross, dot))
    H = _floor_homography(obs_map.bgr, nav_heading)
    if H is not None:
        rmx, rmy = _project_to_floor_mm(H, obs_map.robot_px[0], obs_map.robot_px[1])
        tmx, tmy = _project_to_floor_mm(H, obs_map.target_px[0], obs_map.target_px[1])
        turn_deg = math.degrees(math.atan2(tmy - rmy, tmx - rmx))
    if abs(turn_deg) < 8.0:
        log.info("[turn_to_face_target] already facing target (%.1f°)", turn_deg)
        return None
    log.info("[turn_to_face_target] %.1f° to face target", turn_deg)
    return turn_deg


def dist_to_path(
    robot_px: tuple[int, int],
    path_px: list[tuple[int, int]],
) -> float:
    """Minimum pixel distance from robot_px to any waypoint in path_px."""
    if not path_px:
        return float("inf")
    rx, ry = robot_px
    return min(math.hypot(px[0] - rx, px[1] - ry) for px in path_px)


def update_robot_position(obs_map: ObstacleMap, robot_px: tuple[int, int]) -> None:
    """Update robot position fields in obs_map in-place."""
    obs_map.robot_px = robot_px
    obs_map.robot_grid = _px_to_grid(robot_px, obs_map.w, obs_map.h)


def refresh_approach_path(obs_map: ObstacleMap, plan: NavPlan) -> bool:
    """Recompute the approach goal from the current robot position and replace
    the plan's path with a direct line to it.

    Called on subsequent navigation steps so that the final waypoint stays
    between the robot and the target as the approach angle changes — without
    this, the approach goal is frozen at the step-0 angle and the robot
    spirals around the target.

    Only activates when the robot is within twice the near_target threshold
    of the target (i.e. the final approach phase). Earlier in navigation the
    A* path's obstacle-avoidance waypoints must be preserved.

    Returns True if the path was updated, False if no update was needed
    (e.g. too far from target, already at grasp distance, or no target).
    """
    if obs_map.target_px is None:
        return False
    dist = math.hypot(
        obs_map.robot_px[0] - obs_map.target_px[0],
        obs_map.robot_px[1] - obs_map.target_px[1],
    )
    final_approach_threshold = obs_map.robot_radius_px * _CSPACE_BUFFER_SCALE * _NEAR_TARGET_GRACE * 2.0
    if dist > final_approach_threshold:
        return False
    goal = _approach_goal(obs_map)
    if goal is None:
        return False
    goal_grid, goal_px = goal
    snapped = _nearest_free_cell(obs_map.grid, goal_grid)
    if snapped is None:
        return False
    if snapped != goal_grid:
        goal_px = _grid_to_px(snapped, obs_map.w, obs_map.h)
        goal_grid = snapped
    plan.path_grid = [obs_map.robot_grid, goal_grid]
    plan.path_px = [obs_map.robot_px, goal_px]
    plan.reachable = True
    log.info(
        "[refresh_approach_path] updated approach goal to %s "
        "(robot=%s, target=%s, dist=%.0fpx, threshold=%.0fpx)",
        goal_px, obs_map.robot_px, obs_map.target_px, dist, final_approach_threshold,
    )
    return True


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
    suffix: str = "",
) -> dict[str, str]:
    """Write pipeline debug images to outdir, prefixed a–f for alphabetical sort order.

    a_raw              — unmodified input frame
    b_cleaned          — inpainted frame (robot + target removed) used for depth estimation
    c_depth            — Depth Anything V2 heat-map (INFERNO colormap)
    d_depth_gradient   — floor classification from depth gradients alone (no robot body)
    e_free_mask        — navigable floor after OR-ing in the robot body footprint
    f_cspace           — C-space grid: green=navigable, yellow=inflation buffer, red=obstacle
    g_nav_overlay      — final overlay: obstacle tint + A* path + R/T markers

    suffix is appended to the step prefix (e.g. "_replan") to distinguish replan snapshots
    from the normal per-step saves.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: dict[str, str] = {}
    p = os.path.join  # shorthand
    pfx = f"step_{step:02d}{suffix}"

    # a — raw input
    raw_path = p(outdir, f"{pfx}_a_raw.jpg")
    cv2.imwrite(raw_path, bgr)
    saved["raw"] = raw_path

    # b — inpainted frame used for depth estimation (robot + target removed)
    if obs_map.cleaned_bgr is not None:
        cleaned_path = p(outdir, f"{pfx}_b_cleaned.jpg")
        cv2.imwrite(cleaned_path, obs_map.cleaned_bgr)
        saved["cleaned"] = cleaned_path

    if obs_map.depth_map is not None:
        d = obs_map.depth_map

        # c — depth heat-map
        d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        depth_path = p(outdir, f"{pfx}_c_depth.jpg")
        cv2.imwrite(depth_path, depth_vis)
        saved["depth"] = depth_path

        # d — depth-gradient floor mask (no robot body carve-out)
        hsv_dbg = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow_dbg = cv2.inRange(hsv_dbg, YELLOW_HSV_LO, YELLOW_HSV_HI)
        floor_ref_dbg = cv2.erode(yellow_dbg, np.ones((15, 15), np.uint8))
        if (floor_ref_dbg > 0).sum() < 50:
            floor_ref_dbg = _robot_ring(yellow_dbg)
        grad_mask = _depth_gradient_floor_mask(d, floor_ref_dbg) if floor_ref_dbg is not None else None
        if grad_mask is not None:
            grad_vis = np.zeros_like(bgr)
            grad_vis[grad_mask >  0] = (0, 160, 0)
            grad_vis[grad_mask == 0] = (0, 0, 180)
            grad_path = p(outdir, f"{pfx}_d_depth_gradient.jpg")
            cv2.imwrite(grad_path, grad_vis)
            saved["gradient_mask"] = grad_path

    # e — free mask (gradient floor + robot footprint combined)
    mask_vis = np.zeros_like(bgr)
    mask_vis[obs_map.free_mask >  0] = (0, 160, 0)
    mask_vis[obs_map.free_mask == 0] = (0, 0, 180)
    mask_path = p(outdir, f"{pfx}_e_free_mask.jpg")
    cv2.imwrite(mask_path, mask_vis)
    saved["obstacle_mask"] = mask_path

    # f — C-space planning grid
    cspace = build_cspace_bgr(bgr, obs_map, plan)
    cspace_path = p(outdir, f"{pfx}_f_cspace.jpg")
    cv2.imwrite(cspace_path, cspace)
    saved["cspace"] = cspace_path

    # g — nav overlay
    overlay_bgr = draw_nav_overlay(bgr, obs_map, plan, step)
    overlay_path = p(outdir, f"{pfx}_g_nav_overlay.jpg")
    cv2.imwrite(overlay_path, overlay_bgr)
    saved["nav_overlay"] = overlay_path

    log.info("Navigation debug images (step %d%s) saved to: %s", step, suffix, outdir)
    return saved
