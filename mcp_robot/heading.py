"""
Detect the robot's forward heading in external (DroidCam) frames and overlay
a green arrow showing the direction the gripper is pointing.

The robot has a yellow LEGO body whose chassis presents many long parallel
edges (brick rows, top and bottom panel edges). Those edges all run along the
body's "forward" axis. We:
  1. Color-mask the yellow body to localize it and define a search ROI.
  2. Run a Hough line transform on Canny edges inside that ROI; the dominant
     line orientation gives the forward axis (modulo 180°).
  3. Disambiguate front-vs-back by detecting the dark gripper-and-arm
     assembly nearby — it sits at the FRONT of the body.

This is intentionally a static image-processing approach (no motion required):
the lab lighting and floor are kept clean enough that color masking is more
reliable than optical-flow tracking, which suffers from motor jitter.

Public API:
    detect_heading(bgr) -> Heading | None
    body_hull(bgr) -> np.ndarray | None      # convex hull of the robot's yellow body
    annotate_jpeg_b64(b64) -> str            # returns annotated b64 (or original)
    annotate_jpeg_bytes(buf) -> bytes        # returns annotated bytes (or original)
"""
from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# HSV ranges tuned against tests/fixtures/*/droidcam_*.jpg.
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255].
# Public so navigation.py can reuse the same thresholds for floor exclusion.
YELLOW_HSV_LO = np.array([15, 130, 100], dtype=np.uint8)
YELLOW_HSV_HI = np.array([35, 255, 255], dtype=np.uint8)

# Keep private aliases for any internal usage below (avoids touching every reference).
_YELLOW_HSV_LO = YELLOW_HSV_LO
_YELLOW_HSV_HI = YELLOW_HSV_HI

# Gripper jaws + arm: low value (dark), low-to-medium saturation.
_BLACK_V_MAX = 70
_BLACK_S_MAX = 90

# Minimum yellow blob area (pixels) to accept as the robot body. Tuned for
# 1280x720 droidcam frames; scaled down with image area below.
_MIN_BODY_AREA_FRAC = 0.0025   # 0.25% of frame (handles partially occluded body)
_MIN_GRIPPER_AREA_FRAC = 0.0005  # 0.05% of frame

# Gripper search ROI: expand along the body's long axis (where the arm extends)
# more aggressively than perpendicular, relative to the larger/smaller body dim.
_ROI_EXPAND_ALONG = 1.5   # fraction of max(bw, bh) along the forward/backward axis
_ROI_EXPAND_PERP  = 0.4   # fraction of min(bw, bh) perpendicular to the axis

_ARROW_COLOR_BGR = (0, 220, 0)   # bright green
_ARROW_THICKNESS_FRAC = 0.006    # of image diagonal

_OBJ_COLOR_BGR = (0, 165, 255)   # orange — object bbox, line, angle label


@dataclass
class Heading:
    body_center: tuple[int, int]      # (x, y) pixel coords of body centroid
    arrow_anchor: tuple[int, int]     # (x, y) pixel coords — front-edge midpoint of body
    forward: tuple[float, float]      # unit vector (dx, dy) along body's long axis, gripper-side
    body_area: int                    # pixel count of yellow mask used
    gripper_area: int                 # pixel count of black mask used
    body_radius_px: float = 40.0     # half the longer side of the yellow body bounding box


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _centroid(contour: np.ndarray) -> tuple[int, int] | None:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def body_hull(bgr: np.ndarray) -> np.ndarray | None:
    """
    Locate the robot's yellow body in `bgr` and return its convex hull
    (Nx1x2 int32 contour points), or None if no yellow body is found or
    it's too small to be the robot.

    Locates the body by clustering nearby yellow contours into one shape so
    callers get a reliable centroid + ROI. This hull is NOT used by
    `detect_heading` to compute the heading axis (Hough lines do that) —
    using minAreaRect on the combined hull is unreliable when oblique camera
    angles make the visible yellow region L-shaped.
    """
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    frame_area = h * w

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LO, _YELLOW_HSV_HI)
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.info("body_hull: no yellow contours found in frame")
        return None
    seed = max(contours, key=cv2.contourArea)
    seed_center = _centroid(seed)
    if seed_center is None:
        log.info("body_hull: largest yellow contour has degenerate (zero-area) centroid")
        return None
    sx, sy, sw, sh = cv2.boundingRect(seed)
    cluster_radius = float(np.hypot(sw, sh)) * 1.6
    pts: list[np.ndarray] = [seed.reshape(-1, 2)]
    for c in contours:
        if c is seed or cv2.contourArea(c) < 30:
            continue
        cc = _centroid(c)
        if cc is None:
            continue
        if np.hypot(cc[0] - seed_center[0], cc[1] - seed_center[1]) <= cluster_radius:
            pts.append(c.reshape(-1, 2))
    body_pts = np.vstack(pts)
    body = cv2.convexHull(body_pts)
    area = cv2.contourArea(body)
    min_area = _MIN_BODY_AREA_FRAC * frame_area
    if area < min_area:
        log.info(
            "body_hull: yellow blob too small (%.0fpx < %.0fpx min, %.2f%% of frame) — "
            "robot may be out of frame, far away, or occluded",
            area, min_area, 100.0 * area / frame_area,
        )
        return None
    return body


def detect_heading(bgr: np.ndarray) -> Heading | None:
    """
    Locate the robot in `bgr` and return its forward heading, or None if
    detection is unreliable (robot out of frame, gripper occluded, etc).
    """
    if bgr is None or bgr.size == 0:
        log.info("detect_heading: empty/invalid frame")
        return None
    h, w = bgr.shape[:2]
    frame_area = h * w

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LO, _YELLOW_HSV_HI)
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    # ── 1. Find the yellow body ───────────────────────────────────────────
    body = body_hull(bgr)
    if body is None:
        log.info("detect_heading: aborting — no robot body detected (see body_hull log above)")
        return None
    body_center = _centroid(body)
    if body_center is None:
        log.info("detect_heading: body hull found but centroid degenerate (zero-area moments)")
        return None
    bx, by, bw, bh = cv2.boundingRect(body)

    # ── 2. Forward direction from dominant Hough lines on the body ────────
    # Compute the long axis BEFORE gripper detection so we can use it to
    # score gripper candidates by alignment with the axis (see step 3).
    # The LEGO chassis presents many long parallel edges (brick rows, top
    # and bottom panel edges) all aligned with the body's long axis. We
    # detect Canny edges within a yellow-restricted ROI, run a probabilistic
    # Hough transform, and take the length-weighted modal angle as the long
    # axis. This is robust to oblique camera angles where the projected body
    # shape becomes L-shaped (for which minAreaRect would return a diagonal).
    pad_long_x = int(bw * 0.4)
    pad_long_y = int(bh * 0.4)
    lx0 = max(0, bx - pad_long_x)
    ly0 = max(0, by - pad_long_y)
    lx1 = min(w, bx + bw + pad_long_x)
    ly1 = min(h, by + bh + pad_long_y)
    gray_roi = cv2.cvtColor(bgr[ly0:ly1, lx0:lx1], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 60, 150)
    yellow_dilated = cv2.dilate(yellow_mask[ly0:ly1, lx0:lx1], np.ones((9, 9), np.uint8))
    edges = cv2.bitwise_and(edges, yellow_dilated)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=25, maxLineGap=8,
    )
    if lines is None or len(lines) < 3:
        log.info(
            "detect_heading: body found at %s but too few Hough lines for forward axis (%d found, need >=3)",
            body_center, 0 if lines is None else len(lines),
        )
        return None
    # Length-weighted histogram of line angles in [0, 180).
    angle_hist = np.zeros(180, dtype=np.float32)
    for L in lines:
        # cv2.HoughLinesP returns shape (N, 1, 4) on OpenCV 4.x but (N, 4) on
        # 5.x — reshape(-1) extracts the 4 coords under either convention.
        x1, y1, x2, y2 = L.reshape(-1)
        ldx, ldy = x2 - x1, y2 - y1
        length = math.hypot(ldx, ldy)
        ang = int(math.degrees(math.atan2(ldy, ldx))) % 180
        angle_hist[ang] += length
    # Light circular smoothing so 89° and 91° aren't treated as different bins.
    smoothed = np.convolve(
        np.concatenate([angle_hist[-3:], angle_hist, angle_hist[:3]]),
        np.ones(7) / 7, mode="same",
    )[3:-3]
    dom_deg = int(np.argmax(smoothed))
    a = math.radians(dom_deg)
    axis = (math.cos(a), math.sin(a))

    # ── 3. Find black gripper inside an expanded ROI around the body ──────
    # Expand more along the body's long axis (where the arm protrudes) than
    # perpendicular, so the search window captures the full gripper/arm even
    # when it extends far beyond the body bbox (e.g. chain-link arm at ~90°).
    ax_abs, ay_abs = abs(axis[0]), abs(axis[1])
    pad_along = int(max(bw, bh) * _ROI_EXPAND_ALONG)
    pad_perp  = int(min(bw, bh) * _ROI_EXPAND_PERP)
    pad_x = int(ay_abs * pad_perp + ax_abs * pad_along)
    pad_y = int(ax_abs * pad_perp + ay_abs * pad_along)
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(w, bx + bw + pad_x)
    y1 = min(h, by + bh + pad_y)

    roi_hsv = hsv[y0:y1, x0:x1]
    black_mask = cv2.inRange(
        roi_hsv,
        np.array([0,   0,   0], dtype=np.uint8),
        np.array([179, _BLACK_S_MAX, _BLACK_V_MAX], dtype=np.uint8),
    )
    # Suppress black pixels that overlap the yellow body itself — the gripper
    # is what extends *beyond* the body.
    body_mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(body_mask_full, [body], -1, 255, thickness=cv2.FILLED)
    body_mask_dilated = cv2.dilate(body_mask_full, np.ones((9, 9), np.uint8))
    body_in_roi = body_mask_dilated[y0:y1, x0:x1]
    black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(body_in_roi))
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    # Close with a larger kernel so multi-finger gripper jaws and chain links
    # fuse into one blob (chain links can be ~15 px apart).
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8)
    )

    # Texture (Canny edge density) within the gripper ROI. The real gripper
    # is a mechanical assembly (jaws, chain links) with lots of internal
    # edges; the robot's own cast shadow on the floor is dark enough to pass
    # the black-mask threshold above but is a smooth gradient with almost no
    # internal edges. Used below to down-weight shadow blobs that would
    # otherwise win on area alone.
    gray_gripper_roi = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gripper_edges = cv2.Canny(gray_gripper_roi, 60, 150)

    # Pick the black blob that best represents the gripper. Score by
    # area × |projection onto body long axis| × edge density so that
    # off-axis noise (cables, shadows perpendicular to the forward
    # direction) and large-but-smooth blobs (cast shadows, which can
    # otherwise out-area the real gripper) cannot outscore the actual
    # gripper, which sits at one end of the body's long axis and has
    # visible mechanical texture.
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = _MIN_GRIPPER_AREA_FRAC * frame_area
    bcx, bcy = body_center
    best_score = -1.0
    arrow_anchor: tuple[int, int] | None = None
    gripper_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        gc = _centroid(c)
        if gc is None:
            continue
        cx_full, cy_full = gc[0] + x0, gc[1] + y0
        ddx, ddy = cx_full - bcx, cy_full - bcy
        along = abs(ddx * axis[0] + ddy * axis[1])
        c_mask = np.zeros(black_mask.shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, thickness=cv2.FILLED)
        edge_px = cv2.countNonZero(cv2.bitwise_and(gripper_edges, c_mask))
        texture = edge_px / area
        score = area * along * texture
        if score > best_score:
            best_score = score
            arrow_anchor = (cx_full, cy_full)
            gripper_area = int(area)
    if arrow_anchor is None:
        log.info(
            "detect_heading: body+axis found at %s but no gripper/arm blob detected — "
            "gripper may be occluded or outside the search ROI",
            body_center,
        )
        return None

    # Disambiguate front-vs-back using the gripper centroid: forward points
    # toward the gripper, away from the body interior.
    dx = arrow_anchor[0] - body_center[0]
    dy = arrow_anchor[1] - body_center[1]
    if axis[0] * dx + axis[1] * dy < 0:
        axis = (-axis[0], -axis[1])

    # Anchor the arrow at the body's front edge: project every body-hull
    # vertex onto the forward axis, take the max-projection point. This
    # keeps the arrow rooted on the body itself (not at the gripper centroid,
    # which can sit off-axis when jaws or arm hang asymmetrically).
    hull_xy = body.reshape(-1, 2).astype(np.float32)
    projections = (hull_xy[:, 0] - body_center[0]) * axis[0] + (hull_xy[:, 1] - body_center[1]) * axis[1]
    front_idx = int(np.argmax(projections))
    front_x, front_y = int(hull_xy[front_idx, 0]), int(hull_xy[front_idx, 1])

    return Heading(
        body_center=body_center,
        arrow_anchor=(front_x, front_y),
        forward=axis,
        body_area=int(cv2.contourArea(body)),
        gripper_area=gripper_area,
        body_radius_px=float(max(bw, bh)) / 2.0,
    )


def _draw_arrow(bgr: np.ndarray, heading: Heading) -> np.ndarray:
    h, w = bgr.shape[:2]
    diag = float(np.hypot(w, h))
    thickness = max(2, int(diag * _ARROW_THICKNESS_FRAC))
    # Arrow length: long enough to reach across most of the frame so the user
    # can tell whether "forward" actually meets the target.
    length = int(diag * 0.45)

    start = heading.body_center
    end_x = int(start[0] + heading.forward[0] * length)
    end_y = int(start[1] + heading.forward[1] * length)
    # Clamp end-point inside the frame so the arrowhead is always visible.
    end_x = max(0, min(w - 1, end_x))
    end_y = max(0, min(h - 1, end_y))

    overlay = bgr.copy()
    cv2.arrowedLine(
        overlay, start, (end_x, end_y),
        _ARROW_COLOR_BGR, thickness=thickness, tipLength=0.08,
    )
    # 70% transparent → blend at 30% opacity
    return cv2.addWeighted(overlay, 0.50, bgr, 0.50, 0)


def compute_heading_to_object_angle(heading: Heading, obj_center: tuple[int, int]) -> float:
    """Signed angle in degrees from forward to the vector pointing at obj_center.

    Positive → object is clockwise from forward (image y-down, viewed from above).
    Negative → counter-clockwise.
    """
    bx, by = heading.body_center
    ox, oy = obj_center
    dx, dy = ox - bx, oy - by
    fw = heading.forward
    dot = fw[0] * dx + fw[1] * dy
    cross = fw[0] * dy - fw[1] * dx
    return math.degrees(math.atan2(cross, dot))


def _draw_object_annotation(
    bgr: np.ndarray,
    heading: Heading,
    obj_center: tuple[int, int],
    obj_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Overlay object location on bgr: bbox, line from body to object, angle label."""
    h, w = bgr.shape[:2]
    diag = float(np.hypot(w, h))
    thickness = max(1, int(diag * 0.004))

    out = bgr.copy()
    bx, by = heading.body_center
    ox, oy = obj_center

    angle_deg = compute_heading_to_object_angle(heading, obj_center)
    rot_dir = "CW" if angle_deg > 0 else "CCW"

    if obj_bbox is not None:
        x1, y1, x2, y2 = obj_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), _OBJ_COLOR_BGR, thickness)

    cv2.line(out, (bx, by), (ox, oy), _OBJ_COLOR_BGR, max(1, thickness - 1), cv2.LINE_AA)
    cv2.circle(out, (ox, oy), max(4, thickness * 2), _OBJ_COLOR_BGR, -1)

    angle_text = f"{abs(angle_deg):.0f}{rot_dir}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, diag * 0.0007)
    (tw, th), baseline = cv2.getTextSize(angle_text, font, font_scale, thickness)
    mx = max(0, min(w - tw - 4, (bx + ox) // 2 - tw // 2))
    my = max(th + 6, (by + oy) // 2 - 4)
    cv2.rectangle(out, (mx - 2, my - th - 2), (mx + tw + 2, my + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, angle_text, (mx, my), font, font_scale, _OBJ_COLOR_BGR, thickness, cv2.LINE_AA)

    return out


def annotate_bgr(
    bgr: np.ndarray,
    obj_center: tuple[int, int] | None = None,
    obj_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return `bgr` with a green forward arrow (+ optional object annotation) if heading detected."""
    try:
        heading = detect_heading(bgr)
    except Exception as exc:
        log.debug("Heading detection failed: %s", exc)
        return bgr
    if heading is None:
        return bgr
    out = _draw_arrow(bgr, heading)
    if obj_center is not None:
        out = _draw_object_annotation(out, heading, obj_center, obj_bbox)
    return out


def annotate_jpeg_bytes(buf: bytes) -> bytes:
    """Decode JPEG bytes, overlay arrow, re-encode. Returns original on any failure."""
    try:
        arr = np.frombuffer(buf, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return buf
        annotated = annotate_bgr(bgr)
        if annotated is bgr:  # no change — skip re-encoding
            return buf
        ok, out = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return buf
        return out.tobytes()
    except Exception as exc:
        log.debug("annotate_jpeg_bytes failed: %s", exc)
        return buf


def annotate_jpeg_b64(b64: str) -> str:
    """Same as annotate_jpeg_bytes but base64-in / base64-out."""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return b64
    out = annotate_jpeg_bytes(raw)
    if out is raw:
        return b64
    return base64.b64encode(out).decode()


# ── optical flow annotation ───────────────────────────────────────────────────

_FLOW_COLOR_BGR = (0, 0, 220)   # red arrows
_FLOW_GRID_STEP = 25            # sampling grid spacing (pixels)
_FLOW_MIN_MAG = 1.5             # minimum flow magnitude to draw (pixels)
_FLOW_ARROW_SCALE = 6.0         # visual amplification of arrow length
_FLOW_THICKNESS = 2             # arrow stroke thickness (px)


def _draw_flow_arrows(dst: np.ndarray, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute Farneback optical flow and draw red arrows on a copy of dst."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    h, w = dst.shape[:2]
    out = dst.copy()
    for y in range(_FLOW_GRID_STEP // 2, h, _FLOW_GRID_STEP):
        for x in range(_FLOW_GRID_STEP // 2, w, _FLOW_GRID_STEP):
            dx, dy = flow[y, x]
            mag = math.hypot(dx, dy)
            if mag < _FLOW_MIN_MAG:
                continue
            ex = max(0, min(w - 1, int(x + dx * _FLOW_ARROW_SCALE)))
            ey = max(0, min(h - 1, int(y + dy * _FLOW_ARROW_SCALE)))
            cv2.arrowedLine(out, (x, y), (ex, ey), _FLOW_COLOR_BGR, thickness=_FLOW_THICKNESS, tipLength=0.3)
    return out


def flow_magnitudes(frames_gray: list[np.ndarray]) -> list[float]:
    """Return mean optical-flow magnitude for each consecutive frame pair.

    Result length is len(frames_gray) - 1; index i is the mean magnitude
    between frames[i] and frames[i+1]. Uses the same Farneback parameters
    as _draw_flow_arrows so all flow logic stays consistent.
    """
    mags: list[float] = []
    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mags.append(float(np.mean(np.hypot(flow[..., 0], flow[..., 1]))))
    return mags


def annotate_flow_sequence_bgr(
    annotated: list[np.ndarray],
    raw: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Overlay optical flow red arrows onto a chronological sequence of BGR frames.

    Flow is computed from `raw` (clean, no heading-arrow overlay) so static
    arrow pixels don't produce spurious flow vectors. Arrows are drawn on
    `annotated`. The first frame is returned unchanged (no prior frame).
    """
    src = raw if (raw is not None and len(raw) == len(annotated)) else annotated
    if len(annotated) < 2:
        return list(annotated)
    result: list[np.ndarray] = [annotated[0]]
    prev_gray = cv2.cvtColor(src[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(annotated)):
        curr_gray = cv2.cvtColor(src[i], cv2.COLOR_BGR2GRAY)
        result.append(_draw_flow_arrows(annotated[i], prev_gray, curr_gray))
        prev_gray = curr_gray
    return result


def annotate_flow_sequence_jpeg_b64(
    annotated_b64: list[str],
    raw_b64: list[str] | None = None,
) -> list[str]:
    """JPEG base64 wrapper around annotate_flow_sequence_bgr. Returns original on failure."""
    try:
        def _dec(b: str) -> np.ndarray | None:
            return cv2.imdecode(np.frombuffer(base64.b64decode(b), dtype=np.uint8), cv2.IMREAD_COLOR)

        ann_bgr = [_dec(b) for b in annotated_b64]
        if any(f is None for f in ann_bgr):
            return annotated_b64

        raw_bgr: list[np.ndarray] | None = None
        if raw_b64 and len(raw_b64) == len(annotated_b64):
            decoded_raw = [_dec(b) for b in raw_b64]
            if all(f is not None for f in decoded_raw):
                raw_bgr = decoded_raw  # type: ignore[assignment]

        result_bgr = annotate_flow_sequence_bgr(ann_bgr, raw_bgr)  # type: ignore[arg-type]
        out: list[str] = []
        for i, frame in enumerate(result_bgr):
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            out.append(base64.b64encode(buf.tobytes()).decode() if ok else annotated_b64[i])
        return out
    except Exception as exc:
        log.debug("annotate_flow_sequence_jpeg_b64 failed: %s", exc)
        return annotated_b64
