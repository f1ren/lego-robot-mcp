"""
Visualise detect_heading results on droidcam fixture images.

Saves annotated copies (green forward arrow) to tests/fixtures/heading/annotated/
for manual inspection.  Run with:

    python -m pytest tests/test_heading_annotate.py -s
"""
import math
import pathlib
import unittest

import cv2

FIXTURES  = pathlib.Path(__file__).parent / "fixtures"
OUT_DIR   = FIXTURES / "heading" / "annotated"

# One representative image per scenario, plus the cup fixture that currently fails.
IMAGES = sorted({
    p for pattern in (
        "grasp_readiness/droidcam*.jpg",
        "static_video/droidcam_000.jpg",
        "arm_motion/droidcam_004.jpg",
        "drive_motion/droidcam_005.jpg",
        "gripper_motion/droidcam_008.jpg",
        "move_forward/droidcam_000.jpg",
        "navigation/droidcam_robot_hiding_switch.jpg",
    )
    for p in FIXTURES.glob(pattern)
    if "annotated" not in p.parts
})


class TestHeadingAnnotate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import annotate_bgr, detect_heading
        cls._annotate = staticmethod(annotate_bgr)
        cls._detect   = staticmethod(detect_heading)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_annotate_all_fixtures(self):
        self.assertTrue(IMAGES, f"No fixture images found")
        for img_path in IMAGES:
            with self.subTest(image=img_path.name):
                bgr = cv2.imread(str(img_path))
                self.assertIsNotNone(bgr, f"Could not load {img_path}")

                heading = self._detect(bgr)
                status = (
                    f"body_center={heading.body_center} "
                    f"forward=({heading.forward[0]:.2f},{heading.forward[1]:.2f})"
                    if heading else "NO HEADING DETECTED"
                )
                print(f"\n{img_path.parent.name}/{img_path.name}: {status}")

                annotated = self._annotate(bgr)
                # stamp the subfolder name so images from different dirs don't collide
                stem = f"{img_path.parent.name}__{img_path.stem}"
                out_path = OUT_DIR / f"{stem}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"  -> saved {out_path}")


class TestHeadingDirection(unittest.TestCase):
    """Assert that detect_heading returns the correct compass direction for known fixtures."""

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        cls._detect = staticmethod(detect_heading)

    def _compass(self, forward: tuple[float, float]) -> str:
        dx, dy = forward
        angle = math.degrees(math.atan2(-dy, dx))  # +x=E, -y=N (image y-down)
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        return labels[round(angle / 45) % 8]

    def _assert_heading(self, rel_path: str, expected_compass: str):
        img_path = FIXTURES / rel_path
        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr, f"Could not load {img_path}")
        heading = self._detect(bgr)
        self.assertIsNotNone(heading, f"No heading detected in {rel_path}")
        got = self._compass(heading.forward)
        self.assertEqual(
            got, expected_compass,
            f"{rel_path}: expected {expected_compass}, got {got} "
            f"(forward={heading.forward[0]:.3f},{heading.forward[1]:.3f})",
        )

    def test_gripper_west_current(self):
        """Gripper points west — regression for off-axis cable blob bug."""
        self._assert_heading("heading/droidcam_current.jpg", "W")

    def test_gripper_east_drive_motion(self):
        self._assert_heading("drive_motion/droidcam_005.jpg", "E")

    def test_gripper_east_move_forward(self):
        self._assert_heading("move_forward/droidcam_000.jpg", "E")

    def test_gripper_east_static(self):
        self._assert_heading("static_video/droidcam_000.jpg", "E")

    def test_gripper_north_robot_hiding_switch(self):
        """Gripper points north — robot hiding the switch, new navigation fixture."""
        self._assert_heading("navigation/droidcam_robot_hiding_switch.jpg", "N")

    def test_gripper_north_heading_switch_and_lights_on(self):
        """Gripper points north — fixture captured 2026-06-04 with switch and lights on."""
        self._assert_heading("heading/droidcam_heading_switch_and_lights_on.jpg", "N")

    def test_gripper_southwest_cup_shadow(self):
        """Gripper points south-west, toward the cup — regression for the robot's own cast
        shadow (bigger + smoother than the real gripper blob) winning the area-based score
        and flipping the arrow 180 degrees to point NE at the shadow instead."""
        self._assert_heading("heading/droidcam_cup_gripper_shadow.jpg", "SW")


class TestHeadingBodyCenter(unittest.TestCase):
    """Regression test for body_hull's fragment-clustering fix (see
    mcp_robot/heading.py::_BODY_CLUSTER_RADIUS_FACTOR).

    On 2026-07-18 the robot's yellow chassis was sliced into disjoint blobs by
    the gripper/arm, PCB, and motor housings sitting on top of it. body_hull
    used to cluster fragments by centroid-distance-to-the-single-largest-
    fragment only, which dropped a whole legitimate chunk of chassis on the
    left side (its centroid was ~170px from the seed's — just outside the old
    radius, even though the two fragments' nearest edges were only ~114px
    apart) and biased body_center toward the right wheel instead of the
    chassis's true visual center, so the drawn arrow started off-center.
    """

    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "heading" / "droidcam_split_yellow_body.jpg"

    # Manually verified: the yellow chassis plate spans roughly this region in
    # the raw frame. body_center must land inside it.
    _CHASSIS_GROUND_TRUTH_BBOX = (470, 1020, 710, 1190)

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import body_hull, detect_heading
        cls._detect = staticmethod(detect_heading)
        cls._body_hull = staticmethod(body_hull)

    def test_body_center_spans_both_chassis_fragments(self):
        bgr = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(bgr, f"Could not load {self.FIXTURE}")

        body = self._body_hull(bgr)
        self.assertIsNotNone(body, "body_hull should detect the robot body")
        x, y, w, h = cv2.boundingRect(body)
        self.assertLess(
            x, 550,
            f"body hull left edge={x} — should extend into the left-side chassis fragment "
            "(previously dropped for sitting just outside the seed-only cluster radius)",
        )

        heading = self._detect(bgr)
        self.assertIsNotNone(heading, f"No heading detected in {self.FIXTURE}")
        gx1, gy1, gx2, gy2 = self._CHASSIS_GROUND_TRUTH_BBOX
        cx, cy = heading.body_center
        self.assertTrue(
            gx1 <= cx <= gx2 and gy1 <= cy <= gy2,
            f"body_center {heading.body_center} should land within the chassis "
            f"{self._CHASSIS_GROUND_TRUTH_BBOX} — if it's skewed to one edge, "
            "clustering dropped a real fragment again",
        )


class TestBodyHullRejectsFloorFalsePositive(unittest.TestCase):
    """Regression test for body_hull's oversized-seed retry (see
    mcp_robot/heading.py::_MAX_BODY_AREA_FRAC / _MAX_SEED_RETRIES).

    On 2026-07-18 the wood floor passed the yellow HSV mask as hundreds of
    small contours. The single largest one (81,721px) was a contiguous floor
    patch nowhere near the robot, not the chassis — which, fragmented by the
    gripper/arm/PCB sitting on top of it, had no single fragment anywhere
    near that big. body_hull's nearest-edge clustering (added the same day)
    then transitively chained ~90 of those floor/wall specks onto that seed,
    producing a "body" hull covering 26% of the frame with a centroid ~490px
    from the real robot — the reported arrow started in empty floor.

    body_hull must now reject that oversized cluster and retry from the
    next-largest seed, which lands on the real chassis instead.
    """

    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "heading" / "droidcam_floor_false_positive.jpg"

    # Manually verified: the yellow chassis plate spans roughly this region in
    # the raw frame (bottom-right of a 720x1280 portrait frame).
    _CHASSIS_GROUND_TRUTH_BBOX = (420, 830, 720, 1180)

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import body_hull, detect_heading, _MAX_BODY_AREA_FRAC
        cls._body_hull = staticmethod(body_hull)
        cls._detect = staticmethod(detect_heading)
        cls._max_body_area_frac = _MAX_BODY_AREA_FRAC

    def test_body_hull_lands_on_chassis_not_floor(self):
        bgr = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(bgr, f"Could not load {self.FIXTURE}")
        h, w = bgr.shape[:2]

        body = self._body_hull(bgr)
        self.assertIsNotNone(body, "body_hull should detect the robot body")

        area = cv2.contourArea(body)
        frac = area / (w * h)
        self.assertLessEqual(
            frac, self._max_body_area_frac,
            f"hull area is {frac:.1%} of frame — looks like the floor false positive again",
        )

        x, y, bw, bh = cv2.boundingRect(body)
        cx, cy = x + bw / 2, y + bh / 2
        gx1, gy1, gx2, gy2 = self._CHASSIS_GROUND_TRUTH_BBOX
        self.assertTrue(
            gx1 <= cx <= gx2 and gy1 <= cy <= gy2,
            f"hull center ({cx:.0f},{cy:.0f}) should land within the chassis region "
            f"{self._CHASSIS_GROUND_TRUTH_BBOX} — if it's off in empty floor, "
            "the oversized-seed retry regressed",
        )

    def test_gripper_west_not_swallowed_by_hull_notch(self):
        """Regression for detect_heading's body-suppression mask (see
        mcp_robot/heading.py step 3, "Suppress black pixels...").

        Once body_hull correctly finds the chassis above, its convex hull
        still fills in the concave notch where the arm/gripper mount sits
        recessed between chassis lobes — that filled-in area isn't yellow,
        but the old suppression mask (built from the hull) erased it anyway,
        leaving only the gripper's two jaw-tips: individually too small
        (33px/18px) and ~160px apart, so no candidate ever reached
        _MIN_GRIPPER_AREA_FRAC and detect_heading returned None. Suppressing
        with the raw yellow color mask instead keeps the true chassis pixels
        excluded without erasing the non-yellow gripper sitting in the notch.
        """
        bgr = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(bgr, f"Could not load {self.FIXTURE}")

        heading = self._detect(bgr)
        self.assertIsNotNone(
            heading,
            "detect_heading returned None — gripper likely swallowed by the "
            "hull-notch suppression mask again",
        )
        dx, dy = heading.forward
        angle = math.degrees(math.atan2(-dy, dx))  # +x=E, -y=N (image y-down)
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        got = labels[round(angle / 45) % 8]
        self.assertEqual(
            got, "W",
            f"expected gripper facing W (visible claws point left of chassis), got {got} "
            f"(forward={dx:.3f},{dy:.3f})",
        )


class TestYellowCupNotMistakenForBody(unittest.TestCase):
    """Regression for body_candidates' Technic-hole ranking (2026-10-02).

    A tilted yellow cup (waste bin) on the right of the frame is a single
    smooth 8.2k px yellow contour — larger than any fragment of the robot's
    chassis, which the PCB and black parts split into pieces. Size-ordered
    seeding returned the cup as "the body", no gripper was found next to it,
    and detect_heading returned None, breaking navigate_to/turn_to/drive_to
    and the grasp-readiness gate. The chassis plates have a dense hole grid
    (~31 enclosed holes here) while the cup has ~3, so ranking by holes
    picks the chassis; detect_heading also falls through to the next
    candidate when one yields no gripper.
    """

    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "heading" / "droidcam_yellow_cup_bin.jpg"

    # Manually verified regions in the raw 720x1280 frame.
    _CHASSIS_BBOX = (90, 740, 290, 880)
    _CUP_BBOX = (530, 660, 655, 760)

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import body_hull, detect_heading
        cls._body_hull = staticmethod(body_hull)
        cls._detect = staticmethod(detect_heading)

    def _center_in(self, pt, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

    def test_body_hull_is_chassis_not_cup(self):
        bgr = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(bgr, f"Could not load {self.FIXTURE}")
        body = self._body_hull(bgr)
        self.assertIsNotNone(body, "body_hull should detect the robot body")
        x, y, bw, bh = cv2.boundingRect(body)
        center = (x + bw / 2, y + bh / 2)
        self.assertFalse(self._center_in(center, self._CUP_BBOX), "body_hull picked the yellow cup")
        self.assertTrue(
            self._center_in(center, self._CHASSIS_BBOX),
            f"hull center {center} should land on the chassis {self._CHASSIS_BBOX}",
        )

    def test_heading_north(self):
        bgr = cv2.imread(str(self.FIXTURE))
        heading = self._detect(bgr)
        self.assertIsNotNone(heading, "detect_heading returned None — yellow cup taken as body again?")
        self.assertTrue(self._center_in(heading.body_center, self._CHASSIS_BBOX))
        dx, dy = heading.forward
        angle = math.degrees(math.atan2(-dy, dx))
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        self.assertEqual(labels[round(angle / 45) % 8], "N", f"forward={dx:.3f},{dy:.3f}")


class TestBodyHullJoinsRibbonSplitPlate(unittest.TestCase):
    """Regression for body_candidates' split-plate rule (see
    mcp_robot/heading.py::_PLATE_SPLIT_GAP_FACTOR / _PLATE_MIN_HUE).

    On 2026-10-05 the Pi's white ribbon cable lay across the robot's second
    yellow plate and cut it into two shards (326px and 259px, 12px apart).
    Each was too small on its own to bridge the ~77px gap to the first plate,
    so the hull covered one plate only — 4,854px² instead of ~12,600px² —
    and navigation.mm_per_px read 1.66 instead of ~1.03. The grasp gate then
    reported a 66px front-gap as 110mm (output/logs/mcp_server.log,
    2026-10-05 13:55:16,218).
    """

    FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "heading" / "simpleipcamera_ribbon_split_plate.jpg"

    # Manually verified points in the raw 720x1280 frame, one on each plate.
    _FIRST_PLATE_PT = (431, 741)
    _SECOND_PLATE_PT = (442, 855)

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import body_hull, detect_heading
        from mcp_robot.navigation import mm_per_px
        cls._body_hull = staticmethod(body_hull)
        cls._detect = staticmethod(detect_heading)
        cls._mm_per_px = staticmethod(mm_per_px)
        cls._bgr = cv2.imread(str(cls.FIXTURE))

    def test_hull_spans_both_plates(self):
        self.assertIsNotNone(self._bgr, f"Could not load {self.FIXTURE}")
        body = self._body_hull(self._bgr)
        self.assertIsNotNone(body, "body_hull should detect the robot body")
        for name, pt in (("first", self._FIRST_PLATE_PT), ("second", self._SECOND_PLATE_PT)):
            self.assertGreaterEqual(
                cv2.pointPolygonTest(body, pt, False), 0,
                f"hull {cv2.boundingRect(body)} misses the {name} plate at {pt}",
            )

    def test_mm_per_px_matches_two_plate_frames(self):
        heading = self._detect(self._bgr)
        self.assertIsNotNone(heading, f"No heading detected in {self.FIXTURE}")
        scale = self._mm_per_px(heading.body_area)
        # Same-day frames whose hull already covered both plates read
        # 0.99-1.06 mm/px; the one-plate hull read 1.66.
        self.assertGreater(scale, 0.9)
        self.assertLess(scale, 1.2, f"mm_per_px={scale:.2f} — hull lost a plate again?")

    def test_heading_east(self):
        heading = self._detect(self._bgr)
        self.assertIsNotNone(heading, f"No heading detected in {self.FIXTURE}")
        dx, dy = heading.forward
        angle = math.degrees(math.atan2(-dy, dx))
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        self.assertEqual(labels[round(angle / 45) % 8], "E", f"forward={dx:.3f},{dy:.3f}")


if __name__ == "__main__":
    unittest.main()
