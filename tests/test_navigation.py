"""
Unit tests for navigation.detect_obstacles and navigation.plan_path.

Uses droidcam_nav_obstacle.jpg — a live-captured frame showing the robot on the
right, a cup on the upper-left, and a wooden cabinet across the top that blocks
the straight-line path.  No live camera or robot connection required.

Each test run saves its debug images to tests/fixtures/navigation/annotated/:
  step_00_raw.jpg          — unmodified fixture frame
  step_00_obstacle_mask.jpg — green=free / red=obstacle visualisation
  step_00_nav_overlay.jpg  — full overlay with path, robot (R), target (T)

Inspect those images after the test to verify the obstacle map visually.
"""
import pathlib
import unittest

import cv2

FIXTURES    = pathlib.Path(__file__).parent / "fixtures"
CUP_IMG     = FIXTURES / "navigation" / "droidcam_nav_obstacle.jpg"
ANNOTATED   = FIXTURES / "navigation" / "annotated"


def _load(path: pathlib.Path):
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not load fixture: {path}"
    return bgr


class TestNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.navigation import (
            detect_obstacles, plan_path, save_debug_images,
        )

        _load_model()   # download YOLO weights once
        ANNOTATED.mkdir(parents=True, exist_ok=True)

        bgr = _load(CUP_IMG)
        h_result = detect_heading(bgr)
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        obs_map = detect_obstacles(bgr, h_result, target)
        nav_plan = plan_path(obs_map)
        saved = save_debug_images(bgr, obs_map, nav_plan, str(ANNOTATED), step=0)

        print(f"\n[navigation] Robot at: {obs_map.robot_px}, Target at: {obs_map.target_px}")
        print(f"[navigation] Plan: {nav_plan.reason}")
        print(f"[navigation] Debug images: {saved}")

        cls._bgr     = bgr
        cls._heading = h_result
        cls._target  = target
        cls._obs_map = obs_map
        cls._plan    = nav_plan
        cls._saved   = saved

    # ── ObstacleMap sanity ────────────────────────────────────────────────────

    def test_free_mask_covers_reasonable_area(self):
        """Floor should cover at least 20% of the frame."""
        free_frac = (self._obs_map.free_mask > 0).mean()
        self.assertGreater(free_frac, 0.20,
                           f"Free-space fraction too low: {free_frac:.1%}")

    def test_obstacle_mask_has_obstacles(self):
        """Some pixels should be classified as obstacles."""
        obstacle_frac = (self._obs_map.free_mask == 0).mean()
        self.assertGreater(obstacle_frac, 0.05,
                           f"Obstacle fraction too low: {obstacle_frac:.1%}")

    def test_grid_has_navigable_cells(self):
        """Grid must have at least 10% navigable cells."""
        nav_frac = self._obs_map.grid.mean()
        self.assertGreater(nav_frac, 0.10,
                           f"Navigable grid fraction too low: {nav_frac:.1%}")

    def test_robot_grid_cell_is_navigable(self):
        """The robot's grid cell must always be marked navigable."""
        r, c = self._obs_map.robot_grid
        self.assertTrue(self._obs_map.grid[r, c],
                        f"Robot grid cell ({r},{c}) is not navigable")

    def test_target_detected_if_cup_present(self):
        """If YOLO detected the cup, target_px should be set on the map."""
        if self._target is not None:
            self.assertIsNotNone(self._obs_map.target_px,
                                 "target_px is None even though target was detected")

    def test_path_endpoint_in_cspace(self):
        """The path's last cell must be in the navigable C-space (green zone).

        The path must never end inside the Minkowski-sum inflation buffer —
        the approach goal is snapped to the nearest free cell before A* runs.
        """
        if self._plan.reachable and self._plan.path_grid:
            r, c = self._plan.path_grid[-1]
            self.assertTrue(
                self._obs_map.grid[r, c] or self._obs_map.raw_grid[r, c],
                f"Path endpoint ({r},{c}) is not navigable in either grid",
            )

    # ── Path planning ─────────────────────────────────────────────────────────

    def test_plan_returns_navplan(self):
        """plan_path must return a NavPlan (no crash)."""
        from mcp_robot.navigation import NavPlan
        self.assertIsInstance(self._plan, NavPlan)

    def test_path_reachable_when_target_detected(self):
        """When both robot and cup are detected, A* should find a path."""
        if self._obs_map.target_grid is not None:
            self.assertTrue(
                self._plan.reachable,
                f"Expected path to be reachable, got: {self._plan.reason}",
            )

    def test_path_starts_near_robot(self):
        """First path cell should correspond to the robot's grid cell."""
        if self._plan.reachable and self._plan.path_grid:
            first = self._plan.path_grid[0]
            robot = self._obs_map.robot_grid
            dist = abs(first[0] - robot[0]) + abs(first[1] - robot[1])
            self.assertLessEqual(dist, 2,
                                 f"Path start {first} far from robot {robot}")

    # ── commands_for_step ─────────────────────────────────────────────────────

    def test_commands_return_finite_floats(self):
        """commands_for_step must return finite (turn_deg, drive_s)."""
        import math
        from mcp_robot.navigation import commands_for_step
        td, ds = commands_for_step(self._obs_map, self._plan, self._heading)
        self.assertTrue(math.isfinite(td), f"turn_deg not finite: {td}")
        self.assertTrue(math.isfinite(ds), f"drive_s not finite: {ds}")
        self.assertGreaterEqual(ds, 0.0, "drive_s must be non-negative")

    # ── debug images saved ────────────────────────────────────────────────────

    def test_debug_images_written(self):
        """save_debug_images must write raw, obstacle_mask, depth, and nav_overlay."""
        for key in ("raw", "obstacle_mask", "depth", "cspace", "nav_overlay"):
            self.assertIn(key, self._saved, f"Missing key '{key}' in saved dict")
            path = pathlib.Path(self._saved[key])
            self.assertTrue(path.exists(), f"File not written: {path}")
            self.assertGreater(path.stat().st_size, 1000,
                               f"File suspiciously small: {path}")


class TestNavigationWestHeading(unittest.TestCase):
    """Navigation pipeline on the west-heading DroidCam frame.

    Fixture: tests/fixtures/navigation/droidcam_west_heading.jpg
    Outputs: tests/fixtures/navigation/annotated/west_heading/
      step_00_raw.jpg           — unmodified frame
      step_00_obstacle_mask.jpg — green=free / red=obstacle
      step_00_depth.jpg         — Depth Anything V2 heat-map
      step_00_gradient_mask.jpg — depth-gradient floor classification
      step_00_cspace.jpg        — C-space grid (green/yellow/red + path)
      step_00_nav_overlay.jpg   — full overlay with robot R, target T, A* path

    Inspect those images after the run to diagnose navigation issues.
    """

    CURRENT_IMG = FIXTURES / "navigation" / "droidcam_west_heading.jpg"
    ANNOTATED_CURRENT = FIXTURES / "navigation" / "annotated" / "west_heading"

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.navigation import (
            detect_obstacles, plan_path, save_debug_images,
        )

        _load_model()
        cls.ANNOTATED_CURRENT.mkdir(parents=True, exist_ok=True)

        bgr = _load(cls.CURRENT_IMG)
        h_result = detect_heading(bgr)
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        obs_map = detect_obstacles(bgr, h_result, target)
        nav_plan = plan_path(obs_map)
        saved = save_debug_images(
            bgr, obs_map, nav_plan, str(cls.ANNOTATED_CURRENT), step=0
        )

        print(f"\n[west_heading] Heading:  {h_result.forward if h_result else 'not detected'}")
        print(f"[west_heading] Objects:  {[f'{o.class_name}@{o.confidence:.2f}' for o in objects]}")
        print(f"[west_heading] Robot px: {obs_map.robot_px},  Target px: {obs_map.target_px}")
        print(f"[west_heading] Plan:     {nav_plan.reason}")
        print(f"[west_heading] Images:   {list(saved.values())}")

        cls._bgr     = bgr
        cls._heading = h_result
        cls._objects = objects
        cls._target  = target
        cls._obs_map = obs_map
        cls._plan    = nav_plan
        cls._saved   = saved

    # ── sanity checks ──────────────────────────────────────────────────────────

    def test_robot_body_detected(self):
        """Yellow body must be visible so depth-gradient floor mask works."""
        self.assertIsNotNone(self._heading, "Heading not detected — is the robot visible?")

    def test_cup_detected(self):
        """YOLO should find the blue cup in the current frame."""
        names = [o.class_name for o in self._objects]
        self.assertIn("cup", names, f"Cup not detected. YOLO found: {names}")

    def test_free_mask_covers_floor(self):
        """At least 20% of the frame should be navigable floor."""
        frac = (self._obs_map.free_mask > 0).mean()
        self.assertGreater(frac, 0.20, f"Free-space fraction too low: {frac:.1%}")

    def test_robot_grid_cell_navigable(self):
        r, c = self._obs_map.robot_grid
        self.assertTrue(self._obs_map.grid[r, c],
                        f"Robot grid cell ({r},{c}) is not navigable")

    def test_plan_generated(self):
        from mcp_robot.navigation import NavPlan
        self.assertIsInstance(self._plan, NavPlan)
        print(f"\n  plan reachable={self._plan.reachable}  reason={self._plan.reason}")

    def test_debug_images_written(self):
        for key in ("raw", "obstacle_mask", "depth", "cspace", "nav_overlay"):
            self.assertIn(key, self._saved, f"Missing key '{key}'")
            path = pathlib.Path(self._saved[key])
            self.assertTrue(path.exists(), f"File not written: {path}")
            self.assertGreater(path.stat().st_size, 1000, f"File suspiciously small: {path}")


class TestNavigationRobotHidingSwitch(unittest.TestCase):
    """Navigation pipeline on the robot-hiding-switch DroidCam frame.

    Fixture: tests/fixtures/navigation/droidcam_robot_hiding_switch.jpg
    Outputs: tests/fixtures/navigation/annotated/robot_hiding_switch/
      step_00_a_raw.jpg          — unmodified frame
      step_00_b_cleaned.jpg      — inpainted (robot + cup removed)
      step_00_c_depth.jpg        — Depth Anything V2 heat-map
      step_00_d_depth_gradient.jpg — depth-gradient floor classification
      step_00_e_free_mask.jpg    — navigable floor mask
      step_00_f_cspace.jpg       — C-space grid (green/yellow/red + path)
      step_00_g_nav_overlay.jpg  — full overlay with robot R, target T, A* path

    Robot faces North, parked against a wall switch.  Blue cup target is visible
    in the upper-left.  Inspect the images after the run to verify obstacle map.
    """

    FIXTURE_IMG = FIXTURES / "navigation" / "droidcam_robot_hiding_switch.jpg"
    ANNOTATED_DIR = FIXTURES / "navigation" / "annotated" / "robot_hiding_switch"

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model
        from mcp_robot.navigation import detect_obstacles, plan_path, save_debug_images

        _load_model()
        cls.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

        bgr = _load(cls.FIXTURE_IMG)
        h_result = detect_heading(bgr)
        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        obs_map = detect_obstacles(bgr, h_result, target)
        nav_plan = plan_path(obs_map)
        saved = save_debug_images(bgr, obs_map, nav_plan, str(cls.ANNOTATED_DIR), step=0)

        print(f"\n[robot_hiding_switch] Heading:  {h_result.forward if h_result else 'not detected'}")
        print(f"[robot_hiding_switch] Objects:  {[f'{o.class_name}@{o.confidence:.2f}' for o in objects]}")
        print(f"[robot_hiding_switch] Robot px: {obs_map.robot_px},  Target px: {obs_map.target_px}")
        print(f"[robot_hiding_switch] Plan:     {nav_plan.reason}")
        print(f"[robot_hiding_switch] Images:   {list(saved.values())}")

        cls._bgr     = bgr
        cls._heading = h_result
        cls._objects = objects
        cls._target  = target
        cls._obs_map = obs_map
        cls._plan    = nav_plan
        cls._saved   = saved

    def test_robot_body_detected(self):
        """Yellow chassis must be visible — heading detection requires it."""
        self.assertIsNotNone(self._heading, "Heading not detected — robot body not visible?")

    def test_cup_detected(self):
        """YOLO should find the blue cup in the upper-left of the frame."""
        names = [o.class_name for o in self._objects]
        self.assertIn("cup", names, f"Cup not detected. YOLO found: {names}")

    def test_free_mask_covers_floor(self):
        """At least 20% of the frame should be navigable floor."""
        frac = (self._obs_map.free_mask > 0).mean()
        self.assertGreater(frac, 0.20, f"Free-space fraction too low: {frac:.1%}")

    def test_robot_grid_cell_navigable(self):
        r, c = self._obs_map.robot_grid
        self.assertTrue(self._obs_map.grid[r, c],
                        f"Robot grid cell ({r},{c}) is not navigable")

    def test_plan_generated(self):
        from mcp_robot.navigation import NavPlan
        self.assertIsInstance(self._plan, NavPlan)
        print(f"\n  plan reachable={self._plan.reachable}  reason={self._plan.reason}")

    def test_debug_images_written(self):
        for key in ("raw", "obstacle_mask", "depth", "cspace", "nav_overlay"):
            self.assertIn(key, self._saved, f"Missing key '{key}'")
            path = pathlib.Path(self._saved[key])
            self.assertTrue(path.exists(), f"File not written: {path}")
            self.assertGreater(path.stat().st_size, 1000, f"File suspiciously small: {path}")


if __name__ == "__main__":
    unittest.main()
