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

    def test_target_grid_navigable_when_set(self):
        """Target grid cell must be made navigable for A* to reach it."""
        if self._obs_map.target_grid is not None:
            r, c = self._obs_map.target_grid
            self.assertTrue(self._obs_map.grid[r, c],
                            f"Target grid cell ({r},{c}) is not navigable")

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
        for key in ("raw", "obstacle_mask", "depth", "nav_overlay"):
            self.assertIn(key, self._saved, f"Missing key '{key}' in saved dict")
            path = pathlib.Path(self._saved[key])
            self.assertTrue(path.exists(), f"File not written: {path}")
            self.assertGreater(path.stat().st_size, 1000,
                               f"File suspiciously small: {path}")


if __name__ == "__main__":
    unittest.main()
