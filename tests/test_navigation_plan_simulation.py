"""
Closed-loop simulation of `navigate_to`'s per-step command generation.

Loads the droidcam_robot_hiding_switch.jpg fixture (robot facing North, blue
cup target upper-left — see TestNavigationRobotHidingSwitch in
test_navigation.py), runs the real detect_obstacles -> plan_path pipeline to
get an actual planned route, then dead-reckons a trajectory by repeatedly
calling `commands_for_step` exactly the way `navigate_to` does — turn, drive,
update position, replan on deviation — without any camera or robot connection.

`commands_for_step` now returns wheel-encoder degrees (drive_degrees(), an
exact, repeatable distance) rather than a fixed duration, converted from the
pixel distance to the next waypoint via the robot's apparent body size as a
ruler (mcp_robot.navigation.mm_per_px). The simulation inverts that same
conversion to dead-reckon a pixel trajectory, so — unlike a duration-based
simulation, which could only guess at a px/s travel speed — this is exact: it
uses the identical mm<->px calibration as the production code.

The simulated trajectory is plotted over the originally-planned path and saved
to tests/fixtures/navigation/annotated/plan_simulation/sim_trajectory.jpg for
visual inspection. This is meant to catch regressions in the turn/drive
command synthesis that would otherwise only show up live as "the robot starts
fine, then wanders off with drastic turns".
"""
import math
import pathlib
import unittest

import cv2
import numpy as np

from mcp_robot import config
from mcp_robot.heading import Heading
from mcp_robot.navigation import (
    commands_for_step, detect_obstacles, dist_to_path, near_target, plan_path,
    update_robot_position,
)

FIXTURES    = pathlib.Path(__file__).parent / "fixtures"
FIXTURE_IMG = FIXTURES / "navigation" / "droidcam_robot_hiding_switch.jpg"
ANNOTATED   = FIXTURES / "navigation" / "annotated" / "plan_simulation"

_GRID_ROWS, _GRID_COLS = 60, 80          # must match navigation._GRID_ROWS/_GRID_COLS

_WHEEL_CIRCUMFERENCE_MM = math.pi * config.WHEEL_DIAMETER_MM
_SIM_MAX_STEPS = 40


def _simulate(obs_map, plan, start_heading_deg, body_area_px):
    """Dead-reckon a trajectory by replaying `commands_for_step` in a loop,
    mirroring `navigate_to`'s turn -> drive -> update-position -> replan cycle.

    `body_area_px` is the yellow-body pixel area observed in the fixture
    frame (Heading.body_area); it both feeds commands_for_step's mm
    calibration (via the simulated Heading) and is reused here, identically,
    to convert the resulting wheel-encoder degrees back into a pixel
    distance — the inverse of the same mm<->px relationship, so the
    dead-reckoned trajectory is exact rather than assumed.

    Returns (steps, trajectory, final_plan, n_replans):
      steps      -- list of (turn_deg, drive_deg) commands issued, in order
      trajectory -- list of (x, y) pixel positions visited, including the start
    """
    mm_per_px = math.sqrt(config.ROBOT_BODY_AREA_MM2 / body_area_px)

    heading_deg = start_heading_deg
    trajectory = [obs_map.robot_px]
    steps = []
    n_replans = 0

    for _ in range(_SIM_MAX_STEPS):
        if near_target(obs_map) or not plan.reachable or len(plan.path_px) < 2:
            break

        forward = (math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg)))
        sim_heading = Heading(obs_map.robot_px, obs_map.robot_px, forward, body_area_px, 0)

        turn_deg, drive_deg, reverse = commands_for_step(obs_map, plan, sim_heading)
        if turn_deg == 0.0 and drive_deg == 0.0:
            break
        steps.append((turn_deg, drive_deg, reverse))

        heading_deg += turn_deg
        rad = math.radians(heading_deg)
        travel_mm = (drive_deg / 360.0) * _WHEEL_CIRCUMFERENCE_MM
        travel_px = travel_mm / mm_per_px
        if reverse:
            travel_px = -travel_px
        new_px = (
            int(round(obs_map.robot_px[0] + travel_px * math.cos(rad))),
            int(round(obs_map.robot_px[1] + travel_px * math.sin(rad))),
        )

        deviation = dist_to_path(new_px, plan.path_px)
        update_robot_position(obs_map, new_px)
        trajectory.append(new_px)

        if deviation > obs_map.robot_radius_px:
            plan = plan_path(obs_map, sim_heading)
            n_replans += 1

    return steps, trajectory, plan, n_replans


def _plot(obs_map, planned_path_px, trajectory, out_path):
    """Overlay the originally-planned path (blue) and the simulated,
    dead-reckoned trajectory (green, brightening from start to end) on a
    top-down view of the obstacle grid, and write it to `out_path`."""
    h, w = obs_map.h, obs_map.w
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    obstacle_mask = cv2.resize(
        (~obs_map.raw_grid).astype(np.uint8) * 255,
        (w, h), interpolation=cv2.INTER_NEAREST,
    )
    canvas[obstacle_mask > 0] = (200, 200, 255)

    if len(planned_path_px) > 1:
        pts = np.array(planned_path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (255, 100, 0), 2, cv2.LINE_AA)

    n = len(trajectory)
    for i, pt in enumerate(trajectory):
        brightness = int(80 + 175 * (i + 1) / max(n, 1))
        cv2.circle(canvas, pt, 5, (0, brightness, 0), -1)
        if i > 0:
            cv2.line(canvas, trajectory[i - 1], pt, (0, brightness, 0), 2, cv2.LINE_AA)

    sx, sy = trajectory[0]
    cv2.circle(canvas, (sx, sy), 10, (0, 220, 220), 2)
    cv2.putText(canvas, "S", (sx - 5, sy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 220, 220), 1, cv2.LINE_AA)

    if obs_map.target_px is not None:
        tx, ty = obs_map.target_px
        cv2.circle(canvas, (tx, ty), 10, (0, 165, 255), 2)
        cv2.putText(canvas, "T", (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 1, cv2.LINE_AA)

    label = "blue = planned path   green = simulated (dead-reckoned) trajectory"
    cv2.rectangle(canvas, (4, 4), (len(label) * 7 + 8, 24), (0, 0, 0), cv2.FILLED)
    cv2.putText(canvas, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


class TestNavigationPlanSimulation(unittest.TestCase):
    """Replays the route planned for the robot_hiding_switch fixture through
    `commands_for_step` with no camera or robot, and checks that the
    resulting dead-reckoned trajectory actually makes progress toward the
    target instead of wandering off."""

    @classmethod
    def setUpClass(cls):
        from mcp_robot.heading import detect_heading
        from mcp_robot.grasp_readiness import _yolo_detect, _pick_target, _load_model

        _load_model()
        ANNOTATED.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(FIXTURE_IMG))
        assert bgr is not None, f"Could not load fixture: {FIXTURE_IMG}"

        h_result = detect_heading(bgr)
        assert h_result is not None, "robot_hiding_switch fixture should have a detectable heading"

        objects = _yolo_detect(bgr, target_class="cup")
        target = (
            _pick_target(objects, h_result)
            if (objects and h_result)
            else (max(objects, key=lambda o: o.confidence) if objects else None)
        )

        obs_map = detect_obstacles(bgr, h_result, target)
        plan = plan_path(obs_map, h_result)
        assert plan.reachable, f"robot_hiding_switch fixture plan should be reachable: {plan.reason}"

        cls._initial_path_px = list(plan.path_px)
        start_px, target_px = obs_map.robot_px, obs_map.target_px
        cls._target_px = target_px
        cls._initial_dist = math.hypot(start_px[0] - target_px[0], start_px[1] - target_px[1])

        # Use the fixture's actual detected heading and body area as the
        # simulation's starting orientation/calibration — this is exactly
        # what `navigate_to` would have seen and used for this frame.
        start_heading_deg = math.degrees(math.atan2(h_result.forward[1], h_result.forward[0]))
        steps, trajectory, _, n_replans = _simulate(
            obs_map, plan, start_heading_deg, h_result.body_area
        )

        cls._steps = steps
        cls._trajectory = trajectory
        cls._final_obs_map = obs_map
        cls._n_replans = n_replans

        cls._plot_path = ANNOTATED / "sim_trajectory.jpg"
        _plot(obs_map, cls._initial_path_px, trajectory, cls._plot_path)

        print(f"\n[plan_simulation] heading={h_result.forward} "
              f"({start_heading_deg:.1f}\N{DEGREE SIGN}),  plan: {plan.reason}")
        print(f"[plan_simulation] {len(steps)} steps, {n_replans} replans, "
              f"reached_target={near_target(obs_map)}")
        print(f"[plan_simulation] turn/drive sequence (turn_deg, wheel_deg, reverse): "
              f"{[(round(t, 1), round(d, 0), r) for t, d, r in steps]}")
        print(f"[plan_simulation] plot saved to {cls._plot_path}")

    def test_simulation_produced_steps(self):
        """commands_for_step must actually drive the robot toward the route."""
        self.assertGreater(len(self._steps), 0,
                           "commands_for_step never produced a move on the planned route")

    def test_commands_are_finite(self):
        """Sanity: every issued (turn_deg, drive_deg, reverse) must be a finite, valid command."""
        for turn_deg, drive_deg, reverse in self._steps:
            self.assertTrue(math.isfinite(turn_deg), f"non-finite turn_deg: {turn_deg}")
            self.assertTrue(math.isfinite(drive_deg), f"non-finite drive_deg: {drive_deg}")
            self.assertGreaterEqual(drive_deg, 0.0, f"negative drive_deg: {drive_deg}")
            self.assertIsInstance(reverse, bool, f"reverse not a bool: {reverse}")

    def test_trajectory_converges_on_target(self):
        """The dead-reckoned trajectory must end up closer to the target than
        it started. A 'wanders off' regression would instead leave the robot
        as far from — or farther from — the target than at the outset."""
        end_px = self._trajectory[-1]
        end_dist = math.hypot(end_px[0] - self._target_px[0], end_px[1] - self._target_px[1])
        self.assertLess(
            end_dist, self._initial_dist,
            f"Simulated trajectory ended {end_dist:.0f}px from the target — no closer "
            f"than the {self._initial_dist:.0f}px starting distance. "
            f"commands_for_step may be issuing wrong turn/drive values; "
            f"see {self._plot_path}"
        )

    def test_reaches_target(self):
        """A correct closed-loop replay should reach the C-space buffer distance
        of the target within the step budget — the literal 'does navigation
        arrive' check that a 'wanders off with drastic turns' bug would fail."""
        self.assertTrue(
            near_target(self._final_obs_map),
            f"Simulated robot never reached the target after {len(self._steps)} steps "
            f"({self._n_replans} replans); final position {self._trajectory[-1]}, "
            f"target {self._target_px}. Inspect {self._plot_path} for the trajectory."
        )


if __name__ == "__main__":
    unittest.main()
