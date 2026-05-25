# Grab skill

Execute a complete grab sequence for the target object described in $ARGUMENTS (e.g. `/grab the red ball`).

## Step 1 — Scene assessment (mandatory, never skip)

Call `get_external_camera_image` and `get_robot_state`.

Describe out loud:
- Gripper state: open (~180° between fingers) or not?
- Arm height: lowered (near ground) or raised?
- Object location relative to the robot: far, near, touching front, touching the green arrow?
- Current heading (which way does the green arrow point)?

## Step 2 — Grasp-readiness gate

The object is **ready to be grabbed** only when BOTH of the following are true simultaneously:
1. The object is at least **touching the robot's front body** (gripper end).
2. The green forward-arrow is **well over the object** — not merely touching its edge, but clearly passing through or covering it in the external camera frame.

**If either condition is unmet → do NOT close the gripper.** Instead go to Step 3.
**If both conditions are met → skip to Step 4.**

## Step 3 — Navigate into grab position (repeat until gate passes)

1. Confirm current heading from the green arrow.
2. Drive forward in small increments toward the object.
3. After each drive call, call `get_external_camera_image` and re-evaluate the grasp-readiness gate (Step 2).
4. If the object is off to one side, make a small heading correction first, then drive forward.
5. Repeat until both conditions in Step 2 are satisfied.

## Step 4 — Close gripper

Call `control_gripper action="close"`.

Check the returned `change_description`:
- **Verdict YES / PARTIAL with object grasped** → proceed.
- **Verdict NO or object not grasped** → open gripper, reassess scene (Step 1), replan.

## Step 5 — Lift confirmation

Call `move_arm` to raise the arm slightly and confirm the object is lifted with it. If it falls, report failure and ask the user how to proceed.
