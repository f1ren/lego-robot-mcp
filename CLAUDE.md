# MCP Server for Lego Robot

It controls a 4 motor lego robot connected to a Raspberry Pi via the BuildHat HAT and OV5647 Pi Camera.

# Act
1. **After calling `plan_pddl` and before executing any action, describe what each camera shows:** gripper open or closed, arm high or low, object location relative to the robot, any clearance issues. Do not skip this step — motor position numbers alone are not sufficient.
   - **If a goal or action cannot be completed — including no detection after a full scan — do not immediately conclude the target is absent or the task impossible.** Look at the full camera frame and describe everything visible in the scene, not just the target's presence or absence: objects, surfaces, lighting, any element that reflects the state of the environment. Ask: what would need to be true for this action to succeed, and is anything visible in the scene an unmet prerequisite? If so, and no domain action addresses it, that is a domain gap — reflect on it and consider whether `pddl/robot_domain.pddl` needs a new action, before asking the user to intervene.
   - **Gripper "open" means wide-open**: the angle between the two fingers must be approximately **180 degrees** (fingers pointing in opposite directions). A partially-open gripper is NOT "open" — treat it as closed/partial, because it may not clear the object on approach and may fail to grab it once the jaws close. If the gripper is not at ~180°, plan to open it fully before attempting a grasp.
   - **Always open the gripper as the first step of any grasp action**, even if it visually appears already open. Visual assessment of gripper state is unreliable — the jaws can look open while actually being partially closed. Issuing an explicit open command before approach guarantees the jaws are wide enough to clear and grab the object. This overrides the "skip unnecessary steps" rule for grasps.
   - **Always include a "lower arm" step when planning to grab a ground-level object**, even if the arm visually appears already lowered. VQA models reliably miss subtle arm height differences — issuing an explicit lower command before approach guarantees the arm is at the correct height. This overrides the "skip unnecessary steps" rule for ground-level grasps.
2. **Skip unnecessary steps** Do you need to open the gripper, or is it already open? Do you need to lower the arm, or is it already on the ground? If you close the gripper to grab an object, is the object well located to be grabbed once the jaws close?
3. **Don't trust the MCP as is**. It is a work in progress and should keep on changing. Be skeptic. Did the robot move as expected? Does the motors, mechanincs, and code align as expected? Prefer the images and VQA model results over the sensors numerical input, as it is more robust and data rich.
   - **Never rely on absolute encoder values.** Motor encoder positions reset on power-cycle and drift over time — a stored position like "arm at -28°" is meaningless across sessions. Only use the encoder **delta** (change during a single action) to assess how much a motor moved. The VQA model's change description is the authoritative source of truth for what actually happened.
4. **Directional language:** when describing the robot's heading or turning direction, always use **clockwise / counter-clockwise** (viewed from above) or **compass directions** (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). Never say the robot "turned left" or "turned right" — those terms are ambiguous depending on the observer's perspective. "Left wheel" and "right wheel" are fine as hardware identifiers (the gripper defines the robot's front), but motion outcomes must use CW/CCW or compass terms.
5. **Before any `drive` call, identify the robot's current forward direction.** The gripper is the robot's front — "forward" means moving in the direction the gripper is currently pointing. After any turn the robot's heading changes, so always confirm from the external camera (or the last VQA description) which way the gripper is facing before choosing drive parameters. State this heading explicitly (e.g. "gripper is pointing north") before issuing the command.
   - **Trust the green forward-arrow overlay on external-camera frames.** External (DroidCam) frames are post-processed by `mcp_robot.heading` to overlay a long green arrow rooted at the gripper showing the robot's current "forward" direction. When the arrow is present, use it as the authoritative heading cue and check whether it points at the target. If no arrow is drawn, heading detection failed and you must infer the direction from the image yourself.
   - **Arrow is suppressed for arm/gripper actions.** The green arrow is NOT drawn on frames sent to VQA when evaluating `move_arm`, `control_gripper`, or `put` — because the arrow would obscure the arm and gripper, making it impossible to assess their state. The arrow IS present on frames for `drive`, `move_motor` (wheel ports), and on all frames shown directly to Claude (e.g. `get_robot_state`, `get_external_camera_image`) to support heading decisions.

6. **For any multi-step task (navigate + grasp, navigate + place, or any combination), call `plan_pddl` before calling `get_robot_state`.** Build a nominal problem.pddl from the task description alone — assume `robot-at loc-start`, object at its stated location, `gripper-empty`, and plausible adjacency pairs — then call `plan_pddl`. Do not observe first; getting the plan up front forces you to reason about what the domain says must happen, independently of what you can currently see.

   **After the nominal plan, observe:** call `get_robot_state` (see rule 1 above), then re-run `plan_pddl` with the actual observed initial state.

   Do not manually construct a navigate → open-gripper → lower-arm → pick-up sequence — let the planner emit it. Reserve direct motor tools for single corrective actions (e.g. a small heading adjustment) after the plan is already running.

7. **Always use `navigate_to` to move toward a target.** Never manually plan a `turn` + `drive` sequence for navigation — `navigate_to` handles turning, driving, and obstacle avoidance automatically. Reserve `turn` only for small heading corrections when already within arm's reach of the target.
   - **Always supply a non-empty `target_class_free_text` on the very first `navigate_to` call**, even when `target_class_yolo` is also set. Don't pass `""` and wait for a YOLO-only attempt to fail before adding a free-text fallback — that wastes a step and risks deriving the description later from a stale/annotated frame. Describe the target's color/shape/material as seen in the **raw camera frame**, never from a debug/overlay image (e.g. `step_NN_g_nav_overlay.jpg`'s obstacle-mask tint can make objects look the wrong color).

8. **When plan execution fails and you don't know how to proceed, call `consult_vqa_for_pddl_domain`.** Whenever `plan_pddl` returns an empty plan, a step fails (verdict NO/PARTIAL), or you are otherwise stuck with no clear path forward, call this tool instead of guessing. Do not attempt to diagnose whether the root cause is a domain gap — you can't reliably tell. Pass a 1–3 sentence `failure_context` describing what was tried and why it failed. The tool captures both cameras, attaches the current domain, and asks the VQA: *"What seems to be the problem? Is there anything missing from the domain formalization?"*
   - If the VQA response includes a new PDDL domain, the tool saves it over `pddl/robot_domain.pddl` (backing up the original to `robot_domain.pddl.bak`) and returns `domain_updated: true`.
   - When `domain_updated` is true, immediately re-call `plan_pddl` with the same problem PDDL — the planner will use the updated domain.
   - The updated domain is **not** git-committed; use `git restore pddl/robot_domain.pddl` to return to the original for repeat experiments.

## Tool use and Code synthesis
1. The agentic coder should prefer using the MCP server to control the robot, as it should be more reliable and consistent.
2. **When a tool returns "The user did not allow…", treat it as a deliberate user decision.** Do not attempt to enable the feature by modifying config, setting environment variables, or changing code. Accept the limitation and adapt the plan to work without it.
3. If no appropriate function or tool is missing, the agentic coder should modify the MCP server code itself while testing. For instance, if you want the robot to grasp something, and the function does not exist, you should add the function to the MCP server code and test it. Same goes for moving forward, turning and etc.
3. Inspect the logs of the MCP server for debugging. The logs are available at `mcp_robot/logs/mcp_server.log`.
4. If the action verdict is NO or PARTIAL, or a step produces an unexpected null result (e.g. a scan detects nothing despite the target being expected in the scene), stop and answer: Could there be a problem in the code? Is there an unmet environmental condition visible in the camera frames? Should you fix either before moving on?
   - If the failure or null result looks like a missing precondition, action, or effect rather than a code bug, edit `pddl/robot_domain.pddl` and re-run `plan_pddl` with the corrected domain before continuing.
5. If you need a new primitive function, or any kind of function that you belive will be useful in the future (forward, backward, etc.), code it first, verify it works, and then proceed.
6. **Prefer slower, longer motions over fast, short ones.** High speeds cause the robot to jitter and overshoot, making outcomes harder to control and verify. Slower and larger moves also produce clearer visual changes, making them easier for the Visual Temporal Reasoning model to assess correctly.
7. **Minimum motor speed is 15.** Never pass a speed below 15 to any motor tool. Below this threshold motion is too slow to be reliably detected by the CV pipeline, making it impossible to verify whether the action succeeded.
8. **Always fill `sub_observation` and `sub_action`** on every motor tool call (`drive`, `turn`, `move_arm`, `lower_arm`, `control_gripper`, `move_motor`, `navigate_to`, `scan_for_target`). These become video subtitles. Each must be ~4 words max. `sub_observation`: what was just observed or what the user asked (e.g. "Cup detected ahead"). `sub_action`: what the robot is doing right now (e.g. "Driving toward cup").

## Experience Memory Workflow

This project uses `mcp-memory-service` (`experience-memory` MCP server) to accumulate robot learnings across sessions. The DB lives at `memory/experiences.db`.

**Before any task or code change:** call `memory_search` with keywords relevant to what you're about to do (e.g., `"gripper close"`, `"arm calibration"`, `"drive forward"`). For each returned experience that references specific code (a function, constant, or behaviour):
1. Run `git log --oneline -1` to get the current commit hash.
2. Compare the experience's `commit_hash` field against the current hash. If they differ, grep/read the referenced file+function to verify the logic described still exists as written.
3. If the logic has changed or been removed, **delete the experience** with `memory_delete` before relying on it. If it still applies but the code evolved, **update** it with the new commit hash.

**When storing an experience:** always include in `content`:
- `commit_hash`: output of `git rev-parse --short HEAD` at time of storing
- `validation_version`: a short description of what code state was validated (e.g. `"server.py::_ACTION_VIDEO_FPS=5.0"`)

**When changing code:** if a past experience informed the change, add an inline comment on the changed line(s) with the experience ID and the lesson — e.g. `# exp:abc123 — gripper stalls above 50% speed when arm is extended`. Include the same ID in the commit message. After committing, search for experiences that reference the changed file/function and either update them with the new commit hash or delete them if the lesson no longer applies.

**After every task, failure, code fix, or user feedback:** call `memory_store` with:
- `content`: one clear paragraph — what you tried, what happened, what you learned — plus `commit_hash` and `validation_version` fields
- `tags`: robot body (e.g. `3-wheel-gripper`) + component (`gripper`, `arm`, `drive`, `camera`, `vision`, `buildhat`) + event (`failure`, `success`, `code_fix`, `feedback`)
- `memory_type`: `"learning"` (lesson from outcome), `"error"` (failure + root cause), `"observation"` (factual discovery), or `"decision"` (deliberate design choice)

**Consolidation:** periodically run `memory_consolidate` with `action="run"` and `time_horizon="weekly"` to cluster and compress accumulated entries into higher-level patterns.

**Skill extraction:** when you have enough consolidated learnings, use `memory_list` to pull recent memories and synthesize the patterns. Prefer encoding the lesson as **code** (more precise, cheaper to run). Only create a **skill** when code can't solve it — i.e. when the lesson is a playbook: a class of situations requiring judgment, reflection, or a sequence of code changes rather than a single repeatable action.


## Local VQA (Ollama / Qwen) — Human Review Pause

When `VISION_BACKEND=ollama` is set, the local Qwen model is used for VQA instead of Gemini. After every motor-action tool call (`drive`, `drive_degrees`, `move_arm`, `control_gripper`, `put`, `move_motor`):

1. **Stop.** Do not proceed to the next planned action.
2. **Ask:** "Please review the log — should I proceed?"
3. **Wait for the user's confirmation** before issuing any further motor commands.

Skip the pause when using Gemini (`VISION_BACKEND=gemini` or `VISION_BACKEND=auto` with Gemini succeeding).

## Technical Details

1. The MCP runs in virtual environment.
2. The MCP server hot-reloads on code changes.
3. The MCP server uses persistent SSH connection to the RPi via `paramiko`.