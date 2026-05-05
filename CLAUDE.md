# MCP Server for Lego Robot

It controls a 4 motor lego robot connected to a Raspberry Pi via the BuildHat HAT and OV5647 Pi Camera.

Don't trust the MCP as is. It is a work in progress and should keep on changing. Be skeptic. Did the robot move as expected? Does the motors, mechanincs, and code align as expected? Prefer the vision model results over the sensors input, as it is more robust and data rich.

## Tool use and Code synthesis
1. The agentic coder should prefer using the MCP server to control the robot, as it should be more reliable and consistent.
2. If no appropriate function or tool is missing, the agentic coder should modify the MCP server code itself while testing. For instance, if you want the robot to grasp something, and the function does not exist, you should add the function to the MCP server code and test it.
3. Inspect the logs of the MCP server for debugging. The logs are available at `mcp_robot/logs/mcp_server.log`.
4. If the action verdict is NO or PARTIAL, stop and answer this question: Could there be a problem in the code? Should you fix it before moving on?
5. If you need a new primitive function, or any kind of function that you belive will be useful in the future (forward, backward, etc.), code it first, verify it works, and then proceed.
6. When trying new actions or evaluting the API, start with slower and shorter motions.

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


## Technical Details

1. The MCP runs in virtual environment.
2. The MCP server hot-reloads on code changes.
3. The MCP server uses persistent SSH connection to the RPi via `paramiko`.