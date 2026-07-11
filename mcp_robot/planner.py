"""
PDDL task planner — thin wrapper around pyperplan.

To switch to unified-planning, replace _solve_pddl only.  Everything else
(the domain file, the MCP tool, the action-name format) stays the same.
"""
from __future__ import annotations

import os
import tempfile

_PDDL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pddl",
)
DOMAIN_PATH = os.path.join(_PDDL_DIR, "robot_domain.pddl")

# Experimental override written by consult_vqa_for_pddl_domain. When present,
# solve() uses this instead of DOMAIN_PATH so the original, git-tracked domain
# is never mutated. Gitignored; delete the file to fall back to DOMAIN_PATH.
DOMAIN_FIXED_PATH = os.path.join(_PDDL_DIR, "robot_domain_fixed.pddl")


def solve(problem_pddl: str) -> list[str]:
    """
    Plan from the robot domain + a caller-supplied problem string.

    Uses DOMAIN_FIXED_PATH in place of DOMAIN_PATH when it exists.

    Returns a list of grounded action strings, e.g.
        ["(grasp cup loc-table)", "(navigate loc-table loc-sink)", ...]
    Returns an empty list when no plan exists.

    Raises ImportError if pyperplan is not installed.
    Raises RuntimeError if the domain file is missing.
    """
    domain_path = DOMAIN_FIXED_PATH if os.path.exists(DOMAIN_FIXED_PATH) else DOMAIN_PATH
    if not os.path.exists(domain_path):
        raise RuntimeError(f"PDDL domain file not found: {domain_path}")

    with open(domain_path) as f:
        domain_pddl = f.read()

    return _solve_pddl(domain_pddl, problem_pddl)


def _solve_pddl(domain: str, problem: str) -> list[str]:
    """Run pyperplan on domain+problem strings; return grounded action names."""
    from pyperplan.planner import search_plan, SEARCHES, HEURISTICS  # type: ignore[import]

    with tempfile.TemporaryDirectory() as tmp:
        domain_path = os.path.join(tmp, "domain.pddl")
        problem_path = os.path.join(tmp, "problem.pddl")
        with open(domain_path, "w") as f:
            f.write(domain)
        with open(problem_path, "w") as f:
            f.write(problem)

        plan = search_plan(
            domain_path,
            problem_path,
            SEARCHES["astar"],
            HEURISTICS["hadd"],
        )

    if plan is None:
        return []
    return [op.name for op in plan]
