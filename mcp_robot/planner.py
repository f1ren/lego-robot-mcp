"""
PDDL task planner — thin wrapper around pyperplan.

To switch to unified-planning, replace _solve_pddl only.  Everything else
(the domain file, the MCP tool, the action-name format) stays the same.
"""
from __future__ import annotations

import os
import tempfile

DOMAIN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pddl", "robot_domain.pddl",
)


def solve(problem_pddl: str) -> list[str]:
    """
    Plan from the fixed robot domain + a caller-supplied problem string.

    Returns a list of grounded action strings, e.g.
        ["(pick-up cup loc-table)", "(navigate loc-table loc-sink)", ...]
    Returns an empty list when no plan exists.

    Raises ImportError if pyperplan is not installed.
    Raises RuntimeError if the domain file is missing.
    """
    if not os.path.exists(DOMAIN_PATH):
        raise RuntimeError(f"PDDL domain file not found: {DOMAIN_PATH}")

    with open(DOMAIN_PATH) as f:
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
