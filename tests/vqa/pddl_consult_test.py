#!/usr/bin/env python3
"""
Standalone simulator for the consult_vqa_for_pddl_domain MCP tool.

Lets you replay the exact prompt/image/domain combination sent to Gemini
without needing a connected robot. The question half of the prompt
(DEFAULT_QUESTION) and the extraction/formatting helpers are imported from
neurosymbolic_counselor (github.com/f1ren/NAPC) — the
same package server.py's consult_vqa_for_pddl_domain calls into — so there's
one canonical copy instead of two that can drift.

What's still worth testing from *this* repo specifically: this script calls
vision.ask_with_images(), lego-robot-mcp's real backend (Gemini quota-fallback
switching, Ollama fallback, config.py-driven backend selection) — behavior
the new package's own generic scripts/replay_consult.py knows nothing about
— against this project's real fixture images.

The default images and --context reproduce a real failed call (2026-07-02
10:59:25): robot facing a wall-mounted light switch at close range, blue cup
still not in view after a full 360° scan_for_target sweep found nothing.
Gemini's reply invented a "toggle-lights" action instead of flagging the
real gap (no action to reposition/search elsewhere once a full scan comes up
empty) — use this script to iterate on the prompt until it does. To actually
edit the wording, do it in NAPC's counselor.py and
reinstall (pip install -e, or -U if installed from git) before re-running.

Usage:
  GEMINI_API_KEY=<key> python3 tests/vqa/pddl_consult_test.py
  GEMINI_API_KEY=<key> GEMINI_MODEL=gemini-2.5-pro python3 tests/vqa/pddl_consult_test.py

  # Swap in a different failure scenario
  python3 tests/vqa/pddl_consult_test.py \\
      --front /path/to/pi_camera.jpg --external /path/to/simpleipcamera.jpg \\
      --context "The gripper closed on empty air; the cup was 5cm to the left of center." \\
      --domain pddl/robot_domain.pddl.bak \\
      --plan "(navigate loc-start loc-table)" --plan "(open-gripper)" \\
      --plan "(lower-arm)" --plan "(grasp cup loc-table)"

  # Write out the suggested domain and/or problem if the response contains them
  python3 tests/vqa/pddl_consult_test.py \\
      --save-domain /tmp/suggested_domain.pddl --save-problem /tmp/suggested_problem.pddl

  # Simulate consult_vqa_for_pddl_domain being called before any plan_pddl
  # call this session (no problem PDDL on record yet)
  python3 tests/vqa/pddl_consult_test.py --problem ""

  # Run the same prompt/images N times to check how consistently it lands on
  # the intended fix before treating the wording as settled. --expect checks
  # only the suggested domain/problem blocks, not the surrounding prose — a
  # response can name the right concept while describing the scene and still
  # ship a fix about something unrelated. It's a substring tally only (not
  # sent to the model) — keep it experiment-local rather than encoding it
  # into DEFAULT_QUESTION, which should stay general.
  python3 tests/vqa/pddl_consult_test.py --repeat 5 --expect SUBSTRING1 --expect SUBSTRING2
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES  = Path(__file__).parent.parent / "fixtures" / "pddl_consult"

# mcp-robot is `pip install -e`'d against wherever that command was originally
# run — typically the main checkout, not this worktree. Running this file
# directly (rather than via -m or pytest, both of which put the worktree root
# on sys.path automatically) would otherwise silently import the main
# checkout's mcp_robot instead of this one. Force it explicitly.
sys.path.insert(0, str(REPO_ROOT))
from mcp_robot import config, vision  # noqa: E402
from neurosymbolic_counselor.counselor import DEFAULT_QUESTION, build_prompt  # noqa: E402
from neurosymbolic_counselor.extraction import extract_pddl_domain, extract_pddl_problem  # noqa: E402

DEFAULT_DOMAIN   = REPO_ROOT / "pddl" / "robot_domain.pddl"
DEFAULT_FRONT    = FIXTURES / "pi_camera.jpg"
DEFAULT_EXTERNAL = FIXTURES / "droidcam.jpg"
DEFAULT_CONTEXT  = (
    "Task: lift the blue cup. get_robot_state found no cup in front camera view, "
    "and a full 360° scan_for_target sweep (YOLO class 'cup') also found nothing. "
    "External camera shows a wooden-floor room corner with no cup visible; front "
    "camera shows a wall-mounted light switch at close range."
)
# The actual problem_pddl in play for the 2026-07-02 10:59:25 incident above.
DEFAULT_PROBLEM = (
    "(define (problem lift-cup)\n"
    "  (:domain lego-robot)\n"
    "  (:objects loc-start loc-cup cup)\n"
    "  (:init\n"
    "    (robot-at loc-start)\n"
    "    (object-at cup loc-cup)\n"
    "    (gripper-empty)\n"
    "    (adjacent loc-start loc-cup)\n"
    "    (adjacent loc-cup loc-start)\n"
    "  )\n"
    "  (:goal (holding cup))\n"
    ")"
)


def load_image(path: Path) -> bytes:
    if not path.exists():
        sys.exit(f"Image not found: {path}")
    return path.read_bytes()


def print_call_summary(domain_path: Path, images: list[tuple[str, Path]], prompt: str) -> None:
    print("=" * 70)
    print(f"BACKEND: {config.VISION_BACKEND}  (GEMINI_MODEL={config.GEMINI_MODEL})")
    print(f"DOMAIN : {domain_path}")
    print("IMAGES:")
    for label, path in images:
        print(f"  [{label}] {path}  ({path.stat().st_size} bytes)")
    print()
    print("--- PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---")
    print("=" * 70)
    print()


def run_trial(
    index: int,
    total: int,
    prompt: str,
    labeled_bytes: list[tuple[str, bytes]],
    expect: list[str] | None,
    save_domain: str | None,
    save_problem: str | None,
) -> tuple[bool, bool | None]:
    """Run one vision.ask_with_images call; return (domain_or_problem_block_found, expect_hit_or_None)."""
    if total > 1:
        print(f"\n{'=' * 70}\nTRIAL {index}/{total}\n{'=' * 70}")

    b64_images = [(label, base64.b64encode(data).decode()) for label, data in labeled_bytes]
    print("Calling vision.ask_with_images... ", end="", flush=True)
    response = vision.ask_with_images(prompt, b64_images)
    print("done\n")

    print("--- RESPONSE ---")
    print(response)
    print("--- END RESPONSE ---")

    new_domain = extract_pddl_domain(response)
    domain_found = bool(new_domain)
    new_problem = extract_pddl_problem(response)
    problem_found = bool(new_problem)
    print()
    if domain_found:
        print("[domain] response contains a PDDL domain block.")
        if save_domain:
            path = Path(save_domain if total == 1 else f"{save_domain}.{index}")
            path.write_text(new_domain)  # type: ignore[arg-type]
            print(f"[domain] saved to {path}")
        else:
            print("[domain] pass --save-domain PATH to write it out.")
    else:
        print("[domain] no PDDL domain block found in the response.")

    if problem_found:
        print("[problem] response contains a PDDL problem block.")
        if save_problem:
            path = Path(save_problem if total == 1 else f"{save_problem}.{index}")
            path.write_text(new_problem)  # type: ignore[arg-type]
            print(f"[problem] saved to {path}")
        else:
            print("[problem] pass --save-problem PATH to write it out.")
    else:
        print("[problem] no PDDL problem block found in the response.")

    expect_hit: bool | None = None
    if expect:
        # Checked against the suggested domain/problem blocks only, not the
        # surrounding prose: the model can name the right concept while
        # describing the scene and still fail to encode it as an
        # action/predicate/object (e.g. it mentioned a switch in its
        # diagnosis, then shipped a fix about something else entirely) —
        # matching on prose would call that a hit.
        haystack = (new_domain or "") + "\n" + (new_problem or "")
        expect_hit = any(s.lower() in haystack.lower() for s in expect)
        print(f"[expect] {'HIT' if expect_hit else 'miss'} in suggested domain/problem — looked for: {', '.join(expect)}")

    return (domain_found or problem_found), expect_hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate consult_vqa_for_pddl_domain against saved images")
    ap.add_argument("--domain",   default=str(DEFAULT_DOMAIN), help="path to a robot_domain.pddl file")
    ap.add_argument("--front",    default=str(DEFAULT_FRONT), help="pi_camera (front) image")
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL), help="SimpleIPCamera (external) image")
    ap.add_argument("--context",  default=DEFAULT_CONTEXT, help="failure_context text")
    ap.add_argument("--plan", action="append", metavar="ACTION",
                     help="one grounded PDDL action from the plan being replayed, e.g. "
                          "'(navigate loc-start loc-table)'; repeatable, in plan order. "
                          "Omit to simulate consult_vqa_for_pddl_domain being called before "
                          "any plan_pddl call this session.")
    ap.add_argument("--problem", default=DEFAULT_PROBLEM, metavar="PDDL",
                     help="literal problem_pddl text (not a file path) from the most recent "
                          "plan_pddl call being replayed. Pass '' to simulate "
                          "consult_vqa_for_pddl_domain being called before any plan_pddl call "
                          "this session.")
    ap.add_argument("--save-domain", metavar="PATH", help="if a response contains a new PDDL domain, write it here")
    ap.add_argument("--save-problem", metavar="PATH", help="if a response contains a new PDDL problem, write it here")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                     help="call the model N times with the identical prompt/images and tally results")
    ap.add_argument("--expect", action="append", metavar="SUBSTRING",
                     help="case-insensitive substring to look for in the SUGGESTED PDDL DOMAIN/PROBLEM "
                          "blocks only (not the surrounding prose) — the model can name the right concept "
                          "in its diagnosis without encoding it as an action/predicate/object, so matching "
                          "on prose alone gives false positives. Repeatable; a trial counts as a hit if ANY "
                          "match. Purely a local grading aid — never sent to the model.")
    args = ap.parse_args()

    if not config.GEMINI_API_KEY and config.VISION_BACKEND != "ollama":
        sys.exit("GEMINI_API_KEY env var is not set (or set VISION_BACKEND=ollama to use a local model instead)")
    if args.repeat < 1:
        sys.exit("--repeat must be >= 1")

    domain_path = Path(args.domain)
    if not domain_path.exists():
        sys.exit(f"Domain file not found: {domain_path}")
    domain_text = domain_path.read_text()
    problem_text = args.problem or None

    front_path, external_path = Path(args.front), Path(args.external)
    images = [("pi_camera", front_path), ("simpleipcamera", external_path)]
    labeled_bytes = [(label, load_image(path)) for label, path in images]

    prompt = build_prompt(DEFAULT_QUESTION, args.context, domain_text, problem_text, args.plan)
    print_call_summary(domain_path, images, prompt)

    results = [
        run_trial(i, args.repeat, prompt, labeled_bytes, args.expect,
                  args.save_domain, args.save_problem)
        for i in range(1, args.repeat + 1)
    ]

    if args.repeat > 1:
        domain_hits = sum(1 for found, _ in results if found)
        print(f"\n{'=' * 70}\nSUMMARY over {args.repeat} trials\n{'=' * 70}")
        print(f"domain block found : {domain_hits}/{args.repeat}")
        if args.expect:
            expect_hits = sum(1 for _, hit in results if hit)
            print(f"expected substring  : {expect_hits}/{args.repeat}  (looked for: {', '.join(args.expect)})")


if __name__ == "__main__":
    main()
