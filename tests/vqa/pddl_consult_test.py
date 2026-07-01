#!/usr/bin/env python3
"""
Standalone simulator for the consult_vqa_for_pddl_domain MCP tool.

Lets you replay the exact prompt/image/domain combination sent to Gemini
without needing a connected robot. QUESTION_TEXT is imported from
mcp_robot.vision.CONSULT_DOMAIN_QUESTION — the same constant the production
tool uses — so edit it there and re-run this script until the model actually
spots the missing domain element; the fix then applies to both at once
(the MCP server hot-reloads on code changes).

The default images and --context reproduce a real failed call (2026-06-30
16:09:56): robot wedged against a cabinet, blue cup not in view, scanning
disabled. Gemini's reply invented a "raise-arm" action instead of flagging
the real gap (no way to search/reposition when scanning is off) — use this
script to iterate on the prompt until it does.

Usage:
  GEMINI_API_KEY=<key> python3 tests/vqa/pddl_consult_test.py
  GEMINI_API_KEY=<key> GEMINI_MODEL=gemini-2.5-pro python3 tests/vqa/pddl_consult_test.py

  # Swap in a different failure scenario
  python3 tests/vqa/pddl_consult_test.py \\
      --front /path/to/pi_camera.jpg --external /path/to/droidcam.jpg \\
      --context "The gripper closed on empty air; the cup was 5cm to the left of center." \\
      --domain pddl/robot_domain.pddl.bak

  # Write out the suggested domain if the response contains one
  python3 tests/vqa/pddl_consult_test.py --save-domain /tmp/suggested_domain.pddl

  # Run the same prompt/images N times to check how consistently it lands on
  # the intended fix before treating the wording as settled. --expect is a
  # substring tally only (not sent to the model) — keep it experiment-local
  # rather than encoding it into CONSULT_DOMAIN_QUESTION, which should stay
  # general.
  python3 tests/vqa/pddl_consult_test.py --repeat 5 --expect SUBSTRING1 --expect SUBSTRING2
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES  = Path(__file__).parent.parent / "fixtures" / "pddl_consult"

# mcp-robot is `pip install -e`'d against wherever that command was originally
# run — typically the main checkout, not this worktree. Running this file
# directly (rather than via -m or pytest, both of which put the worktree root
# on sys.path automatically) would otherwise silently import the main
# checkout's mcp_robot instead of this one. Force it explicitly so editing
# CONSULT_DOMAIN_QUESTION in *this* worktree is what actually gets picked up.
sys.path.insert(0, str(REPO_ROOT))
from mcp_robot.vision import _CAMERA_LABELS, CONSULT_DOMAIN_QUESTION as QUESTION_TEXT  # noqa: E402

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-robotics-er-1.6-preview")

DEFAULT_DOMAIN   = REPO_ROOT / "pddl" / "robot_domain.pddl"
DEFAULT_FRONT    = FIXTURES / "pi_camera.jpg"
DEFAULT_EXTERNAL = FIXTURES / "droidcam.jpg"
DEFAULT_CONTEXT  = (
    "The robot needs to lift a blue cup but cannot see it in the current camera "
    "frame. Scanning by rotation is disabled. Neither the external camera nor "
    "the front camera shows the blue cup — the robot is near a wall and the cup "
    "is somewhere in the room but not in view."
)


def build_prompt(question: str, context: str, domain_text: str) -> str:
    """Mirrors the prompt assembly in consult_vqa_for_pddl_domain (server.py)."""
    return (
        f"{question}\n\n"
        f"CONTEXT: {context}\n"
        "Attached are the images from the robot's view and the external camera. "
        "Both were considered, alongside the following PDDL domain:\n\n"
        f"{domain_text}"
    )


def extract_pddl_domain(text: str) -> str | None:
    """Copied verbatim from mcp_robot/server.py::_extract_pddl_domain."""
    m = re.search(r'\(define\s+\(domain\b', text, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def load_image(path: Path) -> bytes:
    if not path.exists():
        sys.exit(f"Image not found: {path}")
    return path.read_bytes()


def print_call_summary(model: str, domain_path: Path, images: list[tuple[str, Path]], prompt: str) -> None:
    print("=" * 70)
    print(f"MODEL : {model}")
    print(f"DOMAIN: {domain_path}")
    print("IMAGES:")
    for label, path in images:
        print(f"  [{label}] {path}  ({path.stat().st_size} bytes)")
    print()
    print("--- PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---")
    print("=" * 70)
    print()


def call_gemini(model: str, prompt: str, labeled_images: list[tuple[str, bytes]]) -> str:
    from google.genai import types

    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    parts: list = [types.Part.from_text(text=prompt)]
    for label, image_bytes in labeled_images:
        desc = _CAMERA_LABELS.get(label, label)
        parts.append(types.Part.from_text(text=f"\n=== {desc} ==="))
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

    max_attempts = 4
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        print(f"Calling Gemini (attempt {attempt}/{max_attempts})... ", end="", flush=True)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
            )
            print("done\n")
            return (resp.text or "").strip()
        except Exception as exc:
            print(f"failed: {exc}")
            if attempt == max_attempts:
                sys.exit(f"All {max_attempts} attempts exhausted.")
            print(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def run_trial(
    index: int,
    total: int,
    model: str,
    prompt: str,
    labeled_bytes: list[tuple[str, bytes]],
    expect: list[str] | None,
    save_domain: str | None,
) -> tuple[bool, bool | None]:
    """Run one Gemini call; return (domain_block_found, expect_hit_or_None)."""
    if total > 1:
        print(f"\n{'=' * 70}\nTRIAL {index}/{total}\n{'=' * 70}")

    response = call_gemini(model, prompt, labeled_bytes)

    print("--- RESPONSE ---")
    print(response)
    print("--- END RESPONSE ---")

    new_domain = extract_pddl_domain(response)
    domain_found = bool(new_domain)
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

    expect_hit: bool | None = None
    if expect:
        expect_hit = any(s.lower() in response.lower() for s in expect)
        print(f"[expect] {'HIT' if expect_hit else 'miss'} — looked for: {', '.join(expect)}")

    return domain_found, expect_hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate consult_vqa_for_pddl_domain against saved images")
    ap.add_argument("--domain",   default=str(DEFAULT_DOMAIN), help="path to a robot_domain.pddl file")
    ap.add_argument("--front",    default=str(DEFAULT_FRONT), help="pi_camera (front) image")
    ap.add_argument("--external", default=str(DEFAULT_EXTERNAL), help="droidcam (external) image")
    ap.add_argument("--context",  default=DEFAULT_CONTEXT, help="failure_context text")
    ap.add_argument("--save-domain", metavar="PATH", help="if a response contains a new PDDL domain, write it here")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                     help="call Gemini N times with the identical prompt/images and tally results")
    ap.add_argument("--expect", action="append", metavar="SUBSTRING",
                     help="case-insensitive substring to look for in each response, for a quick tally "
                          "across --repeat trials (repeatable; a trial counts as a hit if ANY match). "
                          "Purely a local grading aid — never sent to the model.")
    args = ap.parse_args()

    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY env var is not set")
    if args.repeat < 1:
        sys.exit("--repeat must be >= 1")

    domain_path = Path(args.domain)
    if not domain_path.exists():
        sys.exit(f"Domain file not found: {domain_path}")
    domain_text = domain_path.read_text()

    front_path, external_path = Path(args.front), Path(args.external)
    images = [("pi_camera", front_path), ("droidcam", external_path)]
    labeled_bytes = [(label, load_image(path)) for label, path in images]

    prompt = build_prompt(QUESTION_TEXT, args.context, domain_text)
    print_call_summary(GEMINI_MODEL, domain_path, images, prompt)

    results = [
        run_trial(i, args.repeat, GEMINI_MODEL, prompt, labeled_bytes, args.expect, args.save_domain)
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
