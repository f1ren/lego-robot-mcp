#!/usr/bin/env python3
"""
Send PROMPT_TEXT + robot-rotating.gif to Gemini Pro and print the response.

Usage:
  GEMINI_API_KEY=<key> python gemini_pro_test.py
  GEMINI_API_KEY=<key> GEMINI_MODEL=gemini-2.5-pro python gemini_pro_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GIF_PATH       = Path(__file__).parent / "robot-rotating.gif"

PROMPT_TEXT = (
    """You are analysing a 4-motor Lego robot (left wheel, right wheel, arm, gripper). The gripper defines the robot's front.
ACTION COMMANDED: drive left=60 right=60 for 0.8s EXPECTED OUTCOME: Moving the
gripper directly over the paper ball to pick it up. Ball is ~15cm ahead.
Previous attempt at speed 40 for 0.5s barely moved (floor slippage). Goal:
gripper positioned just around the ball, ready to lower and close. Attached is a
video of camera captured during the action.
IMPORTANT: Describe turning direction as clockwise/counter-clockwise (viewed from above) or compass directions (north/south/east/west), not as 'left' or 'right'.
Reply in EXACTLY this format on two
lines: Verdict: YES | NO | PARTIAL —  Changes: <1-2 short sentences on what
actually happened during the motion>"""
)


def main() -> None:
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY env var is not set")
    if not GIF_PATH.exists():
        sys.exit(f"GIF not found: {GIF_PATH}")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("google-genai package not installed — run: pip install google-genai")

    gif_size = GIF_PATH.stat().st_size

    print("=" * 70)
    print(f"MODEL : {GEMINI_MODEL}")
    print(f"GIF   : {GIF_PATH}  ({gif_size:,} bytes)")
    print()
    print("--- PROMPT ---")
    print(PROMPT_TEXT)
    print("--- END PROMPT ---")
    print("=" * 70)
    print()

    client = genai.Client(api_key=GEMINI_API_KEY)

    print("Uploading GIF via Files API... ", end="", flush=True)
    uploaded = client.files.upload(
        file=GIF_PATH,
        config=types.UploadFileConfig(mime_type="image/gif"),
    )
    print(f"done ({uploaded.name})\n")

    max_attempts = 6
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        print(f"Calling Gemini (attempt {attempt}/{max_attempts})... ", end="", flush=True)
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[PROMPT_TEXT, uploaded],
            )
            print("done\n")
            break
        except Exception as exc:
            print(f"failed: {exc}")
            if attempt == max_attempts:
                sys.exit(f"All {max_attempts} attempts exhausted.")
            print(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2
    else:
        sys.exit("All attempts exhausted.")

    print("--- RESPONSE ---")
    print((resp.text or "").strip())
    print("--- END RESPONSE ---")


if __name__ == "__main__":
    main()
