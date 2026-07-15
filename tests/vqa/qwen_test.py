#!/usr/bin/env python3
"""
Standalone Qwen vision tester.

Usage examples:

  # Before/after stills (the fallback path used when streaming is unavailable)
  python qwen_test.py --mode stills \
      --before /tmp/lego-robot-snapshots/before_pi_camera.jpg \
      --after  /tmp/lego-robot-snapshots/after_pi_camera.jpg \
      --action "drive left=40 right=-40 for 1s" \
      --expected "robot turns clockwise"

  # Action video (the main path — pass the whole action_video_* folder)
  python qwen_test.py --mode video \
      --folder /tmp/lego-robot-snapshots/action_video_20260504_134200 \
      --action "drive left=40 right=-40 for 1s" \
      --expected "robot turns clockwise"

  # Clip VQA (used by capture_front/external_video_clip)
  python qwen_test.py --mode clip \
      --folder /tmp/lego-robot-snapshots/action_video_20260504_134200 \
      --camera simpleipcamera

  # Pick the latest action_video_* folder automatically
  python qwen_test.py --mode video --latest

Environment variables (mirrors mcp_robot/config.py defaults):
  OLLAMA_HOST   http://localhost:11434
  OLLAMA_MODEL  qwen2.5vl

Note: _with_change_analysis no longer writes action_video_* snapshot folders
automatically (replaced by the continuous SegmentRecorder in
mcp_robot/recorder.py). The --folder/--latest options below still work against
any action_video_* folders captured manually or by older runs, but new ones
are not produced.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl")

# ── prompts (copied verbatim from vision.py) ──────────────────────────────────

_PROMPT = (
    "You are analysing a 4-motor Lego robot (left wheel, right wheel, arm, "
    "gripper). The gripper defines the robot's front.\n"
    "ACTION COMMANDED: {action}\n"
    "EXPECTED OUTCOME: {expected}\n\n"
    "Below are BEFORE images followed by AFTER images, each labelled by the "
    "camera they came from (pi_camera = robot's front-mounted camera, "
    "simpleipcamera = third-person view).\n\n"
    "IMPORTANT: Describe turning direction as clockwise/counter-clockwise "
    "(viewed from above) or compass directions (north/south/east/west), "
    "not as 'left' or 'right'.\n\n"
    "Reply in EXACTLY this format on two lines:\n"
    "Verdict: YES | NO | PARTIAL — <one short clause justifying the verdict>\n"
    "Changes: <1-2 short sentences describing what actually changed; if "
    "nothing visible changed, say so explicitly>"
)

_VIDEO_PROMPT = (
    "You are analysing {n_frames} sequential frames captured during a 4-motor "
    "Lego robot action (left wheel, right wheel, arm, gripper). The gripper is "
    "the front.\n"
    "ACTION COMMANDED: {action}\n"
    "EXPECTED OUTCOME: {expected}\n\n"
    "The frames below are in chronological order and show the complete motion. "
    "Camera labels: pi_camera = robot eye, simpleipcamera = third-person view.\n\n"
    "IMPORTANT: Describe turning direction as clockwise/counter-clockwise "
    "(viewed from above) or compass directions (north/south/east/west), "
    "not as 'left' or 'right'.\n\n"
    "Reply in EXACTLY this format on two lines:\n"
    "Verdict: YES | NO | PARTIAL — <one short clause justifying the verdict>\n"
    "Changes: <1-2 short sentences on what actually happened during the motion>"
)

_CLIP_PROMPT = (
    "You are analysing a video clip from a 4-motor Lego robot (left wheel, "
    "right wheel, arm, gripper). The gripper defines the robot's front.\n"
    "Camera: {camera}. The {n_frames} images below are sequential frames.\n\n"
    "Describe what you observe: robot position, any motion, visible objects, "
    "and the overall scene state. Be concise (2-4 sentences).\n"
    "IMPORTANT: Describe turning direction as clockwise/counter-clockwise "
    "(viewed from above) or compass directions (north/south/east/west), "
    "not as 'left' or 'right'."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def b64(path: str) -> str:
    return base64.b64encode(load_image(path)).decode()


def load_folder_frames(folder: str) -> list[tuple[str, str, str]]:
    """Return [(filename, camera_label, path)] sorted by filename."""
    import re
    p = Path(folder)
    frames = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # filename like pi_camera_000.jpg → label = pi_camera
            m = re.match(r"^(.+)_(\d+)$", f.stem)
            label = m.group(1) if m else f.stem
            frames.append((f.name, label, str(f)))
    return frames


def latest_action_video_folder(snapshot_dir: str = "/tmp/lego-robot-snapshots") -> str:
    p = Path(snapshot_dir)
    folders = sorted(p.glob("action_video_*"))
    if not folders:
        sys.exit(f"No action_video_* folders found in {snapshot_dir}")
    return str(folders[-1])


def print_call_summary(prompt: str, image_paths: list[tuple[str, str]]) -> None:
    print("=" * 70)
    print(f"MODEL : {OLLAMA_MODEL}")
    print(f"HOST  : {OLLAMA_HOST}")
    print(f"FRAMES: {len(image_paths)}")
    print()
    print("--- PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---")
    print()
    print("--- IMAGES ---")
    for label, path in image_paths:
        size = os.path.getsize(path)
        print(f"  [{label}] {path}  ({size} bytes)")
    print("--- END IMAGES ---")
    print("=" * 70)
    print()


# ── modes ─────────────────────────────────────────────────────────────────────

def run_stills(args: argparse.Namespace) -> None:
    before_paths = [(f"before/{Path(p).stem}", p) for p in args.before]
    after_paths  = [(f"after/{Path(p).stem}",  p) for p in args.after]

    prompt = _PROMPT.format(action=args.action, expected=args.expected)
    prompt += "\n\n=== BEFORE ===\n"
    for label, _ in before_paths:
        prompt += f"[{label}]\n"
    prompt += "\n=== AFTER ===\n"
    for label, _ in after_paths:
        prompt += f"[{label}]\n"

    all_paths = before_paths + after_paths
    print_call_summary(prompt, all_paths)

    images = [load_image(p) for _, p in all_paths]
    _call_ollama(prompt, images)


def run_video(args: argparse.Namespace) -> None:
    folder = latest_action_video_folder() if args.latest else args.folder
    if not folder:
        sys.exit("--folder or --latest required for video mode")

    frames = load_folder_frames(folder)
    if not frames:
        sys.exit(f"No images found in {folder}")

    # Group by camera (preserving per-camera order), matching server behaviour
    cameras: dict[str, list[str]] = {}
    for _, label, path in frames:
        cameras.setdefault(label, []).append(path)

    prompt = _VIDEO_PROMPT.format(action=args.action, expected=args.expected)
    for camera, paths in cameras.items():
        prompt += f"\n\n=== {camera} ({len(paths)} frames) ==="

    labeled_paths = [(label, path) for _, label, path in frames]
    print_call_summary(prompt, labeled_paths)

    images = [load_image(path) for paths in cameras.values() for path in paths]
    _call_ollama(prompt, images)


def run_clip(args: argparse.Namespace) -> None:
    folder = latest_action_video_folder() if args.latest else args.folder
    if not folder:
        sys.exit("--folder or --latest required for clip mode")

    frames = load_folder_frames(folder)
    if not frames:
        sys.exit(f"No images found in {folder}")

    camera = args.camera or "unknown"
    prompt = _CLIP_PROMPT.format(camera=camera, n_frames=len(frames))

    labeled_paths = [(label, path) for _, label, path in frames]
    print_call_summary(prompt, labeled_paths)

    images = [load_image(path) for _, _, path in frames]
    _call_ollama(prompt, images)


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, images: list[bytes]) -> None:
    try:
        import ollama
    except ImportError:
        sys.exit("ollama package not installed — run: pip install ollama")

    print("Calling Ollama... ", end="", flush=True)
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": images}],
    )
    print("done\n")
    print("--- RESPONSE ---")
    print(resp["message"]["content"].strip())
    print("--- END RESPONSE ---")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Test Qwen vision calls in isolation")
    ap.add_argument("--mode", choices=["stills", "video", "clip"], default="video")
    ap.add_argument("--action",   default="unknown action")
    ap.add_argument("--expected", default="something changes")
    ap.add_argument("--camera",   default="simpleipcamera", help="clip mode: camera label")

    # stills mode
    ap.add_argument("--before", nargs="+", metavar="PATH", help="before image file(s)")
    ap.add_argument("--after",  nargs="+", metavar="PATH", help="after image file(s)")

    # video / clip mode
    ap.add_argument("--folder", metavar="DIR",  help="action_video_* folder")
    ap.add_argument("--latest", action="store_true", help="use latest action_video_* folder automatically")

    args = ap.parse_args()

    if args.mode == "stills":
        if not args.before or not args.after:
            ap.error("--before and --after required for stills mode")
        run_stills(args)
    elif args.mode == "video":
        run_video(args)
    elif args.mode == "clip":
        run_clip(args)


if __name__ == "__main__":
    main()
