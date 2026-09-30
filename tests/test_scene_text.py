"""
mcp_robot.scene_text: reading new text off the camera streams, handing it to
NAPC through the events directory, and surfacing NAPC's advisories.

Fixtures (tests/fixtures/scene_text/, both real Pi-camera footage from
2026-09-30, the recycling sign above the three bins):
  sign_dim_pi_camera.jpg   — still frame, dim room (brightness ~45, sharpness ~22)
  sign_pan_pi_camera.mp4   — recorded segment in which the robot pans and the
                              sign comes into view (no text in its first frames)
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from mcp_robot import scene_text
from mcp_robot.scene_text import AdvisoryBoard, SceneTextWatcher, SeenTextRegistry, TextLine

FIXTURES = Path(__file__).parent / "fixtures" / "scene_text"
SIGN_JPG = FIXTURES / "sign_dim_pi_camera.jpg"
PAN_MP4 = FIXTURES / "sign_pan_pi_camera.mp4"


def _b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _all_words(lines) -> set[str]:
    return {w for line in lines for w in scene_text.words(line.text)}


def _line(text: str) -> TextLine:
    return TextLine(text, 0.95, [[100, 100], [200, 100], [200, 130], [100, 130]])


# ── OCR (real RapidOCR, on real frames) ──────────────────────────────────────

@pytest.fixture(scope="module")
def sign_bgr():
    return cv2.imread(str(SIGN_JPG))


def test_reads_the_recycling_sign_in_a_dim_frame(sign_bgr):
    lines = scene_text.read_text(sign_bgr)
    assert {"glass", "plastic", "paper"} <= _all_words(lines)
    assert all(line.score >= 0.7 for line in lines)
    assert "#" not in [line.text for line in lines]  # the icons' low-confidence read


def test_drops_text_cut_off_by_the_frame_edge(sign_bgr):
    # "Glass" spans x=194..288 — crop through it so it touches the left edge.
    lines = scene_text.read_text(np.ascontiguousarray(sign_bgr[:, 240:]))
    assert "glass" not in _all_words(lines)
    assert {"plastic", "paper"} <= _all_words(lines)


def test_filters_low_confidence_digit_only_and_edge_lines(monkeypatch):
    box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    fake_ocr = lambda bgr: ([  # noqa: E731
        (box(100, 100, 200, 130), "Paper only", 0.95),
        (box(100, 200, 200, 230), "#", 0.99),         # an icon
        (box(100, 300, 200, 330), "15:4627", 0.84),   # a clock — new digits every read
        (box(300, 100, 400, 130), "Glass", 0.55),     # low confidence
        (box(0, 400, 90, 430), "Plast", 0.97),        # cut off by the left edge
    ], None)
    monkeypatch.setattr(scene_text, "_get_ocr", lambda: fake_ocr)
    lines = scene_text.read_text(np.zeros((480, 640, 3), dtype=np.uint8))
    assert [line.text for line in lines] == ["Paper only"]


def test_dim_sign_frame_counts_as_clear(sign_bgr):
    brightness, sharpness = scene_text.clarity(sign_bgr)
    assert brightness >= scene_text.config.SCENE_TEXT_MIN_BRIGHTNESS
    assert sharpness >= scene_text.config.SCENE_TEXT_MIN_SHARPNESS


# ── novelty ──────────────────────────────────────────────────────────────────

def test_novelty_is_per_word_and_tolerates_ocr_noise():
    registry = SeenTextRegistry()
    registry.remember([_line("Glass Plastic"), _line("Paper")])
    assert registry.novel([_line("Glass"), _line("Plastic Paper"), _line("Plastlc")]) == []
    assert [l.text for l in registry.novel([_line("Paper"), _line("Metal only")])] == ["Metal only"]
    registry.clear()
    assert len(registry.novel([_line("Paper")])) == 1


# ── the watcher's still/changed/clear gating ─────────────────────────────────

def _scene(seed: int) -> np.ndarray:
    """A textured, clearly-lit frame; different seeds give different views."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 220, size=(24, 32, 3), dtype=np.uint8)
    return cv2.resize(small, (640, 480), interpolation=cv2.INTER_NEAREST)


class _Recorder:
    def __init__(self, texts=("EXIT",)):
        self.texts = texts
        self.calls = 0
        self.findings = []

    def reader(self, bgr):
        self.calls += 1
        return [_line(t) for t in self.texts]


def _watcher(rec: _Recorder, tmp_path, **kwargs) -> SceneTextWatcher:
    return SceneTextWatcher(cameras=("pi_camera",), reader=rec.reader, emit=rec.findings.append,
                            context_frame=lambda camera: None, image_dir=str(tmp_path), threaded=False, **kwargs)


def _feed(watcher, frames, t0=1000.0, dt=0.2):
    for i, frame in enumerate(frames):
        watcher.on_frame("pi_camera", _b64(frame), t0 + i * dt)
    return t0 + len(frames) * dt


def test_still_view_is_read_once_and_new_text_becomes_one_finding(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    _feed(watcher, [_scene(1)] * 10)  # 2s still
    assert rec.calls == 1
    assert len(rec.findings) == 1
    finding = rec.findings[0]
    assert finding["texts"] == ["EXIT"] and finding["camera"] == "pi_camera"
    assert Path(finding["images"][0]["path"]).exists()
    assert finding["ts"] > 1e9 and finding["frame_ts"] >= 1000.0


def test_moving_view_is_never_read(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    _feed(watcher, [_scene(i % 2) for i in range(20)])
    assert rec.calls == 0


def test_changed_view_is_read_again_but_known_text_is_not_reported(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    t = _feed(watcher, [_scene(1)] * 6)
    t = _feed(watcher, [_scene(2)] * 6, t0=t)
    assert rec.calls == 2  # new view → read again
    assert len(rec.findings) == 1  # ... but "EXIT" was already seen

    rec.texts = ("EXIT", "Paper only")
    _feed(watcher, [_scene(3)] * 6, t0=t)
    assert rec.calls == 3
    assert rec.findings[-1]["texts"] == ["Paper only"]
    assert [t["text"] for t in rec.findings[-1]["all_texts"]] == ["EXIT", "Paper only"]


def test_dark_view_is_not_read(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    _feed(watcher, [np.full((480, 640, 3), 5, dtype=np.uint8)] * 10)
    assert rec.calls == 0


def test_reset_makes_text_in_view_new_again(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    t = _feed(watcher, [_scene(1)] * 6)
    watcher.reset()
    _feed(watcher, [_scene(1)] * 6, t0=t)
    assert len(rec.findings) == 2


def test_recorded_pan_onto_the_sign(tmp_path):
    """Real footage + real OCR: nothing is read while the robot pans; once it
    stops with the sign in view, the sign's words come out as one finding."""
    cap = cv2.VideoCapture(str(PAN_MP4))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) > 20

    findings = []
    watcher = SceneTextWatcher(cameras=("pi_camera",), emit=findings.append, context_frame=lambda c: None,
                               image_dir=str(tmp_path), threaded=False)
    t = _feed(watcher, [frames[0]] * 15, dt=1 / 15)  # 1s still, text-free view before the pan
    assert watcher.reads == 1 and findings == []
    t = _feed(watcher, frames, t0=t, dt=1 / 15)  # the pan itself, at the recording's 15 fps
    assert watcher.reads == 1  # moving — nothing read
    _feed(watcher, [frames[-1]] * 15, t0=t, dt=1 / 15)  # robot stops for 1s, sign in view
    assert watcher.reads == 2
    assert len(findings) == 1
    assert {"glass", "plastic", "paper"} <= {w for text in findings[0]["texts"] for w in scene_text.words(text)}


# ── advisory board ───────────────────────────────────────────────────────────

class _Clock:
    def __init__(self, now=10_000.0):
        self.now = now

    def __call__(self):
        return self.now


def _advisory(fid="f1", status="replanning", halt=True, seq=1, updated_at=10_000.0, **extra):
    return {"finding_id": fid, "status": status, "halt": halt, "seq": seq, "updated_at": updated_at,
            "texts": ["Paper"], "camera": "pi_camera", "summary": "Paper goes into the green bin.", **extra}


def _write(events: Path, advisory: dict) -> None:
    path = events / "advisories" / f"{advisory['finding_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(advisory))
    # distinct mtimes so the board's mtime cache sees every rewrite
    stamp = time.time_ns() + advisory["seq"] * 1_000_000
    os.utime(path, ns=(stamp, stamp))


@pytest.fixture
def board(tmp_path):
    clock = _Clock()
    board = AdvisoryBoard(events_dir=str(tmp_path), napc_timeout_s=30, halt_max_age_s=300, clock=clock)
    board.clock = clock
    return board


def test_standing_halt_rides_on_every_result(board, tmp_path):
    _write(tmp_path, _advisory())
    for _ in range(3):
        entries = board.collect()
        assert len(entries) == 1 and entries[0]["halt"] and entries[0]["action"].startswith("HALT")
    assert board.halt_advisory()["finding_id"] == "f1"


def test_new_plan_is_surfaced_once(board, tmp_path):
    _write(tmp_path, _advisory())
    board.collect()
    _write(tmp_path, _advisory(status="new_plan", halt=False, seq=2))
    assert board.halt_advisory() is None
    entries = board.collect()
    assert [e["status"] for e in entries] == ["new_plan"] and "await_advisory" in entries[0]["action"]
    assert board.collect() == []


def test_quiet_statuses_and_acknowledged_advisories_are_not_surfaced(board, tmp_path):
    _write(tmp_path, _advisory("a", status="ignored", halt=False))
    _write(tmp_path, _advisory("b", status="confirmed", halt=False))
    _write(tmp_path, _advisory("c", status="new_plan", halt=False, acknowledged=True))
    assert board.collect() == []


def test_stale_halt_stops_blocking(board, tmp_path):
    _write(tmp_path, _advisory())
    board.clock.now += 301
    assert board.halt_advisory() is None
    assert "note" in board.collect()[0]


def test_dismissed_halt_stops_blocking_until_napc_updates_it(board, tmp_path):
    _write(tmp_path, _advisory())
    assert board.dismiss(["f1", "unknown"]) == ["f1"]
    assert board.halt_advisory() is None and board.collect() == []
    _write(tmp_path, _advisory(status="new_plan", halt=False, seq=2))
    assert [e["status"] for e in board.collect()] == ["new_plan"]


def test_advisories_from_before_this_process_are_ignored(board, tmp_path):
    _write(tmp_path, _advisory(updated_at=9_000.0))
    assert board.collect() == [] and board.halt_advisory() is None


def test_unanswered_finding_is_surfaced_after_timeout(board, tmp_path):
    board.post({"finding_id": "f9", "ts": board.clock.now, "camera": "pi_camera", "texts": ["Paper"],
                "images": [{"path": "/tmp/x.jpg"}]})
    assert (tmp_path / "findings" / "f9.json").exists()
    assert board.collect() == []
    board.clock.now += 31
    entries = board.collect()
    assert entries[0]["status"] == "napc_silent" and "report_scene_text" in entries[0]["message"]
    assert board.collect() == []


def test_attach_advisories_puts_them_first(board, tmp_path, monkeypatch):
    monkeypatch.setattr(scene_text, "get_board", lambda: board)
    _write(tmp_path, _advisory())
    result = scene_text.attach_advisories({"ok": True, "change_description": "moved"})
    assert list(result)[0] == "napc_advisory"

    from mcp.types import TextContent
    listed = scene_text.attach_advisories([TextContent(type="text", text="frame")])
    assert listed[0].text.startswith("NAPC ADVISORY") and listed[1].text == "frame"


# ── server integration ───────────────────────────────────────────────────────

def test_only_the_mcp_registered_tool_attaches_advisories(board, tmp_path, monkeypatch):
    from mcp_robot import server
    monkeypatch.setattr(scene_text, "get_board", lambda: board)
    monkeypatch.setattr(scene_text, "get_watcher", lambda: None)
    _write(tmp_path, _advisory(status="new_plan", halt=False))

    assert "napc_advisory" not in server.scene_text_status()  # internal call: untouched
    registered = server.mcp._tool_manager.get_tool("scene_text_status").fn
    assert registered()["napc_advisory"][0]["status"] == "new_plan"


def test_navigate_to_halts_before_moving(board, tmp_path, monkeypatch):
    from mcp_robot import config, server
    monkeypatch.setattr(scene_text, "get_board", lambda: board)
    _write(tmp_path, _advisory())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    heading = SimpleNamespace(body_center=(100, 100), forward=(1.0, 0.0))
    target = SimpleNamespace(class_name="cup", center=(300, 300), confidence=0.9, note=None)
    obs_map = SimpleNamespace(free_mask=np.ones((10, 10), dtype=np.uint8))
    plan = SimpleNamespace(reason="ok", reachable=True)
    monkeypatch.setattr(config, "SNAPSHOT_DIR", "")
    monkeypatch.setattr(server, "_capture_external_frame", lambda yolo: (frame, heading))
    monkeypatch.setattr(server.grasp_mod, "_yolo_detect", lambda bgr, target_class: [target])
    monkeypatch.setattr(server.grasp_mod, "_pick_target", lambda objects, h: objects[0])
    monkeypatch.setattr(server.nav_mod, "detect_obstacles", lambda *a: obs_map)
    monkeypatch.setattr(server.nav_mod, "plan_path", lambda *a: plan)
    monkeypatch.setattr(server.nav_mod, "draw_nav_overlay", lambda bgr, *a: bgr)
    monkeypatch.setattr(server.nav_mod, "build_cspace_bgr", lambda bgr, *a: bgr)
    monkeypatch.setattr(server.nav_mod, "near_target", lambda m: False)
    monkeypatch.setattr(server.viz, "log_annotated_images", lambda **kw: None)
    turn, drive = Mock(), Mock()
    monkeypatch.setattr(server.robot_mod, "turn", turn)
    monkeypatch.setattr(server.robot_mod, "drive_degrees", drive)

    content = server.navigate_to("cup", "blue cup")
    text = content[-1].text
    assert "Navigation HALTED" in text and "await_advisory" in text
    turn.assert_not_called()
    drive.assert_not_called()
