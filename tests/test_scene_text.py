"""
mcp_robot.scene_text: reading new text off the camera streams and raising it
to the coordinator as a visual signal — outbox line, tool-result attachment,
and the pause until a tool result has carried it.

Fixtures (tests/fixtures/scene_text/, both real Pi-camera footage from
2026-09-30, the recycling sign above the three bins):
  sign_dim_pi_camera.jpg   — still frame, dim room (brightness ~45, sharpness ~22)
  sign_pan_pi_camera.mp4   — recorded segment in which the robot pans and the
                              sign comes into view (no text in its first frames)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from mcp_robot import scene_text
from mcp_robot.scene_text import SceneTextWatcher, SeenTextRegistry, SignalBox, TextLine

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
        self.signals = []

    def reader(self, bgr):
        self.calls += 1
        return [_line(t) for t in self.texts]


def _watcher(rec: _Recorder, tmp_path, **kwargs) -> SceneTextWatcher:
    return SceneTextWatcher(cameras=("pi_camera",), reader=rec.reader, emit=rec.signals.append,
                            context_frame=lambda camera: None, image_dir=str(tmp_path), threaded=False, **kwargs)


def _feed(watcher, frames, t0=1000.0, dt=0.2):
    for i, frame in enumerate(frames):
        watcher.on_frame("pi_camera", _b64(frame), t0 + i * dt)
    return t0 + len(frames) * dt


def test_still_view_is_read_once_and_new_text_becomes_one_signal(tmp_path):
    rec = _Recorder()
    watcher = _watcher(rec, tmp_path)
    _feed(watcher, [_scene(1)] * 10)  # 2s still
    assert rec.calls == 1
    assert len(rec.signals) == 1
    signal = rec.signals[0]
    assert signal["type"] == "visual_signal" and signal["from"] == "neil"
    assert signal["texts"] == ["EXIT"] and signal["camera"] == "pi_camera"
    assert Path(signal["frame"]).exists() and Path(signal["ocr_overlay"]).exists()
    assert signal["context_frame"] is None  # no other camera in this test
    assert signal["ts"] > 1e9 and signal["frame_ts"] >= 1000.0


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
    assert len(rec.signals) == 1  # ... but "EXIT" was already seen

    rec.texts = ("EXIT", "Paper only")
    _feed(watcher, [_scene(3)] * 6, t0=t)
    assert rec.calls == 3
    assert rec.signals[-1]["texts"] == ["Paper only"]
    assert rec.signals[-1]["all_texts"] == ["EXIT", "Paper only"]


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
    assert len(rec.signals) == 2


def test_recorded_pan_onto_the_sign(tmp_path):
    """Real footage + real OCR: nothing is read while the robot pans; once it
    stops with the sign in view, the sign's words come out as one signal."""
    cap = cv2.VideoCapture(str(PAN_MP4))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) > 20

    signals = []
    watcher = SceneTextWatcher(cameras=("pi_camera",), emit=signals.append, context_frame=lambda c: None,
                               image_dir=str(tmp_path), threaded=False)
    t = _feed(watcher, [frames[0]] * 15, dt=1 / 15)  # 1s still, text-free view before the pan
    assert watcher.reads == 1 and signals == []
    t = _feed(watcher, frames, t0=t, dt=1 / 15)  # the pan itself, at the recording's 15 fps
    assert watcher.reads == 1  # moving — nothing read
    _feed(watcher, [frames[-1]] * 15, t0=t, dt=1 / 15)  # robot stops for 1s, sign in view
    assert watcher.reads == 2
    assert len(signals) == 1
    assert {"glass", "plastic", "paper"} <= {w for text in signals[0]["texts"] for w in scene_text.words(text)}


# ── signals: outbox, attachment, pause ───────────────────────────────────────

def _signal(sid="s1", texts=("Paper",)):
    return {"from": "neil", "type": "visual_signal", "id": sid, "time": "12:00:00", "camera": "pi_camera",
            "texts": list(texts), "all_texts": list(texts), "frame": "/tmp/f.jpg", "context_frame": None,
            "ocr_overlay": "/tmp/f_ocr.jpg", "ts": 1.0, "frame_ts": 1.0}


@pytest.fixture
def box(tmp_path, monkeypatch):
    box = SignalBox(outbox=str(tmp_path / "signals" / "neil.jsonl"))
    monkeypatch.setattr(scene_text, "get_signals", lambda: box)
    return box


def test_signal_goes_to_the_outbox_as_one_line_and_stays_pending(box):
    box.add(_signal("s1"))
    box.add(_signal("s2", texts=("Glass", "Plastic")))
    lines = box.outbox.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["s1", "s2"]  # a `tail -F` sees one event each
    assert [s["id"] for s in box.pending()] == ["s1", "s2"]
    assert [s["id"] for s in box.take()] == ["s1", "s2"]
    assert box.pending() == [] and len(box.recent()) == 2


def test_signals_ride_first_on_the_next_result_once(box):
    from mcp.types import TextContent
    box.add(_signal())
    result = scene_text.attach_signals({"ok": True, "change_description": "moved"})
    assert list(result)[0] == "visual_signals" and result["visual_signals"][0]["id"] == "s1"
    assert scene_text.attach_signals({"ok": True}) == {"ok": True}  # already carried

    box.add(_signal("s2"))
    listed = scene_text.attach_signals([TextContent(type="text", text="frame")])
    assert listed[0].text.startswith("NEW VISUAL SIGNAL") and listed[1].text == "frame"


def test_pause_is_on_while_a_signal_is_pending(box, monkeypatch):
    assert not scene_text.should_pause()
    box.add(_signal())
    assert scene_text.should_pause()
    monkeypatch.setattr(scene_text.config, "SCENE_TEXT_PAUSE", False)
    assert not scene_text.should_pause()


def test_paused_motor_command_does_not_move_and_carries_the_signal(box, monkeypatch):
    from mcp_robot import server
    moved = Mock(return_value={"ok": True, "change_description": "gripper opened"})
    monkeypatch.setattr(server, "_with_change_analysis", moved)
    put = server.mcp._tool_manager.get_tool("put").fn  # the MCP-registered wrapper

    box.add(_signal())
    paused = put()
    assert paused["ok"] is False and paused["paused"] is True and "PAUSED" in paused["error"]
    assert paused["visual_signals"][0]["id"] == "s1"
    moved.assert_not_called()

    assert put()["ok"] is True  # re-issued: the signal was shown, so it runs
    moved.assert_called_once()


def test_non_motor_tool_carries_the_signal_without_pausing(box, monkeypatch):
    from mcp_robot import server
    monkeypatch.setattr(scene_text, "get_watcher", lambda: None)
    status = server.mcp._tool_manager.get_tool("scene_text_status").fn

    box.add(_signal())
    result = status()
    assert result["ok"] and result["visual_signals"][0]["id"] == "s1"
    assert result["recent_signals"][0]["texts"] == ["Paper"]
    assert not scene_text.should_pause()  # seen — the next motor command runs


def test_internal_tool_calls_neither_pause_nor_use_up_signals(box, monkeypatch):
    from mcp_robot import server
    monkeypatch.setattr(scene_text, "get_watcher", lambda: None)
    box.add(_signal())
    assert "visual_signals" not in server.scene_text_status()  # the plain function, as click_button calls tools
    assert box.pending()


def test_navigate_to_pauses_before_moving_when_text_is_read_mid_navigation(box, monkeypatch):
    from mcp_robot import config, server

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    heading = SimpleNamespace(body_center=(100, 100), forward=(1.0, 0.0))
    target = SimpleNamespace(class_name="cup", center=(300, 300), confidence=0.9, note=None)
    obs_map = SimpleNamespace(free_mask=np.ones((10, 10), dtype=np.uint8))
    plan = SimpleNamespace(reason="ok", reachable=True)

    def capture(yolo):  # the sign comes into view while navigate_to is looking around
        box.add(_signal())
        return frame, heading

    monkeypatch.setattr(config, "SNAPSHOT_DIR", "")
    monkeypatch.setattr(server, "_capture_external_frame", capture)
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

    content = server.mcp._tool_manager.get_tool("navigate_to").fn("cup", "blue cup")
    assert content[0].text.startswith("NEW VISUAL SIGNAL")  # carried first, by the wrapper
    assert "Navigation PAUSED" in content[-1].text
    turn.assert_not_called()
    drive.assert_not_called()
    assert box.pending() == []
