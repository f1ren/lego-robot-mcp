"""
Logic-focused unit tests for mcp_robot.vision's Gemini backend.

These mock the network boundary (vision._get_gemini_client /
_gemini_generate_with_retry's client argument) with canned responses and
errors, so retry, quota-fallback, and locate_object_vlm JSON-parsing logic
can be exercised offline and deterministically — no GEMINI_API_KEY or
network access required.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from mcp_robot import config, vision


def _tiny_bgr_image(w: int = 200, h: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── retry / backoff (_gemini_generate_with_retry) ────────────────────────────


def test_retry_recovers_after_transient_error(make_fake_client, monkeypatch):
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    client = make_fake_client([Exception("503 Service Unavailable"), "ok"])

    resp = vision._gemini_generate_with_retry(client, "some-model", ["hi"])

    assert resp.text == "ok"
    assert len(client.calls) == 2


def test_non_transient_error_raises_immediately(make_fake_client):
    client = make_fake_client([Exception("400 Bad Request: invalid argument")])

    with pytest.raises(Exception, match="400 Bad Request"):
        vision._gemini_generate_with_retry(client, "some-model", ["hi"])

    assert len(client.calls) == 1  # no retry attempted for a non-transient error


def test_transient_error_exhausts_retries_then_raises(make_fake_client, monkeypatch):
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    client = make_fake_client(
        [Exception("503 Service Unavailable")] * vision._TRANSIENT_MAX_RETRIES
    )

    with pytest.raises(Exception, match="503"):
        vision._gemini_generate_with_retry(client, "some-model", ["hi"])

    assert len(client.calls) == vision._TRANSIENT_MAX_RETRIES


# ── quota fallback (_gemini_describe_video) ──────────────────────────────────


def test_describe_video_falls_back_to_secondary_model_on_quota_error(fake_gemini_client):
    quota_exc = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    fallback_text = "Verdict: YES — moved as expected\nChanges: robot advanced north\nPlan: N/A"
    client = fake_gemini_client([quota_exc, fallback_text])

    frame = base64.b64encode(b"not-a-real-jpeg").decode()
    result = vision._gemini_describe_video("drive forward", "move toward cup", [("pi_camera", frame)])

    assert result == fallback_text
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == config.GEMINI_MODEL
    assert client.calls[1]["model"] == config.GEMINI_FALLBACK_MODEL


def test_locate_object_vlm_falls_back_to_secondary_model_on_quota_error(fake_gemini_client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    quota_exc = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    payload = json.dumps({
        "found": True, "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6,
        "hsv_hue_lo": 100, "hsv_hue_hi": 130, "hsv_sat_min": 50, "hsv_val_min": 60,
        "approx_area_frac": 0.08, "confidence": 0.97, "note": "blue cup on floor",
    })
    client = fake_gemini_client([quota_exc, payload])

    result = vision.locate_object_vlm(_tiny_bgr_image(), "blue cup")

    assert result is not None
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == config.LOCATE_OBJECT_MODEL
    assert client.calls[1]["model"] == config.LOCATE_OBJECT_FALLBACK_MODEL


# ── locate_object_vlm JSON parsing ───────────────────────────────────────────


def test_locate_object_vlm_not_found_returns_none(fake_gemini_client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    fake_gemini_client(['{"found": false}'])

    assert vision.locate_object_vlm(_tiny_bgr_image(), "blue cup") is None


def test_locate_object_vlm_low_confidence_raises(fake_gemini_client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    payload = json.dumps({
        "found": True, "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4,
        "hsv_hue_lo": 100, "hsv_hue_hi": 130, "hsv_sat_min": 40, "hsv_val_min": 40,
        "approx_area_frac": 0.05, "confidence": 0.5, "note": "maybe a cup",
    })
    fake_gemini_client([payload])

    with pytest.raises(vision.LowConfidenceDetection):
        vision.locate_object_vlm(_tiny_bgr_image(), "blue cup")


def test_locate_object_vlm_parses_valid_response(fake_gemini_client, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    payload = json.dumps({
        "found": True, "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6,
        "hsv_hue_lo": 100, "hsv_hue_hi": 130, "hsv_sat_min": 50, "hsv_val_min": 60,
        "approx_area_frac": 0.08, "confidence": 0.97, "note": "blue cup on floor",
    })
    fake_gemini_client([payload])

    result = vision.locate_object_vlm(_tiny_bgr_image(w=200, h=100), "blue cup")

    assert result is not None
    bbox, confidence, note, hsv_lo, hsv_hi, area_frac = result
    assert bbox == (20, 20, 100, 60)  # fractions * (w=200, h=100)
    assert confidence == pytest.approx(0.97)
    assert note == "blue cup on floor"
    assert list(hsv_lo) == [100, 50, 60]
    assert list(hsv_hi) == [130, 255, 255]
    assert area_frac == pytest.approx(0.08)
