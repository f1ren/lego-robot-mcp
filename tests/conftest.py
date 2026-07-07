"""
Shared fixtures for mocking the Gemini network boundary in unit tests.

`fake_gemini_client` / `make_fake_client` let a test replay canned text
responses (or raise canned exceptions) in place of a real
`google.genai.Client`, so retry/fallback/parsing logic in mcp_robot.vision
can be exercised offline and deterministically.

Tests tagged @pytest.mark.live are untouched by any of this — they call the
real API on request (`pytest -m live`) and are excluded from the default run
by the `addopts` in pyproject.toml.
"""
from __future__ import annotations

import pytest

from mcp_robot import vision


class FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    """Stand-in for genai.Client().models — replays canned responses/errors in order."""

    def __init__(self, responses):
        self._queue = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._queue:
            raise AssertionError("FakeGeminiClient: ran out of canned responses")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeGeminiResponse(item)


class FakeGeminiClient:
    """Stand-in for google.genai.Client.

    `responses` is consumed in call order; each entry is either response text
    (str) or an exception instance to raise for that call.
    """

    def __init__(self, responses):
        self.models = _FakeModels(responses)

    @property
    def calls(self):
        return self.models.calls


@pytest.fixture
def make_fake_client():
    """Factory: make_fake_client([...responses]) -> FakeGeminiClient.

    Not wired into vision._get_gemini_client — use this when calling a
    function like _gemini_generate_with_retry that takes the client directly.
    """

    def _make(responses):
        return FakeGeminiClient(responses)

    return _make


@pytest.fixture
def fake_gemini_client(monkeypatch, make_fake_client):
    """Factory: fake_gemini_client([...responses]) -> FakeGeminiClient, wired in
    place of vision._get_gemini_client() for the duration of the test.
    """

    def _install(responses):
        client = make_fake_client(responses)
        monkeypatch.setattr(vision, "_get_gemini_client", lambda: client)
        return client

    return _install


@pytest.fixture(autouse=True)
def _reset_gemini_module_state():
    """Prevent the real Gemini client/model singleton from leaking between tests."""
    yield
    vision._gemini_client = None
    vision._active_model = None
    vision._active_locate_model = None
