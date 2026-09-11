"""Semantic Scholar pacing.

An S2 API key allows one request per second across every endpoint. The clients
once spaced their own calls 0.2s apart, and Barkla job 10492707 had its keyword
search, snippet search and every snippet hydration answered 429 as a result.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.agents.literature import clients
from research_pipeline.config import settings


class _Clock:
    def __init__(self, now: float = 100.0):
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(clients.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(clients.time, "sleep", fake.sleep)
    monkeypatch.setattr(clients, "_semantic_scholar_last_request", float("-inf"))
    return fake


def _response(status, payload=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {"data": []}
    resp.text = ""
    return resp


def test_the_first_request_goes_out_immediately(clock):
    clients._wait_for_semantic_scholar_slot()
    assert clock.sleeps == []


def test_a_request_inside_the_interval_waits_out_the_remainder(clock):
    clients._wait_for_semantic_scholar_slot()
    clock.now += 0.3
    clients._wait_for_semantic_scholar_slot()

    assert clock.sleeps == [pytest.approx(clients.SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS - 0.3)]


def test_a_request_after_the_interval_does_not_wait(clock):
    clients._wait_for_semantic_scholar_slot()
    clock.now += clients.SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS + 0.01
    clients._wait_for_semantic_scholar_slot()

    assert clock.sleeps == []


def test_every_retry_is_paced_not_only_the_first_attempt(clock):
    pace = MagicMock()
    with patch.object(
        clients.requests, "request", side_effect=[_response(429), _response(429), _response(200)]
    ):
        response = clients._request_with_retry("GET", "https://example.org", pace=pace)

    assert response.status_code == 200
    assert pace.call_count == 3


def test_pace_is_not_forwarded_to_requests(clock):
    with patch.object(clients.requests, "request", return_value=_response(200)) as request:
        clients._request_with_retry("GET", "https://example.org", pace=lambda: None, timeout=5)

    assert "pace" not in request.call_args.kwargs
    assert request.call_args.kwargs["timeout"] == 5


def test_keyword_search_back_to_back_queries_are_a_full_interval_apart(clock, monkeypatch):
    monkeypatch.setattr(clients, "settings", replace(settings, semantic_scholar_api_key="fake-key"))
    sent_at: list[float] = []

    def _request(*_args, **_kwargs):
        sent_at.append(clock.now)
        return _response(200)

    with patch.object(clients.requests, "request", side_effect=_request):
        clients.search_semantic_scholar(["q1", "q2", "q3"], max_results=5)

    gaps = [later - earlier for earlier, later in zip(sent_at, sent_at[1:])]
    assert len(gaps) == 2
    assert all(gap >= clients.SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS - 1e-9 for gap in gaps)


def test_a_429_backs_off_longer_than_a_server_error(clock):
    with patch.object(
        clients.requests, "request", side_effect=[_response(429), _response(503), _response(200)]
    ):
        clients._request_with_retry("GET", "https://example.org")

    assert clock.sleeps == [
        clients.RATE_LIMIT_BACKOFF_SECONDS * 1,
        clients.BACKOFF_BASE_SECONDS * 2,
    ]
