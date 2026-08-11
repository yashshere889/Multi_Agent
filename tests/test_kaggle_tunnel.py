"""Tests for the Kaggle cloudflared tunnel helper.

Only the parts that are pure logic and have a real failure mode: parsing the
public URL out of cloudflared's log banner, preferring a cloudflared already
on PATH over downloading one, and readiness polling (a dead process must
surface its log, not time out silently). Actually spawning cloudflared and
reaching the internet is left to the notebook.

scripts/ isn't a package, so the module is loaded by path rather than imported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "tunnel.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tunnel", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tunnel"] = module
    spec.loader.exec_module(module)
    return module


tunnel_mod = _load_module()

CLOUDFLARED_STARTUP_LOG = """\
2024-01-01T00:00:00Z INF Thank you for trying Cloudflare Tunnel.
2024-01-01T00:00:00Z INF +--------------------------------------------------------------------------------------------+
2024-01-01T00:00:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2024-01-01T00:00:00Z INF |  https://random-crossword-words-1234.trycloudflare.com                                    |
2024-01-01T00:00:00Z INF +--------------------------------------------------------------------------------------------+
"""


class _FakeProcess:
    """Minimal Popen stand-in: `exit_after` polls of None, then a return code."""

    def __init__(self, exit_after=None, returncode=1):
        self._polls = 0
        self._exit_after = exit_after
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        self._polls += 1
        if self._exit_after is not None and self._polls > self._exit_after:
            return self.returncode
        return None

    def terminate(self):
        self.terminated = True


def test_url_regex_matches_a_real_cloudflared_log_line():
    match = tunnel_mod._TUNNEL_URL_RE.search(CLOUDFLARED_STARTUP_LOG)
    assert match.group(0) == "https://random-crossword-words-1234.trycloudflare.com"


def test_prefers_a_cloudflared_already_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel_mod.shutil, "which", lambda name: "/usr/local/bin/cloudflared")
    monkeypatch.setattr(
        tunnel_mod,
        "_download_cloudflared",
        lambda install_root: (_ for _ in ()).throw(AssertionError("should not download when already on PATH")),
    )
    assert tunnel_mod.find_or_download_cloudflared(tmp_path) == Path("/usr/local/bin/cloudflared")


def test_downloads_when_not_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(tunnel_mod, "_download_cloudflared", lambda install_root: install_root / "cloudflared")
    assert tunnel_mod.find_or_download_cloudflared(tmp_path) == tmp_path / "cloudflared"


def test_download_reuses_an_existing_cached_binary(monkeypatch, tmp_path):
    cached = tmp_path / "cloudflared"
    cached.write_text("fake binary")
    monkeypatch.setattr(tunnel_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tunnel_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        tunnel_mod.urllib.request,
        "urlretrieve",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-download a cached binary")),
    )
    assert tunnel_mod._download_cloudflared(tmp_path) == cached


def test_download_rejects_unsupported_platforms(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel_mod.platform, "system", lambda: "Darwin")
    with pytest.raises(tunnel_mod.TunnelError, match="No prebuilt cloudflared"):
        tunnel_mod._download_cloudflared(tmp_path)


def test_reads_the_url_once_the_log_reports_it(tmp_path):
    log_path = tmp_path / "cloudflared.log"
    log_path.write_text(CLOUDFLARED_STARTUP_LOG)
    url = tunnel_mod._wait_for_tunnel_url(_FakeProcess(), log_path, timeout=5.0)
    assert url == "https://random-crossword-words-1234.trycloudflare.com"


def test_dead_process_fails_immediately_with_its_log(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel_mod, "READY_POLL_SECONDS", 0)
    log_path = tmp_path / "cloudflared.log"
    log_path.write_text("error: could not reach Cloudflare edge")

    with pytest.raises(tunnel_mod.TunnelError) as excinfo:
        tunnel_mod._wait_for_tunnel_url(_FakeProcess(exit_after=0, returncode=1), log_path, timeout=30.0)

    assert "could not reach Cloudflare edge" in str(excinfo.value)


def test_timeout_terminates_the_process(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel_mod, "READY_POLL_SECONDS", 0)
    process = _FakeProcess()
    log_path = tmp_path / "cloudflared.log"
    log_path.write_text("still connecting...\n")

    with pytest.raises(tunnel_mod.TunnelError, match="did not report"):
        tunnel_mod._wait_for_tunnel_url(process, log_path, timeout=0.01)

    assert process.terminated


def test_stop_tunnel_is_safe_to_call_twice():
    process = _FakeProcess(exit_after=0, returncode=0)
    tunnel = tunnel_mod.Tunnel(public_url="https://x.trycloudflare.com", process=process, log_path=Path("/dev/null"))
    tunnel_mod.stop_tunnel(tunnel)
    tunnel_mod.stop_tunnel(tunnel)
