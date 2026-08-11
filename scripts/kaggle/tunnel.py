"""Expose a local OpenAI-compatible server (llama-server on Kaggle) to the
public internet with a Cloudflare "quick tunnel" — no account, no token.

Kaggle notebooks accept no inbound connections, so a browser-based frontend
running anywhere else — for example this repo's own `research-pipeline serve`
webapp, started on your own machine — can't reach `127.0.0.1:8000` inside the
notebook directly. `cloudflared tunnel --url` punches a temporary hole: it
opens an outbound connection to Cloudflare's edge and hands back a random
`https://*.trycloudflare.com` URL that proxies straight back to the local
port. That URL is unauthenticated (anyone who has it can reach the model) and
lives only as long as the process — the right tradeoff for a short personal
run. Deliberately stdlib-only, like gguf_server.py, so it works before
`pip install -e .` and from a bare GPU box too.

    from tunnel import start_tunnel, stop_tunnel
    t = start_tunnel(8000)
    print(t.public_url)   # hand this + "/v1" to LLM_BASE_URL elsewhere
    ...
    stop_tunnel(t)

Run as a script to hold one open in a terminal:

    python scripts/kaggle/tunnel.py --port 8000
"""

from __future__ import annotations

import argparse
import logging
import platform
import re
import shutil
import stat
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

CLOUDFLARED_RELEASE_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
DEFAULT_INSTALL_ROOT = "/kaggle/temp/llm" if Path("/kaggle").is_dir() else str(Path.home() / ".cache" / "gguf-server")

READY_TIMEOUT_SECONDS = 30.0
READY_POLL_SECONDS = 0.5

# cloudflared prints the quick-tunnel URL inside a boxed banner on stderr,
# e.g. "|  https://random-crossword-words-1234.trycloudflare.com  |".
_TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    """Raised when cloudflared can't be found/downloaded or never reports a URL."""


@dataclass(frozen=True)
class Tunnel:
    public_url: str  # e.g. https://random-words-1234.trycloudflare.com (no path, no trailing slash)
    process: subprocess.Popen
    log_path: Path


def _download_cloudflared(install_root: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise TunnelError(
            f"No prebuilt cloudflared download wired up for {platform.system()}/{platform.machine()} "
            "(only Linux x86_64, which is what every Kaggle GPU image is). Install cloudflared "
            "yourself and make sure it's on PATH."
        )
    install_root.mkdir(parents=True, exist_ok=True)
    binary_path = install_root / "cloudflared"
    if binary_path.exists():
        return binary_path
    logger.info("Downloading cloudflared from %s", CLOUDFLARED_RELEASE_URL)
    urllib.request.urlretrieve(CLOUDFLARED_RELEASE_URL, binary_path)
    binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return binary_path


def find_or_download_cloudflared(install_root: Union[str, Path] = DEFAULT_INSTALL_ROOT) -> Path:
    """Prefers a cloudflared already on PATH; otherwise caches a downloaded
    Linux amd64 binary under install_root."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return Path(on_path)
    return _download_cloudflared(Path(install_root))


def _wait_for_tunnel_url(process: subprocess.Popen, log_path: Path, timeout: float = READY_TIMEOUT_SECONDS) -> str:
    """Polls log_path for cloudflared's public-URL banner. Watches the child
    process as well as the clock, so a cloudflared that can't start (bad
    binary, no outbound network) surfaces its own log immediately instead of
    after the full timeout — the same shape as gguf_server.wait_until_ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else "(no log)"
            raise TunnelError(f"cloudflared exited with code {process.returncode} before reporting a URL:\n{tail}")
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        match = _TUNNEL_URL_RE.search(text)
        if match:
            return match.group(0)
        time.sleep(READY_POLL_SECONDS)

    process.terminate()
    tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else "(no log)"
    raise TunnelError(f"cloudflared did not report a public URL within {timeout:.0f}s:\n{tail}")


def start_tunnel(
    port: int,
    *,
    install_root: Union[str, Path] = DEFAULT_INSTALL_ROOT,
    timeout: float = READY_TIMEOUT_SECONDS,
) -> Tunnel:
    """Launches a cloudflared quick tunnel for http://127.0.0.1:<port> and
    blocks until it reports its public URL."""
    install_root = Path(install_root)
    binary = find_or_download_cloudflared(install_root)
    log_path = install_root / "cloudflared.log"
    log_file = log_path.open("w")
    process = subprocess.Popen(
        [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        url = _wait_for_tunnel_url(process, log_path, timeout=timeout)
    except TunnelError:
        stop_tunnel(process)
        raise

    return Tunnel(public_url=url, process=process, log_path=log_path)


def stop_tunnel(tunnel: Union[Tunnel, subprocess.Popen], timeout: float = 10.0) -> None:
    """Shuts the tunnel down. Safe to call twice."""
    proc = tunnel.process if isinstance(tunnel, Tunnel) else tunnel
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="local port to expose (e.g. gguf_server's DEFAULT_PORT)")
    parser.add_argument("--install-root", default=DEFAULT_INSTALL_ROOT)
    parser.add_argument("--timeout", type=float, default=READY_TIMEOUT_SECONDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tunnel = start_tunnel(args.port, install_root=args.install_root, timeout=args.timeout)
    print(f"public URL: {tunnel.public_url}")
    print(f"  point LLM_BASE_URL at: {tunnel.public_url}/v1")
    print(f"log: {tunnel.log_path}")

    try:
        tunnel.process.wait()
    except KeyboardInterrupt:
        stop_tunnel(tunnel)


if __name__ == "__main__":
    main()
