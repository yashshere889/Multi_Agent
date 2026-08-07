"""Stand up a GGUF model behind an OpenAI-compatible endpoint on a Kaggle GPU.

This is deployment plumbing, not pipeline code: nothing here imports
`research_pipeline`, and the pipeline learns nothing about llama.cpp. The
whole point of the module is that once `serve()` returns, the existing
`LLM_BACKEND=openai` path works unchanged — set LLM_BASE_URL/LLM_MODEL to what
it hands back and every agent talks to a local quantized model instead of a
vLLM node on Barkla. Deliberately stdlib-only (plus huggingface_hub) so it can
run *before* `pip install -e .`, and importable from Colab or any bare GPU box,
not just Kaggle.

    from gguf_server import serve, stop
    server = serve()                       # builds, downloads, launches, waits
    os.environ["LLM_BASE_URL"] = server.base_url
    os.environ["LLM_MODEL"] = server.model_alias

Why build llama.cpp from source rather than download a binary: llama.cpp
publishes no prebuilt Linux **CUDA** archive (only CPU/Vulkan/ROCm/SYCL for
Ubuntu — CUDA binaries are Windows-only), and a CPU build of even a 4B model
makes a full six-agent run take hours. The build is narrowed to one target and
one GPU architecture, which is what keeps it to minutes rather than an hour.

Run it as a script to hold a server open in a terminal instead:

    python scripts/kaggle/gguf_server.py --foreground
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "unsloth/gemma-4-12b-it-GGUF"
# Q4_K_M is the plain, universally-supported 4-bit quant — ~7GB for this 12B,
# which is what makes it fit a single 16GB Kaggle T4 alongside its KV cache.
# unsloth's own "Dynamic 2.0" 4-bit in the same repo is UD-Q4_K_XL — slightly
# larger, quantized per-layer, usually the better accuracy/size trade; swap the
# quant string for it if you'd rather have that.
DEFAULT_QUANT = "Q4_K_M"

# Kaggle gives ~20GB under /kaggle/working, and everything there is packaged as
# notebook output on commit. The build tree and multi-GB weights want neither,
# so they default to scratch and are rebuilt on a fresh session.
DEFAULT_INSTALL_ROOT = "/kaggle/temp/llm" if Path("/kaggle").is_dir() else str(Path.home() / ".cache" / "gguf-server")

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

# Big enough for the widest prompt any agent builds (the Writer's related-work
# batch, ~12k chars of paper abstracts) plus a full 8192-token completion, with
# headroom. Not larger: a 12B at 4-bit is ~7GB of weights, and on a single 16GB
# T4 the KV cache is what runs the card out of memory, so the context window is
# the knob that decides whether the model loads at all.
DEFAULT_CTX_SIZE = 24576
DEFAULT_PORT = 8000

# The name the server advertises at /v1/models, and therefore the only value
# LLM_MODEL may take. Pinned to something short and stable so the notebook
# never has to interpolate a quant filename into an env var.
MODEL_ALIAS = "gemma-4-local"

READY_TIMEOUT_SECONDS = 600.0
READY_POLL_SECONDS = 2.0


class ServerError(RuntimeError):
    """Raised when the model server can't be built, launched, or reached."""


@dataclass(frozen=True)
class Server:
    base_url: str          # ready to assign to LLM_BASE_URL (includes /v1)
    model_alias: str       # ready to assign to LLM_MODEL
    model_path: Path
    log_path: Path
    process: subprocess.Popen


# --------------------------------------------------------------------------
# Environment probing
# --------------------------------------------------------------------------


def _run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def gpu_compute_capability() -> Optional[str]:
    """The CUDA arch to compile for, e.g. "75" for Kaggle's T4, "60" for P100,
    "89" for an L4. Returned as the digits CMAKE_CUDA_ARCHITECTURES wants.

    Probed rather than assumed: Kaggle hands out several different cards, and
    compiling for all of them multiplies the build time by the number of
    architectures for no benefit."""
    if shutil.which("nvidia-smi") is None:
        return None
    probe = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if probe.returncode != 0:
        return None
    first = probe.stdout.strip().splitlines()
    if not first:
        return None
    match = re.match(r"(\d+)\.(\d+)", first[0].strip())
    return f"{match.group(1)}{match.group(2)}" if match else None


def probe_environment() -> dict:
    """Everything worth knowing before committing to a build, gathered in one
    place so a notebook cell can print it and a failure later is diagnosable."""
    gpus: List[str] = []
    if shutil.which("nvidia-smi"):
        query = _run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"])
        if query.returncode == 0:
            gpus = [line.strip() for line in query.stdout.strip().splitlines() if line.strip()]

    nvcc_version = None
    if shutil.which("nvcc"):
        nvcc = _run(["nvcc", "--version"])
        if nvcc.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", nvcc.stdout)
            nvcc_version = match.group(1) if match else "unknown"

    usage = shutil.disk_usage("/")
    return {
        "gpus": gpus,
        "compute_capability": gpu_compute_capability(),
        "nvcc": nvcc_version,
        "cmake": shutil.which("cmake"),
        "git": shutil.which("git"),
        "uv": shutil.which("uv"),
        "internet": _has_internet(),
        "disk_free_gb": round(usage.free / 1e9, 1),
        "python": sys.version.split()[0],
    }


def _has_internet(host: str = "huggingface.co", timeout: float = 5.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_llama_server(install_root: Path, *, force: bool = False, jobs: Optional[int] = None) -> Path:
    """Clones and builds llama.cpp with CUDA, returning the path to the
    `llama-server` binary. Reuses an existing binary unless `force`.

    Two deliberate narrowings keep this to a handful of minutes on Kaggle's
    four cores: only the `llama-server` target is built (not the ~30 example
    binaries), and only for this machine's own CUDA architecture. libcurl is
    switched off because weights come from huggingface_hub here, and its dev
    headers aren't guaranteed to be present."""
    source_dir = install_root / "llama.cpp"
    build_dir = source_dir / "build"
    server_bin = build_dir / "bin" / "llama-server"

    if server_bin.exists() and not force:
        logger.info("Reusing existing llama-server at %s", server_bin)
        return server_bin

    arch = gpu_compute_capability()
    if arch is None:
        raise ServerError(
            "No CUDA GPU detected (nvidia-smi missing or returned nothing). On Kaggle, "
            "set Accelerator to a GPU under Settings and restart the session — a CPU "
            "build would take hours per pipeline run."
        )
    if shutil.which("nvcc") is None:
        raise ServerError(
            "nvcc is not on PATH, so llama.cpp can't be built with CUDA support. "
            "Kaggle's GPU images ship the CUDA toolkit; if you're elsewhere, install it "
            "or point LLM_BASE_URL at a server running somewhere that has it."
        )
    if shutil.which("cmake") is None:
        raise ServerError("cmake is not on PATH. Install it first (`pip install cmake`).")

    install_root.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        logger.info("Cloning llama.cpp into %s", source_dir)
        # Shallow clone of master, not a pinned release tag: Gemma 4 is recent
        # enough that older tags predate its architecture support and would
        # fail to load the GGUF at all.
        clone = _run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(source_dir)])
        if clone.returncode != 0:
            raise ServerError(f"git clone failed: {clone.stderr[-2000:]}")

    logger.info("Configuring llama.cpp for CUDA arch %s", arch)
    configure = _run(
        [
            "cmake",
            "-S", str(source_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]
    )
    if configure.returncode != 0:
        raise ServerError(f"cmake configure failed:\n{configure.stdout[-2000:]}\n{configure.stderr[-2000:]}")

    jobs = jobs or (os.cpu_count() or 4)
    logger.info("Building llama-server with %d jobs (several minutes)", jobs)
    build = _run(["cmake", "--build", str(build_dir), "--config", "Release", "--target", "llama-server", "-j", str(jobs)])
    if build.returncode != 0:
        raise ServerError(f"build failed:\n{build.stdout[-3000:]}\n{build.stderr[-3000:]}")

    if not server_bin.exists():
        raise ServerError(f"build reported success but {server_bin} does not exist")
    logger.info("Built %s", server_bin)
    return server_bin


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def resolve_quant_files(repo_id: str, quant: str) -> List[str]:
    """The GGUF file(s) for a quant, newest-style shard names included.

    Matched against the repo's real file list rather than string-formatted,
    because quant naming varies across unsloth repos (`Q4_K_M` vs `UD-Q4_K_XL`)
    and a guessed filename fails as a 404 halfway through a notebook. Larger
    models are published as `...-00001-of-00002.gguf` shards; llama.cpp is
    handed the first shard and finds the rest itself, but all of them must be
    downloaded, so every match is returned."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(repo_id) if f.endswith(".gguf")]
    # mmproj (vision tower) and mtp (multi-token-prediction) files sit
    # alongside the quants and would otherwise match a bare substring test.
    candidates = [f for f in files if f"-{quant}" in f and not Path(f).name.startswith(("mmproj", "mtp"))]
    if not candidates:
        available = sorted({Path(f).name for f in files})
        raise ServerError(f"No GGUF matching quant {quant!r} in {repo_id}. Available: {available}")
    return sorted(candidates)


def download_model(repo_id: str, quant: str, cache_dir: Path) -> Path:
    """Downloads the quant's file(s), returning the path llama-server should
    open (the first shard, when sharded)."""
    from huggingface_hub import hf_hub_download

    filenames = resolve_quant_files(repo_id, quant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", filenames, repo_id)
    paths = [Path(hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(cache_dir))) for name in filenames]
    return paths[0]


# --------------------------------------------------------------------------
# Serve
# --------------------------------------------------------------------------


def _get_json(url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def wait_until_ready(
    port: int,
    process: subprocess.Popen,
    log_path: Path,
    timeout: float = READY_TIMEOUT_SECONDS,
) -> None:
    """Polls /health until the model is loaded. Watches the child process as
    well as the clock, so an unsupported architecture or an out-of-memory kill
    surfaces its actual log immediately instead of after the full timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:] if log_path.exists() else "(no log)"
            raise ServerError(f"llama-server exited with code {process.returncode} before becoming ready:\n{tail}")
        health = _get_json(f"http://127.0.0.1:{port}/health")
        if health and health.get("status") == "ok":
            return
        time.sleep(READY_POLL_SECONDS)

    process.terminate()
    tail = log_path.read_text(errors="replace")[-3000:] if log_path.exists() else "(no log)"
    raise ServerError(f"llama-server was not ready within {timeout:.0f}s:\n{tail}")


def serve(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    quant: str = DEFAULT_QUANT,
    install_root: str | Path = DEFAULT_INSTALL_ROOT,
    port: int = DEFAULT_PORT,
    ctx_size: int = DEFAULT_CTX_SIZE,
    model_alias: str = MODEL_ALIAS,
    force_rebuild: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Server:
    """Build (if needed), download (if needed), launch, and wait for a local
    OpenAI-compatible server. Returns once /health reports ok."""
    install_root = Path(install_root)
    server_bin = build_llama_server(install_root, force=force_rebuild)
    model_path = download_model(repo_id, quant, install_root / "models")

    log_path = install_root / "llama-server.log"
    command = [
        str(server_bin),
        "--model", str(model_path),
        "--alias", model_alias,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        # Everything on the GPU. Any layer left on the CPU dominates latency
        # across the hundreds of calls a full run makes. On Kaggle's T4 x2
        # llama.cpp splits the layers over both cards by default, so a 4-bit
        # 12B has room to spare; on a single card it just fits.
        "--n-gpu-layers", "999",
        # Render prompts with the GGUF's own chat template. Without it the
        # server falls back to a built-in template that predates Gemma 4 and
        # would format every turn wrongly.
        "--jinja",
        # One request at a time: the pipeline is sequential, and splitting the
        # context across parallel slots would shrink the window each agent gets.
        "--parallel", "1",
    ]
    command += extra_args or []

    # Loopback only, as with the web app: this endpoint has no authentication.
    logger.info("Starting llama-server on port %d (log: %s)", port, log_path)
    log_file = log_path.open("w")
    env = dict(os.environ)
    # The CUDA build produces shared ggml libraries next to the binary; without
    # this the server dies at startup with a linker error rather than anything
    # that names the real problem.
    lib_dir = str(server_bin.parent)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    try:
        wait_until_ready(port, process, log_path)
    except ServerError:
        stop(process)
        raise

    return Server(
        base_url=f"http://127.0.0.1:{port}/v1",
        model_alias=model_alias,
        model_path=model_path,
        log_path=log_path,
        process=process,
    )


def stop(process: subprocess.Popen | Server, timeout: float = 15.0) -> None:
    """Shuts the server down, freeing its VRAM. Safe to call twice."""
    proc = process.process if isinstance(process, Server) else process
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
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="e.g. Q4_K_M, UD-Q4_K_XL, Q5_K_M")
    parser.add_argument("--install-root", default=DEFAULT_INSTALL_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--probe-only", action="store_true", help="print the environment probe and exit")
    parser.add_argument("--foreground", action="store_true", help="hold the server open until interrupted")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.probe_only:
        print(json.dumps(probe_environment(), indent=2))
        return

    server = serve(
        repo_id=args.repo_id,
        quant=args.quant,
        install_root=args.install_root,
        port=args.port,
        ctx_size=args.ctx_size,
        force_rebuild=args.force_rebuild,
    )
    print(f"LLM_BACKEND=openai\nLLM_BASE_URL={server.base_url}\nLLM_MODEL={server.model_alias}")
    print(f"log: {server.log_path}")

    if args.foreground:
        try:
            server.process.wait()
        except KeyboardInterrupt:
            stop(server)


if __name__ == "__main__":
    main()
