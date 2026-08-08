"""Tests for the Kaggle GGUF server helper.

Only the parts that are pure logic and have a real failure mode: quant
resolution (a wrong guess is a 404 halfway through a notebook) and readiness
polling (a dead server must surface its log, not time out silently). Building
and downloading need a GPU and the network, so they're left to the notebook.

scripts/ isn't a package, so the module is loaded by path rather than imported.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "gguf_server.py"

REPO_FILES = [
    "README.md",
    "config.json",
    "mtp-gemma-4-12b-it.gguf",
] + [
    f"gemma-4-12b-it-{quant}.gguf"
    for quant in ("BF16", "Q3_K_M", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_K_M", "UD-Q3_K_XL", "UD-Q4_K_XL")
]


def _load_module():
    spec = importlib.util.spec_from_file_location("gguf_server", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations by looking its
    # own module up in sys.modules, and fails on a module that isn't there yet.
    sys.modules["gguf_server"] = module
    spec.loader.exec_module(module)
    return module


gguf_server = _load_module()


@pytest.fixture
def stub_hub(monkeypatch):
    """Stands in for huggingface_hub with a fixed repo listing."""

    def install(files):
        module = types.ModuleType("huggingface_hub")
        module.HfApi = lambda: types.SimpleNamespace(list_repo_files=lambda repo_id: list(files))
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    return install


def test_resolves_a_single_file_quant(stub_hub):
    stub_hub(REPO_FILES)
    assert gguf_server.resolve_quant_files("unsloth/gemma-4-12b-it-GGUF", "Q4_K_M") == ["gemma-4-12b-it-Q4_K_M.gguf"]


def test_quant_match_is_not_a_bare_substring(stub_hub):
    """Q4_K_M must not also pull in Q4_K_S or UD-Q4_K_XL — the delimiter in the
    match is what keeps a 4-bit request from resolving to a different quant."""
    stub_hub(REPO_FILES)
    assert gguf_server.resolve_quant_files("repo", "Q4_K_M") == ["gemma-4-12b-it-Q4_K_M.gguf"]
    assert gguf_server.resolve_quant_files("repo", "UD-Q4_K_XL") == ["gemma-4-12b-it-UD-Q4_K_XL.gguf"]


def test_companion_ggufs_are_excluded(stub_hub):
    """mmproj (vision tower) and mtp (multi-token-prediction) files sit beside
    the quants; loading one as the model would fail at startup."""
    stub_hub(["gemma-4-12b-it-Q4_K_M.gguf", "mmproj-BF16.gguf", "mtp-gemma-4-12b-it.gguf"])
    assert gguf_server.resolve_quant_files("repo", "Q4_K_M") == ["gemma-4-12b-it-Q4_K_M.gguf"]


def test_sharded_quant_returns_every_shard_first_one_leading(stub_hub):
    """llama.cpp opens the first shard and finds the rest itself, but all of
    them have to be on disk."""
    stub_hub(
        [
            "gemma-4-12b-it-Q4_K_M-00002-of-00002.gguf",
            "gemma-4-12b-it-Q4_K_M-00001-of-00002.gguf",
        ]
    )
    assert gguf_server.resolve_quant_files("repo", "Q4_K_M") == [
        "gemma-4-12b-it-Q4_K_M-00001-of-00002.gguf",
        "gemma-4-12b-it-Q4_K_M-00002-of-00002.gguf",
    ]


def test_unknown_quant_lists_what_is_available(stub_hub):
    stub_hub(REPO_FILES)
    with pytest.raises(gguf_server.ServerError) as excinfo:
        gguf_server.resolve_quant_files("repo", "Q4_K_XXL")
    message = str(excinfo.value)
    assert "Q4_K_XXL" in message
    assert "gemma-4-12b-it-Q4_K_M.gguf" in message


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


def test_ready_when_health_reports_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(gguf_server, "_get_json", lambda url, timeout=5.0: {"status": "ok"})
    gguf_server.wait_until_ready(8000, _FakeProcess(), tmp_path / "log", timeout=5.0)


def test_dead_server_fails_immediately_with_its_log(monkeypatch, tmp_path):
    """A model llama.cpp can't load exits in seconds. Waiting out the full
    timeout would bury the one line that says why."""
    monkeypatch.setattr(gguf_server, "_get_json", lambda url, timeout=5.0: None)
    monkeypatch.setattr(gguf_server, "READY_POLL_SECONDS", 0)
    log_path = tmp_path / "log"
    log_path.write_text("error: unknown model architecture: 'gemma4'")

    with pytest.raises(gguf_server.ServerError) as excinfo:
        gguf_server.wait_until_ready(8000, _FakeProcess(exit_after=0, returncode=1), log_path, timeout=30.0)

    assert "unknown model architecture" in str(excinfo.value)


def test_timeout_terminates_the_process(monkeypatch, tmp_path):
    monkeypatch.setattr(gguf_server, "_get_json", lambda url, timeout=5.0: None)
    monkeypatch.setattr(gguf_server, "READY_POLL_SECONDS", 0)
    process = _FakeProcess()

    with pytest.raises(gguf_server.ServerError, match="not ready"):
        gguf_server.wait_until_ready(8000, process, tmp_path / "log", timeout=0.01)

    assert process.terminated


def test_compute_capability_is_formatted_for_cmake(monkeypatch):
    """CMAKE_CUDA_ARCHITECTURES wants "75", nvidia-smi reports "7.5"."""
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gguf_server,
        "_run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="7.5\n7.5\n", stderr=""),
    )
    assert gguf_server.gpu_compute_capability() == "75"


def test_no_compute_capability_without_a_gpu(monkeypatch):
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: None)
    assert gguf_server.gpu_compute_capability() is None


def _fake_cuda_home(tmp_path, stub_relative_dir):
    """A <cuda_home>/bin/nvcc plus a libcuda.so stub at the given relative
    path, mimicking one of the layouts CUDA toolkit installs actually use."""
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    stub_dir = cuda_home / stub_relative_dir
    stub_dir.mkdir(parents=True)
    (stub_dir / "libcuda.so").write_text("stub")
    return cuda_home, stub_dir, nvcc


def test_finds_stub_in_the_classic_lib64_stubs_layout(monkeypatch, tmp_path):
    _, stub_dir, nvcc = _fake_cuda_home(tmp_path, "lib64/stubs")
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)
    assert gguf_server._find_cuda_driver_stub().resolve() == stub_dir.resolve()


def test_finds_stub_in_the_multiarch_targets_layout(monkeypatch, tmp_path):
    """Newer CUDA 12.x installs put stubs under targets/<arch>/lib/stubs
    instead of a flat lib64/stubs — the layout mismatch that made CMake's own
    FindCUDAToolkit search miss it on Kaggle's image in the first place."""
    _, stub_dir, nvcc = _fake_cuda_home(tmp_path, "targets/x86_64-linux/lib/stubs")
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)
    assert gguf_server._find_cuda_driver_stub().resolve() == stub_dir.resolve()


def test_no_stub_found_returns_none_without_raising(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)
    assert gguf_server._find_cuda_driver_stub() is None


def test_no_stub_search_without_nvcc(monkeypatch):
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: None)
    assert gguf_server._find_cuda_driver_stub() is None


def test_staged_stub_has_both_versioned_and_unversioned_names(monkeypatch, tmp_path):
    """find_library() needs libcuda.so; linking a .so that depends on the
    stub needs libcuda.so.1 too, since that's the SONAME baked into the file
    itself. Both must exist side by side in a directory we can write to."""
    _, stub_dir, nvcc = _fake_cuda_home(tmp_path, "lib64/stubs")
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)

    install_root = tmp_path / "install"
    staged = gguf_server._stage_cuda_driver_stub(install_root)

    real_stub = (stub_dir / "libcuda.so").resolve()
    assert (staged / "libcuda.so").resolve() == real_stub
    assert (staged / "libcuda.so.1").resolve() == real_stub


def test_staging_is_idempotent(monkeypatch, tmp_path):
    _, _, nvcc = _fake_cuda_home(tmp_path, "lib64/stubs")
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)

    install_root = tmp_path / "install"
    first = gguf_server._stage_cuda_driver_stub(install_root)
    second = gguf_server._stage_cuda_driver_stub(install_root)
    assert first == second


def test_staging_returns_none_when_no_stub_exists(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    monkeypatch.setattr(gguf_server.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)
    assert gguf_server._stage_cuda_driver_stub(tmp_path / "install") is None
