import subprocess
from types import SimpleNamespace

from research_pipeline.agents.coder import slurm_submit


def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


# -- count_running_jobs -------------------------------------------------------------------


def test_count_running_jobs_returns_zero_off_cluster(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: None)
    assert slurm_submit.count_running_jobs() == 0


def test_count_running_jobs_counts_squeue_lines(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="101\n102\n103\n"))
    assert slurm_submit.count_running_jobs("someone") == 3


def test_count_running_jobs_blocks_submission_when_squeue_fails(monkeypatch):
    # A failed probe must not read as "queue is empty", or the cap stops
    # capping exactly when the scheduler is unhealthy.
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stderr="boom", returncode=1))
    assert slurm_submit.count_running_jobs() >= 10_000


def test_count_running_jobs_blocks_submission_on_timeout(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="squeue", timeout=30)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert slurm_submit.count_running_jobs() >= 10_000


# -- submit_job ---------------------------------------------------------------------------


def test_submit_job_refuses_when_sbatch_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: None)
    job_id, error = slurm_submit.submit_job(tmp_path / "run.sbatch", tmp_path)
    assert job_id is None
    assert "not on PATH" in error


def test_submit_job_parses_the_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _completed(stdout="Submitted batch job 12345\n")
    )
    job_id, error = slurm_submit.submit_job(tmp_path / "run.sbatch", tmp_path)
    assert job_id == "12345"
    assert error is None


def test_submit_job_reports_a_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _completed(stderr="invalid partition specified", returncode=1),
    )
    job_id, error = slurm_submit.submit_job(tmp_path / "run.sbatch", tmp_path)
    assert job_id is None
    assert "invalid partition" in error


def test_submit_job_reports_unparseable_output(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _completed(stdout="something unexpected")
    )
    job_id, error = slurm_submit.submit_job(tmp_path / "run.sbatch", tmp_path)
    assert job_id is None
    assert "could not parse a job id" in error


def test_submit_job_reports_a_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sbatch", timeout=30)

    monkeypatch.setattr(subprocess, "run", timeout)
    job_id, error = slurm_submit.submit_job(tmp_path / "run.sbatch", tmp_path)
    assert job_id is None
    assert "did not return" in error
