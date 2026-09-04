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


# -- job_state ----------------------------------------------------------------------------


def test_job_state_reports_an_error_off_cluster(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: None)
    state, error = slurm_submit.job_state("101")
    assert state is None
    assert "sacct" in error


def test_job_state_asks_only_about_the_allocation(monkeypatch):
    """Without -X, sacct also returns a row per step, and a .batch step reading
    COMPLETED under a CANCELLED job would have a reconcile pass import results
    from a run that was killed."""
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _completed(stdout="COMPLETED\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert slurm_submit.job_state("101") == ("COMPLETED", None)
    assert "-X" in seen["cmd"]
    assert "101" in seen["cmd"]


def test_job_state_drops_a_trailing_state_qualifier(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _completed(stdout="CANCELLED by 45678\n")
    )

    state, error = slurm_submit.job_state("101")
    assert (state, error) == ("CANCELLED", None)
    assert slurm_submit.is_terminal(state)


def test_job_state_reports_a_purged_or_unknown_job_as_an_error(monkeypatch):
    # No row at all: either the id is wrong or accounting retention dropped it.
    # Either way it is "don't know", never "the job failed".
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="\n  \n"))

    state, error = slurm_submit.job_state("101")
    assert state is None
    assert "no accounting record" in error


def test_job_state_reports_a_failing_sacct_as_an_error(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stderr="boom", returncode=1))

    state, error = slurm_submit.job_state("101")
    assert state is None
    assert "boom" in error


def test_job_state_reports_a_timeout_as_an_error(monkeypatch):
    monkeypatch.setattr(slurm_submit.shutil, "which", lambda name: f"/usr/bin/{name}")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sacct", timeout=30)

    monkeypatch.setattr(subprocess, "run", timeout)

    state, error = slurm_submit.job_state("101")
    assert state is None
    assert "did not return" in error


def test_running_and_pending_states_are_not_terminal():
    for state in ("PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "REQUEUED"):
        assert not slurm_submit.is_terminal(state)
    for state in ("COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED"):
        assert slurm_submit.is_terminal(state)
