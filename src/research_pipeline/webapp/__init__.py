"""Web UI for starting a pipeline run and watching it happen.

`research-pipeline orchestrate` blocks for tens of minutes with no output until
it finishes. This package wraps the same orchestrator graph in a small
FastAPI + HTMX app that shows each stage completing as it completes.

Three processes, not one:

    browser  <--poll--  FastAPI server (app.py)  <--files--  runner (runner.py)

The server never runs a pipeline itself. `POST /runs` spawns
`python -m research_pipeline.webapp.runner <run_dir>` and then only ever reads
files that subprocess writes. That separation is what makes the run
cancellable (a thread inside graph.invoke() cannot be interrupted, a
subprocess takes SIGTERM), keeps a crash in the Coder Agent's sandbox from
taking the server down with it, lets the server restart without losing
in-flight runs, and gives each run its own CODER_EXPERIMENTS_DIR — settings are
read once at import, so that can only be varied per process.

Everything here is optional: the package is imported lazily by `cli.py` so the
CLI keeps working without `uv sync --extra webapp`.
"""

from __future__ import annotations
