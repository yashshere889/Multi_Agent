"""The HTTP layer: forms in, HTML fragments out.

Deliberately thin. Every route either reads a run directory or asks `RunStore`
to start/stop a process — no route ever runs pipeline work itself, so a request
can't block on an agent and the server stays responsive while a run takes an
hour.

The browser keeps up by polling two fragments every couple of seconds rather
than holding a stream open. On a run measured in tens of minutes that costs
nothing, and it survives an SSH tunnel dropping and reconnecting with no
server-side connection state to rebuild — which matters, because the intended
deployment is a tunnel to a compute node.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from research_pipeline import checkpointer
from research_pipeline.config import settings
from research_pipeline.webapp import events, runs, stages

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

# How many log lines the log pane shows. Enough to follow what's happening
# without re-sending a whole run's output on every two-second poll.
LOG_TAIL = 200


def _duration(start: Optional[str], end: Optional[str]) -> str:
    if not start:
        return ""
    try:
        started = datetime.fromisoformat(start)
        finished = datetime.fromisoformat(end) if end else datetime.now(started.tzinfo)
    except ValueError:
        return ""
    total = int((finished - started).total_seconds())
    hours, remainder = divmod(max(total, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m" if hours else (f"{minutes}m {seconds}s" if minutes else f"{seconds}s")


def _short_time(ts: Optional[str]) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S") if ts else ""
    except ValueError:
        return ""


def _is_active(record: dict) -> bool:
    return record.get("status") not in runs.TERMINAL_STATUSES


def _can_resume(record: dict) -> bool:
    """Whether to offer a resume for this run. Both halves matter: the run has
    to have stopped without finishing, and checkpoints have to outlive the
    process that wrote them. Under the default in-memory backend there is
    nothing to resume *to*, and relaunching would quietly redo the whole
    pipeline under this run's id — an hour of work behind a button labelled
    Resume, which is the opposite of what it says."""
    return record.get("status") in runs.RESUMABLE_STATUSES and checkpointer.is_durable()


def _back_to_run(run_id: str, message: str) -> RedirectResponse:
    """POST/redirect/GET, with the refusal carried in the URL so a reload
    doesn't re-submit the action that was just refused."""
    return RedirectResponse(f"/runs/{run_id}?error={quote(message)}", status_code=303)


def _latest_paper(store: runs.RunStore, record: dict) -> Optional[Path]:
    """The PDF worth showing right now: the final one if the run finished, and
    otherwise the newest draft the Writer has produced so far — every iteration
    keeps its own vN.pdf, so a paper is readable long before the run ends."""
    final_result = record.get("final_result") or {}
    if final_result.get("final_paper_path"):
        try:
            candidate = store.resolve_inside(record["run_id"], final_result["final_paper_path"])
            if candidate.exists():
                return candidate
        except runs.RunNotFound:
            logger.warning("Run %s reported a paper path outside its own directory", record["run_id"])

    drafts = sorted((store.run_dir(record["run_id"]) / "outputs").glob("v*.pdf"))
    return drafts[-1] if drafts else None


def create_app(run_store: Optional[runs.RunStore] = None) -> FastAPI:
    """A factory rather than a module-level app, so tests can hand it a
    tmp_path-backed store instead of the configured one."""
    store = run_store or runs.RunStore()

    app = FastAPI(title="Research pipeline", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["duration"] = _duration
    templates.env.filters["short_time"] = _short_time

    def render(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name=name, context=context)

    def progress_context(run_id: str) -> dict:
        record = store.get(run_id)
        stage_events = events.read_events(store.run_dir(run_id), types=[events.STAGE_COMPLETED])
        active = _is_active(record)
        paper = _latest_paper(store, record)
        return {
            "run": record,
            "active": active,
            "can_resume": _can_resume(record),
            "rows": stages.build_progress(stage_events, active, (record.get("params") or {}).get("max_iterations")),
            "paper_name": paper.name if paper else None,
        }

    @app.exception_handler(runs.RunNotFound)
    async def _run_not_found(request: Request, exc: runs.RunNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(request, "index.html", runs=store.list_runs(), error=None, settings=settings)

    @app.get("/partials/runs", response_class=HTMLResponse)
    def runs_fragment(request: Request):
        return render(request, "partials/_runs.html", runs=store.list_runs())

    @app.post("/runs")
    def create_run(
        request: Request,
        question: str = Form(...),
        max_results: int = Form(settings.default_max_results_per_query),
        max_iterations: Optional[int] = Form(None),
        quality_threshold: Optional[int] = Form(None),
    ):
        question = question.strip()
        if not question:
            return render(request, "index.html", runs=store.list_runs(), error="A research question is required.", settings=settings)

        # Refused rather than queued: the pipeline talks to one LLM endpoint, so
        # a second concurrent run competes with the first for it instead of
        # finishing sooner. Saying so is more useful than a silent backlog.
        if store.active_count() >= settings.webapp_max_concurrent_runs:
            return render(
                request,
                "index.html",
                runs=store.list_runs(),
                error=(
                    f"{store.active_count()} run(s) already in progress and WEBAPP_MAX_CONCURRENT_RUNS "
                    f"is {settings.webapp_max_concurrent_runs}. Wait for one to finish, or cancel it."
                ),
                settings=settings,
            )

        record = store.create(
            question,
            {
                "max_results_per_query": max_results,
                "max_iterations": max_iterations,
                "quality_threshold": quality_threshold,
            },
        )
        store.launch(record["run_id"])
        return RedirectResponse(f"/runs/{record['run_id']}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str, error: Optional[str] = None):
        return render(request, "run.html", run=store.get(run_id), error=error)

    @app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
    def progress_fragment(request: Request, run_id: str):
        return render(request, "partials/_progress.html", **progress_context(run_id))

    @app.get("/runs/{run_id}/log", response_class=HTMLResponse)
    def log_fragment(request: Request, run_id: str):
        record = store.get(run_id)
        log_events = events.read_events(store.run_dir(run_id), types=[events.LOG], tail=LOG_TAIL)
        return render(request, "partials/_log.html", run=record, active=_is_active(record), lines=log_events)

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        store.cancel(run_id)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/resume")
    def resume_run(run_id: str):
        """Relaunch a run that stopped without finishing. The runner decides
        what resuming means — it continues from this run's checkpoints if there
        are any — so this route only has to refuse the cases where launching a
        second process would be wrong."""
        record = store.get(run_id)

        # Re-checked here rather than trusted from the button being rendered: a
        # stale page, a bookmarked URL or a curl can all POST this. Two runners
        # on one run directory would interleave events and fight over the same
        # thread_id's checkpoints.
        if record.get("status") not in runs.RESUMABLE_STATUSES:
            return _back_to_run(run_id, f"A {record.get('status')} run cannot be resumed.")

        # Same limit, and the same reasoning, as starting a new run: one LLM
        # endpoint, so a second concurrent run competes with the first.
        if store.active_count() >= settings.webapp_max_concurrent_runs:
            return _back_to_run(run_id, "Too many runs already in progress. Wait for one to finish, or cancel it.")

        store.launch(run_id)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}/paper")
    def paper(run_id: str, file: Optional[str] = None):
        record = store.get(run_id)
        if file:
            # Comes from a link the UI built out of agent output, so it's
            # re-checked here rather than trusted: resolve_inside refuses
            # anything that escapes this run's directory.
            path = store.resolve_inside(run_id, Path("outputs") / Path(file).name)
        else:
            path = _latest_paper(store, record)
        if not path or not path.exists() or path.suffix.lower() != ".pdf":
            raise runs.RunNotFound(f"no such paper for run {run_id}")
        return FileResponse(path, media_type="application/pdf", filename=f"{run_id[:8]}_{path.name}")

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        record = store.get(run_id)
        run_dir = store.run_dir(run_id)
        stage_events = events.read_events(run_dir, types=[events.STAGE_COMPLETED])
        return {
            **record,
            "active": _is_active(record),
            "stages_completed": [e.get("stage") for e in stage_events],
            "stages": stages.build_progress(stage_events, _is_active(record), (record.get("params") or {}).get("max_iterations")),
        }

    @app.get("/api/runs")
    def list_runs():
        return {"runs": store.list_runs()}

    return app
