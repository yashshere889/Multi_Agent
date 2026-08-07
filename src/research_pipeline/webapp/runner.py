"""The subprocess that actually runs a pipeline, one run per process.

    python -m research_pipeline.webapp.runner <run_dir>

Spawned by the web server, which then never touches it again except to read
the files it writes and, on cancel, to send it SIGTERM. Running the graph out
of process is what makes a run cancellable at all — a thread sitting inside
`graph.invoke()` cannot be interrupted — and it keeps a crash in the Coder
Agent's sandbox from taking the server with it.

Progress comes from `graph.stream(..., stream_mode="updates")`, which yields
`{node_name: delta}` as each node returns. The orchestrator graph is used
exactly as the CLI uses it; nothing about it is modified or re-implemented
here.
"""

from __future__ import annotations

import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

from research_pipeline.config import settings
from research_pipeline.orchestrator.graph import build_pipeline_graph
from research_pipeline.webapp import events, runs, stages

logger = logging.getLogger(__name__)


class EventLogHandler(logging.Handler):
    """Copies log records into the run's event stream so the browser sees the
    same progress the CLI prints to stderr."""

    def __init__(self, event_log: events.EventLog) -> None:
        super().__init__()
        self.event_log = event_log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.event_log.log(record.levelname, record.name, record.getMessage())
        except Exception:  # noqa: BLE001
            # A logging handler must never raise into the code that logged:
            # losing a line from the UI is not a reason to fail a run.
            self.handleError(record)


class _PipelineFilter(logging.Filter):
    """Everything this pipeline says, plus warnings and errors from anything
    else. Without the second half a failing HTTP client is invisible; without
    the first, INFO chatter from httpx and urllib3 buries the run."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("research_pipeline.webapp.events"):
            return False  # would recurse through the handler that writes them
        return record.name.startswith("research_pipeline") or record.levelno >= logging.WARNING


def _configure_logging(event_log: events.EventLog, verbose: bool = False) -> logging.Handler:
    # Same stderr format as the CLI, which here lands in the run's stdout.log —
    # so if the event stream itself is what broke, the plain text is still there.
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    # basicConfig silently does nothing when the root logger already has a
    # handler, which would leave the level at WARNING and drop every INFO
    # record before it ever reached the handler below.
    root.setLevel(level)

    handler = EventLogHandler(event_log)
    handler.setLevel(logging.INFO)
    handler.addFilter(_PipelineFilter())
    root.addHandler(handler)
    return handler


def _install_cancel_handler(event_log: events.EventLog, store: runs.RunStore, run_id: str) -> None:
    """Record the cancellation before exiting, so a cancelled run reads as
    cancelled rather than as a process that mysteriously vanished.

    `sys.exit` raises SystemExit, which is a BaseException — so it passes
    straight through the `except Exception` in main() below and through any
    broad handler inside the agents, instead of being mistaken for a run
    failure.
    """

    def handle(signum, _frame):
        event_log.append(events.RUN_CANCELLED, reason=f"received signal {signum}")
        store.update(run_id, status=runs.CANCELLED, finished_at=events.utc_now(), error="Cancelled by the user.")
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def run(run_dir: str | Path) -> dict:
    """Run one pipeline to completion, reporting as it goes. Returns the run's
    final record."""
    run_dir = Path(run_dir)
    store = runs.RunStore(run_dir.parent)
    run_id = run_dir.name
    record = store.get(run_id)

    event_log = events.EventLog(run_dir)
    _configure_logging(event_log)
    _install_cancel_handler(event_log, store, run_id)

    params = record.get("params") or {}
    max_iterations = params.get("max_iterations")
    papers_dir = run_dir / "papers"

    inputs = {
        "research_question": record["question"],
        # Defaulted here rather than left as None: the literature node reads it
        # with `state.get(key, default)`, which only falls back when the key is
        # absent, not when it's present and null. max_iterations and
        # quality_threshold below are checked with `is None` downstream, so
        # they can pass through unset.
        "max_results_per_query": params.get("max_results_per_query") or settings.default_max_results_per_query,
        "download_dir": str(papers_dir),
        "metadata_path": str(papers_dir / "metadata.json"),
        "output_dir": str(run_dir / "outputs"),
        "max_iterations": max_iterations,
        "quality_threshold": params.get("quality_threshold"),
    }

    event_log.append(events.RUN_STARTED, question=record["question"], params=params)
    logger.info("Starting run %s: %s", run_id, record["question"])

    final_result: Optional[dict] = None
    try:
        graph = build_pipeline_graph()
        for update in graph.stream(inputs, config={"configurable": {"thread_id": run_id}}, stream_mode="updates"):
            for node, delta in (update or {}).items():
                if node not in stages.STAGE_ORDER:
                    # LangGraph also emits bookkeeping keys such as __interrupt__.
                    logger.debug("Ignoring a non-stage stream key: %s", node)
                    continue
                event_log.append(events.STAGE_COMPLETED, stage=node, summary=stages.summarize(node, delta or {}))
                if node == stages.FINALIZE:
                    final_result = (delta or {}).get("final_result")

        if final_result is None:
            raise RuntimeError("The graph finished without producing a final_result.")
    except Exception as exc:  # noqa: BLE001
        # Broad for the same reason batch.py's is: this process exists to
        # report on a run, and an unreported failure would leave the UI
        # watching a run that silently stopped existing. The full traceback
        # goes to the event stream and to stdout.log.
        logger.exception("Run %s failed", run_id)
        event_log.append(
            events.RUN_FAILED,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-events.MAX_MESSAGE_CHARS :],
        )
        return store.update(
            run_id,
            status=runs.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=events.utc_now(),
        )

    # The event is written before run.json because RunStore._reconcile trusts
    # the event stream first when recovering a run whose process died.
    event_log.append(events.RUN_COMPLETED, final_result=final_result)
    logger.info(
        "Run %s finished: converged=%s after %s iteration(s)",
        run_id,
        final_result.get("converged"),
        final_result.get("iterations_run"),
    )
    return store.update(run_id, status=runs.COMPLETED, final_result=final_result, finished_at=events.utc_now())


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m research_pipeline.webapp.runner <run_dir>")
    run(argv[0])


if __name__ == "__main__":
    main()
