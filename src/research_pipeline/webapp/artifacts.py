"""What a run actually wrote to disk, as data the UI can render.

The run page shows each stage's *summary* — the numbers an agent reported about
its own work. This module covers the other half: the files themselves. Every
stage already persists its own output (that is the point of `run_<name>_agent()`
writing before returning), the Coder keeps every generated program and every
failed fix attempt, and the Writer keeps every draft, not just the last. All of
it lands under one run directory and, until now, only the newest PDF was
reachable from the browser.

Pure functions over a `Path`, like `stages.py` is pure functions over an event
delta: no FastAPI, no `RunStore`, nothing to mock in a test. The HTTP layer adds
exactly one thing on top — `RunStore.resolve_inside`, which is what actually
refuses a path that escapes the run directory. Nothing here is a security
boundary; treat `rel` values as untrusted until they have been through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

# Directories that are build residue rather than results. `__pycache__` appears
# next to every generated experiment the moment it runs, and a `.venv` is
# thousands of files that would bury the eight that matter — the Coder Agent
# provisions one per experiment (off the quota-bearing filesystem, but a run
# reached by a symlink or an older layout can still surface one here).
SKIP_DIRS = frozenset({"__pycache__", ".venv", "venv", ".git", "node_modules", ".pytest_cache", ".ipynb_checkpoints"})

# A listing is a page someone reads, not an inventory. A run that downloaded 200
# papers is real, so the cap is generous, but it is a cap: the browser should
# never be handed an unbounded table.
MAX_ENTRIES = 750

# Inline preview budget. Past this the viewer shows the head of the file and
# points at the raw download instead of rendering megabytes into the DOM.
MAX_PREVIEW_BYTES = 512 * 1024

# Extensions worth showing *in* the page. Everything else is offered as a
# download — guessing at a binary format is how you paint a terminal with a JPEG.
TEXT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".txt", ".log", ".md", ".py", ".sh", ".sbatch", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
)

PDF = "pdf"
JSON_KIND = "json"
TEXT = "text"
BINARY = "binary"

# Top-level directories in the order a person cares about them, with a heading
# each. A run directory holds a handful of loose files too (run.json,
# events.jsonl, stdout.log), which get their own group rather than being sorted
# in among the outputs.
GROUP_LABELS = {
    "outputs": "Stage outputs and drafts",
    "experiments": "Generated experiments",
    "papers": "Downloaded papers",
    "": "Run files",
}
GROUP_ORDER = ("outputs", "experiments", "papers", "")


@dataclass(frozen=True)
class Artifact:
    """One file under a run directory. `rel` is relative to the run root and is
    what a link round-trips through the server, so it is the only field the
    caller may hand back to `resolve_inside`."""

    rel: str
    name: str
    size: int
    modified: str
    kind: str
    group: str

    @property
    def viewable(self) -> bool:
        """Whether the viewer can render this in the page. A PDF is excluded on
        purpose: it is served raw and opened by the browser's own viewer."""
        return self.kind in (JSON_KIND, TEXT)


def kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDF
    if suffix in (".json", ".jsonl"):
        return JSON_KIND
    if suffix in TEXT_SUFFIXES:
        return TEXT
    return BINARY


def _iter_files(root: Path) -> Iterator[Path]:
    """Every file under `root`, skipping build residue. Sorted at each level so
    the listing is stable between polls rather than reordering under the reader
    — `Path.iterdir` gives directory order, which is not."""
    try:
        children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except (OSError, PermissionError):
        return
    for child in children:
        if child.name in SKIP_DIRS:
            continue
        # Not followed rather than not shown: a symlinked directory could walk
        # out of the run, and resolve_inside would then refuse every link this
        # listing had already rendered.
        if child.is_symlink():
            continue
        if child.is_dir():
            yield from _iter_files(child)
        elif child.is_file():
            yield child


def _modified(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def list_artifacts(run_dir: str | Path, limit: int = MAX_ENTRIES) -> tuple[list[Artifact], bool]:
    """Everything this run wrote, and whether the listing hit `limit`.

    The truncation flag is returned rather than logged because the page has to
    say so: a table that silently stops at 750 rows reads as "that is all there
    is", which for a big sweep would be a lie.
    """
    root = Path(run_dir)
    if not root.is_dir():
        return [], False

    artifacts: list[Artifact] = []
    truncated = False
    for path in _iter_files(root):
        if len(artifacts) >= limit:
            truncated = True
            break
        rel = path.relative_to(root)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        artifacts.append(
            Artifact(
                rel=rel.as_posix(),
                name=rel.as_posix(),
                size=size,
                modified=_modified(path),
                kind=kind_for(path),
                group=rel.parts[0] if len(rel.parts) > 1 else "",
            )
        )
    return artifacts, truncated


def group_artifacts(artifacts: list[Artifact]) -> list[tuple[str, str, list[Artifact]]]:
    """`(group_key, heading, files)` in GROUP_ORDER, dropping empty groups.

    A group not in GROUP_ORDER — some directory a future stage invents — still
    appears, after the known ones and under its own name, rather than vanishing.
    """
    by_group: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        by_group.setdefault(artifact.group, []).append(artifact)

    known = [g for g in GROUP_ORDER if g in by_group]
    unknown = sorted(g for g in by_group if g not in GROUP_ORDER)
    return [(g, GROUP_LABELS.get(g, g), by_group[g]) for g in known + unknown]


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    for unit, scale in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)):
        if size < scale * 1024 or unit == "GB":
            value = size / scale
            return f"{value:.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"
    return f"{size} B"


@dataclass(frozen=True)
class Preview:
    text: str
    truncated: bool
    kind: str
    error: Optional[str] = None


def read_preview(path: Path, max_bytes: int = MAX_PREVIEW_BYTES) -> Preview:
    """The text to show in the viewer.

    JSON is re-indented, because every file this pipeline writes is `json.dump`ed
    compactly enough to be unreadable at width — and re-indenting is also the
    check on whether it *is* valid JSON. A file that fails to parse is shown
    verbatim instead of erroring: a half-written output file, mid-run, is
    exactly when someone wants to look at it.
    """
    kind = kind_for(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return Preview(text="", truncated=False, kind=kind, error=str(exc))

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]

    text = raw.decode("utf-8", errors="replace")
    if kind == JSON_KIND and not truncated and path.suffix.lower() == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
    return Preview(text=text, truncated=truncated, kind=kind)
