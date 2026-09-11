"""What real data is already on disk, stated before anyone plans around it.

The Hypothesis Agent ranks on feasibility "with data that plausibly exists" and
the Experiment Planner chooses a data source, and neither was ever told what
data exists. Barkla job 10495925 ranked a synthetic-only methods benchmark first,
passing over the alternatives for "difficulty in obtaining real-world data",
while real US market returns (Fama-French, 1926-2026) and Shiller's CPI and
long-rate series sat in CODER_DATA_DIR. Its verdict was then withheld, as a
synthetic input's must be.

The inventory is read off the files themselves by `acquire.describe_local`
rather than from a description a human may not have updated, and each file is
listed under the name `provenance._staged_file` matches on — so a plan that
names a listed file resolves to that file. Alias symlinks are left out: they
exist to catch the Coder's phrasings, and listing them would show one dataset
several times under names nobody would choose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_pipeline.config import settings

MAX_FILES = 25
MAX_ROW_CHARS = 240
# Long enough to reach a units line: the Fama-French daily entry shares its notes
# with the monthly one, and at 300 characters "UNITS: percent per period" was cut.
MAX_NOTE_CHARS = 600


def inventory(staging_dir: Path | None) -> list[dict[str, Any]]:
    """One entry per real staged file: name, columns, row count, edge rows, notes."""
    if not staging_dir or not staging_dir.is_dir():
        return []
    # Imported here: the coder package is heavy, and the agents importing this
    # module only need it when a staging directory is actually configured.
    from research_pipeline.agents.coder import acquire

    entries: list[dict[str, Any]] = []
    for candidate in sorted(staging_dir.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file() or candidate.name.startswith("."):
            continue
        # By suffix, not by whether it parses: prose with commas in it reads as a
        # two-column CSV, and the staging README would be listed as a dataset.
        if candidate.suffix.lower() not in acquire.DATA_SUFFIXES:
            continue
        preview = acquire.describe_local(candidate)
        if not preview:
            continue
        entries.append(
            {
                "file": candidate.name,
                "columns": preview.get("columns") or [],
                "row_count": preview.get("row_count"),
                "first_row": (preview.get("sample_rows") or [None])[0],
                "last_row": (preview.get("last_rows") or [None])[-1],
                "notes": preview.get("notes") or "",
            }
        )
        if len(entries) >= MAX_FILES:
            break
    return entries


def _staging_dir() -> Path | None:
    return Path(settings.coder_data_dir) if settings.coder_data_dir else None


def prompt_block(instruction: str, staging_dir: Path | None = None) -> str:
    """The inventory as a prompt block, or "" when nothing is staged — so a
    prompt is byte-identical to what it was before this block existed."""
    entries = inventory(staging_dir if staging_dir is not None else _staging_dir())
    if not entries:
        return ""
    lines = ["", "Real datasets already on disk for experiments (described from the files themselves):"]
    for entry in entries:
        rows = f", {entry['row_count']} rows" if entry["row_count"] is not None else ""
        lines.append(f"- {entry['file']}{rows}; columns: {', '.join(map(str, entry['columns']))}")
        for label in ("first_row", "last_row"):
            if entry[label]:
                lines.append(f"    {label.replace('_', ' ')}: {json.dumps(entry[label], default=str)[:MAX_ROW_CHARS]}")
        if entry["notes"]:
            lines.append(f"    notes: {' '.join(entry['notes'].split())[:MAX_NOTE_CHARS]}")
    lines.append(instruction)
    return "\n".join(lines) + "\n"
