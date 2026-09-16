"""Turns Coder Agent run artefacts into a fine-tuning dataset.

    uv run python scripts/export_preference_data.py --format sft-fix --out fix.jsonl
    uv run python scripts/export_preference_data.py --format dpo --out pairs.jsonl
    uv run python scripts/export_preference_data.py --stats-only outputs/ runs/

Every run of the fix loop already produces the two halves of a training example
and throws away the join between them: `fix_attempts/attempt_<n>/` holds the
code that failed, `fix_history[n]` holds the error that rejected it and whether
the regeneration that followed cleared it. This script walks every
`coder_agent_summary_*.json` it is pointed at, re-assembles those into rows, and
writes JSONL.

Three shapes, because these artefacts genuinely support three and conflating
them would produce a dataset that is subtly wrong:

`--format sft-fix` (the default, and the one to reach for first)
    prompt = the fix prompt that was actually sent (plan, broken code, the
    concrete error); completion = the response that cleared that error. Only
    entries whose `resolved` is True qualify, so every row is a repair a real
    check certified. Nothing is reconstructed and nothing is assumed: both
    halves are verbatim from the transcript. This is the cheapest honest
    dataset here and it targets the loop's actual cost — a fix attempt that
    does not fix anything.

`--format dpo`
    prompt = the request that produced the failing attempt; rejected = that
    attempt's own response; chosen = the response that came after it and
    cleared the error. Note what this does and does not claim: the chosen
    response was produced under a *different*, more informative prompt (it had
    been shown the error). Pairing it against the earlier prompt is deliberate
    and standard for self-correction data — the point being taught is "this
    answer was better for this request", which is true — but it is not two
    samples from one prompt, and a reader of this file should know that.

    A regeneration that was *targeted* (coder_agent._target_sections asked for
    a subset of sections) returns only those sections, which is a complete
    answer to the fix prompt and an incomplete one to the original. Those rows
    are reconstructed by merging the new sections over the previous ones —
    exactly what `_assemble_generation`'s `previous=` merge does at runtime, and
    exactly the program that was then checked — and marked
    `chosen_reconstructed: true`. `--skip-targeted` drops them instead.

`--format sft`
    prompt and completion of the generation that ended the experiment, for
    experiments that reached a given status (default: `completed`). The
    imitation baseline: what a first attempt that worked looks like.

Deliberately standalone, for the same reason scripts/analyze_coder_fix_history.py
is: it imports nothing from `research_pipeline`, so it runs against artefacts
written by any version of the pipeline — including ones on a cluster where the
package isn't installed — without needing that version's code. The two things it
would otherwise import (the delimited wire format, and the run.py template's
section banners) are restated below, next to a note saying what they must stay
in sync with.

Provenance is recorded per row rather than assumed, because it varies:

    prompt_source       "transcript", or null when this artefact predates
                        agents/coder/transcript.py and the prompt is simply
                        gone. `--require-prompt` drops those rows.
    completion_source   "transcript" for the model's verbatim text; "sliced"
                        when it was recovered from the rendered run.py by
                        cutting it back apart at the template's banners, which
                        can recover the code sections and nothing else (no
                        README, no assumptions, no booleans) — such rows carry
                        `partial: true` and the field list they do have.

Run `--stats-only` first. It reports rows per `error_source` against the same
taxonomy analyze_coder_fix_history.py counts failures in, which is how you find
out whether a category has enough examples to train on before spending a GPU
allocation discovering that it doesn't.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("export_preference_data")

DEFAULT_ROOTS = ("outputs", "runs")
SUMMARY_GLOB = "coder_agent_summary_*.json"
TRANSCRIPT_FILENAME = ".transcript.json"

# ---------------------------------------------------------------------------
# The delimited wire format. Restated rather than imported (see the module
# docstring); must stay in sync with research_pipeline/llm_sections.py, which
# owns it. Only the rendering half is needed here — parsing is done by the
# transcripts' own recorded text, which is already whatever the model sent.
# ---------------------------------------------------------------------------


def render_section(field: str, content: str) -> str:
    return f"===BEGIN {field}===\n{content}\n===END {field}==="


def render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(render_section(f, c) for f, c in sections.items())


_SECTION_RE = re.compile(
    r"^[ \t]*===BEGIN[ \t]+(?P<name>[\w.\-/]+)[ \t]*===[ \t]*\r?\n"
    r"(?P<body>.*?)^[ \t]*===END[ \t]+(?P=name)[ \t]*===",
    re.DOTALL | re.MULTILINE,
)


def parse_sections(text: str) -> dict[str, str]:
    """Field names discovered from the text, bodies verbatim. Used to read a
    recorded response back when a row needs its sections rather than its text
    (the targeted-regeneration merge) — not to validate it: a response this
    can't parse is still a legitimate `rejected` sample, and dropping it would
    remove exactly the format failures worth training against."""
    out: dict[str, str] = {}
    for match in _SECTION_RE.finditer(text):
        body = match.group("body")
        if body.endswith("\r\n"):
            body = body[:-2]
        elif body.endswith("\n"):
            body = body[:-1]
        out[match.group("name")] = body
    return out


# ---------------------------------------------------------------------------
# Slicing a rendered run.py back into the sections the model wrote. The fallback
# path, for artefacts written before transcripts existed.
#
# Anchored on the banner comments in agents/coder/templates/run.py.template
# rather than on the template's exact text, so it survives edits to the fixed
# code between the banners — which is most of what changes in that file. Ordered
# most-specific first: "EXPERIMENT EXECUTION" must be tested before the bare
# "CONFIGURATION"/"MODEL" keywords could claim it.
# ---------------------------------------------------------------------------

_FIXED = "__fixed__"
_END = "__end__"

_BANNER_SECTIONS: list[tuple[str, str]] = [
    ("EXPERIMENT METADATA", _FIXED),
    ("ORCHESTRATION", _END),
    ("DATA LOADING", "load_data_function"),
    ("MODEL / ALGORITHM", "build_model_function"),
    ("EXPERIMENT EXECUTION", "run_experiment_function"),
    ("EVALUATION", "evaluate_function"),
    ("HELPER", "helpers"),
    ("CONFIGURATION", "configuration"),
]

# The template's last fixed import, immediately above __AGENT_IMPORTS__. The one
# anchor with nothing structural to hang off — the imports block has no banner
# of its own, it just sits between the template's preamble and the first banner.
_PREAMBLE_LAST_LINE = "from typing import Any"

# What render_experiment_with_spans substitutes for a section the model left
# empty. Mapped back to "" so a sliced reconstruction says the model sent
# nothing, which is what happened, rather than attributing the template's own
# placeholder to it.
_PLACEHOLDERS = {
    "# (no extra imports needed)",
    "# (no extra configuration needed)",
    "# (no helper functions needed)",
}

_BANNER_RULE = "# ═"


def _banner_section(title: str) -> str | None:
    upper = title.upper()
    for keyword, section in _BANNER_SECTIONS:
        if keyword in upper:
            return section
    return None


def _trim(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(lines)
    return "" if body.strip() in _PLACEHOLDERS else body


def slice_run_py(text: str) -> dict[str, str]:
    """The model-written sections of a rendered run.py, keyed as the wire format
    names them. Missing keys mean "couldn't be located", not "empty" — an empty
    section is present with an empty string."""
    lines = text.split("\n")
    # A banner is three consecutive lines: rule, title comment, rule.
    banners: list[tuple[int, int, str]] = []  # (start, end, section)
    i = 0
    while i < len(lines) - 2:
        if lines[i].startswith(_BANNER_RULE) and lines[i + 2].startswith(_BANNER_RULE):
            section = _banner_section(lines[i + 1])
            if section is not None:
                banners.append((i, i + 2, section))
            i += 3
            continue
        i += 1

    sections: dict[str, str] = {}
    for index, (_, end, section) in enumerate(banners):
        if section in (_FIXED, _END):
            continue
        stop = banners[index + 1][0] if index + 1 < len(banners) else len(lines)
        sections[section] = _trim(lines[end + 1 : stop])

    # imports: between the template preamble and the first banner of any kind.
    if banners:
        first_banner = banners[0][0]
        for index in range(first_banner - 1, -1, -1):
            if lines[index].strip() == _PREAMBLE_LAST_LINE:
                sections["imports"] = _trim(lines[index + 1 : first_banner])
                break
    return sections


# ---------------------------------------------------------------------------
# Locating artefacts
# ---------------------------------------------------------------------------


def find_summaries(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.startswith("coder_agent_summary_"):
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob(SUMMARY_GLOB)))
    return sorted(set(found))


def _relative_tail(code_path: Path) -> Path | None:
    """The `experiments/<hid>` tail of a path recorded somewhere else entirely.

    `code_path` is written as whatever it was *at run time*, which is routinely
    a path on another machine — `/kaggle/working/experiments/H1`, or a Barkla
    scratch mount — and the artefacts are then copied somewhere local under a
    different name. The tail from the last `experiments` component onwards is
    the part that survives that move.
    """
    parts = code_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "experiments":
            return Path(*parts[index:])
    return None


def resolve_experiment_dir(
    code_path: str, summary: Path, roots: list[Path], extra_roots: list[Path]
) -> tuple[Path | None, str]:
    """Locate an experiment directory, and say how confidently.

    Returns (path, how) where `how` is one of "exact" (the recorded path is
    still valid), "root" (found under a scan root or an explicit
    --experiments-root), "summary" (found under the summary's own directory),
    or "search" (found by scanning the summary's subtree for a directory of
    that name).

    "search" is reported rather than used silently because it is the one that
    can be wrong: several runs' outputs sitting side by side all contain an
    `experiments/H1`, and pairing one run's summary with another run's code
    would produce training rows whose error never described the code they are
    attached to. The search is confined to the summary's own subtree for that
    reason, and the count is printed at the end.
    """
    candidate = Path(code_path)
    if candidate.is_dir():
        return candidate, "exact"

    tail = _relative_tail(candidate)
    relatives = [candidate] if candidate == Path(candidate.name) else [candidate]
    if tail is not None and tail != candidate:
        relatives.append(tail)
    if not candidate.is_absolute():
        relatives.append(Path(candidate.name))

    # An explicit --experiments-root is the user asserting where these live, so
    # it outranks everything inferred.
    for base in extra_roots:
        for relative in [*relatives, Path(candidate.name)]:
            joined = base / relative
            if joined.is_dir():
                return joined, "root"

    # Then the summary's own neighbourhood: a run's summary and its experiments
    # are normally written under one directory.
    for base in (summary.parent, summary.parent.parent):
        for relative in relatives:
            joined = base / relative
            if joined.is_dir():
                return joined, "summary"

    for base in roots:
        for relative in relatives:
            joined = base / relative
            if joined.is_dir():
                return joined, "root"

    # Last resort, and confined to this summary's own subtree — see the
    # docstring for why it is not allowed to roam the scan roots.
    for base in (summary.parent, summary.parent.parent):
        if not base.is_dir():
            continue
        for match in sorted(base.rglob(candidate.name)):
            if match.is_dir() and match.parent.name == "experiments":
                return match, "search"
    return None, "none"


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# One attempt's artefacts, however much of them survived
# ---------------------------------------------------------------------------


class Attempt:
    """The code an attempt produced plus, when it exists, the exchange that
    produced it. Either half can be missing: an `invalid_format` attempt never
    wrote a run.py, and anything from before transcript.py has no exchange."""

    def __init__(self, directory: Path, ordinal: int | str) -> None:
        self.directory = directory
        self.ordinal = ordinal
        self.transcript: dict[str, Any] | None = None
        transcript_path = directory / TRANSCRIPT_FILENAME
        if transcript_path.is_file():
            loaded = read_json(transcript_path)
            if isinstance(loaded, dict):
                self.transcript = loaded
        self.run_py: str | None = None
        run_py_path = directory / "run.py"
        if run_py_path.is_file():
            try:
                self.run_py = run_py_path.read_text()
            except OSError as exc:
                logger.warning("Could not read %s: %s", run_py_path, exc)
        self.requirements: str | None = None
        requirements_path = directory / "requirements.txt"
        if requirements_path.is_file():
            try:
                self.requirements = requirements_path.read_text()
            except OSError:
                pass

    @property
    def prompt(self) -> str | None:
        if self.transcript:
            return self.transcript.get("user_prompt") or None
        return None

    @property
    def system_prompt(self) -> str | None:
        if self.transcript:
            return self.transcript.get("system_prompt") or None
        return None

    @property
    def raw_response(self) -> str | None:
        """The turn that was parsed — the last one, since `invoke_sections`
        records the repair turn after the turn that needed repairing."""
        if self.transcript:
            responses = self.transcript.get("raw_responses") or []
            if responses:
                return str(responses[-1])
        return None

    @property
    def needed_repair(self) -> bool:
        """Whether the transport's repair round-trip had to run. A completion
        that only parsed on the second ask is a worse example than one that
        parsed first time, and a dataset that can't tell them apart can't say
        so — see transcript.build."""
        if self.transcript:
            return len(self.transcript.get("raw_responses") or []) > 1
        return False

    @property
    def requested_fields(self) -> list[str] | None:
        if self.transcript:
            fields = self.transcript.get("field_names")
            if isinstance(fields, list):
                return [str(f) for f in fields]
        return None

    def completion(self) -> tuple[str | None, str, bool, list[str]]:
        """(text, source, partial, fields). Verbatim from the transcript when
        there is one; otherwise rebuilt from the rendered run.py, which can only
        ever carry the code."""
        raw = self.raw_response
        if raw is not None:
            return raw, "transcript", False, sorted(parse_sections(raw))
        if self.run_py is None:
            return None, "none", True, []
        sections = slice_run_py(self.run_py)
        if self.requirements is not None:
            sections["requirements_txt"] = self.requirements
        if not sections:
            return None, "none", True, []
        return render_sections(sections), "sliced", True, sorted(sections)

    def sections(self) -> dict[str, str]:
        raw = self.raw_response
        if raw is not None:
            return parse_sections(raw)
        if self.run_py is not None:
            sliced = slice_run_py(self.run_py)
            if self.requirements is not None:
                sliced["requirements_txt"] = self.requirements
            return sliced
        return {}


def attempt_ladder(experiment_dir: Path, fix_history: list[dict]) -> list[Attempt]:
    """Every attempt this experiment made, in order, with the code that ended
    the loop last. The final entry is the experiment directory itself: the fix
    loop leaves the accepted (or best — see coder_agent._best_candidate) attempt
    there, so it is the only place that code exists under its own name."""
    ladder = [
        Attempt(
            experiment_dir / "fix_attempts" / f"attempt_{entry.get('attempt', index + 1)}",
            entry.get("attempt", index + 1),
        )
        for index, entry in enumerate(fix_history)
    ]
    ladder.append(Attempt(experiment_dir, "final"))
    return ladder


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _base_row(summary: Path, experiment: dict, experiment_dir: Path) -> dict[str, Any]:
    return {
        "summary": str(summary),
        "experiment_dir": str(experiment_dir),
        "hypothesis_id": experiment.get("hypothesis_id", ""),
        "status": experiment.get("status", ""),
        "starter_used": experiment.get("starter_used", ""),
    }


def build_sft_fix_rows(
    summary: Path, experiment: dict, experiment_dir: Path, skipped: Counter
) -> list[dict]:
    """One row per fix that a real check certified: the fix prompt in, the
    response that cleared the error out. Needs the transcript on the *chosen*
    side only — the prompt and the completion are both in it — so it is the one
    shape that asks nothing of the rendered code."""
    history = experiment.get("fix_history") or []
    ladder = attempt_ladder(experiment_dir, history)
    rows = []
    for index, entry in enumerate(history):
        if not entry.get("resolved"):
            skipped["fix_not_resolved"] += 1
            continue
        chosen = ladder[index + 1]
        if chosen.prompt is None or chosen.raw_response is None:
            skipped["fix_no_transcript"] += 1
            continue
        rows.append(
            {
                **_base_row(summary, experiment, experiment_dir),
                "kind": "sft-fix",
                "attempt": entry.get("attempt", index + 1),
                "error_source": entry.get("error_source", ""),
                "error_summary": entry.get("error_summary", ""),
                "same_error_streak": entry.get("same_error_streak", 0),
                "regenerated_sections": entry.get("regenerated_sections", []),
                "system_prompt": chosen.system_prompt,
                "prompt": chosen.prompt,
                "completion": chosen.raw_response,
                "prompt_source": "transcript",
                "completion_source": "transcript",
                "needed_format_repair": chosen.needed_repair,
                "partial": False,
            }
        )
    return rows


def build_dpo_rows(
    summary: Path,
    experiment: dict,
    experiment_dir: Path,
    skipped: Counter,
    skip_targeted: bool,
) -> list[dict]:
    history = experiment.get("fix_history") or []
    ladder = attempt_ladder(experiment_dir, history)
    rows = []
    for index, entry in enumerate(history):
        if not entry.get("resolved"):
            skipped["dpo_not_resolved"] += 1
            continue
        rejected_attempt = ladder[index]
        chosen_attempt = ladder[index + 1]

        rejected, rejected_source, rejected_partial, _ = rejected_attempt.completion()
        if rejected is None:
            # An attempt with neither a transcript nor a rendered run.py left
            # nothing at all behind — the pre-transcript shape of an
            # invalid_format failure, which returns before any file is written.
            skipped["dpo_no_rejected_artifact"] += 1
            continue

        targeted = bool(entry.get("regenerated_sections"))
        if targeted and skip_targeted:
            skipped["dpo_targeted"] += 1
            continue

        chosen, chosen_source, chosen_partial, chosen_fields = chosen_attempt.completion()
        if chosen is None:
            skipped["dpo_no_chosen_artifact"] += 1
            continue

        reconstructed = False
        if targeted:
            # The regeneration answered only the sections it was asked for, so
            # on its own it is not an answer to the prompt the rejected attempt
            # saw. Merge it over that attempt's sections — the same merge
            # _assemble_generation does with previous=, producing the program
            # that was actually checked next.
            merged = rejected_attempt.sections()
            merged.update(chosen_attempt.sections())
            if not merged:
                skipped["dpo_targeted_unmergeable"] += 1
                continue
            chosen = render_sections(merged)
            chosen_fields = sorted(merged)
            reconstructed = True
            chosen_partial = chosen_partial or rejected_partial

        rows.append(
            {
                **_base_row(summary, experiment, experiment_dir),
                "kind": "dpo",
                "attempt": entry.get("attempt", index + 1),
                "error_source": entry.get("error_source", ""),
                "error_summary": entry.get("error_summary", ""),
                "same_error_streak": entry.get("same_error_streak", 0),
                "regenerated_sections": entry.get("regenerated_sections", []),
                "system_prompt": rejected_attempt.system_prompt,
                "prompt": rejected_attempt.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_source": "transcript" if rejected_attempt.prompt else None,
                "chosen_source": chosen_source,
                "rejected_source": rejected_source,
                "chosen_fields": chosen_fields,
                "chosen_reconstructed": reconstructed,
                "needed_format_repair": chosen_attempt.needed_repair,
                "partial": chosen_partial or rejected_partial,
            }
        )
    return rows


def build_sft_rows(
    summary: Path, experiment: dict, experiment_dir: Path, skipped: Counter, statuses: set[str]
) -> list[dict]:
    status = experiment.get("status", "")
    if statuses and status not in statuses:
        skipped["sft_status"] += 1
        return []
    final = Attempt(experiment_dir, "final")
    if final.prompt is None or final.raw_response is None:
        skipped["sft_no_transcript"] += 1
        return []
    return [
        {
            **_base_row(summary, experiment, experiment_dir),
            "kind": "sft",
            "system_prompt": final.system_prompt,
            "prompt": final.prompt,
            "completion": final.raw_response,
            "prompt_source": "transcript",
            "completion_source": "transcript",
            "fix_attempts": experiment.get("fix_attempts", 0),
            "needed_format_repair": final.needed_repair,
            "generation_kind": (final.transcript or {}).get("kind", ""),
            "partial": False,
        }
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def export(args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    roots = [Path(r) for r in args.roots]
    extra_roots = [Path(r) for r in args.experiments_root]
    summaries = find_summaries(roots)
    rows: list[dict] = []
    skipped: Counter = Counter()
    resolution: Counter = Counter()
    experiments_seen = 0
    unresolved_dirs = 0

    for summary in summaries:
        loaded = read_json(summary)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("experiments"), list):
            logger.warning("Skipping %s: no top-level 'experiments' list", summary)
            continue
        for experiment in loaded["experiments"]:
            if not isinstance(experiment, dict):
                continue
            experiments_seen += 1
            code_path = experiment.get("code_path")
            if not code_path:
                # status == "skipped": no code was ever generated.
                skipped["no_code_path"] += 1
                continue
            experiment_dir, how = resolve_experiment_dir(code_path, summary, roots, extra_roots)
            if experiment_dir is None:
                unresolved_dirs += 1
                skipped["experiment_dir_not_found"] += 1
                continue
            resolution[how] += 1
            if how == "search":
                logger.warning(
                    "%s: %s resolved by searching for a directory named %r — verify it belongs "
                    "to this run before training on it (pass --experiments-root to be explicit)",
                    summary,
                    code_path,
                    Path(code_path).name,
                )
            if args.format == "sft-fix":
                built = build_sft_fix_rows(summary, experiment, experiment_dir, skipped)
            elif args.format == "dpo":
                built = build_dpo_rows(
                    summary, experiment, experiment_dir, skipped, args.skip_targeted
                )
            else:
                built = build_sft_rows(
                    summary, experiment, experiment_dir, skipped, set(args.status)
                )
            for row in built:
                # Carried per row, not only counted: whoever filters this file
                # later needs to be able to drop the guessed ones without
                # re-running the export.
                row["resolved_by"] = how
            rows.extend(built)

    if args.error_source:
        wanted = set(args.error_source)
        before = len(rows)
        rows = [r for r in rows if r.get("error_source", "") in wanted]
        skipped["filtered_error_source"] += before - len(rows)
    if args.require_prompt:
        before = len(rows)
        rows = [r for r in rows if r.get("prompt")]
        skipped["filtered_no_prompt"] += before - len(rows)
    if args.exclude_partial:
        before = len(rows)
        rows = [r for r in rows if not r.get("partial")]
        skipped["filtered_partial"] += before - len(rows)

    by_error: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        bucket = by_error[row.get("error_source", "(none)")]
        bucket["rows"] += 1
        if row.get("prompt"):
            bucket["with_prompt"] += 1
        if row.get("partial"):
            bucket["partial"] += 1
        if row.get("chosen_reconstructed"):
            bucket["reconstructed"] += 1
        if row.get("needed_format_repair"):
            bucket["needed_format_repair"] += 1

    stats = {
        "format": args.format,
        "summaries": len(summaries),
        "experiments": experiments_seen,
        "rows": len(rows),
        "rows_with_prompt": sum(1 for r in rows if r.get("prompt")),
        "rows_partial": sum(1 for r in rows if r.get("partial")),
        "unresolved_experiment_dirs": unresolved_dirs,
        "rows_resolved_by_search": sum(1 for r in rows if r.get("resolved_by") == "search"),
        "experiment_dirs_by_resolution": dict(sorted(resolution.items())),
        "skipped": dict(sorted(skipped.items())),
        "by_error_source": {k: dict(v) for k, v in sorted(by_error.items())},
    }
    return rows, stats


def print_stats(stats: dict[str, Any]) -> None:
    print(f"format:            {stats['format']}")
    print(f"summaries scanned: {stats['summaries']}")
    print(f"experiments seen:  {stats['experiments']}")
    print(f"rows built:        {stats['rows']}")
    print(f"  with a prompt:   {stats['rows_with_prompt']}")
    print(f"  partial:         {stats['rows_partial']}")
    if stats["rows_resolved_by_search"]:
        print(
            f"  !! {stats['rows_resolved_by_search']} rows came from an experiment directory "
            "located by name search — check they belong to their summary's run"
        )
    if stats["unresolved_experiment_dirs"]:
        print(
            f"  !! {stats['unresolved_experiment_dirs']} experiment directories could not be "
            "located — pass --experiments-root"
        )
    if stats["by_error_source"]:
        print("\nby error_source:")
        width = max(len(k) for k in stats["by_error_source"])
        for source, counts in sorted(
            stats["by_error_source"].items(), key=lambda kv: -kv[1]["rows"]
        ):
            extras = ", ".join(f"{k}={v}" for k, v in counts.items() if k != "rows")
            print(
                f"  {source.ljust(width)}  {counts['rows']:>4}"
                + (f"   ({extras})" if extras else "")
            )
    if stats["skipped"]:
        print("\nskipped:")
        for reason, count in stats["skipped"].items():
            print(f"  {reason}: {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Directories (or summary files) to scan. Default: outputs/ runs/",
    )
    parser.add_argument(
        "--format",
        choices=("sft-fix", "dpo", "sft"),
        default="sft-fix",
        help="Which dataset shape to build — see the module docstring.",
    )
    parser.add_argument("--out", type=Path, help="JSONL output path. Omit to print stats only.")
    parser.add_argument("--stats-only", action="store_true", help="Build rows but write nothing.")
    parser.add_argument("--stats-json", type=Path, help="Also write the stats as JSON.")
    parser.add_argument(
        "--experiments-root",
        action="append",
        default=[],
        help="Extra directory to resolve a summary's code_path against, for runs whose "
        "experiments (CODER_EXPERIMENTS_DIR) and summaries were written to different places. "
        "Repeatable.",
    )
    parser.add_argument(
        "--error-source",
        action="append",
        default=[],
        help="Keep only rows for this error_source. Repeatable.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="--format sft only: experiment statuses to keep (default: completed). Repeatable.",
    )
    parser.add_argument(
        "--require-prompt",
        action="store_true",
        help="Drop rows whose prompt is gone (artefacts predating transcript.py).",
    )
    parser.add_argument(
        "--exclude-partial",
        action="store_true",
        help="Drop rows rebuilt from a rendered run.py, which carry code sections only.",
    )
    parser.add_argument(
        "--skip-targeted",
        action="store_true",
        help="--format dpo only: drop pairs whose chosen side had to be reconstructed by "
        "merging a targeted regeneration over the attempt before it.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.format == "sft" and not args.status:
        args.status = ["completed"]

    rows, stats = export(args)

    if args.out and not args.stats_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}\n")

    print_stats(stats)
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
