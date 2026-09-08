"""Harvest preference pairs from the Coder Agent's own fix loop.

Every round of that loop produces, by construction, exactly what direct
preference optimisation needs: a generation that a deterministic check
*rejected*, the error that rejected it, and the regeneration that got past the
same check. This script turns those into a DPO dataset.

    uv run python scripts/dpo/build_preference_dataset.py \
        dissertation/data --out dissertation/data/preference_pairs.jsonl

A pair is emitted only where `fix_history[i].resolved` is true, i.e. the
regeneration that followed genuinely cleared the check that had failed. An
unresolved attempt tells us the generation was bad but not what a better one
looks like, so it is counted and skipped rather than paired against a guess.

Only the *model-written* region of each `run.py` is kept. A rendered experiment
is mostly fixed template — the metadata block and the orchestration footer are
byte-identical between the two halves of every pair — so including it would
spend the context window teaching the model to reproduce text it never wrote.
The region is located by anchoring on the template's own literal chunks either
side of the splice placeholders, so it stays correct if the template changes.

Deliberately outside `research_pipeline`: it reads whatever summary JSON it is
pointed at, including files written by older versions of the schema, and never
imports the package.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMARY_GLOB = "**/coder_agent_summary_*.json"
PLACEHOLDER_RE = re.compile(r"__AGENT_[A-Z_]+__")

# How the fix instruction is reconstructed. The original codegen prompt is not
# retained in run artefacts, so the pair's shared prompt is rebuilt from what
# is: the experiment's own framing plus the error that had to be answered.
PROMPT_TEMPLATE = """\
Write the experiment body for the following study.

Hypothesis: {hypothesis}
Baseline: {baseline}
Success criteria: {success_criteria}

A previous attempt was rejected by an automated check:
[{error_source}] {error_summary}

Produce the imports, configuration, load_data, build_model, run_experiment,
evaluate and helper sections that pass that check."""


SECTION_RE = re.compile(r"^# SECTION \d+ .*$", re.MULTILINE)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a model-written region into its named sections.

    The template rules each spliced region off with a `# SECTION n — NAME`
    banner, so the boundaries are the template's own and need no parsing of the
    generated Python.
    """
    marks = list(SECTION_RE.finditer(body))
    if not marks:
        return [("(whole body)", body)]
    sections = [("(preamble)", body[: marks[0].start()])]
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        sections.append((mark.group(0).strip("# ").strip(), body[mark.start() : end]))
    return sections


def narrow_to_changed_sections(rejected: str, chosen: str) -> tuple[str, str] | None:
    """Keep only the sections that actually differ between the two generations.

    A rendered experiment runs to ~18,000 characters, of which a regeneration
    typically alters one section. Training on the whole body would push the
    differing region past any workable sequence length and spend the context
    teaching the model to reproduce text identical on both sides. Narrowing also
    matches how the agent itself regenerates: a check that localises its failure
    asks for only the sections it implicates (see coder_agent._target_sections).
    """
    left, right = split_sections(rejected), split_sections(chosen)
    if len(left) != len(right):
        # Section structure differs; the whole body is the honest comparison.
        return rejected, chosen
    kept_rejected, kept_chosen = [], []
    for (name, before), (_, after) in zip(left, right):
        if before.strip() != after.strip():
            kept_rejected.append(before.strip())
            kept_chosen.append(after.strip())
    if not kept_rejected:
        return None
    return "\n\n".join(kept_rejected), "\n\n".join(kept_chosen)


def _anchor(lines: list[str], *, from_end: bool) -> str | None:
    """Pick a distinctive template line to anchor the splice boundary on.

    Banner lines are excluded: the template rules every section off with the
    same row of box-drawing characters, so anchoring on one matches the first
    section header rather than the boundary, silently truncating the extracted
    region to a few hundred characters. An anchor must carry letters or digits.
    """
    ordered = reversed(lines) if from_end else iter(lines)
    for line in ordered:
        candidate = line.strip()
        if len(candidate) > 8 and any(ch.isalnum() for ch in candidate):
            return candidate
    return None


def model_written_region(rendered: str, template: str) -> str | None:
    """Return only the spliced, model-written part of a rendered run.py.

    Anchors on the template's literal text immediately before the first splice
    placeholder and immediately after the last, so the boundary is derived from
    the template rather than hard-coded to a line number.
    """
    chunks = PLACEHOLDER_RE.split(template)
    if len(chunks) < 2:
        return None
    prefix = _anchor(chunks[0].split("\n"), from_end=True)
    suffix = _anchor(chunks[-1].split("\n"), from_end=False)
    if prefix is None or suffix is None:
        return None

    start = rendered.find(prefix) if prefix else 0
    start = 0 if start < 0 else start + len(prefix)
    end = rendered.find(suffix, start)
    if end < 0:
        # The footer moved; keep the whole file rather than silently truncating.
        return rendered.strip() or None
    body = rendered[start:end].strip()
    return body or None


def read_code(path: Path, template: str) -> str | None:
    if not path.is_file():
        return None
    try:
        return model_written_region(path.read_text(errors="replace"), template)
    except OSError:
        return None


def experiment_dir(summary_path: Path, experiment: dict) -> Path | None:
    """Locate the experiment directory belonging to a summary.

    `code_path` in the summary is an absolute path on whichever machine produced
    the run, so it cannot be used once the run has been copied elsewhere. Two
    layouts occur and both are checked: the benchmark harness writes the summary
    beside `experiments/`, while a pipeline or coder-only run writes it into an
    `outputs/` subdirectory with `experiments/` as its sibling one level up.
    """
    hypothesis_id = str(experiment.get("hypothesis_id"))
    for base in (summary_path.parent, summary_path.parent.parent):
        candidate = base / "experiments" / hypothesis_id
        if candidate.is_dir():
            return candidate
    return None


def build(roots: list[Path], template: str) -> tuple[list[dict], collections.Counter]:
    pairs: list[dict] = []
    stats: collections.Counter = collections.Counter()

    summaries = sorted({p for root in roots for p in root.glob(SUMMARY_GLOB)})
    stats["summary_files"] = len(summaries)

    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            stats["unreadable_summaries"] += 1
            continue

        for experiment in summary.get("experiments", []) or []:
            stats["experiments"] += 1
            history = experiment.get("fix_history") or []
            if not history:
                continue
            exp_dir = experiment_dir(summary_path, experiment)
            if exp_dir is None:
                stats["experiment_dir_gone"] += len(history)
                continue
            attempts_dir = exp_dir / "fix_attempts"
            results = experiment.get("results") or {}

            for entry in history:
                stats["fix_attempts"] += 1
                source = entry.get("error_source", "unknown")
                if not entry.get("resolved"):
                    stats[f"unresolved:{source}"] += 1
                    continue

                attempt = entry.get("attempt")
                if not isinstance(attempt, int):
                    stats["missing_attempt_number"] += 1
                    continue

                rejected = read_code(attempts_dir / f"attempt_{attempt}" / "run.py", template)
                # The regeneration that cleared the check is the next snapshot if
                # it too was later snapshotted, otherwise the final surviving file.
                chosen = read_code(attempts_dir / f"attempt_{attempt + 1}" / "run.py", template)
                if chosen is None:
                    chosen = read_code(exp_dir / "run.py", template)

                if rejected is None or chosen is None:
                    stats["missing_snapshot"] += 1
                    continue
                narrowed = narrow_to_changed_sections(rejected, chosen)
                if narrowed is None:
                    # Nothing was learned: the check cleared without the body
                    # changing (an environment repair, say). Not a preference.
                    stats["identical_pair"] += 1
                    continue
                rejected, chosen = narrowed

                pairs.append(
                    {
                        "prompt": PROMPT_TEMPLATE.format(
                            hypothesis=results.get("hypothesis") or "(not recorded)",
                            baseline=results.get("baseline") or "(not recorded)",
                            success_criteria=results.get("success_criteria") or "(not recorded)",
                            error_source=source,
                            error_summary=(entry.get("error_summary") or "").strip(),
                        ),
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "error_source": source,
                            "attempt": attempt,
                            "hypothesis_id": experiment.get("hypothesis_id"),
                            "starter_used": experiment.get("starter_used"),
                            "final_status": experiment.get("status"),
                            "run": str(summary_path.parent.name),
                        },
                    }
                )
                stats[f"paired:{source}"] += 1

    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="directories to scan")
    parser.add_argument("--out", type=Path, required=True, help="output .jsonl")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("src/research_pipeline/agents/coder/templates/run.py.template"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    template = args.template.read_text()
    pairs, stats = build(args.roots, template)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")

    logger.info("wrote %d preference pairs to %s", len(pairs), args.out)
    logger.info("")
    for key in ("summary_files", "experiments", "fix_attempts"):
        logger.info("  %-24s %d", key, stats[key])
    logger.info("")
    logger.info("  paired, by error_source:")
    for key, count in sorted(stats.items()):
        if key.startswith("paired:"):
            logger.info("    %-22s %d", key.split(":", 1)[1], count)
    logger.info("")
    logger.info("  not paired:")
    for key, count in sorted(stats.items()):
        if key.startswith("unresolved:"):
            logger.info("    %-22s %d  (never resolved)", key.split(":", 1)[1], count)
    for key in ("missing_snapshot", "identical_pair", "missing_attempt_number", "experiment_dir_gone"):
        if stats[key]:
            logger.info("    %-22s %d", key, stats[key])


if __name__ == "__main__":
    main()
