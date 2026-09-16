"""The prompt and the raw model response behind each generated experiment,
written next to the code they produced.

The fix loop already preserves every attempt's *code* under
`experiments/<hid>/fix_attempts/attempt_<n>/` — which makes each (rejected
generation, the error that rejected it, accepted generation) triple a
preference pair by construction, and that is what those snapshots are kept for.
What they do not preserve is either half of the exchange that produced them:

- The **prompt** is not on disk anywhere. It is reconstructible in principle
  from the plan JSON plus `prompts.py` *at the commit that ran*, which is
  archaeology rather than a dataset — and for a targeted regeneration
  (`coder_agent._target_sections`) it is not reconstructible at all from the
  summary, since the prompt's shape depends on which sections were asked for.
- The **response** is not on disk either. `run.py` is the model's sections
  *after* `sandbox.render_experiment_with_spans` spliced them into the fixed
  template, so training on it would teach a model to emit a file it is never
  asked to emit. The sections are recoverable from a rendered run.py (see
  scripts/export_preference_data.py, which slices them back out), but only for
  an attempt that got as far as being rendered: `invalid_format` and
  `missing_sections` both return before a file is written, so the two failure
  categories where the model's raw text *is* the whole defect are precisely the
  ones that leave nothing behind.

So this module writes both, verbatim, as `.transcript.json` in the experiment
directory — picked up by `_snapshot_attempt` alongside run.py and
requirements.txt, so each attempt's snapshot carries the exchange that made it
and the file left in the experiment directory at the end is the accepted one.

Two deliberate choices:

- **Raw response text, not the parsed sections.** A parsed dict cannot
  represent the answer that failed to parse, and re-rendering one through
  `llm_sections.render_sections` would produce a canonical response the model
  never actually wrote. `llm_sections.invoke_sections`' `on_raw_response`
  callback exists for this: it reports every turn, including a repair turn and
  including a turn nothing could be parsed from.
- **A dotfile.** The experiment directory is scanned for generated Python
  (`sandbox.extract_third_party_imports`) and provisioned as a working
  directory for the experiment itself; a `.`-prefixed name stays out of both,
  the same way `.resolved_requirements.txt` already does.

Nothing here is allowed to end a run. A transcript is a by-product: if writing
one fails — a full disk, a read-only mount, a path that is somehow not a
directory — that is logged and the experiment carries on, exactly as
`coder_agent._record_fix_pattern` treats the fix-pattern store.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = ".transcript.json"

# Stamped into every record so a reader written against a later shape can tell
# what it is holding, rather than inferring it from which keys happen to be
# present — the same problem scripts/analyze_coder_fix_history.py solves by
# treating every field as optional. Bump it only for a change a reader cannot
# absorb by tolerating a missing key.
SCHEMA_VERSION = "coder-transcript-v1"

# The two exchange kinds worth keeping: the first generation of an experiment
# and a regeneration answering a concrete failure. Other `_call_sections`
# callers (shared infrastructure, which names its own sections) are not part of
# a per-experiment attempt and have no experiment directory to be written to.
KIND_GENERATE = "generate"
KIND_FIX = "fix"


def build(
    *,
    kind: str,
    system_prompt: str,
    user_prompt: str,
    raw_responses: list[str],
    field_names: list[str] | None,
    temperature: float | None,
    max_tokens: int | None,
    model: str,
    error: str = "",
) -> dict[str, Any]:
    """One exchange, as the dict `write` persists.

    `raw_responses` is every turn the model produced for this call, in order —
    two entries when `invoke_sections` needed its repair round-trip, one
    otherwise. The last is the one that was parsed (or, when `error` is set, the
    one that still could not be). Keeping the earlier turn matters: a response
    that only parsed after a repair prompt is a worse example than one that
    parsed first time, and a dataset that cannot tell them apart cannot say so.

    `field_names` is what the call *asked* for — `None` meaning "discover them
    from the response". For a targeted regeneration this is the subset, and it
    is the only record of that: the completion is correct for this prompt and
    would be incomplete for any other.
    """
    return {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_responses": list(raw_responses),
        "field_names": list(field_names) if field_names is not None else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Set when the call raised rather than returned — the exchange is kept
        # anyway, because a response no parser could read is a training example
        # about format, not a hole in the data.
        "error": error,
    }


def write(experiment_dir: Path, record: dict[str, Any]) -> Path | None:
    """Persist one exchange as `<experiment_dir>/.transcript.json`, replacing
    whatever the previous attempt left there.

    Returns the path written, or None when transcripts are switched off or the
    write failed. Callers are not expected to check: there is nothing useful to
    do about it either way, which is why nothing raises out of here.
    """
    if not settings.coder_save_transcripts:
        return None
    path = experiment_dir / TRANSCRIPT_FILENAME
    try:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 — see the module docstring
        logger.warning("Could not write %s: %s", path, exc)
        return None
    return path


def read(path: Path) -> dict[str, Any] | None:
    """Load one transcript, or None if it isn't there or isn't readable JSON.

    Tolerant on purpose, for the same reason analyze_coder_fix_history.py is: a
    transcript from an interrupted run is one truncated file among hundreds of
    good ones, and it must not take a whole export down with it.
    """
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read transcript %s: %s", path, exc)
        return None
    if not isinstance(loaded, dict):
        logger.warning("Transcript %s is not an object", path)
        return None
    return loaded
