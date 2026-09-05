"""Hugging Face dataset lookup, so a generated experiment can use a real
dataset instead of inventing one.

Why this exists: the codegen prompt tells the model to synthesize stand-in data
when it can't fetch any, but a plan whose `data_requirements` describe a real,
public dataset used to get either fabricated numbers or — worse — code that
simply assumed a file like `survey_data.csv` would be sitting in the working
directory. Handing the model one concrete, verified dataset (its id, config,
split, real column names and a few real rows) replaces that guess with
something checkable.

Two APIs, both plain HTTP (see https://huggingface.co/docs/dataset-viewer/index):

- the Hub search API (`huggingface.co/api/datasets`) to find candidate ids;
- the Dataset Viewer API (`datasets-server.huggingface.co`) to confirm a
  candidate is actually servable and to read its schema and first rows.

Deliberately **not** the `datasets` Python library, in either mode below.
Generated experiments run in their own throwaway venv, so every extra dependency
is a fresh multi-hundred-MB install (and a local dataset cache) per experiment on
shared HPC scratch; `requests` is already a pipeline dependency and the viewer's
`/rows` endpoint returns paginated JSON.

Two modes, and which one applies is the difference between an experiment that
*samples* a dataset and one that can *train* on it:

- By default the model is handed the `/rows` URL and reads ~100 rows at a time
  over HTTP. Cheap, nothing on disk, and fine for a small experiment.
- With `CODER_DATASET_CACHE_DIR` set, `materialize_dataset` pages that same
  endpoint here, once, and writes the rows to one JSONL file the generated code
  reads with the standard library. Still no new dependency anywhere, and one
  file rather than the thousands a Hugging Face cache directory would create —
  which matters because Barkla's scratch inode quotas bind long before disk
  space does. Shared across every experiment and every run pointed at the same
  directory.

Nothing here ever raises. This lookup is an enhancement to a prompt, never a
precondition for generating code: every failure — no network, a 503, a rate
limit, an unexpected payload shape — degrades to `None`/`[]` and the Coder Agent
generates exactly as it did before, mirroring
`agents/literature/clients.py`'s log-and-degrade contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

HUB_DATASETS_SEARCH_URL = "https://huggingface.co/api/datasets"
DATASET_VIEWER_BASE_URL = "https://datasets-server.huggingface.co"
# The endpoint generated code is pointed at — paginated rows as JSON, no
# dataset download and no local cache. Kept here so prompts.py and the client
# can't disagree about the URL shape the model is told to use.
DATASET_VIEWER_ROWS_URL = f"{DATASET_VIEWER_BASE_URL}/rows"

USER_AGENT = "research-pipeline-coder-agent/0.1 (+https://github.com/)"

# One attempt, short timeout — unlike the literature clients this call is
# optional, and it sits directly in front of a code generation the pipeline is
# waiting on. Retrying a flaky search three times with backoff would trade a
# minute of every experiment's wall clock for a nice-to-have prompt block.
TIMEOUT_SECONDS = 15
DEFAULT_SEARCH_LIMIT = 5
# How many search hits are probed against the viewer before giving up. Each probe
# is up to three more requests, so this bounds the whole lookup at ~1 + 3n.
MAX_CANDIDATES_PROBED = 3
SAMPLE_ROWS = 3
# Sample rows go into a prompt, and one dataset column can hold an entire
# document. Truncated so a single row can't crowd out the experiment plan.
MAX_CELL_CHARS = 200
# A supervised-learning dataset almost always has a train split, and first-rows
# on it is the most representative sample; the rest are fallbacks in order.
PREFERRED_SPLITS = ("train", "training", "validation", "test")

# Dropped when turning a prose data_requirements description into a search query.
# The Hub's `search` parameter matches dataset names/ids, so a full sentence
# matches nothing at all — see _keyword_queries.
_QUERY_STOPWORDS = {
    "a",
    "about",
    "all",
    "and",
    "any",
    "are",
    "collected",
    "collect",
    "data",
    "dataset",
    "datasets",
    "each",
    "for",
    "from",
    "into",
    "least",
    "new",
    "per",
    "real",
    "sample",
    "samples",
    "set",
    "should",
    "small",
    "some",
    "such",
    "the",
    "their",
    "them",
    "then",
    "this",
    "those",
    "using",
    "which",
    "with",
    "within",
}


def _headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    # Optional: public search and the viewer work unauthenticated, a token just
    # buys higher rate limits (and access to gated datasets, which generated
    # code couldn't read anyway).
    if settings.huggingface_api_token:
        headers["Authorization"] = f"Bearer {settings.huggingface_api_token}"
    return headers


def _get_json(url: str, params: dict[str, Any]) -> Any | None:
    """GETs `url` and returns the decoded JSON, or None on any failure at all
    (connection error, non-200, body that isn't JSON)."""
    try:
        response = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("Hugging Face request to %s failed: %s", url, exc)
        return None
    if response.status_code != 200:
        logger.warning(
            "Hugging Face request to %s returned %d: %s",
            url,
            response.status_code,
            response.text[:200],
        )
        return None
    try:
        return response.json()
    except ValueError as exc:
        logger.warning("Hugging Face response from %s was not JSON: %s", url, exc)
        return None


def search_datasets(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Hub dataset search. Returns the raw hit dicts (each with at least an
    "id"), most relevant first, or [] on any failure."""
    if not query.strip():
        return []
    payload = _get_json(HUB_DATASETS_SEARCH_URL, {"search": query, "limit": limit})
    if not isinstance(payload, list):
        return []
    return [hit for hit in payload if isinstance(hit, dict)]


def _keyword_queries(description: str) -> list[str]:
    """Turns a plan's prose data description into Hub search queries.

    The Hub's `search` parameter matches dataset names and ids, not free text, so
    "a survey of 500 undergraduate students measuring sleep quality" finds
    nothing while "survey undergraduate sleep" finds several. Two queries are
    produced, narrow-then-broad, and the caller stops at the first that yields a
    viewer-valid dataset — a deliberate trade: a slightly off-topic real dataset
    the model is told it may ignore is still more useful than no dataset, and one
    extra HTTP call is cheap next to a codegen round-trip.
    """
    # Bare numbers ("500 students", "2024") are dropped along with stopwords:
    # they never help match a dataset *name*, and they crowd out the words that do.
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", description.lower())
        if len(word) > 2 and not word.isdigit() and word not in _QUERY_STOPWORDS
    ]
    queries = []
    for count in (4, 2):
        candidate = " ".join(words[:count])
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries


def _is_viewer_valid(dataset_id: str) -> bool:
    """Whether the Dataset Viewer can actually serve this dataset's rows. A
    dataset that only exists as loose files (or is gated, or too large to
    preview) is no use to generated code, however good the name match is."""
    payload = _get_json(f"{DATASET_VIEWER_BASE_URL}/is-valid", {"dataset": dataset_id})
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("viewer") or payload.get("preview"))


def _pick_split(dataset_id: str) -> tuple[str, str] | None:
    """Returns (config, split) — the preferred split if the dataset has one,
    else whatever it lists first."""
    payload = _get_json(f"{DATASET_VIEWER_BASE_URL}/splits", {"dataset": dataset_id})
    if not isinstance(payload, dict):
        return None
    splits = [entry for entry in payload.get("splits") or [] if isinstance(entry, dict)]
    if not splits:
        return None
    for preferred in PREFERRED_SPLITS:
        for entry in splits:
            if entry.get("split") == preferred and entry.get("config"):
                return str(entry["config"]), str(entry["split"])
    first = splits[0]
    if not first.get("config") or not first.get("split"):
        return None
    return str(first["config"]), str(first["split"])


def _truncate_cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value


def _first_rows(dataset_id: str, config: str, split: str) -> tuple[list[dict], list[dict]] | None:
    """Returns (columns, sample_rows) from the viewer's first-rows endpoint.
    Columns are [{"name", "type"}]; rows are plain dicts with long cells cut."""
    payload = _get_json(
        f"{DATASET_VIEWER_BASE_URL}/first-rows",
        {"dataset": dataset_id, "config": config, "split": split},
    )
    if not isinstance(payload, dict):
        return None

    columns = []
    for feature in payload.get("features") or []:
        if not isinstance(feature, dict) or not feature.get("name"):
            continue
        feature_type = feature.get("type")
        if isinstance(feature_type, dict):
            # {"dtype": "string", "_type": "Value"} for scalars; nested
            # structures (Sequence, ClassLabel) have no dtype, so fall back to
            # the _type name rather than dumping the whole nested schema.
            rendered_type = str(feature_type.get("dtype") or feature_type.get("_type") or "unknown")
        else:
            rendered_type = str(feature_type or "unknown")
        columns.append({"name": str(feature["name"]), "type": rendered_type})
    if not columns:
        return None

    rows = []
    for entry in (payload.get("rows") or [])[:SAMPLE_ROWS]:
        row = entry.get("row") if isinstance(entry, dict) else None
        if isinstance(row, dict):
            rows.append({key: _truncate_cell(value) for key, value in row.items()})
    return columns, rows


def find_dataset_for_experiment(query: str) -> dict | None:
    """Finds one real, viewer-servable dataset matching a plan's data
    description.

    This is the single seam the Coder Agent injects (see its
    `huggingface_lookup_fn` constructor argument), so a test substitutes one
    function instead of faking four HTTP endpoints.

    Returns {"dataset_id", "config", "split", "columns", "sample_rows"} for the
    first candidate the viewer can actually serve, or None if nothing matches —
    including when there's no network, the API is down, or the response shape is
    unrecognisable. Never raises: the caller is about to generate code either
    way, and a failed dataset lookup must not be the reason an experiment
    doesn't get written.
    """
    try:
        for search_query in _keyword_queries(query) or [query.strip()]:
            for hit in search_datasets(search_query)[:MAX_CANDIDATES_PROBED]:
                dataset_id = str(hit.get("id") or "")
                if not dataset_id or not _is_viewer_valid(dataset_id):
                    continue
                picked = _pick_split(dataset_id)
                if picked is None:
                    continue
                config, split = picked
                described = _first_rows(dataset_id, config, split)
                if described is None:
                    continue
                columns, sample_rows = described
                logger.info(
                    "Hugging Face dataset match for %r: %s (config=%s, split=%s, %d columns)",
                    search_query,
                    dataset_id,
                    config,
                    split,
                    len(columns),
                )
                return {
                    "dataset_id": dataset_id,
                    "config": config,
                    "split": split,
                    "columns": columns,
                    "sample_rows": sample_rows,
                }
        logger.info("No viewer-servable Hugging Face dataset found for %r", query[:120])
        return None
    except Exception as exc:  # noqa: BLE001 — see the docstring: never raise
        # _get_json already absorbs every network/decode failure, so reaching
        # here means an unexpected payload shape got past the isinstance guards.
        # Still not worth failing a code generation over.
        logger.warning(
            "Hugging Face dataset lookup for %r failed unexpectedly: %s", query[:120], exc
        )
        return None


# How many rows one /rows request may ask for. The endpoint's own cap; asking
# for more silently returns this many, which would turn a paging loop into an
# infinite one if the caller believed the number it asked for.
ROWS_PAGE_SIZE = 100


def dataset_cache_filename(dataset_id: str, config: str, split: str, max_rows: int) -> str:
    """The one file a materialized dataset lives in.

    The dataset id is in the name so a human can see what a cache directory
    holds, and `max_rows` is in it so raising the cap fetches more rather than
    silently reusing a smaller download that happens to be sitting there. The
    namespace slash becomes `__`, since one file is the whole point — a
    directory per namespace would start reintroducing the inode problem this
    avoids.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "__", f"{dataset_id}__{config}__{split}")
    return f"{stem}__{max_rows}rows.jsonl"


def materialize_dataset(
    hf_dataset: dict, cache_dir: Path, max_rows: int
) -> tuple[Path, int] | None:
    """Download a matched dataset to one local JSONL file. Returns
    (path, row_count), or None if it could not be had.

    Reuses an existing file untouched: the cache is shared across experiments
    and across runs, so the common case in a sweep is that the first plan pays
    for the download and every later one gets it free.

    Written to a temporary name and renamed into place, for the same reason
    run.py's checkpoints are: a process killed mid-download must not leave a
    truncated file that every later run then reads as a complete dataset. That
    is also why there is no separate "done" marker — the final name existing
    *is* the marker.

    Never raises, like everything else in this module. A failure here means the
    prompt falls back to handing over the REST URL, which is exactly what it did
    before this function existed.
    """
    dataset_id = str(hf_dataset.get("dataset_id") or "")
    if not dataset_id or max_rows <= 0:
        return None
    config = str(hf_dataset.get("config") or "default")
    split = str(hf_dataset.get("split") or "train")

    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = cache_dir / dataset_cache_filename(dataset_id, config, split, max_rows)
        if final_path.exists():
            rows = sum(1 for _ in final_path.open(encoding="utf-8"))
            logger.info("Reusing cached %s (%d rows) at %s", dataset_id, rows, final_path)
            return final_path, rows

        tmp_path = final_path.with_suffix(".jsonl.partial")
        written = 0
        with tmp_path.open("w", encoding="utf-8") as fh:
            while written < max_rows:
                payload = _get_json(
                    DATASET_VIEWER_ROWS_URL,
                    {
                        "dataset": dataset_id,
                        "config": config,
                        "split": split,
                        "offset": written,
                        "length": min(ROWS_PAGE_SIZE, max_rows - written),
                    },
                )
                if not isinstance(payload, dict):
                    break
                entries = payload.get("rows") or []
                if not entries:
                    break  # the split is exhausted — fewer rows than the cap
                for entry in entries:
                    row = entry.get("row") if isinstance(entry, dict) else None
                    if isinstance(row, dict):
                        fh.write(json.dumps(row, default=str) + "\n")
                        written += 1
                if len(entries) < ROWS_PAGE_SIZE:
                    break

        if written == 0:
            tmp_path.unlink(missing_ok=True)
            logger.warning("Downloaded no rows for %s — falling back to the rows URL.", dataset_id)
            return None

        os.replace(tmp_path, final_path)
        logger.info("Materialized %s: %d rows to %s", dataset_id, written, final_path)
        return final_path, written
    except Exception as exc:  # noqa: BLE001 — see the docstring: never raise
        logger.warning("Could not materialize %s: %s", dataset_id, exc)
        return None
