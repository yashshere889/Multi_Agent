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

Deliberately **not** the `datasets` Python library. Generated experiments run in
their own throwaway venv, so every extra dependency is a fresh multi-hundred-MB
install (and a local dataset cache) per experiment on shared HPC scratch;
`requests` is already a pipeline dependency and the viewer's `/rows` endpoint
returns paginated JSON that generated code can read directly.

Nothing here ever raises. This lookup is an enhancement to a prompt, never a
precondition for generating code: every failure — no network, a 503, a rate
limit, an unexpected payload shape — degrades to `None`/`[]` and the Coder Agent
generates exactly as it did before, mirroring
`agents/literature/clients.py`'s log-and-degrade contract.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from research_pipeline.agents.coder.dataset_spec import tokenize
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

# Extra endpoints used by the appraisal path. The viewer's /rows is what the
# inspection sample is paged from; /size is what the download budget is checked
# against *before* anything is fetched; the Hub's raw README is the dataset card
# the evidence prompt quotes from.
DATASET_VIEWER_SIZE_URL = f"{DATASET_VIEWER_BASE_URL}/size"
HUB_DATASET_INFO_URL = "https://huggingface.co/api/datasets"
HUB_RAW_URL = "https://huggingface.co/datasets"

# One /rows page. The viewer caps `length` at 100, so a 200-row inspection is
# two calls.
ROWS_PAGE_SIZE = 100
# A dataset card can be a small book. Truncated before it goes into a prompt
# that also has to carry sampled rows and the plan.
MAX_CARD_CHARS = 6000
# Data files pulled when a dataset is downloaded. Restricted so a repo that also
# ships model checkpoints, images or archives doesn't drag them onto shared
# scratch alongside the rows we actually asked for.
DOWNLOAD_ALLOW_PATTERNS = ("*.parquet", "*.csv", "*.tsv", "*.json", "*.jsonl", "README.md")


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
    # full=true adds tags, downloads, likes and cardData (license, task
    # categories, citation) to each hit. The old lookup asked for none of it and
    # used only "id"; the prefilter in dataset_scoring ranks on exactly these
    # fields, so fetching them here saves one /api/datasets/<id> call per
    # candidate.
    payload = _get_json(HUB_DATASETS_SEARCH_URL, {"search": query, "limit": limit, "full": "true"})
    if not isinstance(payload, list):
        return []
    return [hit for hit in payload if isinstance(hit, dict)]


def _keyword_queries(description: str) -> list[str]:
    """Turns a plan's prose data description into Hub search queries.

    The narrow-then-broad fallback used before the appraisal pipeline existed,
    kept because `find_dataset_for_experiment` still uses it. New callers build
    their queries from a DatasetSpec's structured fields instead — see
    `dataset_spec.search_queries`. The tokenizer is shared with that module
    rather than duplicated, so "which words are worth searching for?" has one
    answer here.
    """
    words = tokenize(description)
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


# ---------------------------------------------------------------------------
# The appraisal path: metadata, size, card and paged rows.
#
# Everything below keeps the module's founding contract — never raises, every
# failure degrades to None/[]/{} — because it all feeds an enhancement to a
# prompt, and the Coder is going to generate code either way.
# ---------------------------------------------------------------------------


def dataset_info(dataset_id: str) -> dict:
    """The Hub record: license, tags, cardData, downloads, likes, and `sha` —
    the commit the download is pinned to. `{}` on any failure."""
    if not dataset_id.strip():
        return {}
    payload = _get_json(f"{HUB_DATASET_INFO_URL}/{dataset_id}", {})
    return payload if isinstance(payload, dict) else {}


def dataset_size(dataset_id: str) -> dict:
    """`{"num_rows", "num_bytes"}` for the whole dataset, or `{}`.

    Read *before* any download so the GB budget is enforced against the real
    size rather than discovered halfway through fetching it.
    """
    payload = _get_json(DATASET_VIEWER_SIZE_URL, {"dataset": dataset_id})
    if not isinstance(payload, dict):
        return {}
    size = payload.get("size")
    dataset = size.get("dataset") if isinstance(size, dict) else None
    if not isinstance(dataset, dict):
        return {}
    return {
        "num_rows": int(dataset.get("num_rows") or 0),
        "num_bytes": int(
            dataset.get("num_bytes_original_files")
            or dataset.get("num_bytes_parquet_files")
            or dataset.get("num_bytes_memory")
            or 0
        ),
    }


def dataset_card(dataset_id: str) -> str:
    """The raw README the dataset publishes, truncated. "" when there isn't one.

    Fetched as text rather than JSON, so it doesn't go through `_get_json`.
    """
    if not dataset_id.strip():
        return ""
    url = f"{HUB_RAW_URL}/{dataset_id}/raw/main/README.md"
    try:
        response = requests.get(url, headers=_headers(), timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("Hugging Face card fetch for %s failed: %s", dataset_id, exc)
        return ""
    if response.status_code != 200:
        return ""
    return response.text[:MAX_CARD_CHARS]


def fetch_rows(dataset_id: str, config: str, split: str, limit: int, offset: int = 0) -> list[dict]:
    """Up to `limit` rows from the viewer, paged at ROWS_PAGE_SIZE.

    This is the inspection sample — the rows `dataset_inspect` measures. Returns
    however many it managed to read, including [] — a short read is a smaller
    sample, not an error, and the report records `rows_sampled` either way.
    """
    rows: list[dict] = []
    while len(rows) < limit:
        page = min(ROWS_PAGE_SIZE, limit - len(rows))
        payload = _get_json(
            DATASET_VIEWER_ROWS_URL,
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset + len(rows),
                "length": page,
            },
        )
        if not isinstance(payload, dict):
            break
        entries = payload.get("rows")
        if not isinstance(entries, list) or not entries:
            break
        for entry in entries:
            row = entry.get("row") if isinstance(entry, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        # A short page means the split ran out; asking again just re-reads the
        # tail.
        if len(entries) < page:
            break
    return rows


def describe_candidate(dataset_id: str) -> dict:
    """Everything cheap and non-LLM about one candidate, in one call site.

    `{}` when the viewer can't serve it — which is still the first thing checked,
    because a dataset the viewer won't preview is one whose rows can't be
    inspected, and an uninspectable candidate can't be scored on anything but
    its card.
    """
    if not _is_viewer_valid(dataset_id):
        return {}
    picked = _pick_split(dataset_id)
    if picked is None:
        return {}
    config, split = picked
    described = _first_rows(dataset_id, config, split)
    if described is None:
        return {}
    columns, sample_rows = described
    info = dataset_info(dataset_id)
    size = dataset_size(dataset_id)

    raw_card = info.get("cardData")
    card_data = raw_card if isinstance(raw_card, dict) else {}
    license_id = card_data.get("license")
    if isinstance(license_id, list):
        license_id = license_id[0] if license_id else ""

    return {
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "columns": columns,
        "sample_rows": sample_rows,
        "info": info,
        "cardData": card_data,
        "tags": info.get("tags") or [],
        "downloads": info.get("downloads") or 0,
        "likes": info.get("likes") or 0,
        "revision": str(info.get("sha") or ""),
        "license": str(license_id or ""),
        "num_rows": size.get("num_rows", 0),
        "num_bytes": size.get("num_bytes", 0),
    }


def rows_url_for(dataset_id: str, config: str, split: str, length: int = 100) -> str:
    """The exact REST URL generated code is pointed at on the no-download path.

    safe="" so the namespace slash in "owner/name" is percent-encoded: these are
    query-parameter *values*, and a bare slash there is what makes a hand-built
    dataset-viewer URL 404.
    """
    return (
        f"{DATASET_VIEWER_ROWS_URL}"
        f"?dataset={quote(dataset_id, safe='')}"
        f"&config={quote(config or 'default', safe='')}"
        f"&split={quote(split or 'train', safe='')}"
        f"&offset=0&length={length}"
    )


# ---------------------------------------------------------------------------
# Download and normalization.
#
# huggingface_hub and pyarrow are imported *inside* these functions and sit
# behind the `datasets-download` extra, same arrangement as the checkpointer's
# sqlite/postgres backends: a plain `uv sync` must stay unaffected, and a
# pipeline without them degrades to the REST-URL prompt block that was the only
# behaviour before downloading existed.
# ---------------------------------------------------------------------------


def download_dataset(dataset_id: str, revision: str, dest_dir: Path) -> Path | None:
    """Fetch a dataset's data files at a pinned revision. None on any failure.

    The destination is a *shared* cache keyed by (repo id, revision) by the
    caller, never a per-experiment directory: several plans in one run — and
    several runs on one machine — routinely want the same dataset, and copying
    it per experiment is what would actually justify the shared-scratch concern
    that kept this pipeline REST-only.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed (uv sync --extra datasets-download); "
            "falling back to the Dataset Viewer REST path for %s",
            dataset_id,
        )
        return None

    try:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision or None,
            local_dir=str(dest_dir),
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
            token=settings.huggingface_api_token or None,
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001 — see the module docstring: never raise
        logger.warning("Hugging Face download of %s failed: %s", dataset_id, exc)
        return None


def _read_parquet_rows(path: Path, remaining: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow is not installed; cannot normalize %s", path.name)
        return []
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path.name, exc)
        return []
    return table.slice(0, remaining).to_pylist()


def _read_csv_rows(path: Path, remaining: int, delimiter: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                rows.append(dict(row))
                if len(rows) >= remaining:
                    break
    except OSError as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
    return rows


def _read_json_rows(path: Path, remaining: int) -> list[dict]:
    """Handles both JSON Lines and a single JSON array/object, which is the
    split the Hub's `.json` files actually fall into."""
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return rows

    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except ValueError:
            return rows
        if isinstance(payload, list):
            return [row for row in payload[:remaining] if isinstance(row, dict)]
        return rows

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= remaining:
            break
    return rows


# Extension -> reader, in the order files are preferred. Parquet first because
# it is what the Hub converts everything to and is the most reliably typed.
_NORMALIZE_ORDER = (".parquet", ".jsonl", ".json", ".csv", ".tsv")


def normalize_to_jsonl(source_dir: Path, dest_path: Path, max_rows: int) -> dict:
    """Flatten downloaded data files into one JSON Lines file the generated
    experiment can read with nothing but the standard library.

    This is the point of downloading at all. The pipeline pays the parquet/CSV
    cost once, here, in the pipeline's own environment; the experiment's
    throwaway venv gets a `data.jsonl` and needs no pyarrow, no pandas engine
    and no network — which is what makes a Barkla compute node with no outbound
    route able to run the experiment at all.

    Returns `{"rows_written", "columns", "source_files"}`, or `{}` if nothing
    could be read.
    """
    files: list[Path] = []
    for suffix in _NORMALIZE_ORDER:
        files.extend(sorted(path for path in source_dir.rglob(f"*{suffix}") if path.is_file()))
    if not files:
        logger.warning("No readable data files under %s", source_dir)
        return {}

    rows: list[dict] = []
    used: list[str] = []
    for path in files:
        if len(rows) >= max_rows:
            break
        remaining = max_rows - len(rows)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            batch = _read_parquet_rows(path, remaining)
        elif suffix in (".csv", ".tsv"):
            batch = _read_csv_rows(path, remaining, "\t" if suffix == ".tsv" else ",")
        else:
            batch = _read_json_rows(path, remaining)
        if batch:
            rows.extend(batch)
            used.append(str(path.relative_to(source_dir)))

    if not rows:
        return {}

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write %s: %s", dest_path, exc)
        return {}

    return {"rows_written": len(rows), "columns": columns, "source_files": used}
