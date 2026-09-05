"""Fetch a plan's real data inputs *here*, instead of asking generated code to.

Until now every real input that wasn't already on disk was handed to the model
as a URL, and the model wrote the HTTP call itself inside a throwaway venv. That
places the riskiest, least predictable part of an experiment — a network fetch
against an API whose exact query shape nobody has checked — inside the one loop
that is supposed to be debugging *the experiment*. Three costs followed from it:

- A fetch that 404s or returns an HTML interstitial is indistinguishable, to the
  fix loop, from a defect in the generated source, so it spends
  `CODER_MAX_FIX_ATTEMPTS` on code that was never wrong. Barkla job 10411308 is
  the recorded case (see the comment on `provenance.OPEN_SOURCES`).
- Whether the code fetched the source at all then has to be *inferred* after the
  fact, by `provenance.verify_downloads_used` matching the URL's host against
  the code text — a check that exists only because the fetch happened somewhere
  this pipeline couldn't watch.
- The model is asked to write code against columns it has never seen, because
  nothing read the data before generation.

So this module fetches first. What comes back is validated, written under
`CODER_DATA_CACHE_DIR`, and handed to the model as a **local path plus the real
column names and first rows**. `apply` then rewrites that input from
`real_download` to `real_local`, which `verify_downloads_used` already trusts
unconditionally — correctly, because the bytes are on disk and this process put
them there.

Contract, deliberately identical to `huggingface_client.py`'s: nothing here ever
raises, and every failure — no network, a 503, an unparseable body, a URL that
fails the safety check — degrades to `None`, leaving the input a
`real_download` that the generated code fetches exactly as it did before this
module existed. Acquisition is an enhancement to a prompt, never a precondition.

Safety matters more here than in the paper clients, because a URL reaching this
module can come from a table a human wrote today and from a model tomorrow (see
CLAUDE.md's staged plan for source discovery). So the rules are enforced on
every hop, not on the string that was passed in:

- https only, and the host must resolve to a *global* address — no loopback,
  no link-local, no RFC1918. That is the SSRF guard: without it a suggested
  URL could read this node's metadata service or an internal endpoint.
- Redirects are followed by hand, at most `MAX_REDIRECTS`, re-checking each
  hop — `requests`' own redirect handling would re-validate nothing.
- The body is streamed and abandoned past `max_bytes`, so a wrong URL cannot
  fill a quota'd HPC filesystem.
- An HTML body is rejected rather than saved. This is the specific failure
  `provenance.OPEN_SOURCES` documents: a datacentre IP asking Stooq for CSV
  gets an HTTP 200 carrying a JavaScript proof-of-work page, which pandas then
  parses into garbage instead of failing loudly.
- Nothing downloaded is ever executed, imported, or unpacked.

Reads no settings, same rule as `sandbox.py` and `provenance.py`: the cache
directory and byte cap arrive as arguments, resolved by `coder_agent.py`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests

from research_pipeline.agents.coder import provenance

logger = logging.getLogger(__name__)

USER_AGENT = "research-pipeline-coder-agent/0.1 (+https://github.com/)"

# Generous next to huggingface_client's 15s: that one sits in front of a code
# generation and is a nice-to-have, while this call is the data the experiment
# will actually run on. Still bounded — a hung fetch must not hold a SLURM
# allocation open.
TIMEOUT_SECONDS = 30
MAX_REDIRECTS = 3

# 64 MiB. Big enough for the tabular datasets these experiments use, small
# enough that a wrong URL cannot eat a shared-filesystem quota. Overridable per
# call (CODER_MAX_DOWNLOAD_BYTES) rather than hardcoded, because the right cap
# on localscratch and the right cap on a laptop are not the same number.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024

# Rows kept from a paginated row API. The Dataset Viewer serves at most 100 per
# request, so this is also 50 requests — the point past which paging costs more
# wall clock than an extra row of training data is worth.
MAX_ACQUIRED_ROWS = 5000
PAGE_ROWS = 100

SAMPLE_ROWS = 3
# Sample rows go into a prompt and one column can hold a whole document, so a
# single row must not be able to crowd out the experiment plan. Same reason and
# same number as huggingface_client.MAX_CELL_CHARS.
MAX_CELL_CHARS = 200

# Content types that are never the data we asked for. See the module docstring:
# the HTML entry is the proof-of-work-interstitial failure, and rejecting it
# here is what turns "pandas silently parsed a web page" into "this input was
# not obtained, and the provenance document says so".
REJECTED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/vnd.wap.wml")

FORMAT_CSV = "csv"
FORMAT_TSV = "tsv"
FORMAT_JSON = "json"
FORMAT_JSONL = "jsonl"

_EXTENSIONS = {FORMAT_CSV: ".csv", FORMAT_TSV: ".tsv", FORMAT_JSON: ".json", FORMAT_JSONL: ".jsonl"}

# How the model is told to read each format. Kept beside the writer so the two
# cannot disagree about what is actually on disk.
READ_HINTS = {
    FORMAT_CSV: "comma-separated, with a header row (pandas.read_csv)",
    FORMAT_TSV: "tab-separated, with a header row (pandas.read_csv(..., sep='\\t'))",
    FORMAT_JSON: "a JSON array of objects (pandas.read_json)",
    FORMAT_JSONL: "JSON Lines — one JSON object per line (pandas.read_json(..., lines=True))",
}


@dataclass
class Acquired:
    """One input this pipeline fetched, validated and wrote to disk."""

    url: str
    path: str
    sha256: str
    byte_count: int
    data_format: str
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    sample_rows: list[dict] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The plain-JSON shape threaded through checkpointed graph state.

        Every value here is JSON-able on purpose — CLAUDE.md's rule that a
        non-serializable value never goes into state applies to this record as
        much as to Writer's CitationRegistry.
        """
        return {
            "url": self.url,
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "data_format": self.data_format,
            "columns": list(self.columns),
            "row_count": self.row_count,
            "sample_rows": list(self.sample_rows),
            "from_cache": self.from_cache,
            # Derived, not stored. It travels with the record so provenance.py
            # can tell the model how to read the file without importing this
            # module — which it cannot do, since this module imports it.
            "read_hint": READ_HINTS.get(self.data_format, ""),
        }

    @classmethod
    def from_dict(cls, record: dict) -> Acquired:
        return cls(
            url=str(record.get("url") or ""),
            path=str(record.get("path") or ""),
            sha256=str(record.get("sha256") or ""),
            byte_count=int(record.get("byte_count") or 0),
            data_format=str(record.get("data_format") or ""),
            columns=list(record.get("columns") or []),
            row_count=int(record.get("row_count") or 0),
            sample_rows=list(record.get("sample_rows") or []),
            from_cache=bool(record.get("from_cache")),
        )


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


def _is_public_host(host: str) -> bool:
    """Whether every address `host` resolves to is routable on the public net.

    Every address, not the first: a name that resolves to one public and one
    loopback address is exactly the shape an SSRF attempt takes, and picking a
    winner here would not be the same address `requests` later connects to.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        # is_global already excludes loopback, link-local, RFC1918 and the
        # reserved ranges; multicast is checked separately because it is
        # "global" by that property's definition.
        if not address.is_global or address.is_multicast:
            return False
    return True


def url_is_fetchable(url: str) -> bool:
    """The safety gate, applied to every redirect hop rather than once."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    return _is_public_host(parsed.hostname)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _get(url: str, max_bytes: int) -> tuple[bytes, str] | None:
    """One safety-checked GET, returning (body, content_type) or None.

    Redirects are followed by hand so each hop goes back through
    `url_is_fetchable`; `requests`' own following would validate the first URL
    and then connect wherever it was sent.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not url_is_fetchable(current):
            logger.warning("Refusing to fetch %s: fails the URL safety check", current[:200])
            return None
        try:
            response = requests.get(
                current,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                timeout=TIMEOUT_SECONDS,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            logger.warning("Fetch of %s failed: %s", current[:200], exc)
            return None

        with response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("location") or ""
                if not location:
                    return None
                current = urljoin(current, location)
                continue

            # Any 2xx, not 200 alone. Some open-data portals answer a download
            # with 202 (data.qld.gov.au does), and rejecting that discards a
            # perfectly good CSV over a status code — while `describe` below is
            # the real arbiter of whether the body is data, which is this
            # module's rule everywhere else too.
            if not 200 <= response.status_code < 300:
                logger.warning("Fetch of %s returned HTTP %s", current[:200], response.status_code)
                return None

            content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
            if any(content_type.lower().startswith(t) for t in REJECTED_CONTENT_TYPES):
                # See the module docstring: an HTML body where data was asked
                # for is an interstitial or an error page, and saving it is how
                # a proof-of-work page becomes an experiment's "data".
                logger.warning(
                    "Refusing %s: content-type %s is a web page, not data",
                    current[:200],
                    content_type,
                )
                return None

            body = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        logger.warning(
                            "Abandoning %s: body exceeds the %d-byte cap", current[:200], max_bytes
                        )
                        return None
            except requests.RequestException as exc:
                logger.warning("Fetch of %s failed mid-body: %s", current[:200], exc)
                return None

        return bytes(body), content_type

    logger.warning("Refusing to fetch %s: more than %d redirects", url[:200], MAX_REDIRECTS)
    return None


# --------------------------------------------------------------------------
# Understanding what came back
# --------------------------------------------------------------------------


def _truncate_cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value


def _unwrap_envelope(payload: Any) -> list[dict] | None:
    """Pull the records out of a row-API response.

    Three shapes, in order of how often this pipeline meets them: a bare list of
    objects; the Dataset Viewer's `{"rows": [{"row": {...}}]}`; and the general
    `{"<something>": [ ... ]}` wrapper that most REST catalogues return. Nothing
    is guessed beyond that — a payload whose records can't be identified is not
    acquired, because a file whose columns this module could not read is a file
    the model would be writing blind against.
    """
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
        return records or None

    if not isinstance(payload, dict):
        return None

    rows = payload.get("rows")
    if isinstance(rows, list):
        unwrapped = [
            entry["row"]
            for entry in rows
            if isinstance(entry, dict) and isinstance(entry.get("row"), dict)
        ]
        if unwrapped:
            return unwrapped
        direct = [entry for entry in rows if isinstance(entry, dict)]
        if direct:
            return direct

    for value in payload.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return list(value)
    return None


def _records_from_jsonl(text: str) -> list[dict] | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    records = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        records.append(parsed)
    return records or None


def _describe_delimited(text: str) -> tuple[str, list[str], int] | None:
    """(format, columns, row_count) for a CSV/TSV body, or None."""
    sample = text[:8192]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except (StopIteration, csv.Error):
        return None
    columns = [column.strip() for column in header if column.strip()]
    if len(columns) < 2:
        # One column is what an HTML page or a plain-text error looks like to
        # the CSV reader. Real tabular data has at least two.
        return None
    try:
        row_count = sum(1 for row in reader if any(cell.strip() for cell in row))
    except csv.Error:
        return None
    if row_count < 1:
        return None
    return (FORMAT_TSV if delimiter == "\t" else FORMAT_CSV), columns, row_count


def describe(body: bytes) -> tuple[str, list[str], int, list[dict], bytes] | None:
    """Identify the body by its content, never by its content-type header.

    Returns (format, columns, row_count, sample_rows, bytes-to-write) or None
    when nothing here can vouch for it being tabular data. Content-driven
    because the header routinely lies — a `text/plain` CSV and an
    `application/octet-stream` JSON body are both ordinary — and because the
    one header this module *does* trust it trusts negatively (see
    REJECTED_CONTENT_TYPES).
    """
    try:
        # utf-8-sig, not utf-8: government CSV exports are routinely BOM-prefixed,
        # and a plain utf-8 decode glues \ufeff onto the first column's name. That
        # name then goes into the prompt, so the model is told to select a column
        # that does not exist under the name it was given.
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        records = _unwrap_envelope(payload)
        if records is None:
            return None
        return _describe_records(records)

    records = _records_from_jsonl(text)
    if records is not None:
        return _describe_records(records)

    delimited = _describe_delimited(text)
    if delimited is None:
        return None
    data_format, columns, row_count = delimited
    sample = _sample_from_delimited(text, data_format)
    return data_format, columns, row_count, sample, body


def _describe_records(records: list[dict]) -> tuple[str, list[str], int, list[dict], bytes] | None:
    """Normalize extracted records to JSON Lines.

    A row API's envelope is not a file, so unlike the CSV branch there are no
    original bytes worth keeping — and JSONL is the shape that survives ragged
    records (a column absent from row 1 and present in row 900) which a CSV
    header cannot express. The format is reported as jsonl however the records
    arrived, because that is what is now on disk and what the model will be
    told to read.

    Columns are the union across records, in first-seen order, for the same
    ragged-record reason: reading them off row 1 alone would describe the file
    to the model as narrower than it is.
    """
    if not records:
        return None
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    if not columns:
        return None
    sample = [
        {key: _truncate_cell(value) for key, value in row.items()} for row in records[:SAMPLE_ROWS]
    ]
    written = "\n".join(json.dumps(record, default=str) for record in records) + "\n"
    return FORMAT_JSONL, columns, len(records), sample, written.encode("utf-8")


def _sample_from_delimited(text: str, data_format: str) -> list[dict]:
    delimiter = "\t" if data_format == FORMAT_TSV else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    sample = []
    for row in reader:
        sample.append({key: _truncate_cell(value) for key, value in row.items() if key})
        if len(sample) >= SAMPLE_ROWS:
            break
    return sample


# --------------------------------------------------------------------------
# Paginated row APIs
# --------------------------------------------------------------------------


def is_paginated_rows_url(url: str) -> bool:
    """Whether `url` is a row API this module knows how to page.

    Narrow on purpose: the offset/length pair is the Dataset Viewer's, which is
    where nearly every real input in this pipeline comes from. A URL that just
    happens to carry those parameters pages correctly anyway; anything else is
    fetched once, which is the safe direction to be wrong in.
    """
    query = dict(parse_qsl(urlparse(url).query))
    return "offset" in query and "length" in query


def _with_offset(url: str, offset: int, length: int) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["offset"] = str(offset)
    query["length"] = str(length)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _fetch_paginated(url: str, max_bytes: int, max_rows: int) -> list[dict] | None:
    """Page a row API until `max_rows`, the end of the data, or a failed page.

    A failed page keeps what was already collected rather than discarding it:
    4,900 real rows is an experiment, and the alternative to a short read here
    is not a longer read, it is a synthetic surrogate.
    """
    records: list[dict] = []
    offset = 0
    while len(records) < max_rows:
        page = _get(_with_offset(url, offset, PAGE_ROWS), max_bytes)
        if page is None:
            break
        try:
            payload = json.loads(page[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        batch = _unwrap_envelope(payload)
        if not batch:
            break
        records.extend(batch)
        if len(batch) < PAGE_ROWS:
            break
        offset += len(batch)
    return records[:max_rows] or None


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def _slot(cache_dir: Path, url: str) -> Path:
    """Content-addressed on the URL, so a 100-question sweep fetches once.

    Keyed on the URL rather than the requirement text: two plans asking for the
    same dataset in different words must share one download, and the same words
    pointing at different data must not.
    """
    return cache_dir / hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return (cleaned[:60] or "data").lower()


def _read_cached(slot: Path) -> Acquired | None:
    meta = slot / "meta.json"
    if not meta.is_file():
        return None
    try:
        record = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    acquired = Acquired.from_dict(record)
    # meta.json is written last (see `fetch`), so its presence means the data
    # file was completed — but a scratch filesystem can be cleared underneath a
    # resumed run, so the file is checked rather than assumed.
    if not acquired.path or not Path(acquired.path).is_file():
        return None
    acquired.from_cache = True
    return acquired


# --------------------------------------------------------------------------
# The entry points
# --------------------------------------------------------------------------


def fetch(
    url: str,
    *,
    cache_dir: Path,
    label: str = "",
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = MAX_ACQUIRED_ROWS,
) -> Acquired | None:
    """Fetch one URL to disk and describe it, or return None.

    None is the ordinary outcome for anything that isn't directly fetchable
    tabular data — a base API URL with no query on it, a source needing
    credentials this node doesn't have, an unreachable host. The caller leaves
    that input a `real_download` and the generated code fetches it exactly as
    before, so a miss here costs nothing that was previously working.
    """
    try:
        return _fetch(url, cache_dir, label, max_bytes, max_rows)
    except Exception as exc:  # noqa: BLE001 — see the module docstring: never raise
        logger.warning("Acquisition of %s failed unexpectedly: %s", url[:200], exc)
        return None


def _fetch(url: str, cache_dir: Path, label: str, max_bytes: int, max_rows: int) -> Acquired | None:
    slot = _slot(cache_dir, url)
    cached = _read_cached(slot)
    if cached is not None:
        logger.info("Reusing cached data for %s at %s", url[:120], cached.path)
        return cached

    if is_paginated_rows_url(url):
        records = _fetch_paginated(url, max_bytes, max_rows)
        if records is None:
            return None
        described = _describe_records(records)
    else:
        fetched = _get(url, max_bytes)
        if fetched is None:
            return None
        described = describe(fetched[0])

    if described is None:
        logger.warning("Fetched %s but it is not data this module can vouch for", url[:200])
        return None
    data_format, columns, row_count, sample_rows, payload = described

    slot.mkdir(parents=True, exist_ok=True)
    path = slot / f"{_safe_label(label)}{_EXTENSIONS[data_format]}"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)

    acquired = Acquired(
        url=url,
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        data_format=data_format,
        columns=columns,
        row_count=row_count,
        sample_rows=sample_rows,
    )
    # Written last: meta.json's existence is what `_read_cached` treats as
    # "this slot is complete", so an interrupted fetch must not leave one.
    (slot / "meta.json").write_text(json.dumps(acquired.to_dict(), indent=2, default=str))
    logger.info(
        "Acquired %s -> %s (%d rows, %d columns, %d bytes)",
        url[:120],
        path,
        row_count,
        len(columns),
        len(payload),
    )
    return acquired


def acquire_sources(
    sources: list[provenance.DataSource],
    *,
    cache_dir: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = MAX_ACQUIRED_ROWS,
) -> dict[str, dict]:
    """Fetch every `real_download` that can be fetched. Returns {url: record}.

    A plain dict rather than rewritten sources, because this is what gets
    threaded through checkpointed graph state: the rewrite itself
    (`apply`) is pure and re-runs wherever provenance is needed, the same
    split the Hugging Face lookup already uses.
    """
    acquisitions: dict[str, dict] = {}
    for source in sources:
        if source.kind != provenance.KIND_REAL_DOWNLOAD or not source.uri:
            continue
        if source.uri in acquisitions:
            continue
        acquired = fetch(
            source.uri,
            cache_dir=cache_dir,
            label=source.name,
            max_bytes=max_bytes,
            max_rows=max_rows,
        )
        if acquired is not None:
            acquisitions[source.uri] = acquired.to_dict()
    return acquisitions


def apply(
    sources: list[provenance.DataSource], acquisitions: dict[str, dict] | None
) -> list[provenance.DataSource]:
    """Rewrite each fetched `real_download` as the `real_local` it now is.

    Pure and idempotent — no network, no filesystem beyond what is already
    recorded — so `_provenance_for` stays the cheap function its docstring
    promises and can keep being called at prompt time, after generation, and at
    finalize without refetching anything.

    Promoting the kind is the point. `verify_downloads_used` trusts a
    `real_local` unconditionally and has to interrogate a `real_download` by
    matching its host against the code text; once this process has the bytes,
    that inference is not just unnecessary, it is wrong — code reading a local
    CSV names no host and would be downgraded for it.
    """
    if not acquisitions:
        return sources
    applied: list[provenance.DataSource] = []
    for source in sources:
        record = acquisitions.get(source.uri) if source.uri else None
        if source.kind != provenance.KIND_REAL_DOWNLOAD or not record:
            applied.append(source)
            continue
        acquired = Acquired.from_dict(record)
        applied.append(
            provenance.DataSource(
                name=source.name,
                kind=provenance.KIND_REAL_LOCAL,
                uri=source.uri,
                local_path=acquired.path,
                reason=(
                    f"{source.reason}; fetched by the pipeline to {acquired.path} "
                    f"({acquired.row_count} rows, {acquired.byte_count} bytes, "
                    f"sha256 {acquired.sha256[:12]})"
                ),
                credentials=list(source.credentials),
                usage_verified=source.usage_verified,
                acquired=acquired.to_dict(),
                # Carried, not dropped: a discovered input stays marked as one
                # after it is fetched, or the provenance document would show a
                # keyword-search result as though a human had named it.
                discovered=dict(source.discovered),
            )
        )
    return applied


def local_paths(acquisitions: dict[str, dict] | None) -> list[str]:
    """Every path this run put on disk — what a usage check looks for in code."""
    return [str(record.get("path") or "") for record in (acquisitions or {}).values()]
