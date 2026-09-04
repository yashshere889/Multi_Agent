"""Find a real source for a requirement nobody named one for.

`provenance.resolve` answers "what data is this plan entitled to use?" from
three hand-written tables. That is the right shape for the two things a table
knows and a search cannot: which sources are *restricted* (CMS, UK Biobank —
real data that specifically was not obtained) and which need *credentials*. It
is the wrong shape for the ordinary case. A requirement matching none of the
tables becomes an `unresolved` surrogate, which means the experiment invents its
inputs and the hypothesis verdict is withheld — and for an arbitrary research
plan that is most requirements.

This module searches instead. Given a requirement string it asks a small set of
keyless dataset catalogues for candidates, and hands each candidate to
`acquire.fetch` — which is also the probe: a candidate that comes back as
described tabular data is real, and one that does not is dropped. There is no
separate "is this servable?" step because Phase 1 already answers that question
better than a HEAD request would.

Connectors, in the order they are tried:

    direct   a URL or DOI written into the plan's own `source` field
    ckan     the CKAN portals below — one API shape (`package_search`) shared
             by the UK, Canadian, Australian and EU open-data catalogues
    zenodo   research datasets and their DOIs

Adding a portal is one line in `CKAN_PORTALS`; adding a catalogue with a
different API shape is one function plus one line in `CONNECTORS`.

**The relevance gate is the load-bearing part.** A search that returns *some*
real dataset for "UK air quality" is not the same as returning the right one,
and real-but-wrong data is more dangerous than a surrogate: a surrogate
withholds the hypothesis verdict, while real data claims one. So a candidate
must clear `is_relevant` — deterministic keyword overlap between the requirement
and the candidate's own title and description, the same shape of test
`provenance._staged_file` already uses to match a staged file to a requirement.
That gate is a floor, not a guarantee, which is why every discovered input
records the query, the connector, the catalogue's landing page and the
candidate's title in `data_provenance.json`: a discovered source is a weaker
claim than a staged file, and it must be auditable as such by a human reading
the run afterwards.

Two things this module deliberately does **not** do:

- **Discover against a restricted or credentialed requirement.** Those name real
  data that specifically was not obtained; answering "CMS claims" with a
  cheerful municipal CSV is precisely the over-claim `provenance` exists to
  prevent. Only `unresolved` entries are ever searched for — the same rule
  `provenance.supersede_unresolved` already applies for the same reason.
- **Ask a model anything.** The catalogues are searched with keywords derived
  from the requirement text, and Python decides what came back. Model-proposed
  sources are a later step, and would arrive as another connector here.

Same two contracts as the rest of `agents/coder/`: nothing raises (every failure
degrades to no candidates, leaving the requirement the surrogate it already
was), and nothing reads settings — the cache directory and the caps arrive as
arguments.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from research_pipeline.agents.coder import acquire, provenance

logger = logging.getLogger(__name__)

USER_AGENT = acquire.USER_AGENT
TIMEOUT_SECONDS = 15

# CKAN portals, all speaking the same `package_search` API.
# (label, full package_search URL, where a human reads the dataset). Both URLs
# are stored whole rather than derived from one base: data.europa.eu serves
# package_search directly under its hub path instead of the usual
# /api/3/action/, and assuming the common suffix silently 404'd every request
# to it; and a landing page in the provenance document that 404s for the person
# auditing the run is worse than none. Verified answering 2026-09-04.
# `catalog.data.gov` is deliberately absent — it 404s on both API paths, and a
# portal that never answers costs a request per requirement to learn nothing.
CKAN_PORTALS: list[tuple[str, str, str]] = [
    (
        "data.gov.uk",
        "https://ckan.publishing.service.gov.uk/api/3/action/package_search",
        "https://www.data.gov.uk/dataset",
    ),
    (
        "open.canada.ca",
        "https://open.canada.ca/data/en/api/3/action/package_search",
        "https://open.canada.ca/data/en/dataset",
    ),
    (
        "data.gov.au",
        "https://data.gov.au/data/api/3/action/package_search",
        "https://data.gov.au/dataset",
    ),
    (
        "data.europa.eu",
        "https://data.europa.eu/api/hub/search/ckan/package_search",
        "https://data.europa.eu/data/datasets",
    ),
]

ZENODO_RECORDS_URL = "https://zenodo.org/api/records"

# How many catalogue hits are considered per connector, and how many candidates
# are actually fetched before giving up on a requirement. Each fetch is a real
# download, so the second number is the one that costs wall clock.
MAX_HITS_PER_CONNECTOR = 10
MAX_CANDIDATES_PROBED = 6

# Share of a requirement's content words a candidate must match. See
# `is_relevant` for why this is a fraction rather than a fixed count.
RELEVANCE_FRACTION = 0.6

# Formats this pipeline can actually read (see acquire.describe). Everything
# else a catalogue offers — XLSX, PDF, XML, ZIP, WMS endpoints — is skipped
# rather than downloaded and rejected.
TABULAR_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")
TABULAR_FORMATS = {"CSV", "TSV", "JSON", "JSONL", "NDJSON"}
# GeoJSON is JSON and would parse — into one row per map feature with a nested
# geometry blob for a column. That is not the table the plan asked for, and it
# is worse than nothing because it looks like success.
EXCLUDED_FORMATS = {"GEOJSON", "TOPOJSON"}
# Extensions that veto a resource whatever the catalogue declares it to be.
NON_TABULAR_EXTENSIONS = (".zip", ".gz", ".7z", ".tar", ".xlsx", ".xls", ".pdf", ".xml", ".html")

# Dropped when turning a prose requirement into a catalogue query, and when
# measuring overlap. Same idea as huggingface_client's stop list.
_STOPWORDS = frozenset(
    """
    a an the and or of for from with without to in on at by per over under between
    is are was were be been has have had its it this that these those not
    data dataset datasets database records record source sources file files
    about into using use used via across during within all any new one two
    real public open historical recent daily monthly yearly annual
    """.split()
)


@dataclass
class Candidate:
    """One dataset a catalogue offered for a requirement."""

    name: str
    url: str
    connector: str
    title: str = ""
    description: str = ""
    landing_page: str = ""
    # The catalogue's own name for the *file*, distinct from the dataset title.
    # Recorded because it is often the only thing that distinguishes the
    # measurements in a dataset from its station list, and the person auditing
    # data_provenance.json needs to see which one was taken.
    resource: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "connector": self.connector,
            "title": self.title,
            "resource": self.resource,
            "landing_page": self.landing_page,
        }


# --------------------------------------------------------------------------
# Keywords and relevance
# --------------------------------------------------------------------------


def _is_content_word(token: str) -> bool:
    """Whether `token` is worth searching or matching on.

    The length floor is 3, not 4, and there are two exceptions above it —
    because a data requirement's most discriminating terms are routinely short.
    A plain `len > 3` rule drops `pm2` (from PM2.5), `co2`, `no2`, `EEG`, `GDP`,
    `CMS`, i.e. exactly the words that distinguish "PM2.5 concentrations" from
    every other environmental dataset in a catalogue.

    - a token mixing letters and digits is a measure or a code (`pm2`, `co2`,
      `covid19`), never noise;
    - an all-caps token is an acronym (`EEG`, `GDP`), so case is read from the
      original text rather than after lowercasing;
    - a pure number is dropped, which is what keeps the `5` of "PM2.5" out.
    """
    if token.lower() in _STOPWORDS:
        return False
    if not any(character.isalpha() for character in token):
        return False
    if len(token) >= 3:
        return True
    return token.isupper()


def keywords(text: str) -> set[str]:
    """Content words of `text`, lowercased, stopwords and noise dropped."""
    tokens = re.split(r"[^A-Za-z0-9]+", text or "")
    return {token.lower() for token in tokens if _is_content_word(token)}


def query_for(requirement: str) -> str:
    """The catalogue query for a requirement — its content words, space-joined.

    Prose goes in ("hourly PM2.5 concentrations from urban monitoring stations")
    and keywords come out, because `package_search` and Zenodo's `q` both match
    terms, and a full sentence scores every catalogue's stopword-heavy records.
    """
    words = [word for word in re.split(r"[^A-Za-z0-9]+", requirement or "") if word]
    kept = [word.lower() for word in words if _is_content_word(word)]
    return " ".join(kept[:8]) or " ".join(word.lower() for word in words[:8])


def relevance_score(requirement: str, candidate: Candidate) -> int:
    """How many content words the requirement and the candidate share.

    The resource's own name is scored alongside the dataset's title, because a
    dataset and the files inside it are not the same thing: "Mill Road
    Cambridge: Monitoring Air Quality" is a good match for an air-quality
    requirement while its `Sensor Location_PointLocation` resource is a lookup
    table of monitor coordinates. Scoring both is what lets `find_source` probe
    the measurements before the coordinates.
    """
    wanted = keywords(requirement)
    if not wanted:
        return 0
    offered = keywords(f"{candidate.title} {candidate.description} {candidate.resource}")
    return len(wanted & offered)


def is_relevant(requirement: str, candidate: Candidate) -> bool:
    """Whether `candidate` plausibly *is* the data the requirement asked for.

    The gate that separates "a real dataset" from "the right real dataset". A
    catalogue's relevance ranking is not a substitute for this: `package_search`
    happily returns its best match for a query that matches nothing well, and
    that dataset would otherwise be fetched, described and reported as evidence.

    The threshold is a *majority* of the requirement's content words, not a
    fixed two. A flat two-word bar is nearly vacuous here and measurably so: a
    live "hourly PM2.5 air quality measurements" search returned 396 CKAN
    candidates and all 396 cleared it, because the query handed to the
    catalogue is built from those same words, so every hit shares two of them
    by construction. Requiring most of them is what makes this a filter rather
    than a formality — while the floor of two keeps short requirements, where a
    majority would be one word, from admitting a coincidence.

    Still not a guarantee that the data answers the hypothesis: that judgment
    stays with the human reading `data_provenance.json`, which is why every
    discovered input records what was searched for and what was chosen.
    """
    wanted = keywords(requirement)
    if not wanted:
        return False
    threshold = max(2, math.ceil(len(wanted) * RELEVANCE_FRACTION))
    return relevance_score(requirement, candidate) >= min(threshold, len(wanted))


def _is_tabular(url: str, declared_format: str = "") -> bool:
    """Whether this resource is worth spending one of the four probe downloads on.

    The declared format and the URL both get a veto. A catalogue declaring
    "CSV" for a `.zip` of CSVs is common — Statistics Canada does it — and the
    archive is not something `acquire.describe` can read, so trusting the
    declaration alone burns a probe to learn that.
    """
    declared = (declared_format or "").strip().upper()
    path = urlparse(url).path.lower()
    if declared in EXCLUDED_FORMATS:
        return False
    if path.endswith(NON_TABULAR_EXTENSIONS):
        return False
    if declared in TABULAR_FORMATS:
        return True
    if declared:
        return False
    return path.endswith(TABULAR_EXTENSIONS)


# --------------------------------------------------------------------------
# HTTP, shared by the connectors
# --------------------------------------------------------------------------


def _get_json(url: str, params: dict[str, Any]) -> Any | None:
    """One catalogue request. Never raises; None on anything unexpected.

    Not `acquire._get`: that one enforces a data-payload contract (rejects HTML,
    caps the body, describes what came back) which is right for a dataset and
    wrong for a search response. The safety check is still applied, because a
    portal URL is as much a URL as a dataset one.
    """
    if not acquire.url_is_fetchable(url):
        logger.warning("Refusing catalogue request to %s: fails the URL safety check", url[:200])
        return None
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            logger.info("Catalogue %s returned HTTP %s", url[:120], response.status_code)
            return None
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.info("Catalogue %s unavailable: %s", url[:120], exc)
        return None


# --------------------------------------------------------------------------
# Connectors
# --------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"https?://[^\s,;)\]}>\"']+")
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]}>\"']+")


def search_direct(requirement: str) -> list[Candidate]:
    """A URL or DOI the planner wrote into the requirement itself.

    First, and cheapest: when a plan says "download from https://…", nothing
    needs discovering. A DOI becomes its doi.org resolver URL, which for the
    Zenodo and Dataverse records these plans cite redirects to the record page —
    fetched and dropped by `acquire` unless it is itself data, which is the
    correct outcome rather than a special case worth writing.
    """
    candidates = [
        Candidate(
            name=requirement,
            url=url.rstrip(".,"),
            connector="direct",
            title=requirement,
            description=requirement,
            landing_page=url.rstrip(".,"),
        )
        for url in _URL_PATTERN.findall(requirement or "")
    ]
    candidates.extend(
        Candidate(
            name=requirement,
            url=f"https://doi.org/{doi.rstrip('.,')}",
            connector="direct",
            title=requirement,
            description=requirement,
            landing_page=f"https://doi.org/{doi.rstrip('.,')}",
        )
        for doi in _DOI_PATTERN.findall(requirement or "")
    )
    return candidates


def search_ckan(requirement: str) -> list[Candidate]:
    """Every CKAN portal in CKAN_PORTALS, one `package_search` each.

    Portals are queried in order and their hits concatenated rather than merged
    or ranked: the caller probes candidates in order and stops at the first that
    fetches, so ordering the portals is the ranking.
    """
    query = query_for(requirement)
    if not query:
        return []
    candidates: list[Candidate] = []
    for portal_name, search_url, landing_base in CKAN_PORTALS:
        payload = _get_json(search_url, {"q": query, "rows": MAX_HITS_PER_CONNECTOR})
        results = ((payload or {}).get("result") or {}).get("results")
        if not isinstance(results, list):
            continue
        for dataset in results:
            if not isinstance(dataset, dict):
                continue
            title = str(dataset.get("title") or "")
            notes = str(dataset.get("notes") or "")
            landing = str(dataset.get("url") or "") or f"{landing_base}/{dataset.get('name')}"
            for resource in dataset.get("resources") or []:
                if not isinstance(resource, dict):
                    continue
                url = str(resource.get("url") or "")
                if not url or not _is_tabular(url, str(resource.get("format") or "")):
                    continue
                candidates.append(
                    Candidate(
                        name=requirement,
                        url=url,
                        connector=f"ckan:{portal_name}",
                        title=title,
                        description=notes,
                        landing_page=landing,
                        resource=" ".join(
                            str(resource.get(key) or "") for key in ("name", "description")
                        ).strip(),
                    )
                )
    return candidates


def search_zenodo(requirement: str) -> list[Candidate]:
    """Zenodo records of type `dataset`, and their directly downloadable files."""
    query = query_for(requirement)
    if not query:
        return []
    payload = _get_json(
        ZENODO_RECORDS_URL,
        {"q": query, "size": MAX_HITS_PER_CONNECTOR, "type": "dataset"},
    )
    hits = ((payload or {}).get("hits") or {}).get("hits")
    if not isinstance(hits, list):
        return []
    candidates: list[Candidate] = []
    for record in hits:
        if not isinstance(record, dict):
            continue
        raw_metadata = record.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        title = str(metadata.get("title") or record.get("title") or "")
        description = re.sub(r"<[^>]+>", " ", str(metadata.get("description") or ""))
        landing = str(record.get("doi_url") or (record.get("links") or {}).get("self") or "")
        for entry in record.get("files") or []:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("key") or "")
            url = str((entry.get("links") or {}).get("self") or "")
            if not url or not _is_tabular(filename):
                continue
            candidates.append(
                Candidate(
                    name=requirement,
                    url=url,
                    connector="zenodo",
                    title=title,
                    description=description,
                    landing_page=landing,
                    resource=filename,
                )
            )
    return candidates


# The registry. Order is the search order, cheapest and most specific first.
CONNECTORS: list[tuple[str, Callable[[str], list[Candidate]]]] = [
    ("direct", search_direct),
    ("ckan", search_ckan),
    ("zenodo", search_zenodo),
]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def find_source(
    requirement: str,
    *,
    cache_dir: Path,
    max_bytes: int = acquire.DEFAULT_MAX_BYTES,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]] | None = None,
) -> Candidate | None:
    """One real, fetched, relevant source for `requirement`, or None.

    "Fetched" is not incidental: a candidate is only returned once
    `acquire.fetch` has pulled it and described it as tabular data, so the
    caller never has to trust that a catalogue's URL works. The bytes land in
    the same content-addressed cache the caller then reads, so confirming a
    candidate and acquiring it are one download, not two.
    """
    try:
        return _find_source(requirement, cache_dir, max_bytes, connectors or CONNECTORS)
    except Exception as exc:  # noqa: BLE001 — never raise; a miss is a surrogate
        logger.warning("Source discovery for %r failed unexpectedly: %s", requirement[:120], exc)
        return None


def _find_source(
    requirement: str,
    cache_dir: Path,
    max_bytes: int,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]],
) -> Candidate | None:
    probed = 0
    for connector_name, search in connectors:
        try:
            candidates = search(requirement)
        except Exception as exc:  # noqa: BLE001 — one broken catalogue is not fatal
            logger.warning("Connector %s raised for %r: %s", connector_name, requirement[:80], exc)
            continue

        # Best-scoring first, and among equal scores a URL that looks like a
        # file before one that does not: a dataset's measurements get probed
        # before its station list, and a real CSV before the "Data Query Tool"
        # landing page a catalogue happily declares to be CSV. Ranking only —
        # the gate below still decides admission, so this reorders work but
        # never lets anything new through.
        if connector_name != "direct":
            candidates = sorted(
                candidates,
                key=lambda c: (
                    relevance_score(requirement, c),
                    urlparse(c.url).path.lower().endswith(TABULAR_EXTENSIONS),
                ),
                reverse=True,
            )

        for candidate in candidates:
            # `direct` is exempt: a URL the plan itself names is not a search
            # result that has to argue for its relevance — the planner asserted
            # it, and a keyword gate would reject a bare link with no prose.
            if candidate.connector != "direct" and not is_relevant(requirement, candidate):
                continue
            if probed >= MAX_CANDIDATES_PROBED:
                logger.info("Gave up on %r after probing %d candidates", requirement[:80], probed)
                return None
            probed += 1
            acquired = acquire.fetch(
                candidate.url, cache_dir=cache_dir, label=requirement, max_bytes=max_bytes
            )
            if acquired is None:
                continue
            logger.info(
                "Discovered a source for %r via %s: %s (%d rows) from %s",
                requirement[:80],
                candidate.connector,
                candidate.title[:80] or candidate.url[:80],
                acquired.row_count,
                candidate.landing_page[:120],
            )
            return candidate
    return None


def discover_sources(
    sources: list[provenance.DataSource],
    *,
    cache_dir: Path,
    max_bytes: int = acquire.DEFAULT_MAX_BYTES,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]] | None = None,
) -> dict[str, dict]:
    """Search for every requirement nothing could be resolved for. {name: record}.

    Only `unresolved` surrogates are searched. A restricted or credentialed
    source names real data that specifically was not obtained, and answering it
    with something else found by keyword is the over-claim this whole area of
    the codebase exists to prevent — the same rule, and the same reason, as
    `provenance.supersede_unresolved`.
    """
    discoveries: dict[str, dict] = {}
    for source in sources:
        if source.kind != provenance.KIND_SURROGATE or not source.unresolved:
            continue
        if source.name in discoveries:
            continue
        candidate = find_source(
            source.name, cache_dir=cache_dir, max_bytes=max_bytes, connectors=connectors
        )
        if candidate is not None:
            discoveries[source.name] = {**candidate.to_dict(), "query": query_for(source.name)}
    return discoveries


def apply(
    sources: list[provenance.DataSource], discoveries: dict[str, dict] | None
) -> list[provenance.DataSource]:
    """Turn each discovered requirement into the `real_download` it now has.

    Pure, like `acquire.apply`, and composed with it: this sets the `uri` that
    was found, and `acquire.apply` — running immediately after — promotes it to
    `real_local` because those same bytes are already in the cache. One
    mechanism rather than two, so a discovered input and a table-matched one are
    indistinguishable downstream except in what their `reason` says.

    The reason is where the honesty lives. It names the connector, the query
    that was run and the catalogue record chosen, because a source found by
    keyword search is a weaker claim than one a human staged, and the person
    reading data_provenance.json is the one who can judge it.
    """
    if not discoveries:
        return sources
    applied: list[provenance.DataSource] = []
    for source in sources:
        record = discoveries.get(source.name)
        if record is None or source.kind != provenance.KIND_SURROGATE or not source.unresolved:
            applied.append(source)
            continue
        applied.append(
            provenance.DataSource(
                name=source.name,
                kind=provenance.KIND_REAL_DOWNLOAD,
                uri=str(record.get("url") or ""),
                reason=(
                    f"no source was named for this input, so it was searched for: "
                    f"{record.get('connector', '?')} matched "
                    f"{record.get('title') or record.get('url')!r} "
                    f"for the query {record.get('query', '')!r} "
                    f"(catalogue record: {record.get('landing_page') or 'n/a'}). "
                    "Discovered by keyword search, not named by the plan — check that it "
                    "answers the question before reading the verdict as evidence."
                ),
                discovered=dict(record),
            )
        )
    return applied
