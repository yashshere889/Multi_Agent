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

    direct        a URL or DOI written into the plan's own `source` field
    huggingface   the Hub, via `huggingface_client` — where ML benchmark
                  corpora live, and nowhere else here does
    ckan          the CKAN portals below — one API shape (`package_search`)
                  shared by the UK, Canadian, Australian and EU open-data
                  catalogues
    zenodo        research datasets and their DOIs

Adding a portal is one line in `CKAN_PORTALS`; adding a catalogue with a
different API shape is one function plus one line in `CONNECTORS`.

The Hugging Face connector is here because the government and research
catalogues, between them, hold almost no machine-learning corpus. A requirement
like "documents labelled with a category" finds nothing in any of them, while
the Hub serves 20 Newsgroups and AG News — and a run on "how much of a
document's text must be extracted before its category can be classified
reliably" had to have both hand-staged into `CODER_DATA_DIR` for that reason.
It differs from the others in two ways worth knowing. It offers **parquet**
only, since that is the single format the Hub serves every auto-converted
dataset in, so it declines to nominate anything at all when
`acquire.parquet_supported()` is false rather than spending the probe budget
proving it; and it applies `is_relevant` to the search hits *before* asking the
Hub for their parquet URLs, purely to bound the request count — the caller
applies the same gate again afterwards, so this changes cost and not which
candidates are admitted.

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
- **Call a model.** A model can now take part in sourcing, but only through
  callables the caller injects: `coder_agent` owns the LLM and passes in a
  connector that proposes URLs and a `chooser` that ranks candidates. Neither
  is trusted — a proposed URL is fetched and parsed before it counts as
  anything, and a chooser may only reorder and reject candidates that already
  exist, never invent one. The model nominates, Python rules, and this module
  stays testable without an LLM.

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

from research_pipeline.agents.coder import acquire, huggingface_client, provenance

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

# The Hub's per-dataset parquet index: GET it and you get
# {config: {split: [shard urls]}} for every auto-converted dataset. Asked for
# rather than assembled, because the config is not always "default" — openai/gsm8k
# publishes "main" and "socratic" and no "default" at all, so building the URL
# from a template would 404 on exactly the datasets worth having.
HF_DATASET_PARQUET_URL = "https://huggingface.co/api/datasets/{dataset_id}/parquet"
HF_DATASET_LANDING_URL = "https://huggingface.co/datasets/{dataset_id}"

# How many Hub hits are asked for their parquet index. One request each, spent
# only on hits that already cleared the relevance gate on their search-payload
# title and description — so the connector costs one search plus at most this
# many, however many datasets the Hub returns.
MAX_HF_PARQUET_LOOKUPS = 4
# How many queries the Hub is asked before giving up on a requirement. See
# `_hf_queries` for why a requirement needs several.
MAX_HF_QUERIES = 6

# How many catalogue hits are considered per connector, and how many candidates
# are actually fetched before giving up on a requirement. Each fetch is a real
# download, so the second number is the one that costs wall clock.
MAX_HITS_PER_CONNECTOR = 10
MAX_CANDIDATES_PROBED = 6
# How many candidates a chooser is shown. Bounds the prompt: a live search
# returned 396 admitted candidates, and a model asked to rank all of them would
# spend more tokens on the list than on the experiment.
MAX_CANDIDATES_SHOWN = 20

# Share of a requirement's content words a candidate must match. See
# `is_relevant` for why this is a fraction rather than a fixed count.
RELEVANCE_FRACTION = 0.6

# Formats this pipeline can actually read (see acquire.describe). Everything
# else a catalogue offers — XLSX, PDF, XML, ZIP, WMS endpoints — is skipped
# rather than downloaded and rejected.
#
# Parquet is listed unconditionally rather than gated on
# `acquire.parquet_supported()`, even though reading it needs pyarrow. These
# tables describe what the pipeline reads, not what today's interpreter
# happens to have installed, and the cost of being wrong is asymmetric: a
# stray CKAN parquet resource costs one probe, while making the tables
# runtime-dependent would make the whole format set vary by environment.
# `search_huggingface` is the one place that does check, because parquet is
# the only thing it can offer.
TABULAR_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet")
TABULAR_FORMATS = {"CSV", "TSV", "JSON", "JSONL", "NDJSON", "PARQUET"}
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


# Injected by `coder_agent`, never constructed here. Given the requirement and
# the candidates, returns the indices it would probe, best first — a subset, so
# an empty list means "none of these", which is a legitimate and useful answer.
Chooser = Callable[[str, list["Candidate"]], "list[int] | None"]


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


def _hf_queries(requirement: str) -> list[str]:
    """The ladder of queries to try against the Hub, most specific first.

    `query_for`'s space-joined keywords are the wrong shape here, and measurably
    so. The Hub's `search` matches dataset *names*, not free text, and it
    narrows with every word: measured live, "documents labelled category text"
    returns nothing, "documents labelled" returns nothing, and "text
    classification" returns ten — so a single query built from the whole
    requirement is the one thing guaranteed to miss.

    So: `huggingface_client._keyword_queries` first, reused rather than
    restated because it holds the rule that a number joined to a name is part
    of the name ("conll2003" finds CoNLL-2003 where "conll" does not). Then
    every *adjacent pair* of content words, in the order the requirement wrote
    them. Pairs, because a dataset name is where the requirement's vocabulary
    actually lands — "text classification", "named entity", "news articles" —
    while single words return whatever the Hub ranks highest for a topic and
    lean on `is_relevant` to throw all of it away again.

    The caller stops at the first query that yields an admitted candidate, so
    the length of this list is a cap on a miss, not a cost paid every time.
    """
    queries = [query for query in huggingface_client._keyword_queries(requirement) if query]
    words = [
        word.lower()
        for word in re.split(r"[^A-Za-z0-9]+", requirement or "")
        if _is_content_word(word)
    ]
    for first, second in zip(words, words[1:]):
        pair = f"{first} {second}"
        if pair not in queries:
            queries.append(pair)
    # A one-word requirement has no pairs, and dropping it entirely would make
    # the shortest requirements the only unsearchable ones.
    if len(words) == 1 and words[0] not in queries:
        queries.append(words[0])
    return queries[:MAX_HF_QUERIES]


def _hf_parquet_shard(dataset_id: str) -> tuple[str, str, str] | None:
    """(url, config, split) for one parquet shard of `dataset_id`, or None.

    The preferred split if the dataset publishes one, else the first it lists —
    `huggingface_client.PREFERRED_SPLITS` is reused rather than restated, so
    "which split is the representative one?" keeps having one answer in this
    package. The first shard only: `acquire` caps a fetch by bytes, and one
    shard of a Hub corpus is already more rows than these experiments use.
    """
    index = _get_json(HF_DATASET_PARQUET_URL.format(dataset_id=dataset_id), {})
    if not isinstance(index, dict) or not index:
        return None
    configs = {
        name: splits
        for name, splits in index.items()
        if isinstance(splits, dict) and isinstance(name, str)
    }
    if not configs:
        return None
    config = "default" if "default" in configs else next(iter(configs))
    splits = configs[config]
    ordered = [name for name in huggingface_client.PREFERRED_SPLITS if name in splits]
    ordered += [name for name in splits if name not in ordered]
    for split in ordered:
        shards = splits.get(split)
        if isinstance(shards, list) and shards and isinstance(shards[0], str):
            return shards[0], config, str(split)
    return None


def search_huggingface(requirement: str) -> list[Candidate]:
    """Hub datasets whose parquet export this pipeline can fetch and read.

    The Hub is the only catalogue here that holds ML benchmark corpora, and it
    goes through `huggingface_client` rather than a second search
    implementation of its own.

    Queries come from `_hf_queries` and the first that yields an admitted
    candidate wins, the same shape `find_dataset_for_experiment` uses: a later,
    broader query only ever adds worse matches to a pool that already has one.
    """
    # Declining is the honest answer, not a degradation: every candidate this
    # connector can produce is parquet, so without a reader they would each
    # fetch, fail to parse and burn a probe the other connectors could use.
    if not acquire.parquet_supported():
        logger.info("Skipping the Hugging Face connector: no parquet reader available")
        return []

    for query in _hf_queries(requirement):
        candidates: list[Candidate] = []
        lookups = 0
        for hit in huggingface_client.search_datasets(query, limit=MAX_HITS_PER_CONNECTOR):
            dataset_id = str(hit.get("id") or "")
            if not dataset_id or hit.get("gated") or hit.get("private"):
                # Gated and private datasets need a token that generated code
                # would not have, so they are a miss however well they match.
                continue
            described = Candidate(
                name=requirement,
                url="",
                connector="huggingface",
                title=dataset_id,
                # The card summary plus the Hub's own topical tags. Tags are
                # what carry the vocabulary a plan uses — `text-classification`,
                # `topic-classification` — where a dataset id like `ag_news`
                # shares no word with "documents labelled with a category".
                description=" ".join(
                    [str(hit.get("description") or "")]
                    + [str(tag) for tag in hit.get("tags") or [] if isinstance(tag, str)]
                ),
                landing_page=HF_DATASET_LANDING_URL.format(dataset_id=dataset_id),
            )
            # Pre-filtered here purely to bound the request count; `_find_source`
            # applies the identical gate to whatever comes back, so this cannot
            # admit anything the caller would have rejected.
            if not is_relevant(requirement, described):
                continue
            if lookups >= MAX_HF_PARQUET_LOOKUPS:
                break
            lookups += 1
            shard = _hf_parquet_shard(dataset_id)
            if shard is None:
                # No parquet export: the dataset exists only as loose files, or
                # is too large for the Hub to convert. Nothing to fetch.
                continue
            url, config, split = shard
            described.url = url
            described.resource = f"{config}/{split} (parquet)"
            candidates.append(described)
        if candidates:
            return candidates
    return []


# The registry. Order is the search order, cheapest and most specific first.
# `huggingface` sits ahead of the open-data catalogues rather than after them
# because the two answer disjoint questions: the Hub holds the ML corpora and
# almost no civic statistics, CKAN and Zenodo the reverse. Ordering them costs
# nothing when they disagree and saves four `package_search` calls when the
# requirement is a benchmark corpus — and candidates are pooled and ranked
# together afterwards regardless, so this is request order, not precedence.
CONNECTORS: list[tuple[str, Callable[[str], list[Candidate]]]] = [
    ("direct", search_direct),
    ("huggingface", search_huggingface),
    ("ckan", search_ckan),
    ("zenodo", search_zenodo),
]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def rank_with(
    requirement: str, candidates: list[Candidate], chooser: Chooser | None
) -> list[Candidate]:
    """Order the admitted candidates, and drop any the chooser rejects.

    Without a chooser this is Phase 2's ordering exactly: best keyword score
    first, and among equal scores a URL that looks like a file before one that
    does not.

    With one, the model gets to say which of these files actually contains the
    data — the question keyword overlap cannot answer, and the one a measured
    sweep got wrong twice in five (a geographic reference table returned for a
    request about crime counts; COVID-19 case counts for one about pupil
    absence). Its answer is validated to be indices into the list it was shown,
    so it can reorder and reject but never invent; an empty answer means "none
    of these", which leaves the requirement a labelled surrogate — the right
    outcome, and better than real data answering a different question.

    Rejection matters as much as ordering, and a live run showed why: a chooser
    that ranks the right file first but keeps the wrong ones in the list still
    loses when the right one fails to download, because probing falls straight
    through to a candidate it merely ranked lower. That is why
    DATA_SOURCE_SELECTION_PROMPT asks for a candidate's number only if the
    model would defend it, rather than for a ranking of everything.

    A chooser that raises or returns nothing usable falls back to the keyword
    ordering rather than losing the requirement.
    """
    by_score = sorted(
        candidates,
        key=lambda c: (
            relevance_score(requirement, c),
            urlparse(c.url).path.lower().endswith(TABULAR_EXTENSIONS),
        ),
        reverse=True,
    )
    if chooser is None or len(by_score) < 2:
        return by_score

    shown = by_score[:MAX_CANDIDATES_SHOWN]
    try:
        ranked = chooser(requirement, shown)
    except Exception as exc:  # noqa: BLE001 — a broken chooser is not fatal
        logger.warning("Candidate chooser raised for %r: %s", requirement[:80], exc)
        return by_score

    if ranked is None:
        return by_score
    seen: set[int] = set()
    chosen: list[Candidate] = []
    for index in ranked:
        if isinstance(index, int) and 0 <= index < len(shown) and index not in seen:
            seen.add(index)
            chosen.append(shown[index])
    if not chosen:
        logger.info(
            "Chooser rejected all %d candidates for %r — leaving it a surrogate",
            len(shown),
            requirement[:80],
        )
    return chosen


def find_source(
    requirement: str,
    *,
    cache_dir: Path,
    max_bytes: int = acquire.DEFAULT_MAX_BYTES,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]] | None = None,
    chooser: Chooser | None = None,
) -> Candidate | None:
    """One real, fetched, relevant source for `requirement`, or None.

    "Fetched" is not incidental: a candidate is only returned once
    `acquire.fetch` has pulled it and described it as tabular data, so the
    caller never has to trust that a catalogue's URL works. The bytes land in
    the same content-addressed cache the caller then reads, so confirming a
    candidate and acquiring it are one download, not two.
    """
    try:
        return _find_source(requirement, cache_dir, max_bytes, connectors or CONNECTORS, chooser)
    except Exception as exc:  # noqa: BLE001 — never raise; a miss is a surrogate
        logger.warning("Source discovery for %r failed unexpectedly: %s", requirement[:120], exc)
        return None


def _probe(
    requirement: str, candidates: list[Candidate], cache_dir: Path, max_bytes: int, budget: int
) -> tuple[Candidate | None, int]:
    """Fetch candidates in order until one is real data. Returns (hit, budget left)."""
    for candidate in candidates:
        if budget <= 0:
            logger.info("Out of probe budget for %r", requirement[:80])
            return None, 0
        budget -= 1
        acquired = acquire.fetch(
            candidate.url, cache_dir=cache_dir, label=requirement, max_bytes=max_bytes
        )
        if acquired is None:
            continue
        logger.info(
            "Discovered a source for %r via %s: %s / %s (%d rows) from %s",
            requirement[:80],
            candidate.connector,
            candidate.title[:60] or candidate.url[:60],
            candidate.resource[:60],
            acquired.row_count,
            candidate.landing_page[:120],
        )
        return candidate, budget
    return None, budget


def _find_source(
    requirement: str,
    cache_dir: Path,
    max_bytes: int,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]],
    chooser: Chooser | None = None,
) -> Candidate | None:
    """Direct links first, then everything else pooled and ranked together.

    The pooling is what a chooser needs: asked one catalogue at a time, a model
    can only say "the best of these four", where the honest answer is often
    "none of these, but that Zenodo one". `direct` stays outside the pool and is
    probed as soon as it is found — a URL the plan itself asserted needs no
    ranking, and searching four catalogues to second-guess it is waste.
    """
    budget = MAX_CANDIDATES_PROBED
    gathered: list[Candidate] = []

    for connector_name, search in connectors:
        try:
            candidates = search(requirement)
        except Exception as exc:  # noqa: BLE001 — one broken catalogue is not fatal
            logger.warning("Connector %s raised for %r: %s", connector_name, requirement[:80], exc)
            continue
        if connector_name == "direct":
            # Exempt from the relevance gate too: the planner asserted this URL,
            # and a keyword test would reject a bare link that carries no prose.
            found, budget = _probe(requirement, candidates, cache_dir, max_bytes, budget)
            if found is not None:
                return found
        else:
            gathered.extend(candidates)

    admitted = [c for c in gathered if is_relevant(requirement, c)]
    if not admitted:
        return None
    found, _ = _probe(
        requirement, rank_with(requirement, admitted, chooser), cache_dir, max_bytes, budget
    )
    return found


def discover_sources(
    sources: list[provenance.DataSource],
    *,
    cache_dir: Path,
    max_bytes: int = acquire.DEFAULT_MAX_BYTES,
    connectors: list[tuple[str, Callable[[str], list[Candidate]]]] | None = None,
    chooser: Chooser | None = None,
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
            source.name,
            cache_dir=cache_dir,
            max_bytes=max_bytes,
            connectors=connectors,
            chooser=chooser,
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
