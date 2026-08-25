"""HTTP clients for the external literature-search APIs (arXiv, Semantic Scholar, CORE).

Kept separate from nodes.py so they're plain, testable functions with no
LangGraph state coupling.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

import arxiv
import requests

from research_pipeline.agents.literature.state import Paper
from research_pipeline.config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "research-pipeline-literature-agent/0.1 (+https://github.com/)"

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# citationCount/influentialCitationCount/venue/fieldsOfStudy/tldr cost nothing
# extra — they're additional `fields` on a request we already make — and each
# earns its place downstream: the counts and venue are a deterministic quality
# prior for ranking, fieldsOfStudy lets the Interdisciplinary Agent *check* a
# cross-field paper's discipline instead of trusting the query that found it,
# and tldr is a far denser digest than a truncated abstract for the papers that
# have one. Unknown field names make S2 reject the whole request, so anything
# added here must be a real field on the paper object.
SEMANTIC_SCHOLAR_FIELDS = (
    "title,abstract,authors,year,externalIds,openAccessPdf,url,"
    "citationCount,influentialCitationCount,venue,fieldsOfStudy,tldr"
)

# The reference/citation edges return the same paper object a search does, so
# the same field list applies — prefixed per direction in fetch_related, since
# on those endpoints fields are selected on the *related* paper rather than on
# the one being asked about.
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works/"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.5


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """requests.request with a small exponential backoff on transient failures.

    Non-retryable statuses (e.g. 403) are returned immediately so the caller
    can log the real cause instead of masking it behind three identical retries.
    """
    response: Optional[requests.Response] = None
    last_exc: Optional[requests.RequestException] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("%s %s failed (attempt %d/%d): %s", method, url, attempt, MAX_ATTEMPTS, exc)
        else:
            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response
            logger.warning(
                "%s %s returned retryable status %d (attempt %d/%d)",
                method, url, response.status_code, attempt, MAX_ATTEMPTS,
            )
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_BASE_SECONDS * attempt)
    if response is not None:
        return response
    raise last_exc  # every attempt raised — surface the last connection error


def search_arxiv(queries: List[str], max_results: int) -> List[Paper]:
    # delay_seconds/num_retries above the library defaults (3s/3) because a
    # shared-IP host (e.g. a Kaggle node) can get 429s more aggressively than
    # arXiv's documented per-client rate limit anticipates.
    client = arxiv.Client(delay_seconds=5, num_retries=5)
    papers: List[Paper] = []
    seen_ids = set()
    for query in queries:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        try:
            for result in client.results(search):
                arxiv_id = result.get_short_id()
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)
                papers.append({
                    "source": "arxiv",
                    "arxiv_id": arxiv_id,
                    "title": result.title.strip(),
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary.strip(),
                    "year": result.published.year,
                    "pdf_url": result.pdf_url,
                    "doi": result.doi,
                    "url": result.entry_id,
                    # arXiv publishes no citation data and no tldr, so those
                    # stay None here — the keys exist only to keep one shape
                    # across sources. Its subject categories (cs.LG, q-bio.NC,
                    # …) are a genuine field-of-study signal though, and are the
                    # one place a cross-field paper's discipline can be checked
                    # against something other than the query that found it.
                    "citation_count": None,
                    "influential_citation_count": None,
                    "venue": (result.journal_ref or None) if result.journal_ref else None,
                    "fields_of_study": list(result.categories or []),
                    "tldr": None,
                })
        except (arxiv.ArxivError, requests.RequestException) as exc:
            # A single query hitting arXiv's rate limit shouldn't take down the
            # whole pipeline run — log it and keep whatever other queries find,
            # same graceful-degradation contract as search_semantic_scholar.
            logger.error("arXiv query '%s' failed: %s", query, exc)
            continue
    logger.info("arXiv: found %d unique papers", len(papers))
    return papers


def paper_from_semantic_scholar(paper: dict) -> Paper:
    """Maps one Semantic Scholar paper object onto this pipeline's Paper shape.

    Factored out of search_semantic_scholar once the references/citations
    endpoints started returning the same object: they are the same records
    reached by a different edge, so a second mapping would only be a copy that
    drifts.

    Every field here is nullable in a real response — a paper with no abstract,
    no venue, or no tldr is common — so each is normalized to None/[] rather
    than propagated as whatever null-ish thing the API returned.
    """
    open_access = paper.get("openAccessPdf") or {}
    tldr = paper.get("tldr") or {}
    return {
        "source": "semantic_scholar",
        "paper_id": paper.get("paperId"),
        "title": (paper.get("title") or "").strip(),
        "authors": [a.get("name") for a in paper.get("authors") or []],
        "abstract": (paper.get("abstract") or "").strip(),
        "year": paper.get("year"),
        "pdf_url": open_access.get("url"),
        "doi": (paper.get("externalIds") or {}).get("DOI"),
        "url": paper.get("url"),
        "citation_count": paper.get("citationCount"),
        "influential_citation_count": paper.get("influentialCitationCount"),
        "venue": paper.get("venue") or None,
        "fields_of_study": [f for f in (paper.get("fieldsOfStudy") or []) if f],
        "tldr": (tldr.get("text") or "").strip() or None,
    }


def search_semantic_scholar(queries: List[str], max_results: int) -> List[Paper]:
    if not settings.semantic_scholar_api_key:
        logger.warning("Skipping Semantic Scholar search: SEMANTIC_SCHOLAR_API_KEY is not set")
        return []

    headers = {"x-api-key": settings.semantic_scholar_api_key, "User-Agent": USER_AGENT}
    papers: List[Paper] = []
    seen_ids = set()
    for query in queries:
        params = {"query": query, "limit": max_results, "fields": SEMANTIC_SCHOLAR_FIELDS}
        try:
            resp = _request_with_retry(
                "GET", SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers, timeout=30
            )
        except requests.RequestException as exc:
            logger.error("Semantic Scholar query '%s' failed after retries: %s", query, exc)
            continue
        if resp.status_code != 200:
            logger.error(
                "Semantic Scholar query '%s' failed: %d %s", query, resp.status_code, resp.text[:200]
            )
            continue
        data = resp.json()
        for paper in data.get("data", []):
            paper_id = paper.get("paperId")
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            papers.append(paper_from_semantic_scholar(paper))
        time.sleep(0.2)
    logger.info("Semantic Scholar: found %d unique papers", len(papers))
    return papers


def search_core(queries: List[str], max_results: int) -> List[Paper]:
    if not settings.core_api_key:
        logger.warning("Skipping CORE search: CORE_API_KEY is not set")
        return []

    headers = {"Authorization": f"Bearer {settings.core_api_key}", "User-Agent": USER_AGENT}
    papers: List[Paper] = []
    seen_ids = set()
    for query in queries:
        params = {"q": query, "limit": max_results}
        try:
            resp = _request_with_retry("GET", CORE_SEARCH_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            logger.error("CORE query '%s' failed after retries: %s", query, exc)
            continue
        if resp.status_code != 200:
            logger.error("CORE query '%s' failed: %d %s", query, resp.status_code, resp.text[:200])
            continue
        data = resp.json()
        for work in data.get("results", []):
            core_id = work.get("id")
            if core_id is None or core_id in seen_ids:
                continue
            seen_ids.add(core_id)
            # CORE reports a single fieldOfStudy string rather than a list;
            # wrapped here so the key holds a list on every source.
            field_of_study = (work.get("fieldOfStudy") or "").strip()
            papers.append({
                "source": "core",
                "paper_id": str(core_id),
                "title": (work.get("title") or "").strip(),
                "authors": [a.get("name") for a in (work.get("authors") or []) if a.get("name")],
                "abstract": (work.get("abstract") or "").strip(),
                "year": work.get("yearPublished"),
                "pdf_url": work.get("downloadUrl"),
                "doi": work.get("doi"),
                "url": f"https://core.ac.uk/works/{core_id}",
                "citation_count": None,
                "influential_citation_count": None,
                "venue": (work.get("publisher") or "").strip() or None,
                "fields_of_study": [field_of_study] if field_of_study else [],
                "tldr": None,
            })
        # CORE's free tier is far tighter than Semantic Scholar's (~10 req/10s
        # historically), so this self-throttles harder than the 0.2s used there.
        time.sleep(1.0)
    logger.info("CORE: found %d unique papers", len(papers))
    return papers


# -- citation graph ----------------------------------------------------------
#
# Keyword search is the weakest recall tool available: it finds papers whose
# wording matches, and misses the foundational work everyone in a field cites
# without restating its title. A reference list is the opposite — it is a
# bibliography an author curated by hand, so walking one is high-precision
# recall for exactly what the queries missed.

# Which paper an edge lands on, per direction. "references" walks backward to
# what a seed cites (foundational work); "citations" walks forward to what cites
# it (newer follow-ups).
_EDGE_TARGET = {"references": "citedPaper", "citations": "citingPaper"}

# S2 resolves `arXiv:1706.03762`, not `arXiv:1706.03762v2`.
_ARXIV_VERSION_RE = re.compile(r"v\d+$")


def semantic_scholar_identifier(paper: Paper) -> Optional[str]:
    """The id Semantic Scholar will resolve for a paper we already hold, or None
    if it carries nothing S2 can look up.

    Preference order is deliberate: an S2 paper id needs no resolution at all,
    a DOI is unambiguous, and an arXiv id is last because the same work often
    has both a preprint and a published record. Version suffixes are stripped —
    S2 resolves `arXiv:1706.03762`, not `arXiv:1706.03762v2`.
    """
    if paper.get("source") == "semantic_scholar" and paper.get("paper_id"):
        return str(paper["paper_id"])
    doi = (paper.get("doi") or "").strip()
    if doi:
        return f"DOI:{doi}"
    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return "arXiv:" + _ARXIV_VERSION_RE.sub("", arxiv_id)
    return None


def fetch_related(paper_id: str, direction: str, limit: int) -> List[Paper]:
    """One hop along the citation graph from `paper_id`.

    Returns [] rather than raising on every failure mode — no API key, an id S2
    can't resolve, a rate limit, a malformed response. Expansion is an
    enhancement to a pool that is already usable without it, so a failed hop
    must cost only its own results.
    """
    if direction not in _EDGE_TARGET:
        raise ValueError(f"direction must be one of {sorted(_EDGE_TARGET)}, got {direction!r}")
    if not settings.semantic_scholar_api_key:
        logger.warning("Skipping citation expansion: SEMANTIC_SCHOLAR_API_KEY is not set")
        return []

    target = _EDGE_TARGET[direction]
    fields = ",".join(f"{target}.{field}" for field in SEMANTIC_SCHOLAR_FIELDS.split(","))
    headers = {"x-api-key": settings.semantic_scholar_api_key, "User-Agent": USER_AGENT}

    try:
        resp = _request_with_retry(
            "GET",
            f"{SEMANTIC_SCHOLAR_PAPER_URL}/{paper_id}/{direction}",
            params={"fields": fields, "limit": limit},
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("Citation lookup for %s (%s) failed after retries: %s", paper_id, direction, exc)
        return []
    if resp.status_code != 200:
        # A 404 here is routine, not an outage: plenty of papers have no
        # bibliography S2 has parsed, and many publishers' are absent entirely.
        logger.warning(
            "Citation lookup for %s (%s) returned %d: %s", paper_id, direction, resp.status_code, resp.text[:200]
        )
        return []

    try:
        rows = resp.json().get("data") or []
    except ValueError as exc:
        logger.warning("Citation lookup for %s (%s) returned unparseable JSON: %s", paper_id, direction, exc)
        return []

    papers: List[Paper] = []
    for row in rows:
        related = (row or {}).get(target) or {}
        if not (related.get("title") or "").strip():
            continue
        paper = paper_from_semantic_scholar(related)
        # Recorded so the eval harness — and anyone reading metadata.json — can
        # tell which papers the queries found and which the citation graph did.
        paper["discovered_via"] = direction
        papers.append(paper)

    logger.info("%s of %s: %d paper(s)", direction, paper_id, len(papers))
    time.sleep(0.2)
    return papers
