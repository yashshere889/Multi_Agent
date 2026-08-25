"""HTTP clients for the external literature-search APIs (arXiv, Semantic Scholar, CORE).

Kept separate from nodes.py so they're plain, testable functions with no
LangGraph state coupling.
"""

from __future__ import annotations

import logging
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
            open_access = paper.get("openAccessPdf") or {}
            # Every one of these is nullable in an S2 response — a paper with no
            # abstract, no venue, or no tldr is common — so each is normalized
            # to None/[] rather than propagated as whatever null-ish thing the
            # API returned.
            tldr = paper.get("tldr") or {}
            papers.append({
                "source": "semantic_scholar",
                "paper_id": paper_id,
                "title": (paper.get("title") or "").strip(),
                "authors": [a.get("name") for a in paper.get("authors", [])],
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
            })
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
