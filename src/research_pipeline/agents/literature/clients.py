"""HTTP clients for the external literature-search APIs (arXiv, Semantic Scholar).

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
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,authors,year,externalIds,openAccessPdf,url"

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
            })
        time.sleep(0.2)
    logger.info("Semantic Scholar: found %d unique papers", len(papers))
    return papers
