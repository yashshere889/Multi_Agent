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
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,authors,year,externalIds,openAccessPdf,url"

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works/"

# Passage-level retrieval over the ~275M-passage S2ORC index (dense
# mxbai-embed-large-v1 embeddings unioned with BM25), i.e. the hosted
# "sparse-dense retriever API endpoint" OpenScholar's README pointed at rather
# than the 746GB local datastore that project ships. It is the only source here
# that matches on a paper's *body*, so it finds work whose abstract never says
# what the query is about — and it is the only one that returns real passages,
# which is what finally populates the `full_text` field every downstream agent
# has preferred over the abstract since agents/hypothesis/papers.py was written.
SEMANTIC_SCHOLAR_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
# Snippet results carry only corpusId/title/authors/openAccessInfo — no
# abstract, year, DOI or PDF link — so a snippet-discovered paper is hydrated
# through the same fields the keyword search above already returns, and lands
# downstream indistinguishable from one found any other way.
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SEMANTIC_SCHOLAR_SNIPPET_FIELDS = "snippet.text,snippet.snippetKind,snippet.section"
# The batch endpoint documents 500 ids per POST.
BATCH_ID_CHUNK = 500
# One paper can match many passages; keeping every one of them would let a
# single well-matched paper dominate the Hypothesis Agent's character-budgeted
# batches (agents/hypothesis/papers.py:chunk_papers). Highest-scoring first,
# so the cap drops the weakest matches.
MAX_SNIPPETS_PER_PAPER = 5

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
            })
        # CORE's free tier is far tighter than Semantic Scholar's (~10 req/10s
        # historically), so this self-throttles harder than the 0.2s used there.
        time.sleep(1.0)
    logger.info("CORE: found %d unique papers", len(papers))
    return papers


def _hydrate_snippet_papers(corpus_ids: List[str], headers: dict) -> dict:
    """corpusId -> the same metadata dict shape search_semantic_scholar builds.

    Degrades to `{}` on any failure: a snippet-discovered paper without its
    metadata still has a title, its authors and its passages, which is enough
    for every downstream agent to use and cite it. Losing the passages because
    a second request failed would not be.
    """
    hydrated: dict = {}
    for start in range(0, len(corpus_ids), BATCH_ID_CHUNK):
        chunk = corpus_ids[start : start + BATCH_ID_CHUNK]
        try:
            resp = _request_with_retry(
                "POST",
                SEMANTIC_SCHOLAR_BATCH_URL,
                params={"fields": SEMANTIC_SCHOLAR_FIELDS},
                json={"ids": [f"CorpusId:{cid}" for cid in chunk]},
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            logger.warning("Snippet metadata hydration failed after retries: %s", exc)
            continue
        if resp.status_code != 200:
            logger.warning(
                "Snippet metadata hydration failed: %d %s", resp.status_code, resp.text[:200]
            )
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("Snippet metadata hydration returned non-JSON: %s", exc)
            continue
        if not isinstance(data, list):
            logger.warning("Snippet metadata hydration returned %s, expected a list", type(data).__name__)
            continue
        # The endpoint returns one entry per requested id, in order, with a null
        # for anything it couldn't resolve — so position is the mapping back to
        # the corpusId we asked about. externalIds is preferred where present
        # rather than trusted blindly.
        for requested_id, entry in zip(chunk, data):
            if not isinstance(entry, dict):
                continue
            returned_id = (entry.get("externalIds") or {}).get("CorpusId")
            hydrated[str(returned_id) if returned_id is not None else requested_id] = entry
        time.sleep(0.2)
    return hydrated


def search_semantic_scholar_snippets(queries: List[str], max_results: int) -> List[Paper]:
    """Body-text passage search, grouped back into one Paper per source paper.

    Returns the same `Paper` shape as every other client here, plus `full_text`
    holding the matched passages — so merge_and_dedupe_node folds these into the
    pool on the existing doi/normalized-title key, with no downstream agent
    needing to know this source exists.
    """
    if not settings.semantic_scholar_api_key:
        logger.warning("Skipping Semantic Scholar snippet search: SEMANTIC_SCHOLAR_API_KEY is not set")
        return []

    headers = {"x-api-key": settings.semantic_scholar_api_key, "User-Agent": USER_AGENT}
    # corpusId -> {"paper": <snippet paper block>, "snippets": [(score, text), ...]}
    by_corpus_id: dict = {}
    for query in queries:
        params = {"query": query, "limit": max_results, "fields": SEMANTIC_SCHOLAR_SNIPPET_FIELDS}
        try:
            resp = _request_with_retry(
                "GET", SEMANTIC_SCHOLAR_SNIPPET_URL, params=params, headers=headers, timeout=30
            )
        except requests.RequestException as exc:
            logger.error("Snippet query '%s' failed after retries: %s", query, exc)
            continue
        if resp.status_code != 200:
            logger.error("Snippet query '%s' failed: %d %s", query, resp.status_code, resp.text[:200])
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            logger.error("Snippet query '%s' returned non-JSON: %s", query, exc)
            continue

        for match in data.get("data") or []:
            if not isinstance(match, dict):
                continue
            paper_block = match.get("paper") or {}
            corpus_id = paper_block.get("corpusId")
            snippet = match.get("snippet") or {}
            text = (snippet.get("text") or "").strip()
            if corpus_id is None or not text:
                continue
            corpus_id = str(corpus_id)
            entry = by_corpus_id.setdefault(corpus_id, {"paper": paper_block, "snippets": []})
            # The same passage can be returned for two of the generated queries;
            # keeping both would just repeat it inside one paper's full text.
            if any(text == existing for _, existing in entry["snippets"]):
                continue
            try:
                score = float(match.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            entry["snippets"].append((score, text))
        time.sleep(0.2)

    if not by_corpus_id:
        logger.info("Semantic Scholar snippets: found 0 papers")
        return []

    hydrated = _hydrate_snippet_papers(list(by_corpus_id), headers)

    papers: List[Paper] = []
    for corpus_id, entry in by_corpus_id.items():
        meta = hydrated.get(corpus_id) or {}
        snippet_paper = entry["paper"]
        ranked = sorted(entry["snippets"], key=lambda pair: pair[0], reverse=True)
        full_text = "\n\n".join(text for _, text in ranked[:MAX_SNIPPETS_PER_PAPER])

        open_access = meta.get("openAccessPdf") or {}
        authors = [a.get("name") for a in meta.get("authors", []) if a.get("name")]
        if not authors:
            authors = [str(a) for a in (snippet_paper.get("authors") or [])]
        papers.append({
            "source": "semantic_scholar_snippets",
            "paper_id": meta.get("paperId") or f"CorpusId:{corpus_id}",
            "title": (meta.get("title") or snippet_paper.get("title") or "").strip(),
            "authors": authors,
            "abstract": (meta.get("abstract") or "").strip(),
            "year": meta.get("year"),
            "pdf_url": open_access.get("url"),
            "doi": (meta.get("externalIds") or {}).get("DOI"),
            "url": meta.get("url"),
            "full_text": full_text,
        })

    logger.info(
        "Semantic Scholar snippets: found %d papers (%d hydrated with full metadata)",
        len(papers), len(hydrated),
    )
    return papers
