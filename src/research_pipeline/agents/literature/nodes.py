"""LangGraph node functions for the literature-search agent."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import requests

from research_pipeline.agents.literature.clients import (
    USER_AGENT,
    search_arxiv,
    search_core,
    search_semantic_scholar,
    search_semantic_scholar_snippets,
)
from research_pipeline.agents.literature.state import LiteratureState, Paper
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.reranker import rerank_papers
from research_pipeline.llm_json import strip_fences

logger = logging.getLogger(__name__)

QUERY_GEN_PROMPT = """You are a research assistant helping to search academic databases.
Given the research question below, generate 3 short, focused search queries (keywords or short phrases,
not full sentences) that would find relevant academic papers.
Return ONLY a JSON list of strings, nothing else, no markdown fences.

Research question: {question}
"""

DOWNLOAD_WORKERS = 8


def generate_queries(state: LiteratureState) -> dict:
    question = state["research_question"]
    queries: List[str]
    try:
        chat_model = get_chat_model()
        response = chat_model.invoke(QUERY_GEN_PROMPT.format(question=question))
        # Shared with every other agent, so this one also drops a Nemotron
        # <think> trace rather than only markdown fences.
        parsed = json.loads(strip_fences(response.content))
        assert isinstance(parsed, list) and all(isinstance(q, str) for q in parsed)
        queries = parsed
    except Exception as exc:
        # covers a down/unreachable LLM server, a timeout, or malformed JSON output
        logger.warning("Query generation failed (%s) — falling back to the raw research question", exc)
        queries = [question]

    # case-insensitive de-dupe, preserving order, so near-identical model output
    # doesn't burn extra API calls against arXiv / Semantic Scholar
    seen = set()
    deduped: List[str] = []
    for q in queries:
        q = q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            deduped.append(q)
    queries = deduped or [question]

    logger.info("Generated queries: %s", queries)
    return {"search_queries": queries}


def search_arxiv_node(state: LiteratureState) -> dict:
    max_results = state.get("max_results_per_query", settings.default_max_results_per_query)
    return {"arxiv_papers": search_arxiv(state["search_queries"], max_results)}


def search_semantic_scholar_node(state: LiteratureState) -> dict:
    max_results = state.get("max_results_per_query", settings.default_max_results_per_query)
    return {"semantic_scholar_papers": search_semantic_scholar(state["search_queries"], max_results)}


def search_core_node(state: LiteratureState) -> dict:
    max_results = state.get("max_results_per_query", settings.default_max_results_per_query)
    return {"core_papers": search_core(state["search_queries"], max_results)}


def search_semantic_scholar_snippets_node(state: LiteratureState) -> dict:
    """Body-passage search, gated by ENABLE_SNIPPET_SEARCH.

    Gated in the node rather than by leaving it out of the graph so the fan-in
    shape is the same either way — disabled, it contributes an empty branch,
    exactly like an unset SEMANTIC_SCHOLAR_API_KEY already does.

    The endpoint's limit counts passages, not papers, and one paper commonly
    matches several — so this asks for SNIPPET_PASSAGES_PER_RESULT passages per
    result the run wanted, rather than for max_results_per_query directly (which
    would under-return papers) or a flat count of its own (which would ignore
    --max-results and let this one branch dominate a deliberately small run).
    """
    if not settings.enable_snippet_search:
        logger.info("Skipping snippet search: ENABLE_SNIPPET_SEARCH is off")
        return {"snippet_papers": []}
    max_results = state.get("max_results_per_query", settings.default_max_results_per_query)
    return {
        "snippet_papers": search_semantic_scholar_snippets(
            state["search_queries"], max_results * settings.snippet_passages_per_result
        )
    }


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


# What a losing duplicate is still allowed to contribute to the record that beat
# it. Each of these is either present or absent — never two sources disagreeing
# about the same fact — so filling a gap needs no judgment call. `full_text` is
# why this exists: the snippet search is the only source carrying passages and
# the only one carrying no DOI/year/PDF, so a paper found both ways must end up
# with the richer record's metadata *and* the snippet record's body text,
# whichever of the two happens to win the pdf_url tie-break below.
_FILLABLE_FIELDS = ("abstract", "year", "doi", "url", "full_text")


def _merge_duplicate(existing: Paper, incoming: Paper) -> Paper:
    winner, loser = (
        (incoming, existing)
        if (not existing.get("pdf_url") and incoming.get("pdf_url"))
        else (existing, incoming)
    )
    merged = dict(winner)
    for field in _FILLABLE_FIELDS:
        if not merged.get(field) and loser.get(field):
            merged[field] = loser[field]
    return merged  # type: ignore[return-value]


def merge_and_dedupe_node(state: LiteratureState) -> dict:
    # Snippet results go last so the metadata-complete sources are seen first
    # and win the tie-break on their own terms; a snippet-only paper is still
    # kept, and a snippet duplicate still hands over its passages.
    all_papers = (
        state["arxiv_papers"]
        + state["semantic_scholar_papers"]
        + state["core_papers"]
        + state.get("snippet_papers", [])
    )
    merged: dict[str, Paper] = {}
    for paper in all_papers:
        if not paper.get("title"):
            continue
        key = paper.get("doi") or _normalize_title(paper["title"])
        if key in merged:
            merged[key] = _merge_duplicate(merged[key], paper)
        else:
            merged[key] = paper
    merged_list = list(merged.values())
    with_text = sum(1 for p in merged_list if p.get("full_text"))
    logger.info("Merged to %d unique papers (%d carrying body passages)", len(merged_list), with_text)
    return {"merged_papers": merged_list}


def rerank_papers_node(state: LiteratureState) -> dict:
    """Orders the merged pool by relevance to the research question.

    Sits before download_papers rather than after it so a truncating rerank
    (RERANK_TOP_K) never spends a PDF download on a paper it is about to drop.
    Search here is recall-oriented by construction — four sources, three
    generated queries each, everything merged — and until now nothing judged
    whether what came back was actually about the question.

    No RetryPolicy on this node: rerank_papers swallows its own failures and
    returns the pool untouched, so there is nothing for a retry to catch.
    """
    return {
        "merged_papers": rerank_papers(state.get("research_question"), state["merged_papers"])
    }


def _paper_uid(paper: Paper) -> str:
    """A short, stable-ish identifier used to keep filenames from colliding
    when two papers share a (truncated) title slug."""
    return paper.get("arxiv_id") or paper.get("paper_id") or paper.get("doi") or _normalize_title(paper.get("title", ""))[:12]


def _safe_filename(paper: Paper, ext: str = "pdf") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", paper["title"].strip())[:80].strip("_")
    uid = re.sub(r"[^a-zA-Z0-9]+", "_", _paper_uid(paper))[:20].strip("_")
    return f"{slug}__{uid}.{ext}"


def _looks_like_pdf(content: bytes, content_type: str) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return content[:5] == b"%PDF-"


def _download_one(paper: Paper, download_dir: Path) -> Paper:
    paper = dict(paper)
    pdf_url = paper.get("pdf_url")
    paper["local_path"] = None
    if not pdf_url:
        return paper

    filepath = download_dir / _safe_filename(paper)
    if filepath.exists():
        paper["local_path"] = str(filepath)
        return paper

    try:
        resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        if not _looks_like_pdf(resp.content, resp.headers.get("Content-Type", "")):
            logger.warning(
                "'%s' did not return a PDF (Content-Type=%s) — skipping",
                paper["title"], resp.headers.get("Content-Type"),
            )
        else:
            filepath.write_bytes(resp.content)
            paper["local_path"] = str(filepath)
    except Exception as exc:
        logger.warning("Failed to download '%s': %s", paper["title"], exc)
    return paper


def download_papers_node(state: LiteratureState) -> dict:
    download_dir = Path(state.get("download_dir", "papers"))
    download_dir.mkdir(parents=True, exist_ok=True)
    papers = state["merged_papers"]

    updated_papers: List[Paper] = [None] * len(papers)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        future_to_index = {pool.submit(_download_one, paper, download_dir): i for i, paper in enumerate(papers)}
        for future in as_completed(future_to_index):
            updated_papers[future_to_index[future]] = future.result()

    downloaded = sum(1 for p in updated_papers if p["local_path"])
    logger.info("Downloaded %d/%d PDFs to %s", downloaded, len(updated_papers), download_dir)
    return {"merged_papers": updated_papers}


def save_metadata_node(state: LiteratureState) -> dict:
    metadata_path = Path(state.get("metadata_path", "papers/metadata.json"))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "research_question": state["research_question"],
        "search_queries": state["search_queries"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "papers": state["merged_papers"],
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Saved metadata for %d papers to %s", len(state["merged_papers"]), metadata_path)
    return {"metadata_path": str(metadata_path)}
