"""Builds the literature-search LangGraph agent.

Reference layout for adding a new agent later: give it its own
agents/<name>/{state,clients,nodes,graph}.py following this file's shape,
then register a subcommand for it in cli.py.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import CachePolicy, RetryPolicy

from research_pipeline.agents.literature.nodes import (
    download_papers_node,
    generate_queries,
    merge_and_dedupe_node,
    save_metadata_node,
    search_arxiv_node,
    search_core_node,
    search_semantic_scholar_node,
)
from research_pipeline.agents.literature.state import LiteratureState
from research_pipeline.checkpointer import get_checkpointer, get_node_cache
from research_pipeline.config import settings

# A thin outer safety net over the retries that already exist further down
# (llm.py's client-level max_retries, clients.py's _request_with_retry): it
# catches what escapes them entirely, e.g. a connection error that outlasts the
# client's own budget, or a JSONDecodeError out of a `resp.json()` call. The
# default retry_on is deliberate — it does *not* retry ValueError/RuntimeError
# and friends, which is right here, since a schema failure that survived
# llm_json.py's repair round-trip is a persistent problem, not a flaky one.
_RETRY = RetryPolicy(max_attempts=2)


def build_literature_graph():
    graph = StateGraph(LiteratureState)

    # Cached on the search nodes only: they are the pure "same queries in, same
    # papers out" steps, and they're the ones that cost a third-party API call.
    # Toggled off entirely (rather than given a 0s TTL) when the setting is off,
    # so "disabled" means the node is compiled without a cache policy at all.
    search_cache = (
        {"cache_policy": CachePolicy(ttl=settings.paper_search_cache_ttl_seconds)}
        if settings.enable_paper_search_cache
        else {}
    )

    graph.add_node("generate_queries", generate_queries, retry_policy=_RETRY)
    graph.add_node("search_arxiv", search_arxiv_node, retry_policy=_RETRY, **search_cache)
    graph.add_node("search_semantic_scholar", search_semantic_scholar_node, retry_policy=_RETRY, **search_cache)
    graph.add_node("search_core", search_core_node, retry_policy=_RETRY, **search_cache)
    graph.add_node("merge_and_dedupe", merge_and_dedupe_node)
    # No retry on download_papers on purpose: it is already thread-pooled with
    # per-file partial-success tolerance, so re-running the node on one failed
    # download would re-fetch every paper that already succeeded.
    graph.add_node("download_papers", download_papers_node)
    graph.add_node("save_metadata", save_metadata_node)

    graph.set_entry_point("generate_queries")

    # Fan out: all three searches start as soon as queries are generated
    graph.add_edge("generate_queries", "search_arxiv")
    graph.add_edge("generate_queries", "search_semantic_scholar")
    graph.add_edge("generate_queries", "search_core")

    # Fan in: merge waits for all three branches to complete
    graph.add_edge("search_arxiv", "merge_and_dedupe")
    graph.add_edge("search_semantic_scholar", "merge_and_dedupe")
    graph.add_edge("search_core", "merge_and_dedupe")

    graph.add_edge("merge_and_dedupe", "download_papers")
    graph.add_edge("download_papers", "save_metadata")
    graph.add_edge("save_metadata", END)

    # Checkpointing: a crash mid-run can resume from the last completed node
    # (via the same thread_id) instead of re-querying every API and
    # re-downloading everything. In-memory by default; set CHECKPOINTER_BACKEND
    # to sqlite/postgres to make that survive process restarts too.
    return graph.compile(
        checkpointer=get_checkpointer(),
        # The cache is process-wide, so a second run in the same process reuses
        # the first one's searches. Omitted entirely when caching is off.
        **({"cache": get_node_cache()} if settings.enable_paper_search_cache else {}),
    )
