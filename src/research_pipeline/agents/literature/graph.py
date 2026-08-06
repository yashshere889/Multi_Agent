"""Builds the literature-search LangGraph agent.

Reference layout for adding a new agent later: give it its own
agents/<name>/{state,clients,nodes,graph}.py following this file's shape,
then register a subcommand for it in cli.py.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from research_pipeline.agents.literature.nodes import (
    download_papers_node,
    generate_queries,
    merge_and_dedupe_node,
    save_metadata_node,
    search_arxiv_node,
    search_semantic_scholar_node,
)
from research_pipeline.agents.literature.state import LiteratureState


def build_literature_graph():
    graph = StateGraph(LiteratureState)

    graph.add_node("generate_queries", generate_queries)
    graph.add_node("search_arxiv", search_arxiv_node)
    graph.add_node("search_semantic_scholar", search_semantic_scholar_node)
    graph.add_node("merge_and_dedupe", merge_and_dedupe_node)
    graph.add_node("download_papers", download_papers_node)
    graph.add_node("save_metadata", save_metadata_node)

    graph.set_entry_point("generate_queries")

    # Fan out: both searches start as soon as queries are generated
    graph.add_edge("generate_queries", "search_arxiv")
    graph.add_edge("generate_queries", "search_semantic_scholar")

    # Fan in: merge waits for both branches to complete
    graph.add_edge("search_arxiv", "merge_and_dedupe")
    graph.add_edge("search_semantic_scholar", "merge_and_dedupe")

    graph.add_edge("merge_and_dedupe", "download_papers")
    graph.add_edge("download_papers", "save_metadata")
    graph.add_edge("save_metadata", END)

    # In-memory checkpointing: a crash mid-run can resume from the last
    # completed node (via the same thread_id) instead of re-querying every
    # API and re-downloading everything. Swap for a SqliteSaver if you want
    # that to survive process restarts too.
    return graph.compile(checkpointer=MemorySaver())
