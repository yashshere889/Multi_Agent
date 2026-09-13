"""Papers with Code catalog lookup, so a generated experiment can be grounded in
a published method that has a real, official implementation.

Why this exists: an experiment plan's `methods` list names techniques in prose —
"contrastive encoder with a momentum queue", `reused_from_literature: true` — and
the codegen prompt already tells the model to "implement it as an established
technique (say which one, in a comment)". Nothing checked that the model knew
*which* established technique, and a small quantized model asked to reimplement a
named method from memory will confidently produce something adjacent to it. This
lookup replaces that recall with a citation: the paper that introduced the
closest matching method, its TL;DR, the canonical method names the catalog
attaches to it, and the URL of the authors' own code.

It is the exact counterpart of `huggingface_client.py` — same contract, other
axis. That one answers "what real data can this experiment read?"; this one
answers "what real implementation is this experiment's method supposed to look
like?".

Plain HTTP against the anonymous, read-only catalog API that backs Hugging
Face's `pwc` CLI (https://github.com/huggingface/pwc-cli), whose own transport
defaults to the same base URL and sends no credentials:

- `papers/search` to find candidate papers, filtered to ones that have an
  official implementation;
- `papers/{id}?include_resources=true` to read a candidate's repository list,
  TL;DR and catalog method names.

Deliberately **not** the `pwc` binary or the `pwc-cli` package. Installing a
standalone CLI would put a second, separately-versioned network client and a
downloaded binary on every host the pipeline runs on — including a Barkla login
node already over its file quota — to wrap two GET requests that `requests`
(already a pipeline dependency) makes directly. `PWC_API_URL` is honoured under
its own name so a local mirror of the API works here exactly as it does for the
CLI.

Nothing here ever raises. Like the dataset lookup, this is an enhancement to a
prompt and never a precondition for generating code: every failure — no network,
a 503, a rate limit, an unexpected payload shape — degrades to `[]`/`None` and
the Coder Agent generates exactly as it did before, mirroring
`agents/literature/clients.py`'s log-and-degrade contract.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import requests

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

# The public catalog API the `pwc` CLI itself defaults to. Overridable via
# PWC_API_URL (settings.coder_pwc_api_url) under the CLI's own env var name.
DEFAULT_PWC_API_URL = "https://paperswithcode.co/api/v1"

USER_AGENT = "research-pipeline-coder-agent/0.1 (+https://github.com/)"

# One attempt, short timeout, for the same reason huggingface_client uses one:
# this call sits directly in front of a code generation the pipeline is waiting
# on, and a nice-to-have prompt block is not worth a minute of backoff.
TIMEOUT_SECONDS = 15
# Hits asked for from the search endpoint. A few more than are kept, because the
# per-candidate detail call is what decides whether a hit is actually usable.
SEARCH_LIMIT = 6
# Candidates whose details are fetched before giving up. Bounds the whole lookup
# at 1-2 search calls + this many detail calls.
MAX_CANDIDATES_PROBED = 4
# References rendered into a prompt. Two is enough to show the model the shape of
# the established approach without crowding out the plan and the starter program
# — see coder_agent._bounded_max_tokens for why prompt size is not free here.
MAX_REFERENCES = 2
# Repository URLs kept per paper. The official one plus at most one well-starred
# reimplementation; beyond that the list is noise.
MAX_REPOS_PER_PAPER = 2
# A TL;DR is one or two sentences in the catalog, but it is model-written and
# nothing guarantees that, so it is cut like any other free-text field.
MAX_TLDR_CHARS = 400


def _api_base_url() -> str:
    return (settings.coder_pwc_api_url or DEFAULT_PWC_API_URL).rstrip("/")


def _get_json(path: str, params: dict[str, Any]) -> Any | None:
    """GETs `path` under the catalog API and returns the decoded JSON, or None on
    any failure at all (connection error, non-200, body that isn't JSON)."""
    url = f"{_api_base_url()}/{path.lstrip('/')}"
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("Papers with Code request to %s failed: %s", url, exc)
        return None
    if response.status_code != 200:
        logger.warning(
            "Papers with Code request to %s returned %d: %s",
            url,
            response.status_code,
            response.text[:200],
        )
        return None
    try:
        return response.json()
    except ValueError as exc:
        logger.warning("Papers with Code response from %s was not JSON: %s", url, exc)
        return None


def search_papers(query: str, mode: str = "semantic", limit: int = SEARCH_LIMIT) -> list[dict]:
    """Catalog paper search, restricted to papers with an official
    implementation. Returns the raw hit dicts, most relevant first, or [] on any
    failure.

    The `has_official_implementation` filter is sent as a query parameter *and*
    re-applied to the results here. The API echoes which filters it actually
    honoured in `applied_filters`, which is only worth reading if the caller
    intends to do something different when one was dropped — and there is exactly
    one right answer here either way: a paper with no official code is no use to
    this lookup, so it is dropped locally regardless of what the server did.
    """
    if not query.strip():
        return []
    payload = _get_json(
        "papers/search",
        {
            "q": query,
            "page": 1,
            "page_size": limit,
            "mode": mode,
            "has_official_implementation": "true",
        },
    )
    if not isinstance(payload, dict):
        return []
    return [
        hit
        for hit in payload.get("results") or []
        if isinstance(hit, dict) and hit.get("has_official_implementation") is True
    ]


def _paper_reference(hit: dict) -> str:
    """The id to fetch details by: the arXiv id when there is one (stable, and
    what a human would recognise), else the catalog's own numeric id."""
    return str(hit.get("arxiv_id") or hit.get("id") or "")


def paper_details(paper_reference: str) -> dict | None:
    """One paper's full catalog record, including its repositories. None on any
    failure."""
    if not paper_reference.strip():
        return None
    payload = _get_json(
        f"papers/{quote(paper_reference.strip(), safe='')}", {"include_resources": "true"}
    )
    return payload if isinstance(payload, dict) else None


def _repositories(payload: dict) -> list[dict]:
    """Official repositories first, then the most-starred others. Each entry is
    {"url", "is_official", "stars"}."""
    entries: list[dict[str, Any]] = []
    for repo in payload.get("repositories") or []:
        if not isinstance(repo, dict) or not repo.get("url"):
            continue
        entries.append(
            {
                "url": str(repo["url"]),
                "is_official": repo.get("is_official") is True,
                "stars": int(repo.get("num_stars") or 0),
            }
        )
    entries.sort(key=lambda repo: (not repo["is_official"], -repo["stars"]))
    return entries[:MAX_REPOS_PER_PAPER]


def _method_names(payload: dict) -> list[str]:
    """The catalog's own names for the methods this paper uses — "GCN", "Adam",
    "Layer Normalization". Worth more to the model than the paper title alone:
    they are the vocabulary its training data indexes implementations under."""
    names = []
    for method in payload.get("methods") or []:
        if isinstance(method, dict) and method.get("name"):
            names.append(str(method["name"]))
    return names


def _as_reference(payload: dict) -> dict | None:
    """Reduces a full catalog record to the handful of fields the codegen prompt
    renders. None when the record carries no repository after all — the detail
    call is the authority on that, not the search hit's boolean."""
    repositories = _repositories(payload)
    if not repositories:
        return None
    tldr = str(payload.get("tldr") or "").strip()
    published = str(payload.get("published") or "")
    return {
        "paper_id": str(payload.get("arxiv_id") or payload.get("id") or ""),
        "title": str(payload.get("title") or "").strip(),
        "year": published[:4],
        "citation_count": int(payload.get("citation_count") or 0),
        "tldr": tldr[:MAX_TLDR_CHARS],
        "url_abs": str(payload.get("url_abs") or payload.get("source_url") or ""),
        "repositories": repositories,
        "methods": _method_names(payload),
    }


def find_reference_implementations(query: str, limit: int = MAX_REFERENCES) -> list[dict]:
    """Finds up to `limit` published papers matching `query` that have official
    code, each reduced to {"paper_id", "title", "year", "citation_count",
    "tldr", "url_abs", "repositories", "methods"}.

    This is the single seam the Coder Agent injects (see its `pwc_lookup_fn`
    constructor argument), so a test substitutes one function instead of faking
    two HTTP endpoints.

    `query` is passed as prose, not reduced to keywords — the opposite of
    `huggingface_client._keyword_queries`, and for the opposite reason: the Hub's
    `search` parameter matches dataset *names*, so a sentence matches nothing,
    while this API's `semantic` mode embeds the query and a sentence is exactly
    what it wants. Semantic mode can still come back empty for jargon the index
    has never seen, so a miss retries once in `keyword` mode, which matches
    titles and does find a method named verbatim.

    Returns [] rather than raising, on every failure including no network, an API
    outage, and an unrecognisable payload: the caller is about to generate code
    either way.
    """
    try:
        seen: set[str] = set()
        probed = 0
        references: list[dict] = []
        for mode in ("semantic", "keyword"):
            for hit in search_papers(query, mode=mode):
                reference_id = _paper_reference(hit)
                if not reference_id or reference_id in seen:
                    continue
                seen.add(reference_id)
                # Counted across both modes, not per mode: this bounds the
                # whole lookup's HTTP cost, and a keyword retry after a
                # fruitless semantic pass must not double it.
                if probed >= MAX_CANDIDATES_PROBED:
                    break
                probed += 1
                payload = paper_details(reference_id)
                if payload is None:
                    continue
                reference = _as_reference(payload)
                if reference is None:
                    continue
                references.append(reference)
                if len(references) >= limit:
                    break
            if references:
                break
        if references:
            logger.info(
                "Papers with Code reference implementations for %r: %s",
                query[:120],
                "; ".join(f"{r['paper_id']} ({r['repositories'][0]['url']})" for r in references),
            )
        else:
            logger.info("No Papers with Code reference implementation found for %r", query[:120])
        return references
    except Exception as exc:  # noqa: BLE001 — see the docstring: never raise
        # _get_json already absorbs every network/decode failure, so reaching
        # here means an unexpected payload shape got past the isinstance guards.
        # Still not worth failing a code generation over.
        logger.warning("Papers with Code lookup for %r failed unexpectedly: %s", query[:120], exc)
        return []
