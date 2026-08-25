"""Gold sets: research questions paired with papers a good search *should* find.

Where the papers come from
--------------------------
Not from a model, and not from anyone's memory. `bootstrap_from_survey` fetches
the real reference list of a real survey paper through Semantic Scholar's
`/paper/{id}/references` endpoint and records what it returns. A recent survey on
topic X is a hand-curated, peer-reviewed bibliography of topic X — free ground
truth that someone else already did the work of assembling.

This matters more here than it would elsewhere. A gold set of invented titles
and plausible-looking DOIs would still produce clean-looking recall numbers, and
those numbers would be worse than having none at all: they'd be confidently
wrong, and every later decision would be made against them. The same rule the
pipeline applies to synthetic experiment data applies to its own eval data.

What a gold set is *not*
------------------------
It is not the complete set of relevant papers, and recall against it is not an
absolute score — see metrics.recall. A survey's bibliography is one expert's
view of the field at one moment, it skews toward what was published before the
survey, and a run can legitimately return excellent papers that appear nowhere
in it. Treat these numbers as a relative signal between two configurations of
the pipeline, which is exactly what they're reliable for.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

from research_pipeline.agents.literature.clients import USER_AGENT, _request_with_retry
from research_pipeline.config import settings

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
REFERENCE_FIELDS = "title,externalIds,year,abstract"
# S2 caps a references page at 1000; 100 keeps each response small enough to
# stay well inside the request timeout on a slow link.
REFERENCE_PAGE_SIZE = 100
MAX_REFERENCE_PAGES = 20


class GoldSetError(ValueError):
    """Raised when a gold set file doesn't match the documented shape."""


def validate_gold_entry(data: dict, path: str = "gold") -> None:
    """Raises GoldSetError listing every problem, matching the agents'
    validate_output convention rather than failing on the first one."""
    errors: List[str] = []

    if not isinstance(data, dict):
        raise GoldSetError(f"{path} should be an object, got {type(data).__name__}")

    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{path}.question is missing or empty")

    papers = data.get("papers")
    if not isinstance(papers, list):
        errors.append(f"{path}.papers should be a list")
    elif not papers:
        errors.append(f"{path}.papers is empty — a gold set with no papers can't measure recall")
    else:
        for i, paper in enumerate(papers):
            if not isinstance(paper, dict):
                errors.append(f"{path}.papers[{i}] should be an object")
            elif not any(str(paper.get(k) or "").strip() for k in ("title", "doi", "arxiv_id")):
                # Without at least one identifier there is nothing for
                # metrics.paper_keys to match on, so the entry can only ever
                # count as a miss and would silently depress recall.
                errors.append(f"{path}.papers[{i}] has no title, doi, or arxiv_id to match on")

    if errors:
        raise GoldSetError("; ".join(errors))


def load_gold_entry(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GoldSetError(f"{path} is not valid JSON: {exc}") from exc
    validate_gold_entry(data, path=str(path))
    return data


def load_gold_set(path: str | Path) -> List[dict]:
    """Loads one gold file, or every *.json in a directory (sorted, so a run's
    question order is stable across machines)."""
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob("*.json"))
        if not files:
            raise GoldSetError(f"No gold set files found in {target}/")
        return [load_gold_entry(f) for f in files]
    if not target.exists():
        raise GoldSetError(f"Gold set not found: {target}")
    return [load_gold_entry(target)]


def _reference_page(survey_id: str, offset: int, headers: dict) -> dict:
    resp = _request_with_retry(
        "GET",
        f"{SEMANTIC_SCHOLAR_PAPER_URL}/{survey_id}/references",
        params={"fields": REFERENCE_FIELDS, "limit": REFERENCE_PAGE_SIZE, "offset": offset},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        raise GoldSetError(
            f"Semantic Scholar returned {resp.status_code} for {survey_id}'s references: {resp.text[:200]}"
        )
    return resp.json()


def bootstrap_from_survey(
    survey_id: str,
    question: str,
    *,
    min_year: Optional[int] = None,
    notes: str = "",
) -> dict:
    """Builds a gold entry from a survey paper's real reference list.

    `survey_id` is anything Semantic Scholar resolves — `arXiv:2312.10997`,
    `DOI:10.1145/3626235`, a bare S2 paper id, or a URL-form id.

    `min_year` drops references older than a cutoff. Worth using: a survey's
    bibliography reaches back decades, and a pipeline searching today's
    literature genuinely shouldn't be marked down for missing a 1994 paper that
    only appears as background.
    """
    headers = {"User-Agent": USER_AGENT}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    else:
        # Works unauthenticated, just slowly and with a much tighter rate limit
        # — worth saying out loud, since a shared-IP 429 here looks like an
        # empty bibliography rather than a throttle.
        logger.warning("SEMANTIC_SCHOLAR_API_KEY is not set — the references endpoint will be heavily rate limited")

    try:
        meta = _request_with_retry(
            "GET",
            f"{SEMANTIC_SCHOLAR_PAPER_URL}/{survey_id}",
            params={"fields": "title,year,referenceCount"},
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise GoldSetError(f"Could not reach Semantic Scholar for {survey_id}: {exc}") from exc
    if meta.status_code != 200:
        raise GoldSetError(f"Semantic Scholar returned {meta.status_code} for {survey_id}: {meta.text[:200]}")
    survey = meta.json()

    papers: List[dict] = []
    seen: set[str] = set()
    for page in range(MAX_REFERENCE_PAGES):
        data = _reference_page(survey_id, page * REFERENCE_PAGE_SIZE, headers)
        rows = data.get("data") or []
        for row in rows:
            cited = row.get("citedPaper") or {}
            title = (cited.get("title") or "").strip()
            if not title:
                continue
            year = cited.get("year")
            if min_year is not None and isinstance(year, int) and year < min_year:
                continue
            external = cited.get("externalIds") or {}
            key = (external.get("DOI") or external.get("ArXiv") or title).lower()
            if key in seen:
                continue
            seen.add(key)
            papers.append({
                "title": title,
                "doi": external.get("DOI"),
                "arxiv_id": external.get("ArXiv"),
                "year": year,
            })
        if len(rows) < REFERENCE_PAGE_SIZE:
            break
    else:
        logger.warning(
            "Stopped after %d pages of references for %s — the gold set may be truncated",
            MAX_REFERENCE_PAGES, survey_id,
        )

    if not papers:
        raise GoldSetError(
            f"No usable references found for {survey_id}. Check the id resolves, and that the "
            "paper is one Semantic Scholar has a parsed bibliography for — many publishers' are absent."
        )

    logger.info("Collected %d reference(s) from '%s'", len(papers), survey.get("title"))
    return {
        "question": question,
        "notes": notes,
        # Recorded so a gold set is auditable: anyone can re-fetch this survey
        # and check that the bibliography below is really what it cites.
        "source": {
            "kind": "survey_references",
            "survey_id": survey_id,
            "survey_title": survey.get("title"),
            "survey_year": survey.get("year"),
            "min_year": min_year,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "papers": papers,
    }


def write_gold_entry(entry: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validate_gold_entry(entry, path=str(target))
    target.write_text(json.dumps(entry, indent=2, ensure_ascii=False))
    return target
