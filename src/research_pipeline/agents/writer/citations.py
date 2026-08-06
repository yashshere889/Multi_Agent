"""Citation grounding for the Writer Agent, in NeurIPS's author-year style
(natbib's default: "(Smith et al., 2020)" parenthetical, "Smith et al. (2020)"
narrative — see prompts.py:SYSTEM_PROMPT for exactly how the model is told to
write these).

Every LLM-drafted section is instructed to cite a paper with a marker of the
form `[[cite:ID]]` (parenthetical, like natbib's \\citep) or `[[citet:ID]]`
(narrative, like \\citet — for when the sentence already names the author,
e.g. "As [[citet:1706.03762]] show, ..."), where ID must be one of the ids in
the paper index built here — the *same* id scheme the Hypothesis Agent used
(agents.hypothesis.papers.normalize_paper: prefers arxiv_id, then paper_id,
then doi, then a positional fallback), so citation ids line up with the paper
ids already referenced by gaps/methods_overview/hypotheses. A marker may name
several ids at once (`[[cite:1,2]]`), rendered as one grouped citation.

Author-year citation needs the *whole* set of cited papers before any single
marker can be safely formatted — two different papers might share the same
first-author surname and year (needing a "2020a"/"2020b" suffix to
disambiguate), and that can only be decided once every section has been
drafted. So resolution is two-phase:
  1. `scan()` every drafted section's raw text (marker syntax intact) to
     collect which paper ids are actually cited, and flag any that aren't in
     the index (see `unresolved` below).
  2. `finalize()` once, after every section has been scanned, to assign
     disambiguation suffixes and lock in the References list order
     (alphabetical by first-author surname, then year — not citation order).
  3. `resolve()` each section's text, replacing markers with their final
     formatted citation text.

A marker naming a paper id that isn't in the index is never trusted into the
final text — it's stripped during `scan()`/`resolve()` and recorded in
`unresolved` instead, so a hallucinated citation can never reach the printed
paper (see WriterAgent's validation pass, which surfaces `unresolved` as
review notes rather than raising, since dropping a citation mid-sentence is an
acceptable degradation and a hard failure here would waste the whole
(possibly expensive) drafting pass over one bad marker).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, TypedDict

from research_pipeline.agents.hypothesis.papers import normalize_paper

MARKER_RE = re.compile(r"\[\[(cite|citet):([^\]]+)\]\]")
_ADJACENT_PARENS_RE = re.compile(r"\)\s*\(")


class IndexedPaper(TypedDict):
    id: str
    title: str
    authors: List[str]
    year: Optional[int]
    source: str
    url: Optional[str]


class UnresolvedCitation(TypedDict):
    section: str
    paper_id: str


def build_paper_index(raw_papers: List[dict]) -> Dict[str, IndexedPaper]:
    """Indexes raw Literature Agent paper dicts by the same id scheme
    normalize_paper uses, keeping just what's needed to cite/reference them
    (title/authors/year/source/url) — full text isn't needed downstream of
    Related Work, which gets the raw papers directly."""
    index: Dict[str, IndexedPaper] = {}
    for i, raw in enumerate(raw_papers or []):
        normalized = normalize_paper(raw, i)
        if normalized is None:
            continue
        url = None
        if isinstance(raw, dict):
            url = raw.get("url") or raw.get("pdf_url") or raw.get("doi")
        index[normalized["id"]] = {
            "id": normalized["id"],
            "title": normalized["title"],
            "authors": normalized["authors"],
            "year": normalized["year"],
            "source": normalized["source"],
            "url": url,
        }
    return index


def surname_of(author: str) -> str:
    parts = author.strip().split()
    return parts[-1].rstrip(",") if parts else "Unknown"


def first_author_surname(authors: List[str]) -> str:
    return surname_of(authors[0]) if authors else "Unknown"


def year_text(year: Optional[int]) -> str:
    return str(year) if year is not None else "n.d."


def build_surname_year_lookup(paper_index: Dict[str, IndexedPaper]) -> Dict[Tuple[str, str], List[str]]:
    """Maps (first-author surname lowercased, year text) -> paper ids sharing
    that key — the reverse index the Reviewer Agent uses to check whether an
    author-year citation appearing in the rendered paper text actually
    corresponds to a real paper (see agents.reviewer.checks)."""
    lookup: Dict[Tuple[str, str], List[str]] = {}
    for paper_id, paper in paper_index.items():
        key = (first_author_surname(paper["authors"]).lower(), year_text(paper["year"]))
        lookup.setdefault(key, []).append(paper_id)
    return lookup


def _author_list_text(authors: List[str]) -> str:
    """Short in-text form: surname(s) only, natbib-style ("Smith" /
    "Smith and Jones" / "Smith et al.") — the full name list belongs in the
    References entry (format_reference), not in-text."""
    surnames = [surname_of(a) for a in authors] if authors else []
    if not surnames:
        return "Unknown Author"
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return f"{surnames[0]} et al."


class CitationRegistry:
    """Assigns NeurIPS/natbib-style author-year citations to `[[cite:ID]]` /
    `[[citet:ID]]` markers. See the module docstring for the required
    scan() -> finalize() -> resolve() call order."""

    def __init__(self, paper_index: Dict[str, IndexedPaper]) -> None:
        self.paper_index = paper_index
        self.unresolved: List[UnresolvedCitation] = []
        self._used_ids: set[str] = set()
        self._suffix_by_id: Dict[str, str] = {}
        self._finalized = False

    def scan(self, text: str, section: str) -> None:
        for match in MARKER_RE.finditer(text):
            for raw_id in match.group(2).split(","):
                paper_id = raw_id.strip()
                if paper_id in self.paper_index:
                    self._used_ids.add(paper_id)
                else:
                    self.unresolved.append({"section": section, "paper_id": paper_id})

    def finalize(self) -> None:
        """Assigns "a"/"b"/... disambiguation suffixes to papers that share
        the same first-author surname and year, and locks the References
        order (alphabetical by surname, then year, then title) — must be
        called once, after every section has been scan()'d and before any
        resolve() call."""
        groups: Dict[Tuple[str, str], List[str]] = {}
        for paper_id in self._used_ids:
            paper = self.paper_index[paper_id]
            key = (first_author_surname(paper["authors"]).lower(), year_text(paper["year"]))
            groups.setdefault(key, []).append(paper_id)

        for ids in groups.values():
            if len(ids) == 1:
                self._suffix_by_id[ids[0]] = ""
            else:
                ordered = sorted(ids, key=lambda pid: (self.paper_index[pid]["title"], pid))
                for i, paper_id in enumerate(ordered):
                    self._suffix_by_id[paper_id] = chr(ord("a") + i)
        self._finalized = True

    def _year_with_suffix(self, paper_id: str) -> str:
        paper = self.paper_index[paper_id]
        return f"{year_text(paper['year'])}{self._suffix_by_id.get(paper_id, '')}"

    def _parenthetical_one(self, paper_id: str) -> str:
        paper = self.paper_index[paper_id]
        return f"{_author_list_text(paper['authors'])}, {self._year_with_suffix(paper_id)}"

    def _narrative_one(self, paper_id: str) -> str:
        paper = self.paper_index[paper_id]
        return f"{_author_list_text(paper['authors'])} ({self._year_with_suffix(paper_id)})"

    def resolve(self, text: str, section: str) -> str:
        assert self._finalized, "CitationRegistry.finalize() must be called before resolve()"

        def _sub(match: "re.Match[str]") -> str:
            kind = match.group(1)
            ids = [pid.strip() for pid in match.group(2).split(",")]
            known = [pid for pid in ids if pid in self.paper_index]
            if not known:
                return ""
            if kind == "citet":
                return "; ".join(self._narrative_one(pid) for pid in known)
            return "(" + "; ".join(self._parenthetical_one(pid) for pid in known) + ")"

        resolved = MARKER_RE.sub(_sub, text)
        return _ADJACENT_PARENS_RE.sub("; ", resolved)

    def citation_order(self) -> List[str]:
        """Paper ids in References order: alphabetical by first-author
        surname, then year, then title — not citation/first-appearance
        order, matching a natbib author-year bibliography."""
        return sorted(
            self._used_ids,
            key=lambda pid: (
                first_author_surname(self.paper_index[pid]["authors"]).lower(),
                year_text(self.paper_index[pid]["year"]),
                self.paper_index[pid]["title"],
            ),
        )

    def format_reference(self, paper_id: str) -> str:
        paper = self.paper_index[paper_id]
        authors = paper["authors"]
        if not authors:
            author_text = "Unknown Author"
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} and {authors[1]}"
        elif len(authors) <= 6:
            author_text = ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            author_text = f"{authors[0]} et al."
        line = f"{author_text} ({self._year_with_suffix(paper_id)}). {paper['title']}."
        if paper["url"]:
            line += f" {paper['url']}"
        return line

    def references(self) -> List[str]:
        return [self.format_reference(pid) for pid in self.citation_order()]
