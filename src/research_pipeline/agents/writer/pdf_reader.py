"""Reads text back out of a PDF the Writer Agent rendered (pdf_builder.py),
splitting it into the same section/subsection structure it was written in.

This is a reader for *this pipeline's own* PDF shape — numbered "N Heading"
section markers, "N.k <hypothesis_id>[ -- verdict]" subsection markers (see
pdf_builder.py's module docstring) — not a general-purpose PDF parser. Used by
WriterAgent.revise() (to show the model its own previous draft) and by the
Reviewer Agent (to check the rendered paper against ground truth).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader


def extract_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_into_sections(full_text: str, sections_generated: List[str]) -> Dict[str, str]:
    """sections_generated is WriterAgentOutput's own "sections_generated" list
    (e.g. ["Title", "Abstract", "Introduction", ..., "Future Work",
    "References"]) — the exact headings and order pdf_builder.py rendered,
    with "Title"/"Abstract"/"References" unnumbered and everything between
    numbered 1..N. Returns {heading: body_text}; "Title" is never included
    (it has no body of its own — it's on the same line as the title text).
    A heading whose marker line can't be found in the extracted text (e.g. a
    reader for a PDF this module didn't render) is simply omitted."""
    body_headings = [h for h in sections_generated if h not in ("Title", "Abstract", "References")]
    markers: List[Tuple[str, str]] = [("Abstract", "Abstract")]
    markers += [(f"{i} {heading}", heading) for i, heading in enumerate(body_headings, start=1)]
    markers.append(("References", "References"))

    positions: List[Tuple[int, str]] = []
    for label, key in markers:
        match = re.search(rf"(?m)^\s*{re.escape(label)}\s*$", full_text)
        if match:
            positions.append((match.start(), key))
    positions.sort()

    sections: Dict[str, str] = {}
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        body_start = full_text.find("\n", start)
        body_start = body_start + 1 if body_start != -1 else start
        sections[key] = full_text[body_start:end].strip()
    return sections


def split_hypothesis_subsections(section_text: str, hypothesis_ids: List[str]) -> Dict[str, str]:
    """Further splits a Methods/Results/Discussion section body into its
    per-hypothesis subsections (rendered as "N.k <hypothesis_id>[ --
    verdict]"). Returns {} if none of the given ids' subsection markers are
    found (e.g. a section that isn't split per-hypothesis, like Limitations)."""
    positions: List[Tuple[int, str]] = []
    for hid in hypothesis_ids:
        match = re.search(rf"(?m)^\s*\d+\.\d+\s+{re.escape(hid)}\b.*$", section_text)
        if match:
            positions.append((match.start(), hid))
    if not positions:
        return {}
    positions.sort()

    subsections: Dict[str, str] = {}
    for i, (start, hid) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(section_text)
        body_start = section_text.find("\n", start)
        body_start = body_start + 1 if body_start != -1 else start
        subsections[hid] = section_text[body_start:end].strip()
    return subsections
