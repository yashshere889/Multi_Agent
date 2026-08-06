"""Renders an assembled paper into a PDF styled after a NeurIPS submission:
Times font, a single (not two-) column of roughly NeurIPS's narrower text
width, a centered title/author block, an unnumbered centered "Abstract"
heading, numbered sections/subsections, and a plain centered page number —
not a byte-for-byte reproduction of the NeurIPS LaTeX class (that would mean
replicating its .sty rather than laying out already-drafted text), but the
same recognizable shape.

Pure Python (reportlab) — no LibreOffice, Node/npm, or pandoc required — so it
runs as-is on Barkla (see .env.example) with nothing beyond `uv sync`. No
content decisions are made here; this module only lays out text the Writer
Agent already drafted/validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

_STYLES = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "PaperTitle", parent=_STYLES["Title"], fontName="Times-Bold", fontSize=17, leading=21, spaceAfter=10, alignment=TA_CENTER
)
_AUTHOR_STYLE = ParagraphStyle(
    "Authors", parent=_STYLES["Normal"], fontName="Times-Bold", fontSize=12, leading=15, spaceAfter=2, alignment=TA_CENTER
)
_AFFILIATION_STYLE = ParagraphStyle(
    "Affiliation", parent=_STYLES["Normal"], fontName="Times-Italic", fontSize=10, leading=13, spaceAfter=14, alignment=TA_CENTER
)
_ABSTRACT_HEADING_STYLE = ParagraphStyle(
    "AbstractHeading", parent=_STYLES["Normal"], fontName="Times-Bold", fontSize=11, spaceBefore=4, spaceAfter=6, alignment=TA_CENTER
)
_HEADING_STYLE = ParagraphStyle(
    "SectionHeading", parent=_STYLES["Normal"], fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=14, spaceAfter=6, alignment=TA_LEFT
)
_SUBHEADING_STYLE = ParagraphStyle(
    "SubHeading", parent=_STYLES["Normal"], fontName="Times-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4, alignment=TA_LEFT
)
_BODY_STYLE = ParagraphStyle(
    "Body", parent=_STYLES["Normal"], fontName="Times-Roman", fontSize=10, leading=12, spaceAfter=7, alignment=TA_JUSTIFY
)
_REFERENCE_STYLE = ParagraphStyle(
    "Reference",
    parent=_BODY_STYLE,
    leftIndent=0.3 * inch,
    firstLineIndent=-0.3 * inch,
    spaceAfter=5,
    alignment=TA_LEFT,
)


def _body_flowables(body: str, section_number: int) -> List[Paragraph]:
    """Splits a section's body text on blank lines into paragraphs. A line
    starting with "## " is rendered as a numbered subsection heading ("N.k
    <label>") instead of body text — the light convention writer_agent.py
    uses to insert deterministic per-hypothesis labels (e.g. "## H1 --
    supported") between LLM-authored paragraphs without needing a richer data
    structure to pass across this module boundary."""
    flowables: List[Paragraph] = []
    subsection_number = 0
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            subsection_number += 1
            label = f"{section_number}.{subsection_number} {block[3:].strip()}"
            flowables.append(Paragraph(escape(label), _SUBHEADING_STYLE))
        else:
            flowables.append(Paragraph(escape(block).replace("\n", " "), _BODY_STYLE))
    return flowables


def _draw_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


def build_pdf(
    output_path: Path,
    title: str,
    abstract: str,
    sections: List[Tuple[str, str]],
    references: List[str],
    authors: str = "Anonymous Author(s)",
    affiliation: str = "Anonymous Institution",
) -> None:
    """sections: ordered (heading, body_text) pairs, numbered 1..N in the
    order given. body_text paragraphs are separated by a blank line; a "## "
    prefix on a line marks a numbered subsection (see _body_flowables)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=1.5 * inch,
        rightMargin=1.5 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        title=title,
    )

    story: list = [
        Paragraph(escape(title), _TITLE_STYLE),
        Paragraph(escape(authors), _AUTHOR_STYLE),
        Paragraph(escape(affiliation), _AFFILIATION_STYLE),
    ]
    story.append(Paragraph("Abstract", _ABSTRACT_HEADING_STYLE))
    for block in abstract.split("\n\n"):
        block = block.strip()
        if block:
            story.append(Paragraph(escape(block).replace("\n", " "), _BODY_STYLE))
    story.append(Spacer(1, 0.1 * inch))

    for i, (heading, body) in enumerate(sections, start=1):
        story.append(Paragraph(escape(f"{i} {heading}"), _HEADING_STYLE))
        story.extend(_body_flowables(body, i))

    story.append(Paragraph("References", _HEADING_STYLE))
    if references:
        for ref in references:
            story.append(Paragraph(escape(ref), _REFERENCE_STYLE))
    else:
        story.append(Paragraph("No sources were cited in the text above.", _BODY_STYLE))

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
