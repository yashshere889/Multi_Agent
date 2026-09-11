"""Characters the standard PDF fonts cannot draw must not print as black boxes.

Barkla job 10496057's paper printed "2.789 × 10■■ seconds" for 2.789 × 10⁻⁵,
and lost the non-breaking hyphens and Romanian diacritics in its references.
"""

from research_pipeline.agents.writer import pdf_reader
from research_pipeline.agents.writer.pdf_builder import build_pdf, escape


def test_superscripts_become_markup_not_boxes():
    assert escape("2.789 × 10⁻⁵ s") == "2.789 × 10<super>-5</super> s"


def test_subscripts_become_markup():
    assert escape("CO₂") == "CO<sub>2</sub>"


def test_markup_is_only_added_after_escaping():
    assert escape("a < b ⁽²⁾ & c") == "a &lt; b <super>(2)</super> &amp; c"


def test_unsupported_characters_get_plain_equivalents():
    assert escape("lumped‑parameter") == "lumped-parameter"
    assert escape("Căileanu") == "Caileanu"
    # The plain equivalents are themselves XML-escaped: this is Paragraph markup.
    assert escape("σ ≤ 0.5, Δ ≥ 1") == "sigma &lt;= 0.5, Delta &gt;= 1"


def test_characters_the_font_has_are_left_alone():
    text = "Gödel — “quoted” – café × ± …"
    assert escape(text) == text


def test_a_rendered_paper_has_no_missing_glyph_boxes(tmp_path):
    path = tmp_path / "paper.pdf"
    build_pdf(
        path,
        title="Sensitivity of a lumped‑parameter model",
        abstract="Runtime was 2.789 × 10⁻⁵ seconds (σ ≤ 0.5).",
        sections=[("Results", "Căileanu et al. report CO₂ at 10⁻³.")],
        references=["A. Enescu and Monica Răileanu Szeles (2020). Meta‑analysis."],
    )

    text = pdf_reader.extract_text(path)

    assert "■" not in text
    assert "Caileanu" in text and "Raileanu" in text
    assert "sigma" in text
