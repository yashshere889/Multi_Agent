from research_pipeline.agents.literature.nodes import _looks_like_pdf, _normalize_title, _safe_filename


def test_normalize_title_strips_non_alnum_and_lowercases():
    assert _normalize_title("RAG: A Survey!") == "ragasurvey"


def test_safe_filename_disambiguates_same_title_different_ids():
    paper_a = {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}
    paper_b = {"title": "Attention Is All You Need", "arxiv_id": "1706.03762v2"}
    assert _safe_filename(paper_a) != _safe_filename(paper_b)


def test_looks_like_pdf_trusts_content_type_header():
    assert _looks_like_pdf(b"whatever", "application/pdf; charset=binary") is True


def test_looks_like_pdf_falls_back_to_magic_bytes():
    assert _looks_like_pdf(b"%PDF-1.4 ...", "") is True
    assert _looks_like_pdf(b"<html>not a pdf</html>", "text/html") is False
