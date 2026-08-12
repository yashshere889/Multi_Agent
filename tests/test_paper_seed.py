from research_pipeline.paper_seed import seed_literature_output


def test_output_has_the_keys_downstream_nodes_read():
    result = seed_literature_output([{"title": "RAG Paper", "abstract": "abc"}], "does RAG help?")

    # exactly what run_interdisciplinary_literature_node / run_hypothesis_node read
    assert result["research_question"] == "does RAG help?"
    assert [paper["title"] for paper in result["merged_papers"]] == ["RAG Paper"]
    # ...plus the rest of the LiteratureState shape, empty
    assert result["search_queries"] == []
    assert result["arxiv_papers"] == []
    assert result["semantic_scholar_papers"] == []


def test_papers_are_tagged_as_user_provided():
    result = seed_literature_output([{"title": "RAG Paper"}], "q")
    assert result["merged_papers"][0]["source"] == "user_provided"


def test_a_source_the_caller_set_is_not_overwritten():
    result = seed_literature_output([{"title": "RAG Paper", "source": "arxiv"}], "q")
    assert result["merged_papers"][0]["source"] == "arxiv"


def test_papers_without_a_title_are_dropped():
    result = seed_literature_output(
        [{"title": "RAG Paper"}, {"abstract": "no title"}, {"title": ""}],
        "q",
    )
    assert [paper["title"] for paper in result["merged_papers"]] == ["RAG Paper"]


def test_dedupes_on_doi():
    result = seed_literature_output(
        [
            {"title": "One Title", "doi": "10.1/abc"},
            {"title": "A Completely Different Title", "doi": "10.1/abc"},
        ],
        "q",
    )
    assert len(result["merged_papers"]) == 1


def test_dedupes_on_normalized_title_when_there_is_no_doi():
    result = seed_literature_output(
        [{"title": "RAG: A Paper!"}, {"title": "rag a paper"}, {"title": "Another Paper"}],
        "q",
    )
    assert len(result["merged_papers"]) == 2


def test_a_duplicate_with_a_pdf_url_wins():
    result = seed_literature_output(
        [{"title": "RAG Paper", "abstract": "first"}, {"title": "RAG Paper", "abstract": "second", "pdf_url": "u"}],
        "q",
    )
    assert len(result["merged_papers"]) == 1
    assert result["merged_papers"][0]["pdf_url"] == "u"


def test_a_duplicate_without_a_pdf_url_does_not_displace_one_that_has_it():
    result = seed_literature_output(
        [{"title": "RAG Paper", "pdf_url": "u"}, {"title": "RAG Paper", "abstract": "later"}],
        "q",
    )
    assert result["merged_papers"][0]["pdf_url"] == "u"


def test_ids_are_left_exactly_as_given():
    """No id scheme is invented here — hypothesis/papers.py:normalize_paper
    already has the downstream fallback for papers carrying none."""
    result = seed_literature_output([{"title": "RAG Paper", "arxiv_id": "1234.5678"}, {"title": "Other"}], "q")
    by_title = {paper["title"]: paper for paper in result["merged_papers"]}
    assert by_title["RAG Paper"]["arxiv_id"] == "1234.5678"
    assert "arxiv_id" not in by_title["Other"]
    assert "paper_id" not in by_title["Other"]
    assert "doi" not in by_title["Other"]


def test_the_input_dicts_are_not_mutated():
    original = {"title": "RAG Paper"}
    seed_literature_output([original], "q")
    assert original == {"title": "RAG Paper"}
