"""The loop hands over its best draft, not its last.

Barkla job 10496143's three reviews found 64, 89 and 111 issues — Related Work
alone went 26, 53, 78 as each revision rewrote text no review had faulted — and
the paper handed over was the third.
"""

from research_pipeline.writer_reviewer_loop import ISSUE_CATEGORIES, best_iteration, issue_count


def _review(hallucinations=0, citations=0, results=0, coverage=0, passed=False):
    return {
        "hallucinations": [{"claim": f"h{i}"} for i in range(hallucinations)],
        "citation_issues": [{"issue": f"c{i}"} for i in range(citations)],
        "results_accuracy_issues": [{"issue": f"r{i}"} for i in range(results)],
        "hypothesis_coverage_issues": [{"issue": f"v{i}"} for i in range(coverage)],
        "quality_scores": {"clarity": 2},
        "overall_pass": passed,
    }


def _history(*counts, passing=None):
    return [
        {
            "iteration": i,
            "paper_path": f"/out/v{i}.pdf",
            "review": _review(hallucinations=count, passed=(passing == i)),
        }
        for i, count in enumerate(counts, start=1)
    ]


def test_issue_count_sums_the_four_counted_categories():
    review = _review(hallucinations=3, citations=1, results=2, coverage=1)
    assert issue_count(review) == 7
    assert set(ISSUE_CATEGORIES) == {
        "hallucinations",
        "citation_issues",
        "results_accuracy_issues",
        "hypothesis_coverage_issues",
    }


def test_quality_scores_do_not_count_as_issues():
    assert issue_count({"quality_scores": {"clarity": 1, "flow": 1}}) == 0


def test_the_recorded_run_hands_over_its_first_draft():
    chosen = best_iteration(_history(64, 89, 111))
    assert chosen["iteration"] == 1
    assert chosen["paper_path"] == "/out/v1.pdf"


def test_a_loop_that_improved_hands_over_its_last_draft():
    assert best_iteration(_history(111, 89, 64))["iteration"] == 3


def test_ties_go_to_the_later_iteration():
    assert best_iteration(_history(12, 12, 12))["iteration"] == 3


def test_a_passing_iteration_wins():
    history = _history(40, 0, passing=2)
    assert best_iteration(history)["iteration"] == 2


def test_a_single_iteration_is_its_own_best():
    assert best_iteration(_history(7))["iteration"] == 1
