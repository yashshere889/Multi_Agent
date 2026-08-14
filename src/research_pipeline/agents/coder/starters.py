"""A small library of pre-validated starter programs for common ML/NLP task
shapes, used to ground the Coder Agent's codegen/fix prompts in a real worked
example instead of asking the model to invent structure from scratch every
time.

Each starter is a hand-authored (never model-generated) set of the same seven
splice-relevant sections `prompts.RUN_PY_SECTION_NAMES` names, stored under
`templates/starters/<id>.sections` in the exact `===BEGIN <field>===`/
`===END <field>===` format `llm_sections.py` already defines — reusing that
parser rather than inventing a second convention, and doubling as a positive
test of it. Every starter uses only synthetic data and the standard library
(no numpy/pandas/sklearn), so it compiles, runs, and validates with zero
external dependencies — see `tests/test_coder_starters.py`, which is what
keeps "pre-validated" true on every commit, not just at authoring time.

`select_starter` is a pure, deterministic function of the plan's own text —
no LLM call — matching this codebase's "determinism over model judgment
wherever verifiable" rule for anything that can be checked in Python instead
of asked of the model. It returns None ("general") for anything that doesn't
match a known task shape, which keeps today's abstract, example-free prompt
exactly as it was for those plans.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from research_pipeline.agents.coder import prompts
from research_pipeline.llm_sections import parse_sections

STARTERS_DIR = Path(__file__).resolve().parent / "templates" / "starters"


class StarterProgram(TypedDict):
    id: str
    description: str
    sections: dict[str, str]


def _load_starter(starter_id: str, description: str) -> StarterProgram:
    text = (STARTERS_DIR / f"{starter_id}.sections").read_text()
    sections = parse_sections(text, prompts.RUN_PY_SECTION_NAMES)
    return {"id": starter_id, "description": description, "sections": sections}


# id -> one-line description, shown to the model in the starter block. Order
# here is display order only; match priority is _MATCH_KEYWORDS below.
_DESCRIPTIONS: dict[str, str] = {
    "text_classification": (
        "NLP text classification: tokenize documents, vectorize, train a classifier, "
        "score accuracy/precision/recall/F1."
    ),
    "classification": (
        "Supervised classification on structured/tabular features: train/test split, "
        "fit, predict, score accuracy/precision/recall/F1."
    ),
    "regression": ("Supervised regression: train/test split, fit, predict, score MAE/RMSE/R^2."),
    "clustering": (
        "Unsupervised clustering: fit cluster assignments, score against a validation metric."
    ),
}

STARTERS: dict[str, StarterProgram] = {
    starter_id: _load_starter(starter_id, description)
    for starter_id, description in _DESCRIPTIONS.items()
}

# (starter id, keywords) in priority order — first match wins. Checked as
# case-insensitive substrings against the plan's design + methods + data
# description text. text_classification is checked before classification so
# an NLP classification plan doesn't fall through to the plain tabular
# starter just because "classification" also appears in its text.
_MATCH_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "text_classification",
        (
            "text classification",
            "nlp",
            "corpus",
            "document",
            "sentiment",
            "review",
            "tf-idf",
            "tfidf",
            "bag of words",
            "token",
            "embedding",
            "language model",
        ),
    ),
    (
        "classification",
        (
            "classif",
            "logistic regression",
            "random forest",
            "svm",
            "support vector",
            "decision tree",
            "gradient boosting",
            "xgboost",
            "naive bayes",
            "knn",
            "k-nearest",
        ),
    ),
    (
        "regression",
        (
            "regress",
            "linear regression",
            "ridge",
            "lasso",
            "elastic net",
        ),
    ),
    (
        "clustering",
        (
            "cluster",
            "k-means",
            "kmeans",
            "dbscan",
            "hierarchical clustering",
        ),
    ),
]


def _match_text(plan: dict) -> str:
    design = str(plan.get("design", ""))
    method_names = " ".join(str(m.get("name", "")) for m in (plan.get("methods") or []))
    data_requirements = plan.get("data_requirements") or {}
    data_text = " ".join(str(data_requirements.get(key, "")) for key in ("source", "description"))
    return " ".join([design, method_names, data_text]).lower()


def select_starter(plan: dict) -> StarterProgram | None:
    """The closest-matching starter for `plan`, or None when nothing matches
    (the "general" fallback — today's abstract prompt, no worked example)."""
    text = _match_text(plan)
    for starter_id, keywords in _MATCH_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return STARTERS[starter_id]
    return None
