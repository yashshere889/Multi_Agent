"""What an experiment actually needs from a dataset, written down before
anything is searched for.

The lookup this replaces asked one question — "does a dataset have a name a bit
like these four words?" — and accepted the first answer. That is not a
requirement, so nothing downstream could tell a good match from a bad one: no
license policy to check against, no expected schema to compare columns to, no
target size, and no list of things that should disqualify a candidate.

So the Coder states its requirement first, as a small structured spec:

    task, domain, languages, data_types, desired_examples,
    minimum_quality, license_requirements, avoid

The model drafts it (it is the only party that can read a plan's prose and say
"this wants instruction/code pairs"), but every field is coerced, clamped and
enum-checked here before anything acts on it, and `avoid` is always unioned with
a floor the model is not allowed to drop. If the draft never arrives — the call
raised, the JSON was junk, the feature is switched off — `fallback_spec` builds
one from the plan deterministically, so the rest of the pipeline has a spec
either way and nothing has to special-case its absence.

Pure: no LLM call, no network, no settings, no filesystem. Same rule as
sandbox.py and provenance.py, and for the same reason — the interesting logic
here is decisions about text, which should be testable without any of that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Accepted values for `license_requirements`. Anything else the model invents is
# dropped rather than passed through, since dataset_scoring.license_label
# switches on these exact strings and an unrecognised policy would silently
# behave like "no policy at all".
LICENSE_POLICIES = ("permissive", "share-alike", "noncommercial-ok", "any")

# Unioned into every spec's `avoid`, whatever the model returned. These are the
# three failures the whole appraisal exists to catch, so they are not the
# model's to opt out of: a spec that forgot to mention duplicates would quietly
# disable the duplicate penalty for that plan.
REQUIRED_AVOID = ("synthetic-only", "duplicates", "evaluation-test leakage")

DEFAULT_DESIRED_EXAMPLES = 1000
MAX_DESIRED_EXAMPLES = 100_000_000
DEFAULT_MINIMUM_QUALITY = 0.5

# Dropped when turning prose into a Hub search query. The Hub's `search`
# parameter matches dataset names and ids, not free text, so a full sentence
# matches nothing at all — "a survey of 500 undergraduate students measuring
# sleep quality" finds nothing while "survey undergraduate sleep" finds several.
# Lives here rather than in huggingface_client because it is pure text handling
# with no HTTP in sight; the client imports it.
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "and",
        "any",
        "are",
        "collect",
        "collected",
        "data",
        "dataset",
        "datasets",
        "each",
        "for",
        "from",
        "into",
        "least",
        "new",
        "per",
        "real",
        "sample",
        "samples",
        "set",
        "should",
        "small",
        "some",
        "such",
        "the",
        "their",
        "them",
        "then",
        "this",
        "those",
        "using",
        "which",
        "with",
        "within",
    }
)

# Rough modality guesses for fallback_spec, matched against the plan's prose.
# Deliberately coarse: this path runs only when the model couldn't produce a
# spec, and a coarse spec that lets the search proceed beats no spec at all.
_DATA_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\binstruction|prompt[- ]response|chat|dialog", "instruction"),
    (r"\bcode\b|source code|program|function|repository", "code"),
    (r"\bexplanation|rationale|reasoning|justification", "explanation"),
    (r"\bimage|photo|vision|pixel", "image"),
    (r"\baudio|speech|utterance|waveform", "audio"),
    (r"\bvideo\b|frame sequence", "video"),
    (r"\btabular|spreadsheet|csv|columns of", "tabular"),
    (r"\btime series|timeseries|temporal|longitudinal", "time-series"),
    (r"\bgraph\b|network structure|edge list", "graph"),
    (r"\blabel|classification|annotated|ground truth", "label"),
    (r"\btext\b|corpus|document|sentence|review", "text"),
)

# Same idea for languages, and equally coarse. Natural-language codes only;
# programming languages live in `languages` too when the plan is about code,
# which is why "python" is in here alongside "en".
_LANGUAGE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\bpython\b", "python"),
    (r"\bjavascript\b|\btypescript\b", "javascript"),
    (r"\bjava\b(?!script)", "java"),
    (r"\bc\+\+\b|\bcpp\b", "cpp"),
    (r"\brust\b", "rust"),
    (r"\bchinese\b|\bmandarin\b", "zh"),
    (r"\bspanish\b", "es"),
    (r"\bfrench\b", "fr"),
    (r"\bgerman\b", "de"),
    (r"\bmultilingual\b|\bcross[- ]lingual\b", "multilingual"),
)


def tokenize(text: str) -> list[str]:
    """Content words from prose, in order, ready to be joined into a Hub query.

    Bare numbers ("500 students", "2024") go with the stopwords: they never help
    match a dataset *name*, and they crowd out the words that do.
    """
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and not word.isdigit() and word not in QUERY_STOPWORDS
    ]


@dataclass(frozen=True)
class DatasetSpec:
    """The requirement a candidate dataset is scored against."""

    task: str
    domain: str
    languages: tuple[str, ...]
    data_types: tuple[str, ...]
    desired_examples: int
    minimum_quality: float
    license_requirements: tuple[str, ...]
    avoid: tuple[str, ...]

    @property
    def requires_permissive_license(self) -> bool:
        """Whether an unknown or restrictive license should count against a
        candidate. "any" is an explicit opt-out; an empty policy list is treated
        as permissive-required, since a spec that says nothing about licensing
        is not the same as one that says licensing doesn't matter."""
        return "any" not in self.license_requirements

    @property
    def allows_share_alike(self) -> bool:
        return "share-alike" in self.license_requirements or not self.requires_permissive_license

    @property
    def allows_noncommercial(self) -> bool:
        return "noncommercial-ok" in self.license_requirements or (
            not self.requires_permissive_license
        )

    def avoids(self, item: str) -> bool:
        return any(item in entry for entry in self.avoid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "domain": self.domain,
            "languages": list(self.languages),
            "data_types": list(self.data_types),
            "desired_examples": self.desired_examples,
            "minimum_quality": self.minimum_quality,
            "license_requirements": list(self.license_requirements),
            "avoid": list(self.avoid),
        }


def _clean_str(value: Any, fallback: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())[:200]
    return fallback


def _clean_list(value: Any, limit: int = 12) -> tuple[str, ...]:
    """Lowercased, de-duplicated, order-preserving. Accepts a comma-separated
    string too, which is what the model returns roughly one time in ten."""
    if isinstance(value, str):
        items: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return ()
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            continue
        text = " ".join(str(item).split()).lower()[:60]
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned[:limit])


def _clean_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clean_float(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def validate_spec(payload: Any, plan: dict) -> DatasetSpec:
    """Coerces a model-drafted spec into a `DatasetSpec`. Never raises.

    Every field falls back to what `fallback_spec` would have produced from the
    plan rather than to a global default, so a response that got three fields
    right and dropped the rest still contributes those three.
    """
    base = fallback_spec(plan)
    if not isinstance(payload, dict):
        return base

    policies = tuple(
        policy
        for policy in _clean_list(payload.get("license_requirements"))
        if policy in LICENSE_POLICIES
    )
    avoid = _clean_list(payload.get("avoid"), limit=16)
    return DatasetSpec(
        task=_clean_str(payload.get("task"), base.task),
        domain=_clean_str(payload.get("domain"), base.domain),
        languages=_clean_list(payload.get("languages")) or base.languages,
        data_types=_clean_list(payload.get("data_types")) or base.data_types,
        desired_examples=_clean_int(
            payload.get("desired_examples"), base.desired_examples, 1, MAX_DESIRED_EXAMPLES
        ),
        minimum_quality=_clean_float(
            payload.get("minimum_quality"), base.minimum_quality, 0.0, 1.0
        ),
        license_requirements=policies or base.license_requirements,
        # Union, not replace: see REQUIRED_AVOID.
        avoid=tuple(dict.fromkeys((*avoid, *REQUIRED_AVOID))),
    )


def _plan_prose(plan: dict) -> str:
    requirements = plan.get("data_requirements") or {}
    parts = [
        str(plan.get("objective") or ""),
        str(plan.get("design") or ""),
        str(requirements.get("description") or ""),
        str(requirements.get("source") or ""),
        " ".join(str(step) for step in requirements.get("preprocessing_steps") or []),
        " ".join(str(method.get("name") or "") for method in plan.get("methods") or []),
    ]
    return " ".join(part for part in parts if part)


def fallback_spec(plan: dict) -> DatasetSpec:
    """A spec derived from the plan alone, with no model involved.

    This is what keeps the whole feature optional in the way the rest of the
    Coder's Hugging Face path already is: a failed spec call degrades to a
    coarser requirement, not to no requirement, so search/scoring/critique all
    still run and a candidate is still judged against something.
    """
    prose = _plan_prose(plan)
    lowered = prose.lower()

    data_types = tuple(
        dict.fromkeys(name for pattern, name in _DATA_TYPE_HINTS if re.search(pattern, lowered))
    )
    languages = tuple(
        dict.fromkeys(name for pattern, name in _LANGUAGE_HINTS if re.search(pattern, lowered))
    )
    return DatasetSpec(
        task=_clean_str(plan.get("objective"), "unspecified experimental task"),
        domain=_clean_str(plan.get("design"), "unspecified"),
        languages=languages or ("en",),
        data_types=data_types or ("text",),
        desired_examples=DEFAULT_DESIRED_EXAMPLES,
        minimum_quality=DEFAULT_MINIMUM_QUALITY,
        # No policy stated means permissive-required, not "anything goes" — see
        # DatasetSpec.requires_permissive_license.
        license_requirements=("permissive",),
        avoid=REQUIRED_AVOID,
    )


# Words per Hub query. Measured, not guessed: the Hub's `search` parameter
# matches dataset *names*, and against the live API a two-word query returns a
# full page of hits while three words returns a handful and four returns
# nothing at all ("python code" -> 20, "instruction code python" -> 2,
# "python instruction code pairs" -> 0). Several short queries pooled beats one
# precise query that matches nothing.
MAX_QUERY_WORDS = 2


def _head(words: list[str], count: int) -> list[str]:
    return words[:count]


def search_queries(spec: DatasetSpec, plan: dict, limit: int = 5) -> list[str]:
    """Hub search queries for this spec — several short ones, most specific first.

    Built from the spec's structured fields rather than the first four words of
    the plan's prose, which is what the old `_keyword_queries` did. Two changes
    from that: queries are capped at `MAX_QUERY_WORDS` because anything longer
    matches no dataset name (see the constant), and the caller runs *all* of
    them and pools the hits rather than stopping at the first that returns
    something — scoring happens over the pool, so an extra HTTP call is cheap
    next to appraising a candidate only one angle would have found.

    The angles, in the order they are tried: language+modality, two modalities,
    the plan's own description (the path that still matches when a plan names
    its dataset outright, e.g. "the SQuAD v2 questions"), the domain, the task,
    and finally single salient words as the broad fallback.
    """
    types = [word for data_type in spec.data_types for word in _head(tokenize(data_type), 1)]
    langs = [word for language in spec.languages for word in _head(tokenize(language), 1)]
    task = tokenize(spec.task)
    domain = tokenize(spec.domain)
    requirements = plan.get("data_requirements") or {}
    described = tokenize(str(requirements.get("description") or plan.get("objective") or ""))

    queries: list[str] = []

    def add(*words: str) -> None:
        unique: list[str] = []
        for word in words:
            if word and word not in unique:
                unique.append(word)
        candidate = " ".join(unique[:MAX_QUERY_WORDS])
        if candidate and candidate not in queries:
            queries.append(candidate)

    def pair(words: list[str]) -> None:
        if words:
            add(*_head(words, 2))

    if langs and types:
        add(langs[0], types[0])
    pair(types)
    pair(described)
    pair(domain)
    pair(task)
    # Single words last: broad, but a page of loosely-related candidates the
    # prefilter can rank beats an empty pool.
    for word in (*types, *langs, *domain, *task, *described):
        add(word)

    return queries[:limit]


def from_dict(payload: Any) -> DatasetSpec:
    """Rehydrate a spec from its `to_dict()` form.

    Graph state holds plain dicts (it is checkpointed), so each node that needs
    the spec rebuilds it from the dict already in state rather than the spec
    being threaded around as an object — the same rule the Writer's
    CitationRegistry follows. Tolerant of missing keys for exactly that reason:
    a checkpoint written by an older revision must not crash a resume.
    """
    data = payload if isinstance(payload, dict) else {}
    policies = tuple(
        policy
        for policy in _clean_list(data.get("license_requirements"))
        if policy in LICENSE_POLICIES
    )
    return DatasetSpec(
        task=_clean_str(data.get("task"), "unspecified experimental task"),
        domain=_clean_str(data.get("domain"), "unspecified"),
        languages=_clean_list(data.get("languages")) or ("en",),
        data_types=_clean_list(data.get("data_types")) or ("text",),
        desired_examples=_clean_int(
            data.get("desired_examples"), DEFAULT_DESIRED_EXAMPLES, 1, MAX_DESIRED_EXAMPLES
        ),
        minimum_quality=_clean_float(
            data.get("minimum_quality"), DEFAULT_MINIMUM_QUALITY, 0.0, 1.0
        ),
        license_requirements=policies or ("permissive",),
        avoid=tuple(dict.fromkeys((*_clean_list(data.get("avoid"), limit=16), *REQUIRED_AVOID))),
    )
