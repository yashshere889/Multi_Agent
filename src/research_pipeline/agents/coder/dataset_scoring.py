"""The rubric a candidate dataset is scored against — in Python, not in a prompt.

Asking a model "is this dataset good, 0 to 1?" produces a number with no
defensible construction: it is not comparable between candidates, it cannot be
audited afterwards, and it drifts with phrasing. So the split this package
already uses everywhere else applies here too — **the model produces labels and
evidence, Python produces the number**:

    dataset_score = 0.35 * task_relevance
                  + 0.20 * content_relevance
                  + 0.15 * quality
                  + 0.10 * provenance
                  + 0.10 * schema_fit
                  + 0.10 * license_fit

and each dimension is a small fixed band table, not a free scalar:

    task_relevance   exact 1.0 | related 0.6 | weak 0.3 | unrelated 0.0
    license_fit      permitted 1.0 | unknown 0.3 | incompatible 0.0
    schema_fit       expected 1.0 | partial 0.5 | incompatible 0.0

Who supplies each label is deliberate, and graded by how much judgment it needs:

- **Pure Python, never asked of the model.** `license_label` (an allowlist match
  on the Hub's own license metadata), `quality_score` (arithmetic over
  `dataset_inspect`'s measured rates), `provenance_label` (presence of a
  citation / source datasets / an arXiv link on the Hub record). A model has
  nothing to add to any of these and everything to lose by guessing.
- **Hybrid.** `schema_fit_label` — the model maps each required data type to a
  column *name*, and Python checks those names exist in the real schema and
  turns the coverage fraction into a band. The model reads intent; Python checks
  the claim.
- **Model label plus evidence.** `task_relevance` and `content_relevance` only.

`score()` reads labels. Any `score` field in a model response is dropped by
`coerce_appraisal` before it reaches here, and there is a test that a response
claiming 0.99 on an all-`unrelated` dataset still scores near zero.

The critic's findings come from a fixed vocabulary for the same reason: a
free-text objection can't be routed, but `CRITIC_HARD_FAILS` membership can.

Pure: no network, no LLM, no settings, no filesystem.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from research_pipeline.agents.coder.dataset_inspect import InspectionReport
from research_pipeline.agents.coder.dataset_spec import DatasetSpec, tokenize

WEIGHTS: dict[str, float] = {
    "task_relevance": 0.35,
    "content_relevance": 0.20,
    "quality": 0.15,
    "provenance": 0.10,
    "schema_fit": 0.10,
    "license_fit": 0.10,
}

# A weight table that doesn't sum to 1 makes every score silently
# incomparable to the configured minimum. Checked at import rather than with a
# bare `assert`, which `python -O` strips.
if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
    raise RuntimeError(f"dataset_scoring.WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values())}")

TASK_RELEVANCE: dict[str, float] = {"exact": 1.0, "related": 0.6, "weak": 0.3, "unrelated": 0.0}
CONTENT_RELEVANCE: dict[str, float] = {"exact": 1.0, "related": 0.6, "weak": 0.3, "unrelated": 0.0}
SCHEMA_FIT: dict[str, float] = {"expected": 1.0, "partial": 0.5, "incompatible": 0.0}
LICENSE_FIT: dict[str, float] = {"permitted": 1.0, "unknown": 0.3, "incompatible": 0.0}
PROVENANCE: dict[str, float] = {"documented": 1.0, "partial": 0.5, "unknown": 0.3, "absent": 0.0}

BANDS: dict[str, dict[str, float]] = {
    "task_relevance": TASK_RELEVANCE,
    "content_relevance": CONTENT_RELEVANCE,
    "schema_fit": SCHEMA_FIT,
    "license_fit": LICENSE_FIT,
    "provenance": PROVENANCE,
}

# The band an unrecognised label falls to. Always the pessimistic end: a model
# that answers "moderately relevant, I think" has not established relevance, and
# guessing upward on its behalf is how an unjustified dataset gets accepted.
_FALLBACK_BAND: dict[str, str] = {
    "task_relevance": "unrelated",
    "content_relevance": "unrelated",
    "schema_fit": "incompatible",
    "license_fit": "unknown",
    "provenance": "unknown",
}

# Phrasings seen from the model mapped onto the band they mean. Matched as
# substrings against the lowercased response, longest first, so "not related"
# can't be swallowed by "related".
_LABEL_SYNONYMS: dict[str, tuple[tuple[str, str], ...]] = {
    "task_relevance": (
        ("exact", "exact"),
        ("direct", "exact"),
        ("same task", "exact"),
        ("unrelated", "unrelated"),
        ("not related", "unrelated"),
        ("none", "unrelated"),
        ("weak", "weak"),
        ("loose", "weak"),
        ("tangential", "weak"),
        ("marginal", "weak"),
        ("related", "related"),
        ("adjacent", "related"),
        ("similar", "related"),
    ),
    "schema_fit": (
        ("incompatible", "incompatible"),
        ("mismatch", "incompatible"),
        ("none", "incompatible"),
        ("expected", "expected"),
        ("full", "expected"),
        ("complete", "expected"),
        ("partial", "partial"),
        ("partially", "partial"),
    ),
    "license_fit": (
        ("incompatible", "incompatible"),
        ("prohibited", "incompatible"),
        ("permitted", "permitted"),
        ("permissive", "permitted"),
        ("allowed", "permitted"),
        ("unknown", "unknown"),
        ("unclear", "unknown"),
    ),
    "provenance": (
        ("documented", "documented"),
        ("absent", "absent"),
        ("undocumented", "absent"),
        ("partial", "partial"),
        ("unknown", "unknown"),
    ),
}
_LABEL_SYNONYMS["content_relevance"] = _LABEL_SYNONYMS["task_relevance"]

REQUIREMENT_STATUSES = ("pass", "fail", "unknown")

# --------------------------------------------------------------------------
# License policy — decided here, never asked of the model.
# --------------------------------------------------------------------------

PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsl-1.0",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-3.0",
        "cc-by-2.0",
        "odc-by",
        "odbl",
        "pddl",
        "isc",
        "zlib",
        "unlicense",
        "wtfpl",
        "postgresql",
        "artistic-2.0",
        "openrail",
        "cdla-permissive-1.0",
        "cdla-permissive-2.0",
    }
)
SHARE_ALIKE_LICENSES: frozenset[str] = frozenset(
    {
        "cc-by-sa-4.0",
        "cc-by-sa-3.0",
        "gpl-3.0",
        "gpl-2.0",
        "lgpl-3.0",
        "agpl-3.0",
        "mpl-2.0",
        "epl-2.0",
        "cdla-sharing-1.0",
        "osl-3.0",
    }
)
# Substrings that make a license incompatible with a permissive requirement
# regardless of the exact identifier: noncommercial and no-derivatives clauses.
_RESTRICTIVE_MARKERS: tuple[str, ...] = (
    "-nc",
    "nc-",
    "noncommercial",
    "non-commercial",
    "-nd",
    "nd-",
    "noderiv",
)
# Identifiers that say nothing: present, but no more informative than absent.
_UNINFORMATIVE_LICENSES: frozenset[str] = frozenset({"other", "unknown", "none", "unlicensed", ""})


def license_label(license_id: str | None, spec: DatasetSpec) -> str:
    """The license band, from the Hub's own metadata and the spec's policy.

    Never asked of the model: the Hub publishes a machine-readable identifier,
    and "which licenses does this project accept?" is a policy question the
    project answers, not one a dataset card gets a vote on. A missing or `other`
    identifier is `unknown` (0.3), not `incompatible` — plenty of usable
    datasets are simply sloppy about metadata — but it never scores as permitted.
    """
    identifier = (license_id or "").strip().lower()
    if identifier in _UNINFORMATIVE_LICENSES:
        return "unknown"
    if not spec.requires_permissive_license:
        return "permitted"
    if any(marker in identifier for marker in _RESTRICTIVE_MARKERS):
        return "permitted" if spec.allows_noncommercial else "incompatible"
    if identifier in PERMISSIVE_LICENSES:
        return "permitted"
    if identifier in SHARE_ALIKE_LICENSES:
        return "permitted" if spec.allows_share_alike else "incompatible"
    # A real identifier this project has no opinion about. Unknown, not
    # incompatible — the allowlist is not exhaustive and never will be.
    return "unknown"


def license_evidence(license_id: str | None, label: str) -> str:
    identifier = (license_id or "").strip().lower() or "(none published)"
    return f"Hub license metadata is {identifier!r}; project policy classifies that as {label}."


# --------------------------------------------------------------------------
# Quality — arithmetic over measured rates, never asked of the model.
# --------------------------------------------------------------------------

# How much each measured defect can cost, at most. Capped individually so one
# pathological rate can't drive the whole dimension negative on its own and
# hide the others.
_MAX_DUPLICATE_PENALTY = 0.40
_MAX_EMPTY_PENALTY = 0.30
_MAX_MALFORMED_PENALTY = 0.30
_MAX_REPETITION_PENALTY = 0.20
# Repetition below this is normal (label columns, boilerplate license headers)
# and is not penalised at all.
_REPETITION_TOLERANCE = 0.30


def quality_score(report: InspectionReport) -> float:
    """The quality dimension, 0 to 1, from `dataset_inspect`'s measured rates.

    Size enters as a multiplier rather than another subtraction: a spotless
    dataset one tenth the size the plan asked for is genuinely usable-but-thin,
    which is a scaling of its worth, not a defect to deduct.
    """
    if report.rows_sampled <= 0:
        # Nothing could be sampled, so nothing is established. Same principle as
        # an UNKNOWN requirement: not zero, but nowhere near clean.
        return 0.3

    base = 1.0
    base -= min(_MAX_DUPLICATE_PENALTY, report.duplicate_rate)
    base -= min(_MAX_EMPTY_PENALTY, report.empty_rate * 1.5)
    base -= min(_MAX_MALFORMED_PENALTY, report.malformed_rate * 1.5)
    base -= min(_MAX_REPETITION_PENALTY, max(0.0, report.repetition_score - _REPETITION_TOLERANCE))
    base = max(0.0, base)

    # Halved at zero rows available, untouched at or above the requested count.
    return max(0.0, min(1.0, base * (0.5 + 0.5 * report.size_adequacy)))


def quality_evidence(report: InspectionReport) -> str:
    if report.rows_sampled <= 0:
        return "No rows could be sampled, so quality could not be established."
    return (
        f"Measured over {report.rows_sampled} sampled rows: "
        f"{report.duplicate_rate:.0%} duplicates, {report.empty_rate:.0%} empty, "
        f"{report.malformed_rate:.0%} malformed, repetition {report.repetition_score:.2f}, "
        f"size adequacy {report.size_adequacy:.2f}."
    )


# --------------------------------------------------------------------------
# Provenance — read off the Hub record, never asked of the model.
# --------------------------------------------------------------------------


def provenance_label(info: dict | None) -> str:
    """How well documented this dataset's origin is, from the Hub record alone."""
    if not isinstance(info, dict) or not info:
        return "absent"
    raw_card = info.get("cardData")
    card = raw_card if isinstance(raw_card, dict) else {}
    tags = [str(tag).lower() for tag in info.get("tags") or []]

    documented = any(
        [
            bool(card.get("citation")),
            bool(card.get("source_datasets")),
            bool(card.get("paperswithcode_id")),
            any(tag.startswith("arxiv:") for tag in tags),
            any(tag.startswith("doi:") for tag in tags),
        ]
    )
    if documented:
        return "documented"
    if card and (card.get("license") or card.get("task_categories") or card.get("language")):
        return "partial"
    if info.get("id"):
        return "unknown"
    return "absent"


def provenance_evidence(info: dict | None, label: str) -> str:
    if label == "documented":
        return "Hub record carries a citation, source datasets, or a paper/DOI link."
    if label == "partial":
        return "Hub record has structured card metadata but names no upstream source or citation."
    if label == "unknown":
        return "Hub record exists but publishes no card metadata establishing where the data came from."
    return "No Hub record metadata was retrievable for this dataset."


# --------------------------------------------------------------------------
# Schema fit — the model maps, Python verifies.
# --------------------------------------------------------------------------


def schema_fit_label(
    column_mapping: Any, spec: DatasetSpec, actual_columns: list[str]
) -> tuple[str, dict[str, str]]:
    """Coverage of the spec's required data types by real columns.

    The model proposes `{"instruction": "prompt", "code": "completion"}`; this
    keeps only the entries naming a column that actually exists in the schema
    the viewer reported, then bands the coverage fraction. A mapping onto an
    invented column name is exactly the hallucination this arrangement exists to
    catch, and it is dropped silently rather than argued with.
    """
    required = [data_type for data_type in spec.data_types if data_type]
    if not required:
        return "partial", {}

    available = {name.lower(): name for name in actual_columns}
    resolved: dict[str, str] = {}
    if isinstance(column_mapping, dict):
        for data_type, column in column_mapping.items():
            if not isinstance(column, str) or not column.strip():
                continue
            match = available.get(column.strip().lower())
            if match and str(data_type).lower() in {r.lower() for r in required}:
                resolved[str(data_type).lower()] = match

    coverage = len(resolved) / len(required)
    if coverage >= 1.0:
        return "expected", resolved
    if coverage >= 0.5:
        return "partial", resolved
    return "incompatible", resolved


def schema_evidence(resolved: dict[str, str], spec: DatasetSpec, label: str) -> str:
    if not spec.data_types:
        return "The spec names no required data types, so schema fit is treated as partial."
    missing = [data_type for data_type in spec.data_types if data_type.lower() not in resolved]
    mapped = ", ".join(f"{key} -> {value}" for key, value in sorted(resolved.items())) or "none"
    suffix = f"; unmatched: {', '.join(missing)}" if missing else ""
    return f"Required types matched to real columns ({mapped}){suffix}. Banded as {label}."


# --------------------------------------------------------------------------
# Model-supplied labels.
# --------------------------------------------------------------------------


def coerce_label(dimension: str, value: Any) -> str:
    """A model's relevance answer mapped onto one of the dimension's bands.

    Unrecognised, empty, or non-string answers fall to the pessimistic band —
    see `_FALLBACK_BAND`.
    """
    bands = BANDS.get(dimension, {})
    fallback = _FALLBACK_BAND.get(dimension, next(iter(bands), ""))
    if not isinstance(value, str):
        return fallback
    text = value.strip().lower()
    if text in bands:
        return text
    for needle, band in _LABEL_SYNONYMS.get(dimension, ()):
        if needle in text:
            return band
    return fallback


def coerce_requirements(payload: Any) -> dict[str, dict[str, str]]:
    """The per-requirement `{status, evidence}` block, normalized.

    An unrecognised status becomes `unknown`, not `pass` — the prompt's whole
    instruction is to mark UNKNOWN rather than assume, and a coercion that
    rounded ambiguity up to `pass` would undo it.
    """
    requirements: dict[str, dict[str, str]] = {}
    if not isinstance(payload, dict):
        return requirements
    for name, entry in list(payload.items())[:24]:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status not in REQUIREMENT_STATUSES:
            status = "unknown"
        evidence = entry.get("evidence")
        requirements[str(name)[:80]] = {
            "status": status,
            "evidence": (
                " ".join(str(evidence).split())[:400]
                if isinstance(evidence, str) and evidence.strip()
                else "No evidence was provided."
            ),
        }
    return requirements


# --------------------------------------------------------------------------
# Critic vocabulary.
# --------------------------------------------------------------------------

# The critic answers a fixed checklist, so its objections can be routed rather
# than merely read. A free-text finding cannot be mapped to a veto or a penalty;
# these can.
CRITIC_FINDING_CODES: tuple[str, ...] = (
    "description_mismatch",
    "mostly_irrelevant",
    "unusable_schema",
    "substantial_duplication",
    "mostly_empty_or_broken",
    "synthetic_only",
    "undocumented_provenance",
    "unknown_license",
    "suspiciously_tiny",
    "unrelated_material",
    "evaluation_contamination",
    "personal_information",
)

# Findings that end the candidate outright. Contaminated or personal data must
# not be used at all, whatever else is true of it.
CRITIC_HARD_FAILS: frozenset[str] = frozenset(
    {
        "evaluation_contamination",
        "personal_information",
    }
)

# `description_mismatch` is a hard fail only when the measured inspection found
# something that backs it. Barkla job 10334394 is why: the critic vetoed
# Ammok/apple_stock_price at 0.95 on the evidence "the sampled rows show prices
# in the range of 0.1-0.12 ... Apple's stock has historically traded in the
# hundreds of dollars range". Those are split-adjusted prices — AAPL floated at
# $22 in December 1980 and has since split 2:1, 2:1, 2:1, 7:1 and 4:1, a factor
# of 224, so $0.10 is exactly right — and the objection cited no card, only a
# misremembered fact about the world. Meanwhile the inspection had measured 0%
# malformed, 0% empty and repetition 0.01.
#
# The code's own definition is "the rows contradict what the CARD claims", which
# is a comparison between two things in front of the model. When nothing
# measured agrees that the data is off, an unverifiable recollection should cost
# the candidate something without overriding six measured dimensions on its own.
#
# `unusable_schema` is a hard fail only when Python's own verification agrees —
# that is, when `schema_fit` was banded `incompatible`. The rubric already
# measured schema fit by checking the model's column mapping against the real
# schema, and `partial` deliberately means "usable but imperfect". Letting the
# critic veto a `partial` lets it overturn a dimension that was already scored,
# on the same evidence, which is the invented-verdict problem this whole module
# exists to prevent — one level up from the invented score.
#
# Barkla job 10334335 is why this is conditional: Ammok/apple_stock_price scored
# 0.90 (task and content both exact) with schema_fit=partial, because the spec
# asked for `stock_symbol` and a single-ticker Apple series has no symbol column
# — every row is the same symbol. The critic hard-failed it on exactly that
# already-scored fact and the run went back to synthesized data.
CONDITIONAL_HARD_FAILS: dict[str, str] = {"unusable_schema": "incompatible"}


def inspection_corroborates_a_mismatch(report: InspectionReport) -> bool:
    """Whether anything measured backs a claim that the data isn't what it says.

    A genuine description mismatch almost always leaves a measurable trace: rows
    missing declared columns, empty records, a script no requested language
    uses, benchmark text where none was promised, or nothing sampleable at all.
    A dataset that measured clean on every one of those has no corroboration for
    "this data is wrong".
    """
    return bool(
        report.rows_sampled == 0
        or report.malformed_rate > 0.05
        or report.empty_rate > 0.05
        or report.unexpected_language
        or report.contamination_hits
    )


# Soft findings, subtracted from the score. Fixed values, because a penalty the
# model got to size would be the invented score coming back in through a
# different door.
CRITIC_PENALTIES: dict[str, float] = {
    # An uncorroborated description_mismatch — see CONDITIONAL_HARD_FAILS above.
    # Sized so a strong candidate survives an unsupported objection (0.95 -> 0.80,
    # still accepted) while a marginal one does not (0.80 -> 0.65, rejected).
    # The claim counts for something; it is not decisive on its own.
    "description_mismatch": 0.15,
    "mostly_irrelevant": 0.20,
    "substantial_duplication": 0.10,
    "mostly_empty_or_broken": 0.15,
    "synthetic_only": 0.15,
    "undocumented_provenance": 0.05,
    "unknown_license": 0.05,
    "suspiciously_tiny": 0.10,
    "unrelated_material": 0.10,
}


@dataclass(frozen=True)
class CriticFinding:
    code: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "evidence": self.evidence}


def coerce_findings(payload: Any) -> list[CriticFinding]:
    """Findings from a critic response, keeping only the fixed vocabulary.

    A code outside `CRITIC_FINDING_CODES` is dropped: the critic is asked to
    justify a rejection in terms the pipeline can act on, and an invented
    category is neither a veto nor a penalty, so silently honouring it would
    mean an unroutable objection changed the outcome anyway.
    """
    entries: list[Any]
    if isinstance(payload, dict):
        entries = list(payload.get("findings") or [])
    elif isinstance(payload, list):
        entries = list(payload)
    else:
        return []

    findings: list[CriticFinding] = []
    seen: set[str] = set()
    for entry in entries[:24]:
        if isinstance(entry, str):
            code, evidence = entry.strip().lower(), ""
        elif isinstance(entry, dict):
            code = str(entry.get("code") or entry.get("finding") or "").strip().lower()
            evidence = " ".join(str(entry.get("evidence") or "").split())[:400]
        else:
            continue
        code = code.replace("-", "_").replace(" ", "_")
        if code not in CRITIC_FINDING_CODES or code in seen:
            continue
        seen.add(code)
        findings.append(CriticFinding(code=code, evidence=evidence or "No evidence was provided."))
    return findings


# --------------------------------------------------------------------------
# The score itself.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreComponents:
    """One band label (or, for quality, one computed value) per dimension.

    There is deliberately no field here for a model-supplied number. The only
    way into this dataclass is a label from a fixed table or a value Python
    computed, which is what makes `score()` reproducible from the record.
    """

    task_relevance: str = "unrelated"
    content_relevance: str = "unrelated"
    quality: float = 0.0
    provenance: str = "unknown"
    schema_fit: str = "incompatible"
    license_fit: str = "unknown"
    evidence: dict[str, str] = field(default_factory=dict)

    def values(self) -> dict[str, float]:
        return {
            "task_relevance": TASK_RELEVANCE[self.task_relevance],
            "content_relevance": CONTENT_RELEVANCE[self.content_relevance],
            "quality": max(0.0, min(1.0, self.quality)),
            "provenance": PROVENANCE[self.provenance],
            "schema_fit": SCHEMA_FIT[self.schema_fit],
            "license_fit": LICENSE_FIT[self.license_fit],
        }

    def labels(self) -> dict[str, str]:
        return {
            "task_relevance": self.task_relevance,
            "content_relevance": self.content_relevance,
            "quality": f"{self.quality:.2f}",
            "provenance": self.provenance,
            "schema_fit": self.schema_fit,
            "license_fit": self.license_fit,
        }


def score(components: ScoreComponents) -> float:
    """The weighted sum. The only place in this pipeline a dataset score is
    produced."""
    values = components.values()
    return round(sum(WEIGHTS[name] * values[name] for name in WEIGHTS), 4)


@dataclass
class ScoredDataset:
    """A candidate, everything measured about it, and what was decided."""

    dataset_id: str
    components: ScoreComponents
    report: InspectionReport
    base_score: float = 0.0
    score: float = 0.0
    revision: str = ""
    config: str = ""
    split: str = ""
    license: str = ""
    size_bytes: int = 0
    rows: int = 0
    columns: list[dict] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    requirements: dict[str, dict[str, str]] = field(default_factory=dict)
    findings: list[CriticFinding] = field(default_factory=list)
    decision: str = "pending"
    reasons_for_rejection: list[str] = field(default_factory=list)
    local_path: str = ""
    rows_url: str = ""
    downloaded_at: str = ""

    def as_record(self) -> dict[str, Any]:
        """The provenance record written beside the experiment.

        Everything needed to re-derive the score without re-running anything:
        the per-dimension bands and their evidence, the measured inspection
        statistics, the pinned revision, and — when it was rejected — why.
        """
        return {
            "dataset": self.dataset_id,
            "revision": self.revision,
            "downloaded_at": self.downloaded_at,
            "decision": self.decision,
            "reason": self.components.evidence.get("task_relevance", ""),
            "score": round(self.score, 4),
            "base_score": round(self.base_score, 4),
            "weights": dict(WEIGHTS),
            "bands": self.components.labels(),
            "evidence": {name: round(value, 4) for name, value in self.components.values().items()},
            "evidence_notes": dict(self.components.evidence),
            "requirements": self.requirements,
            "column_mapping": dict(self.column_mapping),
            "critic_findings": [finding.to_dict() for finding in self.findings],
            "reasons_for_rejection": list(self.reasons_for_rejection),
            "license": self.license,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "config": self.config,
            "split": self.split,
            "local_path": self.local_path,
            "rows_url": self.rows_url,
            "inspection": self.report.to_dict(),
        }


def rescore(scored: ScoredDataset) -> ScoredDataset:
    """(Re)compute `base_score` and `score` from the components. Called after
    building a candidate and again after the critic applies penalties."""
    scored.base_score = score(scored.components)
    scored.score = scored.base_score
    return scored


def apply_critic(
    scored: ScoredDataset, findings: list[CriticFinding], spec: DatasetSpec
) -> ScoredDataset:
    """Fold critic findings into the decision.

    A hard-fail code vetoes outright, whatever the score was — the point of
    running an adversarial pass at all is that some objections are not tradeable
    against a strong task match. Soft codes subtract their fixed penalty.

    Two codes are conditional. `synthetic_only` vetoes when the spec's `avoid`
    names it (which `REQUIRED_AVOID` guarantees by default) and is otherwise a
    penalty, since a plan that explicitly wants synthetic data exists.
    `unusable_schema` vetoes only when `schema_fit` was banded `incompatible`,
    and `description_mismatch` only when the measured inspection found something
    that backs it — see CONDITIONAL_HARD_FAILS for why the critic does not get
    to overturn what Python already verified.
    """
    scored.findings = findings
    penalty = 0.0
    reasons: list[str] = list(scored.reasons_for_rejection)

    for finding in findings:
        hard = finding.code in CRITIC_HARD_FAILS or (
            finding.code == "synthetic_only" and spec.avoids("synthetic-only")
        )
        # A conditional veto only lands when Python's own band agrees with it.
        required_band = CONDITIONAL_HARD_FAILS.get(finding.code)
        if required_band is not None:
            hard = scored.components.schema_fit == required_band
        if finding.code == "description_mismatch":
            hard = inspection_corroborates_a_mismatch(scored.report)
        if hard:
            reasons.append(f"{finding.code}: {finding.evidence}")
        else:
            penalty += CRITIC_PENALTIES.get(finding.code, 0.0)

    scored.score = round(max(0.0, scored.base_score - penalty), 4)
    scored.reasons_for_rejection = reasons
    if reasons:
        scored.decision = "reject"
    return scored


def decide(scored: ScoredDataset, minimum_score: float) -> ScoredDataset:
    """Accept or reject, with the reasons spelled out. Pure threshold logic —
    no model is consulted about whether its own candidate is good enough."""
    if scored.decision == "reject":
        return scored

    reasons = list(scored.reasons_for_rejection)
    if scored.components.license_fit == "incompatible":
        reasons.append("License requirement not satisfied")
    if scored.components.schema_fit == "incompatible":
        reasons.append("Schema does not match the requested format")
    if scored.score < minimum_score:
        reasons.append(f"Score {scored.score:.2f} is below the {minimum_score:.2f} threshold")

    scored.reasons_for_rejection = reasons
    scored.decision = "reject" if reasons else "accept"
    return scored


# --------------------------------------------------------------------------
# The deterministic prefilter that decides who is worth an LLM call.
# --------------------------------------------------------------------------

# Weights for ranking candidates *before* any model sees them. Separate from
# WEIGHTS on purpose: this is a cheap triage over metadata only (no rows have
# been sampled yet), and it exists to spend the appraisal budget on the three
# most plausible candidates rather than the first three the Hub returned.
_PREFILTER_WEIGHTS: dict[str, float] = {
    "name_overlap": 0.40,
    "license": 0.25,
    "popularity": 0.20,
    "size": 0.15,
}


def _popularity(downloads: Any) -> float:
    """log10-scaled downloads, saturating at 1e6. Linear download counts would
    let one megapopular but irrelevant dataset dominate the whole ranking."""
    try:
        count = max(0, int(downloads or 0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, math.log10(count + 1) / 6.0)


def prefilter_score(candidate: dict, spec: DatasetSpec) -> float:
    """Triage score for one candidate, from Hub metadata alone."""
    wanted = set()
    for text in (spec.task, spec.domain, *spec.data_types, *spec.languages):
        wanted.update(tokenize(str(text)))

    haystack = " ".join(
        [
            str(candidate.get("dataset_id") or candidate.get("id") or "")
            .replace("/", " ")
            .replace("-", " ")
            .replace("_", " "),
            " ".join(str(tag) for tag in candidate.get("tags") or []),
            str((candidate.get("cardData") or {}).get("pretty_name") or ""),
        ]
    )
    found = set(tokenize(haystack))
    overlap = len(wanted & found) / len(wanted) if wanted else 0.0

    license_id = (candidate.get("cardData") or {}).get("license")
    if isinstance(license_id, list):
        license_id = license_id[0] if license_id else ""
    license_value = LICENSE_FIT[license_label(license_id, spec)]

    rows = candidate.get("num_rows") or rows_from_size_categories(candidate.get("cardData"))
    size = 1.0 if spec.desired_examples <= 0 else min(1.0, float(rows) / spec.desired_examples)

    parts = {
        "name_overlap": overlap,
        "license": license_value,
        "popularity": _popularity(candidate.get("downloads")),
        "size": size,
    }
    return round(sum(_PREFILTER_WEIGHTS[name] * parts[name] for name in _PREFILTER_WEIGHTS), 4)


def prefilter(candidates: list[dict], spec: DatasetSpec, limit: int) -> list[dict]:
    """Rank candidates by metadata alone and keep the best `limit`.

    Candidates whose license is outright incompatible are dropped here rather
    than ranked low: paying for an appraisal of a dataset the project may not
    use is the one wasted call that is certain to be wasted.
    """
    keepable: list[tuple[float, dict]] = []
    for candidate in candidates:
        license_id = (candidate.get("cardData") or {}).get("license")
        if isinstance(license_id, list):
            license_id = license_id[0] if license_id else ""
        if license_label(license_id, spec) == "incompatible":
            continue
        keepable.append((prefilter_score(candidate, spec), candidate))

    keepable.sort(key=lambda pair: pair[0], reverse=True)
    ranked = []
    for value, candidate in keepable[: max(0, limit)]:
        enriched = dict(candidate)
        enriched["prefilter_score"] = value
        ranked.append(enriched)
    return ranked


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())[:400]
    return "No evidence was provided."


def coerce_appraisal(payload: Any) -> dict[str, Any]:
    """Normalize an evidence response into exactly the fields scoring consumes.

    The returned dict has **no score key and no way to add one**. That is the
    point: the prompt tells the model not to return a score, and this makes it
    not matter whether it obeyed. Everything else falls to its pessimistic band
    via `coerce_label`, so a malformed or missing response yields a candidate
    that scores near zero rather than one that scores by accident.
    """
    data = payload if isinstance(payload, dict) else {}
    mapping = data.get("column_mapping")
    return {
        "task_relevance": coerce_label("task_relevance", data.get("task_relevance")),
        "task_relevance_evidence": _evidence_text(data.get("task_relevance_evidence")),
        "content_relevance": coerce_label("content_relevance", data.get("content_relevance")),
        "content_relevance_evidence": _evidence_text(data.get("content_relevance_evidence")),
        "column_mapping": mapping if isinstance(mapping, dict) else {},
        "requirements": coerce_requirements(data.get("requirements")),
    }


# Hub `size_categories` tags, mapped to the low end of the range they name. Used
# by the prefilter, which runs before the viewer's /size is known — the low end
# because a triage step should under-promise, and "n>1T" is a tag, not a claim
# anyone is going to hold the uploader to.
_SIZE_CATEGORY_ROWS: dict[str, int] = {
    "n<1k": 0,
    "1k<n<10k": 1_000,
    "10k<n<100k": 10_000,
    "100k<n<1m": 100_000,
    "1m<n<10m": 1_000_000,
    "10m<n<100m": 10_000_000,
    "100m<n<1b": 100_000_000,
    "1b<n<10b": 1_000_000_000,
    "n>1t": 1_000_000_000_000,
}


def rows_from_size_categories(card_data: Any) -> int:
    """A rough row count from the Hub's `size_categories`, or 0 if it says
    nothing useful."""
    if not isinstance(card_data, dict):
        return 0
    categories = card_data.get("size_categories")
    if isinstance(categories, str):
        categories = [categories]
    if not isinstance(categories, list):
        return 0
    best = 0
    for entry in categories:
        best = max(best, _SIZE_CATEGORY_ROWS.get(str(entry).strip().lower(), 0))
    return best


def _dataclass_kwargs(cls: Any, payload: dict) -> dict[str, Any]:
    """Only the keys `cls` actually declares. A checkpoint written by an earlier
    revision of this module must not crash a resume with an unexpected keyword."""
    allowed = {field_.name for field_ in dataclasses.fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def to_state(scored: ScoredDataset) -> dict[str, Any]:
    """The full candidate as plain JSON, for graph state.

    Distinct from `as_record()`, which is the audit artifact written beside the
    experiment: this one round-trips losslessly through `from_state`, that one
    is shaped for a human (and for the Writer) to read.
    """
    return dataclasses.asdict(scored)


def from_state(payload: Any) -> ScoredDataset:
    """Rebuild a candidate from `to_state`. Tolerant of missing/extra keys."""
    data = dict(payload) if isinstance(payload, dict) else {}
    components = data.get("components")
    report = data.get("report")
    findings = data.get("findings")
    data["components"] = ScoreComponents(
        **_dataclass_kwargs(ScoreComponents, components if isinstance(components, dict) else {})
    )
    data["report"] = InspectionReport(
        **_dataclass_kwargs(InspectionReport, report if isinstance(report, dict) else {})
    )
    data["findings"] = [
        CriticFinding(**_dataclass_kwargs(CriticFinding, entry))
        for entry in (findings or [])
        if isinstance(entry, dict)
    ]
    data.setdefault("dataset_id", "")
    return ScoredDataset(**_dataclass_kwargs(ScoredDataset, data))
