"""What a dataset's rows actually are, measured rather than described.

A dataset card is marketing copy the uploader wrote; the rows are the thing the
experiment will run on, and the two disagree often enough that trusting the
first is how a run ends up training on 40% duplicates, or on a "code" dataset
that is mostly empty strings, or on rows lifted straight out of the benchmark
the experiment is about to evaluate against. So a candidate is sampled — 200
rows by default, paged from the Dataset Viewer — and the sample is measured
here: duplication, emptiness, malformed records, templated repetition, script
mix, benchmark contamination, PII.

Every number this produces is deterministic and computed in Python. None of it
is asked of the model, and none of it can be overridden by the model — the
appraisal prompt is shown these statistics as *findings*, not as questions.
`dataset_scoring.quality_score` turns them into the rubric's quality dimension,
and `dataset_scoring.CRITIC_HARD_FAILS` uses the contamination and PII counts to
back a veto with something better than an opinion.

Pure: no network, no LLM, no settings, no filesystem — same rule as sandbox.py.
The rows arrive as plain dicts from whoever fetched them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Substrings that mean a row probably came out of an evaluation benchmark. A hit
# is not proof — a dataset legitimately *about* HumanEval will match — which is
# why this is a count fed to a critic that must justify the call, rather than an
# automatic rejection on its own.
CONTAMINATION_MARKERS: tuple[str, ...] = (
    "humaneval",
    "mbpp",
    "gsm8k",
    "mmlu",
    "hellaswag",
    "truthfulqa",
    "bigbench",
    "big-bench",
    "arc-challenge",
    "winogrande",
    "squad_v2",
    "glue benchmark",
    "superglue",
    "test set",
    "leaderboard submission",
)

# Deliberately narrow and high-precision. A false positive here escalates to a
# human-visible rejection reason, so patterns that fire on ordinary prose
# (anything date- or number-shaped) are left out on purpose.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"),  # phone
    re.compile(r"\b(?:4\d{3}|5[1-5]\d{2})[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),  # card
)

# Unicode blocks, coarsely. Enough to answer "are these rows in the script this
# spec asked for?" without adding a language-detection dependency to a package
# whose whole point is that generated experiments install nothing extra.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("latin", 0x0041, 0x024F),
    ("greek", 0x0370, 0x03FF),
    ("cyrillic", 0x0400, 0x04FF),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("devanagari", 0x0900, 0x097F),
    ("cjk", 0x3040, 0x9FFF),
    ("hangul", 0xAC00, 0xD7AF),
)

# Which script a spec's `languages` entry implies. Programming languages and
# "multilingual" map to no expectation at all rather than to latin — a
# multilingual corpus that is 60% CJK is doing exactly what it says.
_LANGUAGE_SCRIPTS: dict[str, str] = {
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "it": "latin",
    "pt": "latin",
    "nl": "latin",
    "sv": "latin",
    "pl": "latin",
    "tr": "latin",
    "id": "latin",
    "vi": "latin",
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    "zh": "cjk",
    "ja": "cjk",
    "ko": "hangul",
    "ar": "arabic",
    "fa": "arabic",
    "he": "hebrew",
    "hi": "devanagari",
    "el": "greek",
}

# Rows are truncated before hashing/scanning. A single row can be a whole
# source file, and the statistics below don't get more accurate for reading all
# of it — but they do get quadratically slower.
MAX_SCANNED_CHARS = 4000
# How much of a row's longest text field defines its "template prefix". Long
# enough that two genuinely different examples rarely collide, short enough that
# a shared generated preamble does.
TEMPLATE_PREFIX_CHARS = 48
# Mean value length below which a column is treated as categorical — a label, a
# language tag, a source name — and excluded from the repetition measure. A
# constant categorical column is normal and informative; a constant *content*
# column is templated filler. Without this split, `lang="python"` on every row
# of a perfectly good 156k-example dataset scores it as maximally repetitive,
# which is what m-a-p/CodeFeedback-Filtered-Instruction did against the live Hub.
MIN_CONTENT_CHARS = 20


@dataclass(frozen=True)
class InspectionReport:
    """Deterministic statistics over a sampled slice of one dataset."""

    rows_sampled: int = 0
    num_rows_total: int = 0
    duplicate_rate: float = 0.0
    empty_rate: float = 0.0
    malformed_rate: float = 0.0
    repetition_score: float = 0.0
    size_adequacy: float = 0.0
    contamination_hits: int = 0
    pii_hits: int = 0
    unexpected_language: bool = False
    dominant_script: str = ""
    column_fill: dict[str, float] = field(default_factory=dict)
    script_mix: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_sampled": self.rows_sampled,
            "num_rows_total": self.num_rows_total,
            "duplicates_estimate": round(self.duplicate_rate, 4),
            "invalid_rows": round(self.malformed_rate, 4),
            "empty_rows": round(self.empty_rate, 4),
            "repetition_score": round(self.repetition_score, 4),
            "size_adequacy": round(self.size_adequacy, 4),
            "contamination_hits": self.contamination_hits,
            "pii_hits": self.pii_hits,
            "unexpected_language": self.unexpected_language,
            "dominant_script": self.dominant_script,
            "column_fill": {name: round(value, 4) for name, value in self.column_fill.items()},
            "script_mix": {name: round(value, 4) for name, value in self.script_mix.items()},
            "notes": list(self.notes),
        }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _row_text(row: dict) -> str:
    """Every scalar in the row flattened into one lowercased string, capped.
    Containers are JSON-dumped so a nested list of strings still gets scanned."""
    parts: list[str] = []
    for value in row.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(str(value))
        elif isinstance(value, (list, dict, tuple)):
            try:
                parts.append(json.dumps(value, default=str))
            except (TypeError, ValueError):
                parts.append(str(value))
    return " ".join(parts)[:MAX_SCANNED_CHARS].lower()


def _row_fingerprint(row: dict) -> str:
    """A stable key for exact-duplicate detection. Whitespace is collapsed so
    two records differing only in trailing newlines count as the duplicates they
    are."""
    try:
        normalized = {
            key: " ".join(str(value).split()) if isinstance(value, str) else value
            for key, value in sorted(row.items())
        }
        return json.dumps(normalized, sort_keys=True, default=str)[:MAX_SCANNED_CHARS]
    except (TypeError, ValueError):
        return str(sorted(row.items()))[:MAX_SCANNED_CHARS]


def _longest_text(row: dict) -> str:
    texts = [value for value in row.values() if isinstance(value, str)]
    return max(texts, key=len) if texts else ""


def _script_mix(sample: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for char in sample:
        if not char.isalpha():
            continue
        point = ord(char)
        for name, low, high in _SCRIPT_RANGES:
            if low <= point <= high:
                counts[name] += 1
                total += 1
                break
        else:
            counts["other"] += 1
            total += 1
    if not total:
        return {}
    return {name: count / total for name, count in counts.most_common()}


def _expected_scripts(languages: tuple[str, ...] | list[str]) -> set[str]:
    """Scripts implied by a spec's languages. Empty means "no expectation" —
    which is the right answer for a programming-language or multilingual spec,
    and is why `unexpected_language` stays False rather than defaulting to
    suspicion."""
    expected = {_LANGUAGE_SCRIPTS[code] for code in languages if code in _LANGUAGE_SCRIPTS}
    return expected


def _repetition_score(rows: list[dict], columns: list[str]) -> float:
    """How templated the sample looks, 0 (varied) to 1 (all identical).

    Two cheap signals, maxed: the largest share any single value holds in any
    one *content* column, and the largest share of rows whose longest text field
    opens with the same 48 characters. The second catches generated filler that
    varies only in its tail, which the first misses entirely.

    Categorical columns are excluded from the first signal — see
    MIN_CONTENT_CHARS. "Every row says python" is a fact about the schema;
    "every row opens with the same paragraph" is the defect being looked for.
    """
    if not rows:
        return 0.0
    best = 0.0

    for column in columns:
        values = [
            " ".join(str(row[column]).split())
            for row in rows
            if column in row and not _is_empty(row.get(column))
        ]
        # A column with two filled cells always has a 50%+ mode; that is noise,
        # not repetition, so short columns are skipped rather than scored.
        if len(values) < 5:
            continue
        if sum(len(value) for value in values) / len(values) < MIN_CONTENT_CHARS:
            continue  # categorical, not content — see MIN_CONTENT_CHARS
        mode_count = Counter(values).most_common(1)[0][1]
        best = max(best, mode_count / len(values))

    prefixes = [
        " ".join(_longest_text(row).split())[:TEMPLATE_PREFIX_CHARS]
        for row in rows
        if _longest_text(row).strip()
    ]
    if len(prefixes) >= 5:
        prefix_count = Counter(prefixes).most_common(1)[0][1]
        best = max(best, prefix_count / len(prefixes))

    return best


def inspect_rows(
    rows: list[dict],
    columns: list[dict] | list[str],
    languages: tuple[str, ...] | list[str] = (),
    desired_examples: int = 0,
    num_rows_total: int = 0,
) -> InspectionReport:
    """Measure a sampled slice of a dataset. Never raises.

    `columns` accepts either the viewer's `[{"name", "type"}]` shape or a plain
    list of names. `languages` and `desired_examples` come from the DatasetSpec;
    both are optional, and omitting them just drops the two checks that need
    them (language expectation, size adequacy) rather than failing.
    """
    names: list[str] = []
    for column in columns:
        if isinstance(column, dict) and column.get("name"):
            names.append(str(column["name"]))
        elif isinstance(column, str) and column:
            names.append(column)

    clean_rows = [row for row in rows if isinstance(row, dict)]
    sampled = len(clean_rows)
    notes: list[str] = []

    if not sampled:
        return InspectionReport(
            num_rows_total=num_rows_total,
            notes=["no rows could be sampled from this dataset"],
        )

    fingerprints: list[str] = []
    empty_rows = 0
    malformed_rows = 0
    contamination_hits = 0
    pii_hits = 0
    filled: Counter[str] = Counter()
    script_sample: list[str] = []

    for row in clean_rows:
        fingerprints.append(_row_fingerprint(row))

        values = list(row.values())
        if not values or all(_is_empty(value) for value in values):
            empty_rows += 1
        # Malformed means the record doesn't have the shape the dataset itself
        # declares — a column the schema promises is simply absent from this
        # row. A *present but null* column is emptiness, counted above; the two
        # are different problems and are scored separately.
        if names and any(name not in row for name in names):
            malformed_rows += 1

        for name in names:
            if name in row and not _is_empty(row[name]):
                filled[name] += 1

        text = _row_text(row)
        script_sample.append(text)
        if any(marker in text for marker in CONTAMINATION_MARKERS):
            contamination_hits += 1
        if any(pattern.search(text) for pattern in _PII_PATTERNS):
            pii_hits += 1

    duplicate_rate = 1.0 - (len(set(fingerprints)) / sampled)
    empty_rate = empty_rows / sampled
    malformed_rate = malformed_rows / sampled
    repetition = _repetition_score(clean_rows, names)

    mix = _script_mix(" ".join(script_sample)[: MAX_SCANNED_CHARS * 4])
    dominant = next(iter(mix), "")
    expected = _expected_scripts(languages)
    unexpected = bool(expected) and bool(dominant) and dominant not in expected

    total = num_rows_total or sampled
    adequacy = 1.0 if desired_examples <= 0 else min(1.0, total / desired_examples)

    if duplicate_rate >= 0.2:
        notes.append(f"{duplicate_rate:.0%} of sampled rows are exact duplicates")
    if empty_rate >= 0.1:
        notes.append(f"{empty_rate:.0%} of sampled rows are entirely empty")
    if malformed_rate >= 0.1:
        notes.append(f"{malformed_rate:.0%} of sampled rows are missing a declared column")
    if repetition >= 0.5:
        notes.append(f"sampled rows are highly repetitive (score {repetition:.2f})")
    if unexpected:
        notes.append(f"sampled text is predominantly {dominant}, which no requested language uses")
    if contamination_hits:
        notes.append(f"{contamination_hits} sampled rows mention an evaluation benchmark by name")
    if pii_hits:
        notes.append(f"{pii_hits} sampled rows contain something shaped like personal data")
    if desired_examples > 0 and adequacy < 1.0:
        notes.append(f"{total} rows available against {desired_examples} requested")
    for name in names:
        if filled[name] / sampled < 0.5:
            notes.append(f"column {name!r} is empty in {1 - filled[name] / sampled:.0%} of rows")

    return InspectionReport(
        rows_sampled=sampled,
        num_rows_total=total,
        duplicate_rate=duplicate_rate,
        empty_rate=empty_rate,
        malformed_rate=malformed_rate,
        repetition_score=repetition,
        size_adequacy=adequacy,
        contamination_hits=contamination_hits,
        pii_hits=pii_hits,
        unexpected_language=unexpected,
        dominant_script=dominant,
        column_fill={name: filled[name] / sampled for name in names},
        script_mix=mix,
        notes=notes,
    )
