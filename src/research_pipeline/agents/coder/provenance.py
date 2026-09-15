"""Decide what data an experiment is actually allowed to use, and say so.

The failure this prevents is quiet and worse than a crash. A generated
experiment that invents its inputs still produces posterior means, credible
intervals and a `meets_success_criteria` flag, and every one of those numbers
looks exactly like a result. The Writer then reads that flag: `writer_agent`
maps `false` to **"refuted"**, so a run on synthesized data can currently be
written up as a refutation of a hypothesis that was never tested.

So each input the plan names resolves to one declared kind, and the verdict is
computed here in Python rather than asked of a model:

    real_local          a file staged on disk (e.g. CMS extracts under a DUA)
    real_download       fetchable from a named open source
    synthetic_surrogate generated, with the reason the real source was unusable

If any input is a surrogate, `meets_success_criteria` becomes the string
`"unknown"`, which `writer_agent` already maps to "inconclusive". The experiment
still runs and its metrics are still reported — a working, reviewable pipeline
on surrogate data is a legitimate deliverable — it just does not get to claim
support or refutation it has no evidence for.

This complements the Hugging Face dataset search rather than replacing it: that
finds a real dataset when one exists, and this records what happened when one
does not. Reads no settings and calls no model, same rule as sandbox.py.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

KIND_REAL_LOCAL = "real_local"
KIND_REAL_DOWNLOAD = "real_download"
KIND_SURROGATE = "synthetic_surrogate"
REAL_KINDS = {KIND_REAL_LOCAL, KIND_REAL_DOWNLOAD}

VERDICT_EVIDENCE = "real data — findings are interpretable as evidence for the hypothesis"
VERDICT_SURROGATE = (
    "synthetic surrogate data — the pipeline is exercised but the findings are NOT "
    "interpretable as evidence for or against the hypothesis"
)
VERDICT_MIXED = (
    "mixed real and synthetic inputs — findings are NOT interpretable as evidence; the "
    "synthetic inputs are listed in data_provenance.json"
)
VERDICT_UNCONFIRMED = (
    "real data, but from a source nobody named — it was found by keyword search and has not "
    "been confirmed to answer this question, so the findings are NOT interpretable as evidence "
    "until a human checks the discovered sources listed in data_provenance.json"
)

# Sources that are real but categorically not openly downloadable. Naming them
# is the difference between "I could not obtain CMS claims, here is why" and an
# experiment quietly making some up.
RESTRICTED_SOURCES: list[tuple[str, str]] = [
    (
        r"\bCMS\b|medicare|medicaid|hospital claims",
        "CMS claims require a Data Use Agreement and are not openly downloadable",
    ),
    (r"\bUK ?Biobank\b", "UK Biobank requires an approved application"),
    (r"\bMIMIC\b", "MIMIC requires PhysioNet credentialing and a signed DUA"),
    (
        r"\bNHS\b digital|hospital episode statistics|\bHES\b",
        "NHS HES data requires a Data Access Request",
    ),
    (r"\bSEER\b", "SEER research data requires a signed data-use agreement"),
    (
        r"electronic health record|\bEHR\b|patient[- ]level",
        "patient-level records require ethics approval and a data agreement",
    ),
]

# Public data behind free registration. Separate from the open sources below
# because "public" and "fetchable right now" are different things: without the
# key every request is a 401, which no amount of regenerating the code can fix.
CREDENTIALED_SOURCES: list[tuple[str, str, str, tuple[str, ...], str]] = [
    (
        r"\bEPA\b|air quality system|\bAQS\b",
        "EPA AQS",
        "https://aqs.epa.gov/data/api",
        ("AQS_EMAIL", "AQS_KEY"),
        "register free at https://aqs.epa.gov/data/api/signup, then export AQS_EMAIL and AQS_KEY",
    ),
    (
        r"\bNOAA\b|climate data online",
        "NOAA CDO",
        "https://www.ncei.noaa.gov/cdo-web/api/v2",
        ("NOAA_CDO_TOKEN",),
        "request a token at https://www.ncdc.noaa.gov/cdo-web/token, then export NOAA_CDO_TOKEN",
    ),
]

OPEN_SOURCES: list[tuple[str, str, str]] = [
    (r"american community survey|\bACS\b|census", "US Census ACS", "https://api.census.gov/data"),
    (r"world bank", "World Bank", "https://api.worldbank.org/v2"),
    (r"\bWHO\b|global health observatory", "WHO GHO", "https://ghoapi.azureedge.net/api"),
    (r"open ?street ?map|\bOSM\b", "OpenStreetMap", "https://overpass-api.de/api"),
    (r"\bNDVI\b|landsat|\bUSGS\b", "USGS EarthExplorer", "https://earthexplorer.usgs.gov"),
    (r"hugging ?face", "Hugging Face datasets", "https://datasets-server.huggingface.co/rows"),
    # Deliberately no equity/stock-price entry. Stooq, Yahoo Finance's chart
    # endpoint and the like look keyless but are not reachable from a compute
    # node: Stooq answers a datacentre IP with an HTTP 200 carrying a JavaScript
    # proof-of-work page rather than CSV (which pandas then parses as garbage
    # instead of failing loudly), and Yahoo returns 429. Barkla job 10411308
    # spent every fix attempt on the resulting 404s and then synthesized anyway.
    # An entry naming a source this network cannot fetch is worse than no entry:
    # it turns "no real source" into a promise the generated code cannot keep.
    # Real market data reaches an experiment here through the Hugging Face
    # dataset search instead.
    (
        r"coin ?gecko|crypto(currency)? price|bitcoin|ethereum",
        "CoinGecko",
        "https://api.coingecko.com/api/v3",
    ),
    (
        r"open-?meteo|weather (data|history)|temperature record",
        "Open-Meteo",
        "https://archive-api.open-meteo.com/v1/archive",
    ),
    (r"open ?alex|scholarly (metadata|citation)", "OpenAlex", "https://api.openalex.org"),
    (
        r"eurostat",
        "Eurostat",
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
    ),
    (r"wikipedia pageviews|wikimedia", "Wikimedia REST", "https://wikimedia.org/api/rest_v1"),
]


@dataclass
class DataSource:
    """One resolved input, and the honest story of where it came from."""

    name: str
    kind: str
    uri: str = ""
    local_path: str = ""
    reason: str = ""
    credentials: list[str] = field(default_factory=list)
    # True only for the last-resort branch of `resolve`: nothing was staged and
    # no source could be *identified* for this requirement. That is a different
    # statement from a restricted or credentialed source, which names real data
    # that specifically was not obtained — those must keep withholding the
    # verdict, and this one can be answered by a dataset found elsewhere. See
    # `supersede_unresolved`.
    unresolved: bool = False
    # Set for an input whose use by the generated code has already been
    # established by a stronger check than `verify_downloads_used` can make —
    # the Hugging Face dataset, confirmed by the code naming its id. Checking
    # such a source again by URL host would downgrade code that reads it from a
    # downloaded local copy, or from a URL built some way other than the one
    # handed over in the prompt.
    usage_verified: bool = False
    # Set by `acquire.apply` when this pipeline fetched the input itself: the
    # url it came from, a sha256 of the bytes, the format written, and the real
    # columns and first rows read off it. Empty for a staged file (nothing was
    # fetched) and for an input the generated code is still expected to fetch.
    # Carried on the DataSource rather than as five more fields because it is
    # one indivisible fact — "these exact bytes, from there, at that time" — and
    # because it lands verbatim in data_provenance.json, which is the record
    # someone re-running this experiment reads.
    acquired: dict[str, Any] = field(default_factory=dict)
    # Set by `discover.apply` when no source was *named* for this requirement
    # and one was searched for: the connector, the query run, the catalogue
    # record chosen and its landing page. Kept distinct from `acquired` because
    # the two answer different questions — `acquired` says these bytes are real
    # and here, `discovered` says nobody asked for this particular dataset by
    # name, so whether it answers the question is a judgment a human still has
    # to make. See discover.py's module docstring on the relevance gate.
    discovered: dict[str, Any] = field(default_factory=dict)
    # Set by `resolve` when the requirement asks for data to be *generated*.
    # Distinct from `unresolved`, which this keeps alongside it, because that
    # flag answers two questions at once and only one of them changes here:
    # "may a search look for this?" (no — see is_synthesis_request) and "may a
    # real input that turned up under another name supersede it?" (yes, and it
    # must — a plan naming synthetic data whose code demonstrably reads a real
    # dataset instead has a phantom requirement, exactly the case
    # supersede_unresolved exists for).
    synthesis_request: bool = False
    # Set by `coder_agent` for a staged file: its columns, first and last rows
    # and any trailing placeholder columns, read off the bytes by
    # `acquire.describe_local`. Empty for everything else.
    preview: dict[str, Any] = field(default_factory=dict)

    @property
    def is_real(self) -> bool:
        return self.kind in REAL_KINDS

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "uri": self.uri,
            "local_path": self.local_path,
            "reason": self.reason,
            "credentials": self.credentials,
        }
        # Additive: an input nobody fetched produces exactly the document this
        # wrote before acquisition existed.
        if self.acquired:
            document["acquired"] = self.acquired
        if self.discovered:
            document["discovered"] = self.discovered
        if self.preview:
            document["preview"] = self.preview
        return document


def split_requirements(source: str, description: str = "") -> list[str]:
    """Break the planner's `source` field into the distinct inputs it names.

    One string routinely covers several datasets ("EPA AQS for PM2.5; CMS claims
    for admissions; Census ACS for deprivation"), and each has its own
    availability story.

    `description` is deliberately not split alongside it — that field describes
    the *derived* dataset the inputs are merged into, and splitting it invents a
    phantom input nothing can resolve. It is used only when `source` is empty.
    """
    text = (source or description or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;\n]|(?<=[a-z])\s+and\s+(?=[A-Z])", text) if p.strip()]
    return parts or [text]


def _match(patterns, text: str):
    for entry in patterns:
        if re.search(entry[0], text, re.IGNORECASE):
            return entry
    return None


# ---------------------------------------------------------------------------
# Reading a requirement: its content words, the datasets it names, and the
# staged file it answers to. Content words live here rather than in discover.py
# — which imports this module — so the staged-file matcher and the catalogue
# relevance gate cannot drift into two definitions of "a word worth matching".
# ---------------------------------------------------------------------------

# Dropped when turning a prose requirement into a catalogue query, and when
# measuring overlap. Same idea as huggingface_client's stop list.
_STOPWORDS = frozenset(
    """
    a an the and or of for from with without to in on at by per over under between
    is are was were be been has have had its it this that these those not
    data dataset datasets database records record source sources file files
    about into using use used via across during within all any new one two
    real public open historical recent daily monthly yearly annual
    """.split()
)


def _is_content_word(token: str) -> bool:
    """Whether `token` is worth searching or matching on.

    The length floor is 3, not 4, and there are two exceptions above it —
    because a data requirement's most discriminating terms are routinely short.
    A plain `len > 3` rule drops `pm2` (from PM2.5), `co2`, `no2`, `EEG`, `GDP`,
    `CMS`, i.e. exactly the words that distinguish "PM2.5 concentrations" from
    every other environmental dataset in a catalogue.

    - a token mixing letters and digits is a measure or a code (`pm2`, `co2`,
      `covid19`), never noise;
    - an all-caps token is an acronym (`EEG`, `GDP`), so case is read from the
      original text rather than after lowercasing;
    - a pure number is dropped, which is what keeps the `5` of "PM2.5" out.
    """
    if token.lower() in _STOPWORDS:
        return False
    if not any(character.isalpha() for character in token):
        return False
    if len(token) >= 3:
        return True
    return token.isupper()


def keywords(text: str) -> set[str]:
    """Content words of `text`, lowercased, stopwords and noise dropped."""
    tokens = re.split(r"[^A-Za-z0-9]+", text or "")
    return {token.lower() for token in tokens if _is_content_word(token)}


# What introduces an example in a planner's data requirement. The parenthesised
# "(e.g., X or Y)" is the dominant shape: 17 of batch 10460809's 36
# requirements used it, and most of the rest named their dataset after a colon.
_EXAMPLE_MARKER = re.compile(
    r"(?:\(\s*|\b)(?:e\.\s?g\.?|i\.\s?e\.?|such as|for example|for instance)\s*[,:]?\s*"
    r"(?P<span>[^()]*)\)?",
    re.IGNORECASE,
)
# An alternative that opens with one of these is a hedge, not a name: "or
# similar lending dataset", "other public repositories".
_FILLER_LEADS = frozenset({"any", "comparable", "equivalent", "etc", "other", "others", "similar"})
# More named alternatives than this and a search is being asked to guess.
MAX_NAMED_ALTERNATIVES = 3


def _alternatives_in(span: str) -> list[str]:
    """The alternatives one span lists, hedges dropped. A URL is never split."""
    parts = [span] if "://" in span else re.split(r",|;|\s+or\s+|\s*/\s*", span)
    found: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" .")
        if not cleaned or cleaned.split()[0].lower() in _FILLER_LEADS or not keywords(cleaned):
            continue
        found.append(cleaned)
    return found


def named_alternatives(requirement: str) -> list[str]:
    """The specific datasets a requirement names, in the order worth trying them.

    The planner usually knows which dataset it means, and says so in a clause
    nothing was reading: "public dataset (e.g., UCI Adult dataset or similar
    tabular dataset)", "public dataset: UCI Electricity Load Diagrams". Every
    matcher downstream was handed the whole string, and the words most of them
    could act on were the generic ones at the front.

    Order: a name after a colon, then examples, then a "from" clause — the first
    two name *datasets*, the last usually names a *repository* ("from UCI Machine
    Learning Repository"). Returns [] when nothing is named, and never the
    requirement itself, which every caller tries last anyway.
    """
    text = requirement or ""
    found: list[str] = []
    colon = re.match(r"^(?P<head>[^:()]+):\s*(?P<span>.+)$", text)
    if colon and colon.group("span").startswith("//"):
        colon = None
    if colon:
        span = _EXAMPLE_MARKER.sub("", colon.group("span"))
        found += _alternatives_in(re.sub(r"\(([^)]*)\)", r" \1", span))
    for example in _EXAMPLE_MARKER.finditer(text):
        found += _alternatives_in(example.group("span"))
    source = re.search(r"\bfrom\s+(?P<span>[^()]+)", text)
    if source and not colon:
        found += _alternatives_in(source.group("span"))
    if not found and " / " in text:
        found += _alternatives_in(text)

    ordered: list[str] = []
    seen = {text.strip().lower()}
    for alternative in found:
        if alternative.lower() not in seen:
            seen.add(alternative.lower())
            ordered.append(alternative)
    return ordered


# Words a staged *filename* attracts because it describes data rather than
# naming it. Someone staging a file writes a description, and on 5 September a
# set of keyword-packed aliases was staged so vague requirements would find 20
# Newsgroups. One of them —
# dataset_benchmark_standard_collection_annotated_labeled_..._newsgroups_train.csv
# — then answered every requirement that said "benchmark" or "standard", and
# "Public benchmark datasets" published a *supported* verdict off 20 Newsgroups.
_STAGING_STOPWORDS = _STOPWORDS | frozenset(
    {
        "available",
        "benchmark",
        "benchmarks",
        "collection",
        "collections",
        "instance",
        "instances",
        "name",
        "names",
        "publicly",
        "repository",
        "sample",
        "samples",
        "similar",
        "standard",
    }
)
# Never enough to make a match on their own — "daily" is in half the staged
# filenames — but the only thing that tells the daily and monthly versions of one
# series apart, so they break ties.
_FREQUENCY_WORDS = frozenset(
    {"annual", "daily", "hourly", "monthly", "quarterly", "weekly", "yearly"}
)
# Content words a phrase must share with a filename, or all of them when it has
# fewer. See _staged_file for how this number was chosen.
STAGED_MATCH_MIN_WORDS = 2


def _singular(word: str) -> str:
    """Fold a plural — "documents" to "document" — so the two sides need not agree.

    Crude on purpose, and harmless for it: applied to both sides alike, a word it
    mangles ("categories" to "categorie") is mangled identically on each.
    """
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _staging_words(text: str) -> set[str]:
    """Content words for matching a phrase against a staged filename.

    `keywords` with three differences, each measured against the cluster's own
    staging directory. Two-letter tokens count on both sides: `keywords` keeps
    "AG" only in upper case, which a filename never is, so "AG News" could not
    find ag_news_*.csv. Plurals are folded. And `_STAGING_STOPWORDS` drops the
    descriptive vocabulary a staged filename attracts.
    """
    words: set[str] = set()
    for token in re.split(r"[^A-Za-z0-9]+", text or ""):
        lowered = token.lower()
        if len(lowered) < 2 or lowered in _STAGING_STOPWORDS:
            continue
        if not any(character.isalpha() for character in lowered):
            continue
        words.add(_singular(lowered))
    return words


def _frequency_words(text: str) -> set[str]:
    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", text or "")} & _FREQUENCY_WORDS


def _staged_file(staging_dir: Path | None, requirement: str) -> tuple[Path, str] | None:
    """A file the user staged for this requirement, and the phrase that matched it.

    Not an exact name match — somebody staging data names the file after the
    data, not after the planner's phrasing of it — but not one shared word
    either, which is what this was until it resolved 28 of batch 10460809's 36
    requirements to a staged file, most of them to one alias of 20 Newsgroups.
    Across every requirement that has ever hit a staged file (38 distinct), that
    rule matched 25 that could not answer their question, and 19 experiments
    published a verdict off one of them: nine "supported", ten "refuted". "public
    EHR dataset (e.g., MIMIC-III or eICU)" was among them, reaching 20 Newsgroups
    before the restricted-source rule was ever consulted — and a staged input is
    `real_local` and named, so no confirmation gate could catch it afterwards.

    So the datasets the requirement names are tried before the requirement
    itself, and a file must share `STAGED_MATCH_MIN_WORDS` content words with the
    phrase (all of them, when it has fewer). Measured against the same 38: no
    implausible match survives, and the four correct ones do — the S&P 500 series
    for "S&P 500 daily closing prices" (twice), AG News for a financial-news
    requirement, 20 Newsgroups for "20 Newsgroups". `discover.is_relevant`'s
    majority rule was measured too and rejected: it loses "CMS Medicare claims
    for admissions" against a file named medicare_claims_2020.

    Ties go to the file whose frequency ("daily", "monthly") agrees, then to the
    real file over an alias symlinked to it, then to the shorter — more specific
    — filename, then to a training split.
    """
    if not staging_dir or not staging_dir.is_dir():
        return None
    files = [
        path
        for path in sorted(staging_dir.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and not path.name.lower().startswith("readme")
    ]
    for phrase in [*named_alternatives(requirement)[:MAX_NAMED_ALTERNATIVES], requirement]:
        wanted = _staging_words(phrase)
        if not wanted:
            continue
        needed = min(STAGED_MATCH_MIN_WORDS, len(wanted))
        frequency = _frequency_words(phrase)
        best: tuple[tuple[int, int, bool, int, bool], Path] | None = None
        for path in files:
            offered = _staging_words(path.stem)
            overlap = len(wanted & offered)
            if overlap < needed:
                continue
            rank = (
                overlap,
                len(frequency & _frequency_words(path.stem)),
                not path.is_symlink(),
                -len(offered),
                "train" in path.stem.lower(),
            )
            if best is None or rank > best[0]:
                best = (rank, path)
        if best is not None:
            return best[1], phrase
    return None


# The vocabulary a synthesis instruction is built from. Split into three sets
# rather than one alternation because the rule is "the requirement says nothing
# but this", not "the requirement contains one of these".
_SYNTHESIS_WORDS = {
    "artificial",
    "dummy",
    "fake",
    "generate",
    "generated",
    "generating",
    # Nouns for the act rather than the product. Not a guess: batch 10460809's
    # planner wrote "synthetic generation" as the entire data requirement in
    # four of its 36 plans, and the first version of this list — built from
    # phrasings invented for its own unit tests — missed every one. Barkla job
    # 10492707's planner wrote the same phrase and both searches went looking,
    # returning a code-generation corpus for a pension-liability experiment.
    "generation",
    "generator",
    "mock",
    "random",
    "randomly",
    "simulate",
    "simulated",
    "simulation",
    "simulations",
    "synthesis",
    "synthesised",
    "synthesized",
    "synthetic",
    "toy",
}
# Nouns that name data in general rather than any particular data.
_GENERIC_DATA_WORDS = {
    "data",
    "dataset",
    "datasets",
    "example",
    "examples",
    "input",
    "inputs",
    "matrices",
    "matrix",
    "observation",
    "observations",
    "record",
    "records",
    "row",
    "rows",
    "sample",
    "samples",
    "series",
    "set",
    "sets",
    "value",
    "values",
}
# Shape adjectives that still say nothing about provenance.
_SHAPE_WORDS = {
    "categorical",
    "numeric",
    "numerical",
    "simple",
    "small",
    "tabular",
    "test",
}
_FILLER_WORDS = {"a", "an", "the", "of", "with", "and", "some"}


def is_synthesis_request(requirement: str) -> bool:
    """Whether this requirement asks for data to be generated rather than found.

    Measured, not guessed: across benchmark runs 10460710/10460812/10460813 the
    planner wrote the bare word "synthetic" as a data requirement, and because
    no table matched it the requirement became `unresolved` — this package's
    signal to go looking. Both searches then obliged. The Hub matched
    `gretelai/synthetic_text_to_sql` on the word itself and handed a text-to-SQL
    corpus to tabular classification, timeseries forecasting, high-complexity
    CPU and bootstrap resampling in the same run.

    Searching for a synthesis instruction is a category error rather than a bad
    match: no dataset anywhere "is" the synthetic data a plan intends to
    generate, so every hit is wrong by construction and an honest miss is the
    best available outcome.

    The test is deliberately strict — every word must be synthesis vocabulary, a
    generic data noun, a shape adjective or filler — so it recognises "synthetic
    data" and declines to judge "simulated portfolio returns" or "synthetic
    control estimates for German reunification". Those name, or may name, real
    data. The asymmetry is on purpose: a requirement wrongly matched here is a
    real dataset nobody ever searches for, while one wrongly missed is searched,
    and since the Hub match is now `discovered` its verdict is withheld anyway.
    Over-matching costs data; under-matching costs a search.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", requirement.lower()) if w]
    if not words:
        return False
    meaningful = [w for w in words if w not in _FILLER_WORDS]
    if not any(w in _SYNTHESIS_WORDS for w in meaningful):
        return False
    return all(
        w in _SYNTHESIS_WORDS or w in _GENERIC_DATA_WORDS or w in _SHAPE_WORDS for w in meaningful
    )


def resolve(
    requirements: list[str],
    *,
    staging_dir: Path | None = None,
    network_available: bool = False,
) -> list[DataSource]:
    """Decide, per requirement, what data the experiment will actually use.

    Order is deliberate: a staged file beats anything inferred, a restricted
    source is never guessed at, a credentialed one is real only with its key,
    and a surrogate is the last resort and always labelled as one.
    """
    resolved: list[DataSource] = []
    for requirement in requirements:
        staged = _staged_file(staging_dir, requirement)
        if staged:
            path, matched_on = staged
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_REAL_LOCAL,
                    local_path=str(path),
                    reason=f"staged locally at {path}"
                    + (
                        f", matched on the dataset the plan named: {matched_on!r}"
                        if matched_on != requirement
                        else ""
                    ),
                )
            )
            continue

        if is_synthesis_request(requirement):
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_SURROGATE,
                    reason=(
                        "the plan asked for generated data, so this input is synthesized "
                        "by design rather than for want of a source"
                    ),
                    unresolved=True,
                    synthesis_request=True,
                )
            )
            continue

        credentialed = _match(CREDENTIALED_SOURCES, requirement)
        if credentialed:
            _, label, uri, variables, howto = credentialed
            missing = [v for v in variables if not os.environ.get(v)]
            if not missing:
                resolved.append(
                    DataSource(
                        name=requirement,
                        kind=KIND_REAL_DOWNLOAD,
                        uri=uri,
                        reason=f"{label}, credentials found in {', '.join(variables)}",
                        credentials=list(variables),
                    )
                )
            else:
                resolved.append(
                    DataSource(
                        name=requirement,
                        kind=KIND_SURROGATE,
                        uri=uri,
                        reason=(
                            f"{label} requires an API key and {', '.join(missing)} "
                            f"{'is' if len(missing) == 1 else 'are'} not set ({howto}); "
                            "a documented surrogate is generated instead"
                        ),
                    )
                )
            continue

        restricted = _match(RESTRICTED_SOURCES, requirement)
        if restricted:
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_SURROGATE,
                    reason=(
                        f"{restricted[1]}. Nothing was staged for it, so a documented "
                        "surrogate is generated instead."
                    ),
                )
            )
            continue

        open_source = _match(OPEN_SOURCES, requirement)
        if open_source and network_available:
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_REAL_DOWNLOAD,
                    uri=open_source[2],
                    reason=f"open source {open_source[1]}, reachable from this node",
                )
            )
            continue

        resolved.append(
            DataSource(
                name=requirement,
                kind=KIND_SURROGATE,
                uri=open_source[2] if open_source else "",
                reason=(
                    f"open source {open_source[1]} identified, but this node has no outbound "
                    "network; a documented surrogate is generated instead"
                    if open_source
                    else "no open source identified for this input and nothing staged locally; "
                    "a documented surrogate is generated instead"
                ),
                unresolved=True,
            )
        )
    return resolved


def verify_downloads_used(sources: list[DataSource], code: str) -> list[DataSource]:
    """Downgrade a `real_download` the generated code never actually fetches.

    `resolve` can only say a source *is* openly fetchable; whether the code went
    and fetched it is a different question, and until it is asked a plan whose
    model quietly synthesized instead still earns the "real data — findings are
    interpretable as evidence" stamp. That is the over-claiming direction, the
    one this module exists to prevent, and every entry added to OPEN_SOURCES
    widens the exposure to it.

    The host of the declared URI is the fixed trace — the same reasoning as
    `sandbox.check_hf_dataset_usage` matching on the dataset id rather than
    walking the AST for a particular call shape: the model can write the request
    in more ways than are worth enumerating, but it cannot fetch the source
    without naming its host. `real_local` is left alone (a file on disk is real
    whatever the code string looks like), and so is anything already a surrogate.

    The question is asked of the experiment as a whole, not of each requirement
    separately: if *any* declared real input is demonstrably obtained, nothing is
    downgraded. One requirement is routinely satisfied by another entry's data —
    a matched Hub dataset answers the "Hugging Face" the plan asked for, and the
    generated code may then read it in a form that never names the REST host —
    and per-requirement matching turns that into a phantom surrogate. What stays
    caught is the failure this guards: an experiment that declared real inputs,
    fetched none of them, and synthesized instead.
    """
    if not code:
        return sources

    def obtained(source: DataSource) -> bool:
        if source.kind == KIND_REAL_LOCAL or source.usage_verified:
            return True
        host = urlparse(source.uri).netloc
        return bool(host) and host in code

    if any(obtained(s) for s in sources if s.is_real):
        return sources

    verified: list[DataSource] = []
    for source in sources:
        host = urlparse(source.uri).netloc
        if source.kind != KIND_REAL_DOWNLOAD or not host or source.usage_verified:
            verified.append(source)
            continue
        verified.append(
            DataSource(
                name=source.name,
                kind=KIND_SURROGATE,
                uri=source.uri,
                reason=(
                    f"{source.reason}, but the generated code never fetches {host} — it "
                    "appears to have synthesized this input instead, so the verdict is "
                    "withheld rather than credited to data that was never read"
                ),
                credentials=source.credentials,
            )
        )
    return verified


def supersede_unresolved(sources: list[DataSource], code: str) -> list[DataSource]:
    """Drop requirements that a real input the code actually reads already answers.

    `resolve` produces one entry per requirement *string the planner wrote*, and
    a dataset found by the Hugging Face search is added as its own entry — so a
    plan naming "Yahoo Finance or public stock market data" that ends up reading
    50,000 rows of real daily prices from the Hub scores one real input and one
    surrogate, and is reported as mixed. Nothing synthetic went into that
    experiment; the surrogate is a phantom, describing a requirement the dataset
    met under a different name.

    Two conditions, both deterministic, both required:

    - some real input is demonstrably read by the code (`verify_downloads_used`
      has already downgraded any that are not), and
    - the code defines no `synthesize_*` generator — the name `prompt_block`
      instructs a surrogate to use, so its absence is the trace that nothing was
      invented.

    Only `unresolved` entries are superseded. A restricted source (CMS, UK
    Biobank) or one missing its API key names real data that specifically was
    not obtained, and no amount of other data answers it — those keep withholding
    the verdict, which is the whole point of this module.
    """
    if not code or not any(s.is_real for s in sources):
        return sources
    if re.search(r"\bdef\s+synthesize_\w*", code):
        return sources
    kept = [s for s in sources if not (s.unresolved and s.kind == KIND_SURROGATE)]
    return kept if any(s.is_real for s in kept) else sources


def all_real(sources: list[DataSource]) -> bool:
    return bool(sources) and all(s.is_real for s in sources)


def _unconfirmed(source: DataSource) -> bool:
    # A URL the plan wrote into its own `source` field is not a search result:
    # `discover.search_direct` fetched exactly what the plan named, which is the
    # standing a table match has. Counting it as found-by-search withheld the
    # verdict from precisely the plans that did the right thing and named their
    # data. `verify_downloads_used` still checks the code actually read it.
    return (
        source.is_real
        and bool(source.discovered)
        and source.discovered.get("connector") != "direct"
    )


def needs_confirmation(sources: list[DataSource]) -> bool:
    """Whether any input is real but was *found* rather than named.

    A separate question from `all_real`, and the reason the two exist side by
    side. A discovered input is genuinely real data — it is on disk, it has a
    checksum, nothing was invented — so it passes every synthetic-data test in
    this module. What it has not passed is the question of whether it answers
    *this* question, and `discover.is_relevant` is a keyword floor, not an
    answer to that.

    Measured, not assumed: a live sweep of five requirements against the open
    catalogues returned one clearly correct dataset, two that were real and
    plausible and wrong (a geographic reference table for a request about crime
    counts; COVID-19 case counts for one about pupil absence), and two honest
    misses. Granting a verdict on the middle two would be worse than the
    surrogate they replaced — a refutation computed on the wrong real data
    reads as defensible in a way one computed on invented data does not, which
    inverts this module's entire purpose.

    So the metrics are still reported and the experiment still runs on real,
    messy, genuinely useful data; the verdict simply waits for a human to look
    at the landing page recorded for each discovered input and say yes. Staging
    the confirmed file under CODER_DATA_DIR is what turns it into a `real_local`
    nobody has to second-guess.
    """
    return any(_unconfirmed(s) for s in sources)


# Phrases in which the model reports having *substituted* for the data the plan
# asked for. Each was taken from an `assumptions_made` entry in a real run whose
# verdict was asserted anyway (coder benchmark 10431703/10431840, cases 03, 04
# and 11). They are grouped only for reporting; any match has the same effect.
_DECLARED_SUBSTITUTION = re.compile(
    r"\bproxy\b|\bstand[- ]in\b|\bsubstitut|\binstead of\b|\bin place of\b"
    r"|\bas a replacement\b",
    re.I,
)
# ...but only when the sentence is about the *data*. These words are generic
# English for "I did X rather than Y", and a plan's methods get simplified far
# more often than its inputs get swapped: Barkla job 10522998 withheld the
# verdict on a run that read the staged AG News CSV exactly as instructed,
# because one assumption said "keeps top 50% of candidates **instead of**
# complex theoretical bounds" — a statement about the pruning rule, not the
# corpus. That misread is now systematic rather than incidental, because
# prompts.REFERENCE_IMPLEMENTATION_NOTE asks the model to record precisely this
# kind of divergence from a published method ("a smaller model, fewer epochs, a
# simplified variant"), in precisely this phrasing.
#
# The other two patterns need no such guard: _DECLARED_SYNTHETIC_USE and
# _DECLARED_FABRICATED_LABELS name data or labels in the pattern itself.
_ABOUT_DATA = re.compile(
    r"\bdata\b|\bdataset\b|\bcorpus\b|\bcorpora\b|\bfile\b|\bcsv\b|\bjson\b"
    r"|\brows?\b|\brecords?\b|\bsamples?\b|\bdocuments?\b|\bcolumns?\b"
    r"|\blabels?\b|\binputs?\b",
    re.I,
)
_DECLARED_FABRICATED_LABELS = re.compile(
    r"keyword[- ]based (?:sentiment )?label|true labels? (?:are|aren'?t|is|isn'?t) "
    r"(?:not )?(?:provided|available)|heuristic label|pseudo[- ]label|synthetic label",
    re.I,
)
_DECLARED_SYNTHETIC_USE = re.compile(
    r"synthetic (?:data|dataset|corpus)|simulated (?:data|dataset)|generated (?:the )?data",
    re.I,
)
# What makes a statement hypothetical rather than a report of what happened.
# "falls back to synthetic data if the fetch fails" describes the guarded read
# that `sandbox.check_data_fallback` *requires*; it is not a confession.
_HYPOTHETICAL = re.compile(
    r"\bif\b|\bwhen\b|\bunless\b|\bshould\b|\bin case\b|\bfall(?:s|ing)? back\b"
    r"|\bwould\b|\bmay\b",
    re.I,
)


def declared_substitutions(assumptions: list[str]) -> list[str]:
    """The assumptions in which the model states it used something else.

    Returns only *assertions*. A conditional clause describing a fallback path
    that may never have executed is excluded, because generating one is required
    behaviour rather than a defect.

    And only assertions about the **data**. The generic substitution wording
    ("instead of", "in place of") is also how a model describes simplifying a
    *method*, which every experiment does and which the reference-implementation
    prompt now explicitly asks it to write down — so that pattern must also
    mention data to count. See _ABOUT_DATA.
    """
    found = []
    for assumption in assumptions or []:
        text = str(assumption)
        if _HYPOTHETICAL.search(text):
            continue
        if (
            (_DECLARED_SUBSTITUTION.search(text) and _ABOUT_DATA.search(text))
            or _DECLARED_FABRICATED_LABELS.search(text)
            or _DECLARED_SYNTHETIC_USE.search(text)
        ):
            found.append(text.strip())
    return found


def honour_declared_substitution(
    sources: list[DataSource], assumptions: list[str]
) -> list[DataSource]:
    """Withhold the verdict when the model says it substituted for the real data.

    `resolve` establishes that an input *could* be obtained and
    `verify_downloads_used` that the code went and fetched something from the
    right host. Neither asks whether what was fetched is the data the plan
    called for. Two observed failures sit exactly in that gap, and both asserted
    a verdict rather than withholding one:

      * a real dataset fetched from the right host but answering a different
        question — benchmark case 03 pulled `HuggingFaceFW/fineweb` "as a proxy
        for movie review sentiment corpus" and derived its labels by keyword,
        then reported the hypothesis supported;
      * a file genuinely present on disk whose *contents* are synthetic, which
        `real_local` cannot distinguish from real data.

    In every observed instance the model declared what it had done, in
    `assumptions_made`, and nothing read the declaration. This reads it, and
    treats an asserted substitution the way a surrogate is already treated: the
    metrics stand, the verdict does not.

    Every real source is downgraded rather than a guessed subset, because the
    assumption text does not say which input it refers to and withholding is the
    safe direction — the same reasoning that makes an unresolvable
    `data_requirements` block a surrogate rather than a pass.

    **This is not a check on reality.** It honours what the model reported; a
    substitution it never mentions is not caught, and cannot be by reading prose.
    It closes the declared case only, which is the case observed in practice.
    """
    declared = declared_substitutions(assumptions)
    if not declared:
        return sources
    reason = (
        "the generated code reports substituting for the data this plan requires: "
        + "; ".join(declared[:2])
    )
    downgraded = []
    for source in sources:
        if source.kind in REAL_KINDS:
            downgraded.append(
                DataSource(
                    name=source.name,
                    kind=KIND_SURROGATE,
                    uri=source.uri,
                    local_path=source.local_path,
                    reason=reason,
                    credentials=list(source.credentials),
                )
            )
        else:
            downgraded.append(source)
    return downgraded


def verdict(sources: list[DataSource]) -> str:
    """The methodological validity stamp — computed, never asked of the model."""
    if not sources:
        # No resolvable inputs is not evidence of real ones. Silence here would
        # let a plan with an unparseable data_requirements block claim support.
        return VERDICT_SURROGATE
    if all_real(sources):
        return VERDICT_UNCONFIRMED if needs_confirmation(sources) else VERDICT_EVIDENCE
    return VERDICT_MIXED if any(s.is_real for s in sources) else VERDICT_SURROGATE


def as_document(sources: list[DataSource]) -> dict[str, Any]:
    return {
        "inputs": [s.to_dict() for s in sources],
        "methodological_validity": verdict(sources),
        "all_inputs_real": all_real(sources),
        "surrogate_count": sum(1 for s in sources if s.kind == KIND_SURROGATE),
        # Separate from all_inputs_real on purpose: these inputs *are* real. What
        # is missing is a human confirming that they answer the question asked.
        "unconfirmed_discovered_inputs": [s.name for s in sources if _unconfirmed(s)],
    }


def write(sources: list[DataSource], path: Path) -> dict[str, Any]:
    document = as_document(sources)
    path.write_text(json.dumps(document, indent=2))
    return document


WITHHELD_SURROGATE = (
    "One or more inputs are synthetic surrogates, so these metrics describe the pipeline's "
    "behaviour on generated data and say nothing about the real-world hypothesis. See "
    "data_provenance.json."
)

WITHHELD_UNCONFIRMED = (
    "One or more inputs are real data found by keyword search rather than named by the plan, and "
    "nothing has confirmed they answer this question. The experiment ran on real data and its "
    "metrics are reported, but a verdict would be a claim about a dataset nobody chose. Check the "
    "landing pages in data_provenance.json; staging the confirmed file under CODER_DATA_DIR makes "
    "the verdict reachable."
)

# Kept as the old name so anything importing it still works; the surrogate case
# is what it always meant.
WITHHELD_BECAUSE = WITHHELD_SURROGATE


def apply_to_results(results: dict, sources: list[DataSource]) -> dict:
    """Withhold the hypothesis verdict when any input is synthetic, or real but
    unconfirmed.

    `meets_success_criteria` becomes the string "unknown" rather than False.
    That distinction is the whole point: `writer_agent` maps False to "refuted"
    and "unknown" to "inconclusive", so returning False here would have the
    paper claim a refutation off invented numbers. The metrics themselves are
    left untouched and still reported — they describe what the pipeline did,
    which is worth reading; they just no longer carry a verdict about the world.
    """
    return apply_document_to_results(results, as_document(sources))


def apply_document_to_results(results: dict, document: dict) -> dict:
    """The same withholding, decided from an already-computed provenance
    document rather than live DataSources.

    Exists for `reconcile.py`: a SLURM job's results arrive in a later process
    than the one that resolved its inputs, and re-resolving there would ask a
    machine that may not have the staging directory mounted whether a file the
    submitting machine could see is real. The document that run wrote is the
    answer; this reads it. `apply_to_results` above goes through here too, so
    there is one implementation of what withholding means — including the
    unconfirmed-discovery case, which the document carries as
    `unconfirmed_discovered_inputs` precisely so it survives being written to
    disk and read back in another process.

    An empty or absent document withholds. Not knowing where an experiment's
    inputs came from is not evidence that they were real — the same reason
    `verdict` treats an empty source list as surrogate.
    """
    all_inputs_real = bool(document.get("all_inputs_real"))
    unconfirmed = bool(document.get("unconfirmed_discovered_inputs"))
    if all_inputs_real and not unconfirmed:
        return results

    stamped = dict(results)
    # setdefault for the same reason compute_provenance uses it: both gates can
    # fire on one result, and whichever runs first records the model's real
    # claim before replacing it.
    stamped.setdefault(
        "model_reported_meets_success_criteria", results.get("meets_success_criteria")
    )
    stamped["meets_success_criteria"] = "unknown"
    stamped["methodological_validity"] = document.get("methodological_validity", VERDICT_SURROGATE)
    # Real-but-unconfirmed is a different problem from synthetic, and says a
    # different thing about what would make the verdict reachable — one needs a
    # human to confirm the dataset, the other needs real data to exist at all.
    # A surrogate anywhere is the more serious of the two, so it wins when both
    # are true.
    reason = WITHHELD_UNCONFIRMED if all_inputs_real else WITHHELD_SURROGATE
    existing = stamped.get("verdict_withheld_because")
    stamped["verdict_withheld_because"] = f"{existing} {reason}" if existing else reason
    return stamped


def _acquired_lines(source: DataSource) -> list[str]:
    """What the model is told about a file this pipeline fetched for it.

    The columns and first rows are the part that matters: before acquisition
    the model wrote `load_data` against a schema it had only been told about in
    prose, and guessed column names accordingly. These were read off the actual
    bytes now on disk.
    """
    acquired = source.acquired
    if not acquired:
        return []
    lines = [
        f"   Fetched by the pipeline from {acquired.get('url', '')} — "
        f"{acquired.get('row_count', 0)} rows, "
        f"sha256 {str(acquired.get('sha256', ''))[:12]}.",
        f"   Format: {acquired.get('read_hint') or acquired.get('data_format', '')}",
    ]
    columns = acquired.get("columns") or []
    if columns:
        lines.append(f"   Columns: {', '.join(str(column) for column in columns)}")
    sample = acquired.get("sample_rows") or []
    if sample:
        lines.append(f"   First rows: {json.dumps(sample, default=str)[:1200]}")
    return lines


def _staged_lines(source: DataSource) -> list[str]:
    """What the model is told about a staged file: read off its bytes, first and
    last rows both, and — stated as fact — any columns ending in placeholders."""
    preview = source.preview
    if not preview:
        return []
    count = preview.get("row_count")
    size = f" — {count} rows" if count is not None else ""
    lines = [
        f"   Staged file{size}. Format: {preview.get('read_hint') or preview.get('data_format', '')}"
    ]
    columns = preview.get("columns") or []
    if columns:
        lines.append(f"   Columns: {', '.join(str(column) for column in columns)}")
    if preview.get("sample_rows"):
        lines.append(f"   First rows: {json.dumps(preview['sample_rows'], default=str)[:1200]}")
    if preview.get("last_rows"):
        lines.append(f"   Last rows: {json.dumps(preview['last_rows'], default=str)[:1200]}")
    if preview.get("notes"):
        lines.append(
            "   Notes from the staging README (source, units, caveats — follow them): "
            + " ".join(str(preview["notes"]).split())
        )
    placeholders = preview.get("trailing_placeholders") or {}
    if placeholders:
        detail = ", ".join(f"{column} (last {n} rows)" for column, n in placeholders.items())
        lines.append(
            "   PLACEHOLDERS: these columns are 0 or blank at the end of the file although "
            f"populated elsewhere — unpublished periods, not real values: {detail}. Drop those "
            "rows, or that column, before computing anything from it."
        )
    return lines


def _discovered_lines(source: DataSource) -> list[str]:
    """Told to the model because a discovered dataset can be real and wrong.

    Nobody named this file: it was found by keyword search against a catalogue.
    The columns are real, so the model must work with the columns it is given
    rather than the ones the plan imagined — and where the fit is poor, saying
    so in assumptions_made is the honest outcome, not quietly synthesizing
    something that matches the plan better.
    """
    if not source.discovered:
        return []
    return [
        f"   NOTE: no source was named for this input. It was found by searching "
        f"{source.discovered.get('connector', 'a catalogue')} for "
        f"{source.discovered.get('query', '')!r} "
        f"(record: {source.discovered.get('landing_page') or 'n/a'}).",
        "   Use the columns it actually has, not the ones the plan assumed. If it does not "
        "fit the experiment, say so in assumptions_made rather than synthesizing a "
        "replacement.",
    ]


def prompt_block(sources: list[DataSource]) -> str:
    """What the code generator is told about its inputs.

    Surrogates are named as surrogates here too, with an instruction to write
    and label a generator — not to assume some file will be there, which is the
    assumption that produced an experiment expecting placeholder CSVs nothing
    had created.
    """
    if not sources:
        return ""

    lines = [
        "RESOLVED DATA INPUTS — use exactly these, and nothing else:",
    ]
    for index, source in enumerate(sources, start=1):
        lines.append(f"\n{index}. {source.name}")
        if source.kind == KIND_REAL_LOCAL:
            lines.append(f"   REAL, already on disk at: {source.local_path}")
            lines.extend(_acquired_lines(source))
            lines.extend(_staged_lines(source))
            lines.extend(_discovered_lines(source))
            lines.append("   Read it directly. Do not download anything for this input.")
        elif source.kind == KIND_REAL_DOWNLOAD:
            lines.append(f"   REAL, fetch from: {source.uri}")
            if source.credentials:
                lines.append(
                    "   Credentials are in the environment — read them with "
                    + " and ".join(f"os.environ['{v}']" for v in source.credentials)
                    + ". Never hardcode them and never print them."
                )
            lines.append(
                "   Wrap the fetch in try/except and raise a clear error on failure. Do NOT "
                "silently fall back to made-up numbers."
            )
        else:
            lines.append(f"   SURROGATE — {source.reason}")
            lines.append(
                "   Write an explicit, seeded generator function whose name starts with "
                "`synthesize_`, with a docstring stating that it is synthetic and why. Give it "
                "realistic ranges and a realistic correlation structure so the pipeline is "
                "genuinely exercised, and say in assumptions_made that this input is synthetic."
            )
    lines.append(
        "\nNever write a file under data/ and then read it back as if it were real, and never "
        "assume an unstated file already exists."
    )
    return "\n".join(lines)
