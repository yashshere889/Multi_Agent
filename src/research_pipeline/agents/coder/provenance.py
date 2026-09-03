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
    (
        r"\bFRED\b|federal reserve economic data",
        "FRED",
        "https://api.stlouisfed.org/fred/series/observations",
        ("FRED_API_KEY",),
        "request a free key at https://fredaccount.stlouisfed.org/apikeys, then export FRED_API_KEY",
    ),
]

OPEN_SOURCES: list[tuple[str, str, str]] = [
    (r"american community survey|\bACS\b|census", "US Census ACS", "https://api.census.gov/data"),
    (r"world bank", "World Bank", "https://api.worldbank.org/v2"),
    (r"\bWHO\b|global health observatory", "WHO GHO", "https://ghoapi.azureedge.net/api"),
    (r"open ?street ?map|\bOSM\b", "OpenStreetMap", "https://overpass-api.de/api"),
    (r"\bNDVI\b|landsat|\bUSGS\b", "USGS EarthExplorer", "https://earthexplorer.usgs.gov"),
    (r"hugging ?face", "Hugging Face datasets", "https://datasets-server.huggingface.co/rows"),
    # Daily OHLC equity/index history, keyless. Stooq serves plain CSV over a
    # stable GET (Yahoo Finance's chart endpoint is the same shape but rate-limits
    # server IPs harder), so it is the one named here. A run asking to compare
    # forecasting models against "S&P 500 daily closing prices" resolved to a
    # surrogate before this entry existed — no open source matched, so the
    # verdict was withheld on an experiment that could have used real prices.
    (
        r"yahoo ?finance|stock (price|market)|share price|equity price|"
        r"\bS&?P ?500\b|\bOHLC\b|ticker|stock exchange|market index",
        "Stooq daily OHLC (keyless CSV; ^SPX for the S&P 500)",
        "https://stooq.com/q/d/l/",
    ),
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

    @property
    def is_real(self) -> bool:
        return self.kind in REAL_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "uri": self.uri,
            "local_path": self.local_path,
            "reason": self.reason,
            "credentials": self.credentials,
        }


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


def _staged_file(staging_dir: Path | None, requirement: str) -> Path | None:
    """A file the user staged for this requirement, matched on shared keywords.

    Not an exact name match: somebody staging data names the file after the
    data, not after the planner's phrasing of it.
    """
    if not staging_dir or not staging_dir.is_dir():
        return None
    words = {w for w in re.split(r"[^a-z0-9]+", requirement.lower()) if len(w) > 3}
    if not words:
        return None
    best: tuple[int, Path] | None = None
    for candidate in staging_dir.rglob("*"):
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        name_words = {w for w in re.split(r"[^a-z0-9]+", candidate.stem.lower()) if len(w) > 3}
        overlap = len(words & name_words)
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, candidate)
    return best[1] if best else None


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
            resolved.append(
                DataSource(
                    name=requirement,
                    kind=KIND_REAL_LOCAL,
                    local_path=str(staged),
                    reason=f"staged locally at {staged}",
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
    a matched Hub dataset downloaded to disk answers the "Hugging Face" the plan
    asked for, and the generated code then reads a local file and never names the
    REST host — and per-requirement matching turns that into a phantom surrogate.
    What stays caught is the failure this guards: an experiment that declared
    real inputs, fetched none of them, and synthesized instead.
    """
    if not code:
        return sources

    def obtained(source: DataSource) -> bool:
        if source.kind == KIND_REAL_LOCAL:
            return True
        host = urlparse(source.uri).netloc
        return bool(host) and host in code

    if any(obtained(s) for s in sources if s.is_real):
        return sources

    verified: list[DataSource] = []
    for source in sources:
        host = urlparse(source.uri).netloc
        if source.kind != KIND_REAL_DOWNLOAD or not host:
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


def all_real(sources: list[DataSource]) -> bool:
    return bool(sources) and all(s.is_real for s in sources)


def verdict(sources: list[DataSource]) -> str:
    """The methodological validity stamp — computed, never asked of the model."""
    if not sources:
        # No resolvable inputs is not evidence of real ones. Silence here would
        # let a plan with an unparseable data_requirements block claim support.
        return VERDICT_SURROGATE
    if all_real(sources):
        return VERDICT_EVIDENCE
    return VERDICT_MIXED if any(s.is_real for s in sources) else VERDICT_SURROGATE


def as_document(sources: list[DataSource]) -> dict[str, Any]:
    return {
        "inputs": [s.to_dict() for s in sources],
        "methodological_validity": verdict(sources),
        "all_inputs_real": all_real(sources),
        "surrogate_count": sum(1 for s in sources if s.kind == KIND_SURROGATE),
    }


def write(sources: list[DataSource], path: Path) -> dict[str, Any]:
    document = as_document(sources)
    path.write_text(json.dumps(document, indent=2))
    return document


def apply_to_results(results: dict, sources: list[DataSource]) -> dict:
    """Withhold the hypothesis verdict when any input is synthetic.

    `meets_success_criteria` becomes the string "unknown" rather than False.
    That distinction is the whole point: `writer_agent` maps False to "refuted"
    and "unknown" to "inconclusive", so returning False here would have the
    paper claim a refutation off invented numbers. The metrics themselves are
    left untouched and still reported — they describe what the pipeline did,
    which is worth reading; they just no longer carry a verdict about the world.
    """
    if all_real(sources):
        return results

    stamped = dict(results)
    stamped["model_reported_meets_success_criteria"] = results.get("meets_success_criteria")
    stamped["meets_success_criteria"] = "unknown"
    stamped["methodological_validity"] = verdict(sources)
    stamped["verdict_withheld_because"] = (
        "One or more inputs are synthetic surrogates, so these metrics describe the pipeline's "
        "behaviour on generated data and say nothing about the real-world hypothesis. See "
        "data_provenance.json."
    )
    return stamped


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
