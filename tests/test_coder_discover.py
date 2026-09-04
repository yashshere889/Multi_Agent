"""Tests for agents/coder/discover.py — finding a source for a requirement
nobody named one for.

No test here touches the network: catalogue responses go through a faked
`discover._get_json`, and the fetch-as-probe through a faked `acquire.fetch`.
The catalogue payload shapes below are trimmed copies of real responses from
data.gov.uk's CKAN and zenodo.org, verified 2026-09-04.
"""

from __future__ import annotations

import pytest

from research_pipeline.agents.coder import acquire, discover, provenance

# --------------------------------------------------------------------------
# Keywords, queries, relevance
# --------------------------------------------------------------------------


def test_query_for_drops_stopwords_and_short_tokens():
    assert discover.query_for("Hourly PM2.5 data from urban monitoring stations") == (
        "hourly pm2 urban monitoring stations"
    )


def test_query_for_falls_back_when_everything_is_a_stopword():
    # Better a weak query than none: an empty `q` matches the whole catalogue.
    assert discover.query_for("the data of the records") == "the data of the records"
    assert discover.query_for("") == ""


def _candidate(title="", description="", connector="ckan:test", url="https://x.example/d.csv"):
    return discover.Candidate(
        name="req", url=url, connector=connector, title=title, description=description
    )


def test_is_relevant_needs_two_shared_content_words():
    requirement = "urban air quality monitoring measurements"
    assert discover.is_relevant(requirement, _candidate(title="Urban Air Quality Monitoring"))
    # One shared word is a coincidence, not a match.
    assert not discover.is_relevant(requirement, _candidate(title="Urban Tree Canopy Survey"))


def test_is_relevant_accepts_one_word_when_that_is_all_there_is():
    assert discover.is_relevant("earthquakes", _candidate(title="Global earthquakes catalogue"))


def test_is_relevant_reads_the_description_too():
    assert discover.is_relevant(
        "hospital admission rates",
        _candidate(title="NHS Statistics", description="Monthly hospital admission counts"),
    )


def test_is_relevant_rejects_an_unrelated_best_match():
    """The gate that matters. A catalogue returns its best hit for a query that
    matches nothing well, and real-but-wrong data claims a verdict a surrogate
    would have withheld."""
    assert not discover.is_relevant(
        "bicycle collision casualty records",
        _candidate(title="Livestock population estimates", description="Sheep and cattle counts"),
    )


def test_is_relevant_rejects_an_empty_requirement():
    assert not discover.is_relevant("", _candidate(title="Anything at all"))


# --------------------------------------------------------------------------
# Format filtering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "declared", "expected"),
    [
        ("https://x.example/a.csv", "CSV", True),
        ("https://x.example/a", "JSON", True),
        ("https://x.example/a.xlsx", "XLSX", False),
        ("https://x.example/a.pdf", "PDF", False),
        ("https://x.example/wms", "WMS", False),
        # GeoJSON parses as JSON and would yield one row per map feature with a
        # nested geometry blob — success-shaped garbage, so it is excluded.
        ("https://x.example/a.geojson", "GeoJSON", False),
        # No declared format: fall back to the path's extension.
        ("https://x.example/download/a.csv?x=1", "", True),
        ("https://x.example/download/a.zip", "", False),
    ],
)
def test_is_tabular(url, declared, expected):
    assert discover._is_tabular(url, declared) is expected


# --------------------------------------------------------------------------
# Connectors
# --------------------------------------------------------------------------


def test_search_direct_finds_a_url_the_plan_named():
    found = discover.search_direct("Download from https://example.org/rates.csv, monthly.")
    assert [c.url for c in found] == ["https://example.org/rates.csv"]
    assert found[0].connector == "direct"


def test_search_direct_resolves_a_doi():
    found = discover.search_direct("See 10.5281/zenodo.5251825 for the released data.")
    assert [c.url for c in found] == ["https://doi.org/10.5281/zenodo.5251825"]


def test_search_direct_finds_nothing_in_prose():
    assert discover.search_direct("Air quality measurements for UK cities") == []


CKAN_PAYLOAD = {
    "success": True,
    "result": {
        "results": [
            {
                "name": "ni-air-quality",
                "title": "NI Air Quality",
                "notes": "Air quality monitoring measurements for Northern Ireland.",
                "url": "",
                "resources": [
                    {"url": "http://airqualityni.co.uk/atom.xml", "format": "XML"},
                    {"url": "https://airqualityni.co.uk/data.csv", "format": "CSV"},
                ],
            }
        ]
    },
}

ZENODO_PAYLOAD = {
    "hits": {
        "hits": [
            {
                "doi_url": "https://doi.org/10.5281/zenodo.5251825",
                "metadata": {
                    "title": "Urban air quality monitoring series",
                    "description": "<p>Hourly <b>air quality</b> monitoring measurements.</p>",
                },
                "files": [
                    {"key": "figure.xlsx", "links": {"self": "https://zenodo.org/f/figure.xlsx"}},
                    {"key": "series.csv", "links": {"self": "https://zenodo.org/f/series.csv"}},
                ],
            }
        ]
    }
}


def _serve_catalogue(monkeypatch, payload):
    calls = []

    def fake_get_json(url, params):
        calls.append((url, params))
        return payload

    monkeypatch.setattr(discover, "_get_json", fake_get_json)
    return calls


def test_search_ckan_reads_the_real_response_shape(monkeypatch):
    calls = _serve_catalogue(monkeypatch, CKAN_PAYLOAD)
    found = discover.search_ckan("air quality monitoring")

    # One request per portal, all to package_search with the keyword query.
    assert len(calls) == len(discover.CKAN_PORTALS)
    assert calls[0][0].endswith("/api/3/action/package_search")
    assert calls[0][1]["q"] == "air quality monitoring"
    # The XML resource is skipped; only the CSV becomes a candidate.
    assert {c.url for c in found} == {"https://airqualityni.co.uk/data.csv"}
    assert found[0].title == "NI Air Quality"
    assert found[0].connector.startswith("ckan:")
    # Landing page falls back to a page a human can actually open.
    assert found[0].landing_page.endswith("/ni-air-quality")


def test_search_ckan_survives_an_unexpected_payload(monkeypatch):
    for payload in ({}, {"result": None}, {"result": {"results": "nope"}}, None):
        _serve_catalogue(monkeypatch, payload)
        assert discover.search_ckan("air quality") == []


def test_search_zenodo_reads_the_real_response_shape(monkeypatch):
    _serve_catalogue(monkeypatch, ZENODO_PAYLOAD)
    found = discover.search_zenodo("air quality monitoring")

    assert [c.url for c in found] == ["https://zenodo.org/f/series.csv"]  # the .xlsx is skipped
    assert found[0].landing_page == "https://doi.org/10.5281/zenodo.5251825"
    assert "<b>" not in found[0].description  # HTML stripped before keyword matching
    assert discover.is_relevant("air quality monitoring", found[0])


def test_search_zenodo_survives_an_unexpected_payload(monkeypatch):
    for payload in ({}, {"hits": {}}, {"hits": {"hits": [None, 3]}}, None):
        _serve_catalogue(monkeypatch, payload)
        assert discover.search_zenodo("air quality") == []


def test_a_catalogue_request_to_an_unsafe_url_is_refused(monkeypatch):
    """The same URL gate acquire.py applies — a portal URL is still a URL."""
    monkeypatch.setattr(discover.acquire, "url_is_fetchable", lambda url: False)
    monkeypatch.setattr(
        discover.requests, "get", lambda *a, **k: pytest.fail("no request may be made")
    )
    assert discover._get_json("https://internal.example/api", {}) is None


# --------------------------------------------------------------------------
# find_source
# --------------------------------------------------------------------------


def _fake_acquire(monkeypatch, succeeds_for=()):
    """Fake acquire.fetch: succeeds only for the URLs listed. Records probes."""
    probed = []

    def fake_fetch(url, *, cache_dir, label="", max_bytes=0, max_rows=0):
        probed.append(url)
        if url not in succeeds_for:
            return None
        return acquire.Acquired(
            url=url,
            path=f"/cache/{len(probed)}.csv",
            sha256="a" * 64,
            byte_count=10,
            data_format="csv",
            columns=["a", "b"],
            row_count=5,
        )

    monkeypatch.setattr(discover.acquire, "fetch", fake_fetch)
    return probed


def _connector(name, candidates):
    return (name, lambda requirement: list(candidates))


def test_find_source_returns_the_first_candidate_that_actually_fetches(monkeypatch, tmp_path):
    good = _candidate(title="air quality monitoring", url="https://x.example/good.csv")
    bad = _candidate(title="air quality monitoring", url="https://x.example/dead.csv")
    probed = _fake_acquire(monkeypatch, succeeds_for={"https://x.example/good.csv"})

    found = discover.find_source(
        "air quality monitoring",
        cache_dir=tmp_path,
        connectors=[_connector("fake", [bad, good])],
    )
    assert found is not None and found.url == "https://x.example/good.csv"
    assert probed == ["https://x.example/dead.csv", "https://x.example/good.csv"]


def test_find_source_never_probes_an_irrelevant_candidate(monkeypatch, tmp_path):
    """The relevance gate runs before the download, so an unrelated hit costs
    nothing and — more importantly — can never be returned."""
    unrelated = _candidate(title="Livestock population estimates", url="https://x.example/a.csv")
    probed = _fake_acquire(monkeypatch, succeeds_for={"https://x.example/a.csv"})

    found = discover.find_source(
        "bicycle collision casualties",
        cache_dir=tmp_path,
        connectors=[_connector("fake", [unrelated])],
    )
    assert found is None
    assert probed == []


def test_find_source_exempts_a_directly_named_url_from_the_relevance_gate(monkeypatch, tmp_path):
    """A bare link the planner wrote has no prose to score, and it was asserted
    rather than searched for."""
    named = discover.Candidate(
        name="req", url="https://x.example/named.csv", connector="direct", title="", description=""
    )
    _fake_acquire(monkeypatch, succeeds_for={"https://x.example/named.csv"})
    found = discover.find_source(
        "totally unrelated words here",
        cache_dir=tmp_path,
        connectors=[_connector("direct", [named])],
    )
    assert found is not None and found.url == "https://x.example/named.csv"


def test_find_source_stops_after_the_probe_cap(monkeypatch, tmp_path):
    many = [
        _candidate(title="air quality monitoring", url=f"https://x.example/{i}.csv")
        for i in range(10)
    ]
    probed = _fake_acquire(monkeypatch, succeeds_for=set())
    assert (
        discover.find_source(
            "air quality monitoring",
            cache_dir=tmp_path,
            connectors=[_connector("fake", many)],
        )
        is None
    )
    assert len(probed) == discover.MAX_CANDIDATES_PROBED


def test_one_broken_connector_does_not_stop_the_others(monkeypatch, tmp_path):
    def boom(requirement):
        raise RuntimeError("catalogue is down")

    good = _candidate(title="air quality monitoring", url="https://x.example/good.csv")
    _fake_acquire(monkeypatch, succeeds_for={"https://x.example/good.csv"})
    found = discover.find_source(
        "air quality monitoring",
        cache_dir=tmp_path,
        connectors=[("broken", boom), _connector("fake", [good])],
    )
    assert found is not None and found.url == "https://x.example/good.csv"


def test_find_source_never_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        discover, "_find_source", lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom"))
    )
    assert discover.find_source("anything", cache_dir=tmp_path) is None


# --------------------------------------------------------------------------
# discover_sources / apply
# --------------------------------------------------------------------------


def _unresolved(name="bicycle collision casualties"):
    return provenance.DataSource(
        name=name,
        kind=provenance.KIND_SURROGATE,
        reason="no open source identified",
        unresolved=True,
    )


def test_discover_sources_only_searches_unresolved_requirements(monkeypatch, tmp_path):
    """A restricted or credentialed source names real data that specifically was
    not obtained. Answering it with a keyword match is the over-claim this whole
    area of the codebase exists to prevent."""
    searched = []
    monkeypatch.setattr(
        discover,
        "find_source",
        lambda requirement, **kwargs: searched.append(requirement) or None,
    )
    sources = [
        _unresolved("bicycle collisions"),
        provenance.DataSource(
            name="CMS claims", kind=provenance.KIND_SURROGATE, reason="requires a DUA"
        ),
        provenance.DataSource(
            name="EPA AQS",
            kind=provenance.KIND_SURROGATE,
            reason="needs an API key",
            uri="https://aqs",
        ),
        provenance.DataSource(
            name="staged", kind=provenance.KIND_REAL_LOCAL, local_path="/data/x.csv"
        ),
        provenance.DataSource(
            name="World Bank", kind=provenance.KIND_REAL_DOWNLOAD, uri="https://api.worldbank.org"
        ),
    ]
    discover.discover_sources(sources, cache_dir=tmp_path)
    assert searched == ["bicycle collisions"]


def test_discover_sources_records_what_was_searched_and_chosen(monkeypatch, tmp_path):
    monkeypatch.setattr(
        discover,
        "find_source",
        lambda requirement, **kwargs: discover.Candidate(
            name=requirement,
            url="https://x.example/found.csv",
            connector="ckan:data.gov.uk",
            title="Reported road collisions",
            landing_page="https://www.data.gov.uk/dataset/collisions",
        ),
    )
    found = discover.discover_sources(
        [_unresolved("bicycle collision casualties")], cache_dir=tmp_path
    )
    record = found["bicycle collision casualties"]
    assert record["url"] == "https://x.example/found.csv"
    assert record["connector"] == "ckan:data.gov.uk"
    assert record["landing_page"] == "https://www.data.gov.uk/dataset/collisions"
    assert record["query"] == "bicycle collision casualties"


def test_apply_turns_a_discovered_requirement_into_a_download():
    sources = [_unresolved()]
    discoveries = {
        sources[0].name: {
            "url": "https://x.example/found.csv",
            "connector": "ckan:data.gov.uk",
            "title": "Reported road collisions",
            "landing_page": "https://www.data.gov.uk/dataset/collisions",
            "query": "bicycle collision casualties",
        }
    }
    applied = discover.apply(sources, discoveries)
    assert applied[0].kind == provenance.KIND_REAL_DOWNLOAD
    assert applied[0].uri == "https://x.example/found.csv"
    assert applied[0].discovered["connector"] == "ckan:data.gov.uk"
    # The reason has to carry the audit trail — this is a weaker claim than a
    # staged file and the document must say so.
    assert "searched for" in applied[0].reason
    assert "data.gov.uk/dataset/collisions" in applied[0].reason
    assert "check that it answers the question" in applied[0].reason


def test_apply_is_a_no_op_without_discoveries():
    sources = [_unresolved()]
    assert discover.apply(sources, {}) == sources
    assert discover.apply(sources, None) == sources


def test_apply_never_rewrites_a_restricted_surrogate():
    restricted = provenance.DataSource(
        name="CMS claims", kind=provenance.KIND_SURROGATE, reason="requires a DUA"
    )
    discoveries = {"CMS claims": {"url": "https://x.example/anything.csv", "connector": "ckan"}}
    assert discover.apply([restricted], discoveries) == [restricted]


# --------------------------------------------------------------------------
# The composition: discovered -> downloaded -> local, and what it reports
# --------------------------------------------------------------------------


def test_a_discovered_source_becomes_a_real_local_input_and_stays_marked():
    """discover.apply sets the uri, acquire.apply promotes it — one mechanism,
    so a discovered input travels the same path as a named one and differs only
    in what its provenance says."""
    sources = [_unresolved()]
    discoveries = {
        sources[0].name: {
            "url": "https://x.example/found.csv",
            "connector": "ckan:data.gov.uk",
            "title": "Reported road collisions",
            "landing_page": "https://www.data.gov.uk/dataset/collisions",
            "query": "bicycle collision casualties",
        }
    }
    acquisitions = {
        "https://x.example/found.csv": acquire.Acquired(
            url="https://x.example/found.csv",
            path="/cache/ab/found.csv",
            sha256="b" * 64,
            byte_count=99,
            data_format="csv",
            columns=["date", "casualties"],
            row_count=42,
        ).to_dict()
    }

    applied = acquire.apply(discover.apply(sources, discoveries), acquisitions)
    assert applied[0].kind == provenance.KIND_REAL_LOCAL
    assert applied[0].local_path == "/cache/ab/found.csv"
    assert provenance.all_real(applied) is True  # nothing was invented...
    # ...but nobody named this dataset, so the verdict waits for a human.
    assert provenance.verdict(applied) == provenance.VERDICT_UNCONFIRMED
    # Still marked as discovered after the promotion — otherwise the document
    # would show a keyword-search result as though a human had named it.
    assert applied[0].discovered["connector"] == "ckan:data.gov.uk"

    document = provenance.as_document(applied)
    assert document["inputs"][0]["discovered"]["landing_page"] == (
        "https://www.data.gov.uk/dataset/collisions"
    )
    assert document["inputs"][0]["acquired"]["row_count"] == 42

    block = provenance.prompt_block(applied)
    assert "no source was named for this input" in block
    assert "ckan:data.gov.uk" in block
    assert "Columns: date, casualties" in block
    assert "say so in assumptions_made rather than synthesizing" in block


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The terms that actually discriminate between datasets are short.
        ("PM2.5 concentrations", {"pm2", "concentrations"}),
        ("CO2 emissions", {"co2", "emissions"}),
        ("EEG recordings", {"eeg", "recordings"}),
        ("GDP per capita", {"gdp", "capita"}),
        ("air quality", {"air", "quality"}),
        # ...but a bare number is not one of them.
        ("2020 2021 rainfall", {"rainfall"}),
        ("the data of the records", set()),
    ],
)
def test_keywords_keeps_short_scientific_terms(text, expected):
    assert discover.keywords(text) == expected


def test_a_measure_code_is_enough_to_match():
    """`pm2` shared between a requirement and a candidate is a strong signal,
    and the length floor used to throw it away."""
    assert discover.is_relevant(
        "PM2.5 concentrations", _candidate(title="Hourly PM2.5 concentrations by station")
    )


def test_a_declared_format_cannot_override_an_archive_extension():
    """Statistics Canada publishes .zip resources declared as "CSV". Trusting
    the declaration burns one of the four probe downloads to learn that an
    archive is not a table."""
    assert discover._is_tabular("https://x.example/38100111-eng.zip", "CSV") is False
    assert discover._is_tabular("https://x.example/tables.xlsx", "CSV") is False
    assert discover._is_tabular("https://x.example/real.csv", "CSV") is True


def test_every_ckan_portal_names_a_full_search_url():
    """The suffix is not derivable: data.europa.eu serves package_search under
    its hub path, not under /api/3/action/, and assuming otherwise 404'd every
    request to it."""
    for _, search_url, landing in discover.CKAN_PORTALS:
        assert search_url.startswith("https://")
        assert search_url.endswith("/package_search")
        assert landing.startswith("https://")


def test_relevance_scores_the_resource_name_too():
    """A dataset and the files inside it are not the same thing: an air-quality
    dataset's station list is not its measurements."""
    measurements = _candidate(title="Monitoring Air Quality")
    measurements.resource = "Hourly PM2.5 readings by station"
    locations = _candidate(title="Monitoring Air Quality")
    locations.resource = "Sensor Location PointLocation"

    requirement = "hourly PM2.5 air quality measurements"
    assert discover.relevance_score(requirement, measurements) > discover.relevance_score(
        requirement, locations
    )


def test_find_source_probes_the_best_scoring_candidate_first(monkeypatch, tmp_path):
    locations = _candidate(title="Monitoring Air Quality", url="https://x.example/loc.csv")
    locations.resource = "Sensor Location PointLocation"
    measurements = _candidate(title="Monitoring Air Quality", url="https://x.example/hourly.csv")
    measurements.resource = "Hourly PM2.5 readings"
    probed = _fake_acquire(
        monkeypatch, succeeds_for={"https://x.example/loc.csv", "https://x.example/hourly.csv"}
    )

    found = discover.find_source(
        "hourly PM2.5 air quality measurements",
        cache_dir=tmp_path,
        # Deliberately offered worst-first, the order a catalogue returned them.
        connectors=[_connector("fake", [locations, measurements])],
    )
    assert probed == ["https://x.example/hourly.csv"]
    assert found is not None and found.url == "https://x.example/hourly.csv"


def test_the_chosen_resource_is_recorded_for_the_auditor(monkeypatch):
    _serve_catalogue(monkeypatch, CKAN_PAYLOAD)
    found = discover.search_ckan("air quality monitoring")
    assert found[0].to_dict()["resource"] is not None


# --------------------------------------------------------------------------
# A discovered input is real, and still does not get to claim a verdict
# --------------------------------------------------------------------------


def _discovered_local(name="crime counts by police force"):
    """What discover.apply + acquire.apply produce for a discovered input."""
    return provenance.DataSource(
        name=name,
        kind=provenance.KIND_REAL_LOCAL,
        uri="https://x.example/found.csv",
        local_path="/cache/ab/found.csv",
        reason="searched for",
        acquired={"sha256": "c" * 64, "row_count": 404, "columns": ["CSP Name"]},
        discovered={
            "connector": "ckan:data.gov.uk",
            "title": "Police recorded crime data",
            "resource": "Geographic reference table",
            "landing_page": "https://www.data.gov.uk/dataset/crime",
            "query": "crime counts police force",
        },
    )


def test_a_discovered_input_is_real_but_needs_confirmation():
    sources = [_discovered_local()]
    assert provenance.all_real(sources) is True  # nothing was invented
    assert provenance.needs_confirmation(sources) is True
    assert provenance.verdict(sources) == provenance.VERDICT_UNCONFIRMED


def test_a_named_input_still_gets_a_verdict():
    """The conservative path must not swallow the ordinary one."""
    staged = provenance.DataSource(
        name="staged cohort", kind=provenance.KIND_REAL_LOCAL, local_path="/data/cohort.csv"
    )
    assert provenance.needs_confirmation([staged]) is False
    assert provenance.verdict([staged]) == provenance.VERDICT_EVIDENCE


def test_a_discovered_input_withholds_the_hypothesis_verdict():
    """The measured reason this exists: a live sweep returned two datasets that
    were real, plausible and wrong. A refutation computed on the wrong real data
    reads as defensible in a way one computed on invented data does not."""
    results = {"metrics": {"r2": 0.81}, "meets_success_criteria": False}
    stamped = provenance.apply_to_results(results, [_discovered_local()])

    assert stamped["meets_success_criteria"] == "unknown"  # not False — not "refuted"
    assert stamped["model_reported_meets_success_criteria"] is False
    assert stamped["metrics"] == {"r2": 0.81}  # the metrics are still reported
    assert "found by keyword search" in stamped["verdict_withheld_because"]
    assert "CODER_DATA_DIR" in stamped["verdict_withheld_because"]


def test_the_document_names_which_inputs_are_unconfirmed():
    document = provenance.as_document([_discovered_local()])
    assert document["all_inputs_real"] is True
    assert document["unconfirmed_discovered_inputs"] == ["crime counts by police force"]
    assert document["methodological_validity"] == provenance.VERDICT_UNCONFIRMED


def test_a_run_with_no_discovery_writes_the_document_it_always_did():
    staged = provenance.DataSource(
        name="staged", kind=provenance.KIND_REAL_LOCAL, local_path="/data/x.csv"
    )
    document = provenance.as_document([staged])
    assert document["unconfirmed_discovered_inputs"] == []
    assert "discovered" not in document["inputs"][0]


# --------------------------------------------------------------------------
# The chooser: a model may reorder and reject, never invent
# --------------------------------------------------------------------------


def _scored(title, resource, url):
    c = _candidate(title=title, url=url)
    c.resource = resource
    return c


def test_rank_with_falls_back_to_keyword_order_without_a_chooser():
    a = _scored("Air quality", "Sensor locations", "https://x.example/loc.csv")
    b = _scored("Air quality", "Hourly PM2.5 measurements", "https://x.example/hourly.csv")
    assert discover.rank_with("hourly PM2.5 measurements", [a, b], None) == [b, a]


def test_rank_with_follows_the_choosers_order():
    a = _scored("Air quality", "Hourly PM2.5 measurements", "https://x.example/hourly.csv")
    b = _scored("Air quality", "Sensor locations", "https://x.example/loc.csv")
    assert discover.rank_with("air quality", [a, b], lambda r, c: [1, 0]) == [b, a]


def test_rank_with_drops_candidates_the_chooser_leaves_out():
    """Rejecting is the point: the file with the right title is routinely a
    station list or a data dictionary, not the data."""
    a = _scored("Air quality", "Hourly PM2.5", "https://x.example/hourly.csv")
    b = _scored("Air quality", "Geographic reference table", "https://x.example/ref.csv")
    assert discover.rank_with("air quality", [a, b], lambda r, c: [0]) == [a]


def test_an_empty_choice_means_none_of_these_and_is_honoured():
    a = _scored("Livestock", "Sheep counts", "https://x.example/a.csv")
    b = _scored("Livestock", "Cattle counts", "https://x.example/b.csv")
    assert discover.rank_with("air quality", [a, b], lambda r, c: []) == []


def test_a_chooser_cannot_invent_a_candidate():
    a = _scored("Air quality", "Hourly PM2.5", "https://x.example/a.csv")
    b = _scored("Air quality", "Sensor locations", "https://x.example/b.csv")
    # Out of range, negative, duplicated, and the wrong type — all dropped.
    ranked = discover.rank_with("air quality", [a, b], lambda r, c: [99, -1, 1, 1, "0", None])
    assert ranked == [b]


def test_a_raising_chooser_falls_back_to_keyword_order():
    a = _scored("Air quality", "Sensor locations", "https://x.example/loc.csv")
    b = _scored("Air quality", "Hourly PM2.5 measurements", "https://x.example/hourly.csv")

    def boom(requirement, candidates):
        raise RuntimeError("model is down")

    assert discover.rank_with("hourly PM2.5 measurements", [a, b], boom) == [b, a]


def test_a_chooser_returning_none_falls_back_rather_than_rejecting_everything():
    """None and [] must not mean the same thing: [] is "I looked and none fit",
    None is "no answer was obtained"."""
    a = _scored("Air quality", "Hourly PM2.5", "https://x.example/a.csv")
    b = _scored("Air quality", "Sensor locations", "https://x.example/b.csv")
    assert discover.rank_with("air quality", [a, b], lambda r, c: None) == [a, b]


def test_the_chooser_is_not_called_for_a_single_candidate(monkeypatch):
    only = _scored("Air quality", "Hourly PM2.5", "https://x.example/a.csv")

    def never(requirement, candidates):
        pytest.fail("no model call is worth making to rank one candidate")

    assert discover.rank_with("air quality", [only], never) == [only]


def test_the_chooser_sees_at_most_the_shown_cap():
    many = [_scored("Air quality", f"file {i}", f"https://x.example/{i}.csv") for i in range(50)]
    seen = {}
    discover.rank_with("air quality", many, lambda r, c: seen.setdefault("n", len(c)) and [])
    assert seen["n"] == discover.MAX_CANDIDATES_SHOWN


# --------------------------------------------------------------------------
# Pooling across connectors, and where `direct` sits
# --------------------------------------------------------------------------


def test_candidates_are_pooled_across_connectors_before_ranking(monkeypatch, tmp_path):
    """Asked one catalogue at a time, a chooser can only pick the best of four
    when the honest answer is "none of these, but that Zenodo one"."""
    # Both clear the relevance gate — this test is about pooling, not the gate.
    ckan = _scored("Air quality hourly measurements", "Station list", "https://x.example/ckan.csv")
    zenodo = _scored(
        "Air quality hourly measurements", "Hourly readings", "https://x.example/zenodo.csv"
    )
    _fake_acquire(monkeypatch, succeeds_for={"https://x.example/zenodo.csv"})

    pools = []
    found = discover.find_source(
        "air quality hourly measurements",
        cache_dir=tmp_path,
        connectors=[_connector("ckan", [ckan]), _connector("zenodo", [zenodo])],
        chooser=lambda r, c: pools.append([x.url for x in c]) or [1],
    )
    assert pools == [["https://x.example/ckan.csv", "https://x.example/zenodo.csv"]]
    assert found is not None and found.url == "https://x.example/zenodo.csv"


def test_a_directly_named_url_short_circuits_the_catalogues(monkeypatch, tmp_path):
    """Searching four catalogues to second-guess a URL the plan asserted is waste."""
    named = discover.Candidate(name="req", url="https://x.example/named.csv", connector="direct")
    _fake_acquire(monkeypatch, succeeds_for={"https://x.example/named.csv"})

    found = discover.find_source(
        "air quality",
        cache_dir=tmp_path,
        connectors=[
            _connector("direct", [named]),
            ("ckan", lambda r: pytest.fail("no catalogue search once a named URL worked")),
        ],
    )
    assert found is not None and found.url == "https://x.example/named.csv"


def test_a_dead_direct_url_falls_through_to_the_catalogues(monkeypatch, tmp_path):
    named = discover.Candidate(name="req", url="https://x.example/dead.csv", connector="direct")
    live = _scored("Air quality monitoring", "Hourly readings", "https://x.example/live.csv")
    _fake_acquire(monkeypatch, succeeds_for={"https://x.example/live.csv"})

    found = discover.find_source(
        "air quality monitoring",
        cache_dir=tmp_path,
        connectors=[_connector("direct", [named]), _connector("ckan", [live])],
    )
    assert found is not None and found.url == "https://x.example/live.csv"


def test_the_probe_budget_is_shared_across_connectors(monkeypatch, tmp_path):
    """A direct link that fails must not hand the catalogues a fresh budget."""
    direct = [
        discover.Candidate(name="r", url=f"https://x.example/d{i}.csv", connector="direct")
        for i in range(4)
    ]
    catalogue = [
        _scored("Air quality monitoring", "readings", f"https://x.example/c{i}.csv")
        for i in range(6)
    ]
    probed = _fake_acquire(monkeypatch, succeeds_for=set())
    discover.find_source(
        "air quality monitoring",
        cache_dir=tmp_path,
        connectors=[_connector("direct", direct), _connector("ckan", catalogue)],
    )
    assert len(probed) == discover.MAX_CANDIDATES_PROBED
