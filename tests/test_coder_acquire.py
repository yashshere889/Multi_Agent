"""Tests for agents/coder/acquire.py — fetching an experiment's data here
rather than inside the generated code.

No test in this file touches the network. `requests.get` and
`socket.getaddrinfo` are both faked, the latter because the SSRF guard resolves
every host it is asked about and a real lookup would make these tests depend on
DNS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.agents.coder import acquire, provenance, sandbox

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, status_code=200, headers=None, body=b"", raises_at=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._raises_at = raises_at

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "location" in self.headers

    @property
    def is_permanent_redirect(self):
        return self.status_code in (301, 308)

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Every host resolves to a public address unless a test says otherwise."""
    monkeypatch.setattr(
        acquire.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )


def _serve(monkeypatch, responses):
    """Serve `responses` (a list, or a {url: response} map) and record calls."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if isinstance(responses, dict):
            return responses[url]
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    return calls


def _csv_response(body=b"city,pm25\nLiverpool,12.4\nManchester,15.1\n"):
    return FakeResponse(headers={"content-type": "text/csv"}, body=body)


# --------------------------------------------------------------------------
# The URL safety gate
# --------------------------------------------------------------------------


def test_url_is_fetchable_requires_https():
    assert acquire.url_is_fetchable("https://example.org/data.csv") is True
    assert acquire.url_is_fetchable("http://example.org/data.csv") is False
    assert acquire.url_is_fetchable("file:///etc/passwd") is False
    assert acquire.url_is_fetchable("https://") is False


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # RFC1918
        "192.168.1.1",  # RFC1918
        "169.254.169.254",  # the cloud metadata service — the SSRF prize
        "::1",  # IPv6 loopback
    ],
)
def test_url_is_fetchable_rejects_non_public_addresses(monkeypatch, address):
    family = 10 if ":" in address else 2
    monkeypatch.setattr(
        acquire.socket, "getaddrinfo", lambda host, port: [(family, 1, 6, "", (address, 0))]
    )
    assert acquire.url_is_fetchable("https://internal.example/data.csv") is False


def test_url_is_fetchable_rejects_a_host_with_any_private_address(monkeypatch):
    """One public and one loopback address is the shape a rebinding attempt
    takes; picking the public one would not be the address requests connects to."""
    monkeypatch.setattr(
        acquire.socket,
        "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0)), (2, 1, 6, "", ("127.0.0.1", 0))],
    )
    assert acquire.url_is_fetchable("https://split-horizon.example/d.csv") is False


def test_url_is_fetchable_rejects_an_unresolvable_host(monkeypatch):
    def boom(host, port):
        raise OSError("no such host")

    monkeypatch.setattr(acquire.socket, "getaddrinfo", boom)
    assert acquire.url_is_fetchable("https://nope.example/data.csv") is False


def test_fetch_refuses_a_redirect_into_a_private_address(monkeypatch, tmp_path):
    """The hop is what gets checked, not the string that was passed in."""
    resolutions = {
        "public.example": ("93.184.216.34", 2),
        "internal.example": ("127.0.0.1", 2),
    }
    monkeypatch.setattr(
        acquire.socket,
        "getaddrinfo",
        lambda host, port: [(resolutions[host][1], 1, 6, "", (resolutions[host][0], 0))],
    )
    _serve(
        monkeypatch,
        {
            "https://public.example/d.csv": FakeResponse(
                status_code=302, headers={"location": "https://internal.example/secrets"}
            )
        },
    )
    assert acquire.fetch("https://public.example/d.csv", cache_dir=tmp_path) is None


def test_fetch_follows_a_redirect_to_a_public_host(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        {
            "https://a.example/d.csv": FakeResponse(
                status_code=302, headers={"location": "https://b.example/real.csv"}
            ),
            "https://b.example/real.csv": _csv_response(),
        },
    )
    acquired = acquire.fetch("https://a.example/d.csv", cache_dir=tmp_path)
    assert acquired is not None
    assert acquired.columns == ["city", "pm25"]


def test_fetch_gives_up_on_a_redirect_loop(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        [FakeResponse(status_code=302, headers={"location": "https://a.example/again"})],
    )
    assert acquire.fetch("https://a.example/d.csv", cache_dir=tmp_path) is None


# --------------------------------------------------------------------------
# What comes back over the wire
# --------------------------------------------------------------------------


def test_fetch_rejects_an_html_body(monkeypatch, tmp_path):
    """The Stooq failure: HTTP 200 carrying a proof-of-work page, which pandas
    would otherwise parse into garbage instead of failing loudly."""
    _serve(
        monkeypatch,
        [
            FakeResponse(
                headers={"content-type": "text/html; charset=utf-8"},
                body=b"<html><body>checking your browser</body></html>",
            )
        ],
    )
    assert acquire.fetch("https://stooq.example/q/d/l", cache_dir=tmp_path) is None


def test_fetch_rejects_a_non_200(monkeypatch, tmp_path):
    _serve(monkeypatch, [FakeResponse(status_code=404, body=b"nope")])
    assert acquire.fetch("https://example.org/missing.csv", cache_dir=tmp_path) is None


def test_fetch_abandons_a_body_past_the_cap(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        [FakeResponse(headers={"content-type": "text/csv"}, body=b"a,b\n" + b"1,2\n" * 100_000)],
    )
    assert acquire.fetch("https://example.org/huge.csv", cache_dir=tmp_path, max_bytes=1024) is None


def test_fetch_returns_none_when_requests_raises(monkeypatch, tmp_path):
    def boom(url, **kwargs):
        raise acquire.requests.RequestException("connection reset")

    monkeypatch.setattr(acquire.requests, "get", boom)
    assert acquire.fetch("https://example.org/d.csv", cache_dir=tmp_path) is None


def test_fetch_never_raises_on_an_unexpected_failure(monkeypatch, tmp_path):
    """The never-raise contract: an experiment must not go ungenerated because
    a download broke in a way nothing anticipated."""

    def boom(url, **kwargs):
        raise MemoryError("something entirely unexpected")

    monkeypatch.setattr(acquire.requests, "get", boom)
    assert acquire.fetch("https://example.org/d.csv", cache_dir=tmp_path) is None


# --------------------------------------------------------------------------
# Identifying the payload
# --------------------------------------------------------------------------


def test_describe_reads_csv_columns_and_row_count():
    described = acquire.describe(b"city,pm25,year\nLiverpool,12.4,2020\nLeeds,11.0,2021\n")
    assert described is not None
    data_format, columns, row_count, sample, payload = described
    assert data_format == acquire.FORMAT_CSV
    assert columns == ["city", "pm25", "year"]
    assert row_count == 2
    assert sample[0] == {"city": "Liverpool", "pm25": "12.4", "year": "2020"}
    assert payload == b"city,pm25,year\nLiverpool,12.4,2020\nLeeds,11.0,2021\n"


def test_describe_reads_tsv():
    described = acquire.describe(b"city\tpm25\nLiverpool\t12.4\nLeeds\t11.0\n")
    assert described is not None
    assert described[0] == acquire.FORMAT_TSV
    assert described[1] == ["city", "pm25"]


def test_describe_reads_a_json_array_of_objects():
    body = json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]).encode()
    described = acquire.describe(body)
    assert described is not None
    assert described[0] == acquire.FORMAT_JSONL
    assert described[1] == ["a", "b"]
    assert described[2] == 2
    # Normalized to one JSON object per line, whatever the envelope was.
    assert described[4] == b'{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n'


def test_describe_unwraps_the_dataset_viewer_envelope():
    body = json.dumps(
        {
            "features": [{"name": "text"}],
            "rows": [
                {"row_idx": 0, "row": {"text": "hello", "label": 1}},
                {"row_idx": 1, "row": {"text": "world", "label": 0}},
            ],
            "num_rows_total": 2,
        }
    ).encode()
    described = acquire.describe(body)
    assert described is not None
    assert described[1] == ["text", "label"]
    assert described[2] == 2


def test_describe_takes_the_union_of_ragged_record_keys():
    """A column absent from row 1 and present later must still be described,
    which is exactly what a CSV header could not express."""
    body = json.dumps([{"a": 1}, {"a": 2, "b": 9}]).encode()
    described = acquire.describe(body)
    assert described is not None
    assert described[1] == ["a", "b"]


def test_describe_rejects_a_single_column_body():
    """One column is what a plain-text error page looks like to a CSV reader."""
    assert acquire.describe(b"Internal Server Error\nplease try again later\n") is None


def test_describe_rejects_json_with_no_identifiable_records():
    assert acquire.describe(b'{"status": "ok", "count": 3}') is None


def test_describe_rejects_an_empty_or_undecodable_body():
    assert acquire.describe(b"") is None
    assert acquire.describe(b"\xff\xfe\x00binary") is None


def test_describe_truncates_a_huge_cell():
    body = json.dumps([{"doc": "x" * 5000, "n": 1}]).encode()
    described = acquire.describe(body)
    assert described is not None
    assert len(described[3][0]["doc"]) == acquire.MAX_CELL_CHARS + 1  # + the ellipsis


# --------------------------------------------------------------------------
# Writing, and the cache
# --------------------------------------------------------------------------


def test_fetch_writes_the_file_and_its_metadata(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    acquired = acquire.fetch(
        "https://example.org/air.csv", cache_dir=tmp_path, label="EPA AQS PM2.5"
    )
    assert acquired is not None
    written = Path(acquired.path)
    assert written.is_file()
    assert written.parent.parent == tmp_path
    assert written.name == "epa_aqs_pm2.5.csv"
    assert written.read_bytes() == b"city,pm25\nLiverpool,12.4\nManchester,15.1\n"
    assert acquired.row_count == 2
    assert acquired.byte_count == len(written.read_bytes())
    assert len(acquired.sha256) == 64
    assert acquired.from_cache is False
    meta = json.loads((written.parent / "meta.json").read_text())
    assert meta["url"] == "https://example.org/air.csv"
    assert meta["sha256"] == acquired.sha256
    # No .partial left behind.
    assert not list(written.parent.glob("*.partial"))


def test_fetch_serves_the_second_call_from_cache(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, [_csv_response()])
    first = acquire.fetch("https://example.org/air.csv", cache_dir=tmp_path)
    second = acquire.fetch("https://example.org/air.csv", cache_dir=tmp_path)
    assert len(calls) == 1  # a 100-question sweep downloads each dataset once
    assert second is not None and first is not None
    assert second.path == first.path
    assert second.sha256 == first.sha256
    assert second.from_cache is True


def test_fetch_refetches_when_the_cached_file_is_gone(monkeypatch, tmp_path):
    """Scratch filesystems get cleared under a resumed run, so meta.json's
    presence is not taken as proof the data file survived."""
    calls = _serve(monkeypatch, [_csv_response(), _csv_response()])
    first = acquire.fetch("https://example.org/air.csv", cache_dir=tmp_path)
    assert first is not None
    Path(first.path).unlink()
    again = acquire.fetch("https://example.org/air.csv", cache_dir=tmp_path)
    assert len(calls) == 2
    assert again is not None and again.from_cache is False


def test_different_urls_do_not_share_a_cache_slot(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        {
            "https://example.org/a.csv": _csv_response(b"x,y\n1,2\n"),
            "https://example.org/b.csv": _csv_response(b"p,q\n3,4\n"),
        },
    )
    first = acquire.fetch("https://example.org/a.csv", cache_dir=tmp_path)
    second = acquire.fetch("https://example.org/b.csv", cache_dir=tmp_path)
    assert first is not None and second is not None
    assert first.path != second.path
    assert second.columns == ["p", "q"]


# --------------------------------------------------------------------------
# Paginated row APIs
# --------------------------------------------------------------------------


def _viewer_page(offset, total):
    rows = [
        {"row_idx": index, "row": {"text": f"row-{index}", "label": index % 2}}
        for index in range(offset, min(offset + acquire.PAGE_ROWS, total))
    ]
    return FakeResponse(
        headers={"content-type": "application/json"},
        body=json.dumps({"rows": rows, "num_rows_total": total}).encode(),
    )


def test_is_paginated_rows_url():
    assert acquire.is_paginated_rows_url("https://x.example/rows?dataset=d&offset=0&length=100")
    assert not acquire.is_paginated_rows_url("https://x.example/data.csv")


def test_fetch_pages_a_rows_api_past_the_first_page(monkeypatch, tmp_path):
    total = 250
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        offset = int(dict(acquire.parse_qsl(acquire.urlparse(url).query))["offset"])
        return _viewer_page(offset, total)

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    acquired = acquire.fetch(
        "https://datasets-server.example/rows?dataset=d&config=default&split=train"
        "&offset=0&length=100",
        cache_dir=tmp_path,
    )
    assert acquired is not None
    assert acquired.row_count == total
    assert acquired.data_format == acquire.FORMAT_JSONL
    assert len(calls) == 3
    assert "offset=100" in calls[1]
    assert len(Path(acquired.path).read_text().strip().splitlines()) == total


def test_fetch_stops_paging_at_the_row_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        acquire.requests,
        "get",
        lambda url, **kwargs: _viewer_page(
            int(dict(acquire.parse_qsl(acquire.urlparse(url).query))["offset"]), 100_000
        ),
    )
    acquired = acquire.fetch(
        "https://datasets-server.example/rows?dataset=d&offset=0&length=100",
        cache_dir=tmp_path,
        max_rows=150,
    )
    assert acquired is not None
    assert acquired.row_count == 150


def test_a_failed_page_keeps_the_rows_already_collected(monkeypatch, tmp_path):
    """4,900 real rows is an experiment. The alternative to a short read is not
    a longer read, it is a synthetic surrogate."""
    state = {"calls": 0}

    def fake_get(url, **kwargs):
        state["calls"] += 1
        if state["calls"] > 2:
            return FakeResponse(status_code=503, body=b"")
        offset = int(dict(acquire.parse_qsl(acquire.urlparse(url).query))["offset"])
        return _viewer_page(offset, 100_000)

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    acquired = acquire.fetch(
        "https://datasets-server.example/rows?dataset=d&offset=0&length=100", cache_dir=tmp_path
    )
    assert acquired is not None
    assert acquired.row_count == 2 * acquire.PAGE_ROWS


# --------------------------------------------------------------------------
# acquire_sources / apply
# --------------------------------------------------------------------------


def _download(name="EPA AQS PM2.5", uri="https://example.org/air.csv", **kw):
    return provenance.DataSource(
        name=name, kind=provenance.KIND_REAL_DOWNLOAD, uri=uri, reason="openly fetchable", **kw
    )


def test_acquire_sources_fetches_only_real_downloads(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, [_csv_response()])
    sources = [
        _download(),
        provenance.DataSource(
            name="CMS claims", kind=provenance.KIND_SURROGATE, reason="needs a DUA"
        ),
        provenance.DataSource(
            name="staged cohort", kind=provenance.KIND_REAL_LOCAL, local_path="/data/cohort.csv"
        ),
    ]
    acquisitions = acquire.acquire_sources(sources, cache_dir=tmp_path)
    assert list(acquisitions) == ["https://example.org/air.csv"]
    assert len(calls) == 1


def test_acquire_sources_skips_a_source_that_cannot_be_fetched(monkeypatch, tmp_path):
    """A base API URL with no query on it is the ordinary miss: it returns
    something that isn't tabular data, so the input stays a real_download and
    the generated code fetches it exactly as before."""
    _serve(monkeypatch, [FakeResponse(headers={"content-type": "text/html"}, body=b"<html>")])
    sources = [_download(uri="https://api.worldbank.org/v2")]
    assert acquire.acquire_sources(sources, cache_dir=tmp_path) == {}


def test_apply_promotes_a_fetched_download_to_real_local(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    sources = [_download()]
    acquisitions = acquire.acquire_sources(sources, cache_dir=tmp_path)

    applied = acquire.apply(sources, acquisitions)
    assert applied[0].kind == provenance.KIND_REAL_LOCAL
    assert applied[0].uri == "https://example.org/air.csv"  # origin is kept
    assert applied[0].local_path.endswith(".csv")
    assert applied[0].acquired["columns"] == ["city", "pm25"]
    assert "fetched by the pipeline" in applied[0].reason
    assert provenance.all_real(applied) is True


def test_apply_is_a_no_op_without_acquisitions():
    sources = [_download()]
    assert acquire.apply(sources, {}) == sources
    assert acquire.apply(sources, None) == sources


def test_apply_is_idempotent(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    sources = [_download()]
    acquisitions = acquire.acquire_sources(sources, cache_dir=tmp_path)
    once = acquire.apply(sources, acquisitions)
    assert acquire.apply(once, acquisitions) == once


def test_apply_leaves_surrogates_alone(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    surrogate = provenance.DataSource(
        name="CMS claims", kind=provenance.KIND_SURROGATE, reason="needs a DUA"
    )
    sources = [_download(), surrogate]
    applied = acquire.apply(sources, acquire.acquire_sources(sources, cache_dir=tmp_path))
    assert [s.kind for s in applied] == [provenance.KIND_REAL_LOCAL, provenance.KIND_SURROGATE]
    assert provenance.verdict(applied) == provenance.VERDICT_MIXED


# --------------------------------------------------------------------------
# The point of all of it: what the rest of the Coder Agent then sees
# --------------------------------------------------------------------------


def test_an_acquired_input_survives_verify_downloads_used(monkeypatch, tmp_path):
    """The core win. Code reading a local CSV names no host, so as a
    real_download it would have been downgraded to a surrogate and had its
    verdict withheld; as the real_local it now is, it is trusted."""
    _serve(monkeypatch, [_csv_response()])
    sources = [_download()]
    applied = acquire.apply(sources, acquire.acquire_sources(sources, cache_dir=tmp_path))
    code = f"df = pandas.read_csv({applied[0].local_path!r})"

    assert provenance.verify_downloads_used(sources, code)[0].kind == provenance.KIND_SURROGATE
    assert provenance.verify_downloads_used(applied, code)[0].kind == provenance.KIND_REAL_LOCAL
    assert provenance.verdict(provenance.verify_downloads_used(applied, code)) == (
        provenance.VERDICT_EVIDENCE
    )


def test_prompt_block_hands_over_the_real_columns(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    sources = [_download()]
    applied = acquire.apply(sources, acquire.acquire_sources(sources, cache_dir=tmp_path))
    block = provenance.prompt_block(applied)
    assert "REAL, already on disk at:" in block
    assert "Columns: city, pm25" in block
    assert "Liverpool" in block  # a real first row, not a described one
    assert "pandas.read_csv" in block
    assert "synthesize_" not in block


def test_provenance_document_records_the_checksum(monkeypatch, tmp_path):
    _serve(monkeypatch, [_csv_response()])
    sources = [_download()]
    applied = acquire.apply(sources, acquire.acquire_sources(sources, cache_dir=tmp_path))
    document = provenance.as_document(applied)
    assert document["all_inputs_real"] is True
    assert document["inputs"][0]["acquired"]["sha256"] == applied[0].acquired["sha256"]
    assert document["inputs"][0]["acquired"]["url"] == "https://example.org/air.csv"


def test_an_unfetched_input_writes_the_document_it_always_did():
    """`acquired` is additive: a plan nothing was fetched for produces exactly
    the provenance document this wrote before acquisition existed."""
    assert "acquired" not in provenance.as_document([_download()])["inputs"][0]


def test_check_hf_dataset_usage_accepts_reading_the_downloaded_file():
    """Once the dataset is on disk the prompt tells the model *not* to make an
    HTTP request, so the dataset id never appears in correct code."""
    hf_dataset = {"dataset_id": "owner/name"}
    code = "df = pandas.read_json('/cache/ab12/owner_name.jsonl', lines=True)"

    assert sandbox.check_hf_dataset_usage("", code, [], hf_dataset)  # flagged without the path
    assert (
        sandbox.check_hf_dataset_usage("", code, [], hf_dataset, ["/cache/ab12/owner_name.jsonl"])
        == []
    )


def test_local_paths_lists_every_acquired_file():
    assert acquire.local_paths({"https://a": {"path": "/x/a.csv"}}) == ["/x/a.csv"]
    assert acquire.local_paths({}) == []
    assert acquire.local_paths(None) == []


def test_describe_strips_a_byte_order_mark():
    """Government CSV exports are routinely BOM-prefixed. Decoding as plain
    utf-8 glues ﻿ onto the first column name, which then reaches the model
    as a column it is told to select and cannot find."""
    described = acquire.describe("﻿OBJECTID,name\n1,Wigan\n".encode())
    assert described is not None
    assert described[1] == ["OBJECTID", "name"]
    assert described[3][0] == {"OBJECTID": "1", "name": "Wigan"}


def test_fetch_accepts_any_2xx_and_lets_the_body_decide(monkeypatch, tmp_path):
    """Some open-data portals answer a CSV download with 202. `describe` is the
    real arbiter of whether a body is data, so the status check must not be the
    thing that discards a good file."""
    _serve(
        monkeypatch,
        [
            FakeResponse(
                status_code=202,
                headers={"content-type": "text/csv"},
                body=b"year,casualties\n2020,15\n",
            )
        ],
    )
    acquired = acquire.fetch("https://portal.example/d.csv", cache_dir=tmp_path)
    assert acquired is not None
    assert acquired.columns == ["year", "casualties"]


def test_a_2xx_with_a_non_data_body_is_still_rejected(monkeypatch, tmp_path):
    _serve(
        monkeypatch,
        [
            FakeResponse(
                status_code=202,
                headers={"content-type": "text/csv"},
                body=b"job queued, try again later\n",
            )
        ],
    )
    assert acquire.fetch("https://portal.example/d.csv", cache_dir=tmp_path) is None
