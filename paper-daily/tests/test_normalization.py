import logging
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_arxiv import (
    fetch_arxiv_html_recent,
    normalize_arxiv_abs_html,
    normalize_arxiv_entry,
    parse_arxiv_entries_xml,
    parse_arxiv_recent_heading_date,
    parse_arxiv_recent_ids,
    parse_arxiv_recent_refs,
)
from fetch_openreview import normalize_openreview_note
from fetch_openalex import abstract_from_inverted_index, normalize_openalex_work


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_arxiv_schema_normalization():
    entry = Obj(
        id="https://arxiv.org/abs/2605.12345v1",
        title=" Sensor Forecasting with Sequence Modeling ",
        summary="We forecast sensor signals with temporal representations.",
        published="2026-05-14T00:00:00Z",
        updated="2026-05-14T01:00:00Z",
        authors=[{"name": "Alice"}, {"name": "Bob"}],
        links=[
            {"rel": "alternate", "href": "https://arxiv.org/abs/2605.12345v1"},
            {"type": "application/pdf", "href": "https://arxiv.org/pdf/2605.12345v1"},
        ],
        tags=[{"term": "cs.LG"}, {"term": "q-bio.NC"}],
    )

    paper = normalize_arxiv_entry(entry)

    assert paper["source"] == "arxiv"
    assert paper["id"] == "2605.12345v1"
    assert paper["title"] == "Sensor Forecasting with Sequence Modeling"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["pdf_url"].endswith("2605.12345v1")
    assert paper["categories"] == ["cs.LG", "q-bio.NC"]


def test_openreview_schema_normalization():
    note = Obj(
        id="abc123",
        forum="forum123",
        cdate=1778716800000,
        mdate=1778720400000,
        content={
            "title": {"value": "Language Model Assisted Time-Series Forecasting"},
            "authors": {"value": ["Carol", "Dave"]},
            "abstract": {"value": "A method for adaptive sequence forecasting."},
            "keywords": {"value": ["time series", "sequence modeling"]},
        },
    )

    paper = normalize_openreview_note(note, "ICLR")

    assert paper["source"] == "openreview"
    assert paper["id"] == "abc123"
    assert paper["venue"] == "ICLR"
    assert paper["authors"] == ["Carol", "Dave"]
    assert "openreview.net/forum" in paper["url"]
    assert "time series" in paper["categories"]


def test_arxiv_stdlib_xml_parser_supports_real_feed_shape():
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2605.12345v1</id>
        <updated>2026-05-14T01:00:00Z</updated>
        <published>2026-05-14T00:00:00Z</published>
        <title>Sequence Modeling for Sensor Forecasting</title>
        <summary>We forecast sensor signals.</summary>
        <author><name>Alice</name></author>
        <arxiv:primary_category term="q-bio.NC" />
        <category term="cs.LG" />
        <link href="http://arxiv.org/abs/2605.12345v1" rel="alternate" type="text/html" />
        <link href="http://arxiv.org/pdf/2605.12345v1" rel="related" type="application/pdf" title="pdf" />
      </entry>
    </feed>
    """

    entries = parse_arxiv_entries_xml(feed)
    paper = normalize_arxiv_entry(entries[0])

    assert paper["id"] == "2605.12345v1"
    assert paper["authors"] == ["Alice"]
    assert paper["categories"] == ["q-bio.NC", "cs.LG"]
    assert paper["pdf_url"].endswith("2605.12345v1")


def test_arxiv_html_fallback_parsers_support_recent_and_abs_pages():
    recent_html = """
    <dl id='articles'>
      <h3>Fri, 15 May 2026 (showing first 25 of 261 entries )</h3>
      <dt><a href ="/abs/2605.15188" title="Abstract">arXiv:2605.15188</a></dt>
      <dt><a href="/abs/2605.15188" title="Abstract">duplicate</a></dt>
      <dt><a href="/abs/2605.15183v2" title="Abstract">arXiv:2605.15183</a></dt>
    </dl>
    """
    assert parse_arxiv_recent_ids(recent_html) == ["2605.15188", "2605.15183"]
    assert parse_arxiv_recent_heading_date("Fri, 15 May 2026 (showing first 25 of 261 entries )").startswith(
        "2026-05-15"
    )
    assert parse_arxiv_recent_refs(recent_html) == [
        ("2605.15188", "2026-05-15T00:00:00+00:00"),
        ("2605.15183", "2026-05-15T00:00:00+00:00"),
    ]

    abs_html = """
    <meta name="citation_title" content="A Test Paper" />
    <meta name="citation_author" content="Alice" />
    <meta name="citation_author" content="Bob" />
    <meta name="citation_date" content="2026/05/14" />
    <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2605.15188" />
    <meta name="citation_abstract" content="An abstract about neural time series." />
    <td class="tablecell subjects">
      <span class="primary-subject">Machine Learning (cs.LG)</span>; Signal Processing (eess.SP)
    </td>
    """
    paper = normalize_arxiv_abs_html("2605.15188", abs_html, recent_listed_at="2026-05-15T00:00:00+00:00")
    assert paper["title"] == "A Test Paper"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["categories"] == ["cs.LG", "eess.SP"]
    assert paper["published_at"].startswith("2026-05-15")
    assert paper["abs_page_published_at"].startswith("2026-05-14")


def test_arxiv_html_recent_uses_listing_date_when_abs_metadata_lags(monkeypatch):
    recent_html = """
    <dl id="articles">
      <h3>Fri, 15 May 2026 (showing 1 of 1 entries )</h3>
      <dt><a href="/abs/2605.15188" title="Abstract">arXiv:2605.15188</a></dt>
    </dl>
    """
    abs_html = """
    <meta name="citation_title" content="A Test Paper" />
    <meta name="citation_author" content="Alice" />
    <meta name="citation_date" content="2026/05/14" />
    <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2605.15188" />
    <meta name="citation_abstract" content="An abstract about neural time series." />
    <td class="tablecell subjects">
      <span class="primary-subject">Machine Learning (cs.LG)</span>
    </td>
    """

    def fake_http_get_text(url, retries=2, retry_after_seconds=60):
        if "/list/" in url:
            return recent_html
        if "/abs/2605.15188" in url:
            return abs_html
        raise AssertionError(url)

    monkeypatch.setattr("fetch_arxiv.http_get_text", fake_http_get_text)

    papers, warnings = fetch_arxiv_html_recent(
        {
            "html_fallback_per_category": 25,
            "html_fallback_max_results": 10,
            "html_fallback_sleep_seconds": 0,
        },
        ["cs.LG"],
        date(2026, 5, 15),
        3,
        10,
        logging.getLogger("test_arxiv_html_recent"),
    )

    assert warnings == []
    assert [paper["id"] for paper in papers] == ["2605.15188"]
    assert papers[0]["published_at"].startswith("2026-05-15")
    assert papers[0]["recent_listed_at"].startswith("2026-05-15")


def test_arxiv_html_recent_does_not_reuse_latest_batch_when_no_new_listing_exists(monkeypatch):
    recent_html = """
    <dl id="articles">
      <h3>Fri, 15 May 2026 (showing 1 of 1 entries )</h3>
      <dt><a href="/abs/2605.15188" title="Abstract">arXiv:2605.15188</a></dt>
    </dl>
    """
    abs_html = """
    <meta name="citation_title" content="A Test Paper" />
    <meta name="citation_author" content="Alice" />
    <meta name="citation_date" content="2026/05/14" />
    <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2605.15188" />
    <meta name="citation_abstract" content="An abstract about neural time series." />
    <td class="tablecell subjects">
      <span class="primary-subject">Machine Learning (cs.LG)</span>
    </td>
    """

    def fake_http_get_text(url, retries=2, retry_after_seconds=60):
        if "/list/" in url:
            return recent_html
        if "/abs/2605.15188" in url:
            return abs_html
        if "/abs/" in url:
            raise AssertionError("old batch should not be fetched")
        raise AssertionError(url)

    monkeypatch.setattr("fetch_arxiv.http_get_text", fake_http_get_text)

    papers, warnings = fetch_arxiv_html_recent(
        {
            "html_fallback_per_category": 25,
            "html_fallback_max_results": 10,
            "html_fallback_sleep_seconds": 0,
        },
        ["cs.LG"],
        date(2026, 5, 18),
        3,
        10,
        logging.getLogger("test_arxiv_weekend_fallback"),
    )

    assert warnings == []
    assert papers == []


def test_openreview_dict_note_normalization_for_http_api_fallback():
    note = {
        "id": "dict123",
        "forum": "forum456",
        "cdate": 1778716800000,
        "mdate": 1778720400000,
        "content": {
            "title": {"value": "Adaptive Sensor Forecasting"},
            "authors": {"value": ["Eve"]},
            "abstract": {"value": "OpenReview HTTP API shape."},
            "keywords": {"value": ["time series", "forecasting"]},
        },
    }

    paper = normalize_openreview_note(note, "TMLR")

    assert paper["id"] == "dict123"
    assert paper["title"] == "Adaptive Sensor Forecasting"
    assert paper["authors"] == ["Eve"]
    assert "time series" in paper["categories"]


def test_openalex_work_normalization():
    work = {
        "id": "https://openalex.org/W123456789",
        "doi": "https://doi.org/10.1000/test",
        "display_name": "Adaptive Neural Decoding with EEG Signals",
        "publication_date": "2026-05-15",
        "updated_date": "2026-05-16T12:00:00Z",
        "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ],
        "abstract_inverted_index": {
            "We": [0],
            "decode": [1],
            "neural": [2],
            "signals": [3],
        },
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "source": {"display_name": "Example Journal"},
        },
        "best_oa_location": {
            "pdf_url": "https://example.org/paper.pdf",
        },
        "topics": [{"display_name": "Brain-computer interface"}],
        "concepts": [{"display_name": "Electroencephalography"}],
        "type": "article",
        "cited_by_count": 7,
    }

    paper = normalize_openalex_work(work)

    assert paper["source"] == "openalex"
    assert paper["id"] == "W123456789"
    assert paper["title"] == "Adaptive Neural Decoding with EEG Signals"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["abstract"] == "We decode neural signals"
    assert paper["url"] == "https://doi.org/10.1000/test"
    assert paper["pdf_url"] == "https://example.org/paper.pdf"
    assert paper["published_at"].startswith("2026-05-15")
    assert paper["venue"] == "Example Journal"
    assert "Brain-computer interface" in paper["categories"]
    assert paper["doi"] == "https://doi.org/10.1000/test"
    assert paper["cited_by_count"] == 7


def test_openalex_abstract_from_inverted_index_skips_bad_positions():
    abstract = abstract_from_inverted_index({"Adaptive": [0], "decoding": ["1"], "bad": ["x"]})

    assert abstract == "Adaptive decoding"
