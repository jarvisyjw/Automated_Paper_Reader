import argparse
import json
import sys
import logging
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_arxiv import fetch_arxiv_html_recent
from daily_papers import compare_with_previous_candidates, run_pipeline
from utils import NetworkPreflightError, network_preflight_urls, write_json_atomic


def test_fetch_stage_generates_candidates_not_report(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
research_profile:
  language: zh
  positive_keywords:
    - time series
    - sequence modeling
  negative_keywords:
    - image generation
  arxiv_categories:
    - cs.LG
sources:
  arxiv:
    enabled: false
    lookback_days: 3
    max_results: 100
  openreview:
    enabled: false
    venues: []
    max_results: 100
output:
  report_dir: reports
  data_dir: data
  log_dir: logs
retrieval:
  candidate_limit: 80
""",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config=str(config_path),
        date="2026-05-14",
        stage="fetch",
        lookback_days=None,
        force=True,
        sources=None,
    )
    result = run_pipeline(args)

    candidates_path = Path(result["candidates_path"])
    assert candidates_path.exists()
    assert (tmp_path / "data" / "raw" / "2026-05-14.json").exists()
    assert not (tmp_path / "reports" / "2026-05-14.md").exists()
    assert not (tmp_path / "data" / "processed" / "2026-05-14_top10.json").exists()


def test_write_json_atomic_falls_back_when_replace_is_blocked(tmp_path, monkeypatch):
    target = tmp_path / "out.json"

    def deny_replace(*args, **kwargs):
        raise PermissionError("rename blocked by sandbox")

    monkeypatch.setattr("utils._direct_json_write_enabled", lambda: False)
    monkeypatch.setattr("utils.os.replace", deny_replace)

    write_json_atomic(target, {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_network_preflight_failure_exits_before_writing_empty_candidates(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
research_profile:
  positive_keywords: []
  negative_keywords: []
  arxiv_categories:
    - cs.LG
sources:
  arxiv:
    enabled: true
    lookback_days: 3
    max_results: 1
  openreview:
    enabled: false
output:
  report_dir: reports
  data_dir: data
  log_dir: logs
retrieval:
  candidate_limit: 80
""",
        encoding="utf-8",
    )

    def fail_preflight(urls, timeout=10, retries=1):
        raise NetworkPreflightError([{"url": urls[0], "error": "blocked"}])

    monkeypatch.setattr("daily_papers.check_network_preflight", fail_preflight)
    args = argparse.Namespace(
        config=str(config_path),
        date="2026-05-14",
        stage="fetch",
        lookback_days=None,
        force=True,
        sources=None,
    )

    with pytest.raises(NetworkPreflightError):
        run_pipeline(args)

    assert not (tmp_path / "data" / "processed" / "2026-05-14_candidates.json").exists()


def test_fetch_stage_records_duplicate_candidate_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
research_profile:
  positive_keywords:
    - time series
  negative_keywords: []
  arxiv_categories:
    - cs.LG
sources:
  arxiv:
    enabled: true
    lookback_days: 3
    max_results: 10
  openreview:
    enabled: false
output:
  report_dir: reports
  data_dir: data
  log_dir: logs
retrieval:
  candidate_limit: 10
""",
        encoding="utf-8",
    )
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    previous_candidates = [
        paper_fixture("2605.00001", "Useful Neural Time Series Method"),
        paper_fixture("2605.00002", "Another Sequence Modeling Paper"),
    ]
    write_json_atomic(processed_dir / "2026-05-14_candidates.json", previous_candidates)

    def fake_fetch_arxiv(source_config, research_profile, report_date, lookback, logger):
        return list(reversed(previous_candidates)), []

    monkeypatch.setattr("daily_papers.run_network_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("daily_papers.fetch_arxiv", fake_fetch_arxiv)

    args = argparse.Namespace(
        config=str(config_path),
        date="2026-05-15",
        stage="fetch",
        lookback_days=None,
        force=True,
        sources=None,
    )
    result = run_pipeline(args)

    assert result["duplicate_check"]["status"] == "duplicate_of_previous"
    assert result["duplicate_check"]["duplicate_of_date"] == "2026-05-14"
    assert result["duplicate_check"]["same_order"] is False
    raw_payload = json.loads((tmp_path / "data" / "raw" / "2026-05-15.json").read_text(encoding="utf-8"))
    assert raw_payload["duplicate_check"]["status"] == "duplicate_of_previous"


def test_fetch_stage_accepts_openalex_as_configured_source(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
research_profile:
  positive_keywords:
    - neural decoding
  negative_keywords: []
sources:
  arxiv:
    enabled: false
  openreview:
    enabled: false
  openalex:
    enabled: true
    lookback_days: 3
    max_results: 10
    search_queries:
      - neural decoding
output:
  report_dir: reports
  data_dir: data
  log_dir: logs
retrieval:
  candidate_limit: 10
""",
        encoding="utf-8",
    )

    def fake_fetch_openalex(source_config, research_profile, report_date, lookback, logger):
        return [openalex_paper_fixture("W123", "Adaptive Neural Decoding")], []

    monkeypatch.setattr("daily_papers.run_network_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("daily_papers.fetch_openalex", fake_fetch_openalex)

    args = argparse.Namespace(
        config=str(config_path),
        date="2026-05-15",
        stage="fetch",
        lookback_days=None,
        force=True,
        sources=None,
    )
    result = run_pipeline(args)
    raw_payload = json.loads((tmp_path / "data" / "raw" / "2026-05-15.json").read_text(encoding="utf-8"))
    candidates = json.loads((tmp_path / "data" / "processed" / "2026-05-15_candidates.json").read_text())

    assert result["source_counts"] == {"openalex": 1}
    assert raw_payload["sources"] == ["openalex"]
    assert candidates[0]["source"] == "openalex"


def test_network_preflight_includes_openalex_when_enabled():
    urls = network_preflight_urls(
        {
            "sources": {
                "arxiv": {"enabled": False},
                "openreview": {"enabled": False},
                "openalex": {"enabled": True},
            }
        },
        ["openalex"],
    )

    assert urls == ["https://api.openalex.org/"]


def test_compare_with_previous_candidates_reports_changes(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    write_json_atomic(
        processed_dir / "2026-05-14_candidates.json",
        [paper_fixture("2605.00001", "Old Paper")],
    )

    result = compare_with_previous_candidates(
        [paper_fixture("2605.00002", "New Paper")],
        processed_dir,
        date(2026, 5, 15),
    )

    assert result["status"] == "changed"
    assert result["new_count"] == 1
    assert result["removed_count"] == 1


def test_fetch_stage_recommends_no_new_note_for_empty_fresh_batch(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
research_profile:
  positive_keywords: []
  negative_keywords: []
  arxiv_categories:
    - cs.LG
sources:
  arxiv:
    enabled: true
    lookback_days: 3
    max_results: 10
  openreview:
    enabled: false
output:
  report_dir: reports
  data_dir: data
  log_dir: logs
retrieval:
  candidate_limit: 10
""",
        encoding="utf-8",
    )

    monkeypatch.setattr("daily_papers.run_network_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("daily_papers.fetch_arxiv", lambda *args, **kwargs: ([], []))

    args = argparse.Namespace(
        config=str(config_path),
        date="2026-05-18",
        stage="fetch",
        lookback_days=None,
        force=True,
        sources=None,
    )
    result = run_pipeline(args)
    raw_payload = json.loads((tmp_path / "data" / "raw" / "2026-05-18.json").read_text(encoding="utf-8"))

    assert result["candidate_count"] == 0
    assert result["recommended_action"] == "write_no_new_batch_note"
    assert raw_payload["recommended_action"] == "write_no_new_batch_note"
    assert json.loads((tmp_path / "data" / "processed" / "2026-05-18_candidates.json").read_text()) == []


def test_arxiv_html_fallback_uses_target_recent_batch_when_abs_dates_lag(monkeypatch):
    recent_html = """
    <dl id="articles">
      <h3>Fri, 15 May 2026 (showing first 25 of 261 entries )</h3>
      <dt><a href="/abs/2605.15188" title="Abstract">arXiv:2605.15188</a></dt>
    </dl>
    """
    abs_html = """
    <meta name="citation_title" content="Weekend-safe fallback paper" />
    <meta name="citation_author" content="Alice" />
    <meta name="citation_date" content="2026/05/14" />
    <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2605.15188" />
    <meta name="citation_abstract" content="An abstract about temporal adaptation." />
    <td class="tablecell subjects">
      <span class="primary-subject">Machine Learning (cs.LG)</span>
    </td>
    """

    def fake_http_get_text(url, retries=0, retry_after_seconds=60):
        if "list/cs.LG/recent" in url:
            return recent_html
        if "abs/2605.15188" in url:
            return abs_html
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("fetch_arxiv.http_get_text", fake_http_get_text)

    papers, warnings = fetch_arxiv_html_recent(
        {
            "html_fallback_per_category": 25,
            "html_fallback_max_results": 10,
            "html_fallback_sleep_seconds": 0,
        },
        ["cs.LG"],
        date(2026, 5, 15),
        lookback_days=3,
        max_results=10,
        logger=logging.getLogger("test"),
    )

    assert warnings == []
    assert len(papers) == 1
    assert papers[0]["published_at"].startswith("2026-05-15")
    assert papers[0]["recent_listed_at"].startswith("2026-05-15")


def test_arxiv_html_fallback_does_not_reuse_old_batch(monkeypatch):
    recent_html = """
    <dl id="articles">
      <h3>Fri, 15 May 2026 (showing 1 of 1 entries )</h3>
      <dt><a href="/abs/2605.15188" title="Abstract">arXiv:2605.15188</a></dt>
    </dl>
    """

    def fake_http_get_text(url, retries=0, retry_after_seconds=60):
        if "list/cs.LG/recent" in url:
            return recent_html
        if "abs/" in url:
            raise AssertionError("old recent-list batch should not be reused")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("fetch_arxiv.http_get_text", fake_http_get_text)

    papers, warnings = fetch_arxiv_html_recent(
        {
            "html_fallback_per_category": 25,
            "html_fallback_max_results": 10,
            "html_fallback_sleep_seconds": 0,
        },
        ["cs.LG"],
        date(2026, 5, 18),
        lookback_days=3,
        max_results=10,
        logger=logging.getLogger("test_no_reuse"),
    )

    assert warnings == []
    assert papers == []


def paper_fixture(paper_id, title):
    return {
        "id": paper_id,
        "source": "arxiv",
        "title": title,
        "authors": ["Alice"],
        "abstract": "A method for neural time series and sequence modeling.",
        "url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
        "published_at": "2026-05-14T00:00:00Z",
        "updated_at": "2026-05-14T00:00:00Z",
        "venue": "cs.LG",
        "categories": ["cs.LG"],
    }


def openalex_paper_fixture(paper_id, title):
    paper = paper_fixture(paper_id, title)
    paper.update(
        {
            "source": "openalex",
            "url": f"https://openalex.org/{paper_id}",
            "pdf_url": "",
            "venue": "OpenAlex Test Venue",
            "categories": ["Brain-computer interface"],
        }
    )
    return paper
