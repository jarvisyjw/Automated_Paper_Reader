"""OpenAlex retrieval and normalization."""

from __future__ import annotations

import os
import time as time_module
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode

from utils import (
    isoformat_or_empty,
    normalize_title,
    normalize_whitespace,
    requests_get,
    unique_preserve_order,
    validate_paper_schema,
)


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_date",
        "updated_date",
        "authorships",
        "abstract_inverted_index",
        "primary_location",
        "best_oa_location",
        "locations",
        "topics",
        "concepts",
        "type",
        "cited_by_count",
        "open_access",
    ]
)


def fetch_openalex(
    source_config: dict[str, Any],
    research_profile: dict[str, Any],
    target_date: date,
    lookback_days: int,
    logger,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch recent works from OpenAlex. Network errors are returned as warnings."""

    warnings: list[str] = []
    if not source_config.get("enabled", True):
        return [], warnings

    max_results = int(source_config.get("max_results", 100))
    queries = build_openalex_queries(source_config, research_profile)
    if not queries:
        warning = "OpenAlex source is enabled but no search queries could be built."
        warnings.append(warning)
        logger.warning(warning)
        return [], warnings

    logger.info("Fetching OpenAlex works, queries=%s, max_results=%s", len(queries), max_results)
    papers: list[dict[str, Any]] = []
    sleep_seconds = float(source_config.get("sleep_seconds", 0.2))

    for query in queries:
        if len(papers) >= max_results:
            break
        try:
            query_papers = fetch_openalex_query(
                query,
                source_config,
                target_date,
                lookback_days,
                max_results=max_results - len(papers),
            )
            papers.extend(query_papers)
        except Exception as exc:
            warning = f"OpenAlex query failed for {query!r}: {exc}"
            warnings.append(warning)
            logger.warning(warning)
        if sleep_seconds > 0:
            time_module.sleep(sleep_seconds)

    papers = dedupe_openalex_results(papers)[:max_results]
    logger.info("Fetched %s OpenAlex works after date filtering", len(papers))
    return papers, warnings


def build_openalex_queries(source_config: dict[str, Any], research_profile: dict[str, Any]) -> list[str]:
    configured = source_config.get("search_queries")
    if configured:
        if isinstance(configured, str):
            return unique_preserve_order([configured])
        return unique_preserve_order(str(query) for query in configured)

    max_queries = int(source_config.get("max_auto_queries", 5))
    profile_terms = list(research_profile.get("background_topics", [])) + list(
        research_profile.get("positive_keywords", [])
    )
    terms = [normalize_whitespace(str(term)) for term in profile_terms]
    terms = [term for term in terms if term]
    return unique_preserve_order(terms)[:max_queries]


def fetch_openalex_query(
    query: str,
    source_config: dict[str, Any],
    target_date: date,
    lookback_days: int,
    max_results: int,
) -> list[dict[str, Any]]:
    start_date = target_date - timedelta(days=max(1, int(lookback_days)) - 1)
    filters = [
        f"from_publication_date:{start_date.isoformat()}",
        f"to_publication_date:{target_date.isoformat()}",
    ]
    work_types = source_config.get("types", [])
    if work_types:
        filters.append("type:" + "|".join(normalize_whitespace(str(item)) for item in work_types if item))

    params = {
        "search": query,
        "filter": ",".join(filters),
        "per_page": min(max(1, int(source_config.get("per_page", 50))), max(1, max_results)),
        "sort": str(source_config.get("sort", "publication_date:desc")),
        "select": OPENALEX_SELECT_FIELDS,
    }
    mailto = normalize_whitespace(str(source_config.get("mailto") or ""))
    if mailto:
        params["mailto"] = mailto
    api_key_env = normalize_whitespace(str(source_config.get("api_key_env") or "OPENALEX_API_KEY"))
    api_key = normalize_whitespace(str(source_config.get("api_key") or os.environ.get(api_key_env, "")))
    if api_key:
        params["api_key"] = api_key

    data = openalex_get_json(
        f"{OPENALEX_WORKS_URL}?{urlencode(params)}",
        retries=int(source_config.get("retries", 1)),
        retry_after_seconds=int(source_config.get("retry_after_seconds", 10)),
        timeout=int(source_config.get("timeout_seconds", 30)),
    )
    results = data.get("results", []) if isinstance(data, dict) else []
    papers: list[dict[str, Any]] = []
    for work in results:
        paper = normalize_openalex_work(work)
        if paper.get("title"):
            papers.append(paper)
    return papers


def openalex_get_json(
    url: str,
    retries: int = 1,
    retry_after_seconds: int = 10,
    timeout: int = 30,
) -> dict[str, Any]:
    last_response = None
    for attempt in range(max(1, retries + 1)):
        response = requests_get(url, timeout=timeout)
        last_response = response
        if response.status_code != 429 or attempt >= retries:
            response.raise_for_status()
            return response.json()
        retry_after = response.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else max(1, retry_after_seconds)
        time_module.sleep(delay)

    assert last_response is not None
    last_response.raise_for_status()
    return last_response.json()


def normalize_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex work into the project paper schema."""

    openalex_id = normalize_whitespace(str(work.get("id", "")))
    short_id = openalex_id.rstrip("/").rsplit("/", 1)[-1] if openalex_id else ""
    title = normalize_whitespace(str(work.get("display_name") or work.get("title") or ""))
    primary_location = work.get("primary_location") or {}
    best_oa_location = work.get("best_oa_location") or {}
    if not isinstance(primary_location, dict):
        primary_location = {}
    if not isinstance(best_oa_location, dict):
        best_oa_location = {}

    doi = normalize_whitespace(str(work.get("doi") or ""))
    paper = {
        "id": short_id,
        "source": "openalex",
        "title": title,
        "authors": extract_openalex_authors(work.get("authorships", [])),
        "abstract": openalex_abstract(work),
        "url": first_nonempty(
            doi,
            str(primary_location.get("landing_page_url") or ""),
            str(best_oa_location.get("landing_page_url") or ""),
            openalex_id,
        ),
        "pdf_url": first_openalex_pdf_url(work),
        "published_at": isoformat_or_empty(work.get("publication_date")),
        "updated_at": isoformat_or_empty(work.get("updated_date") or work.get("publication_date")),
        "venue": openalex_location_source_name(primary_location) or openalex_location_source_name(best_oa_location),
        "categories": extract_openalex_categories(work),
        "doi": doi,
        "openalex_url": openalex_id,
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "openalex_type": normalize_whitespace(str(work.get("type") or "")),
    }
    return validate_paper_schema(paper)


def extract_openalex_authors(authorships: Any) -> list[str]:
    authors: list[str] = []
    if not isinstance(authorships, list):
        return authors
    for item in authorships:
        if not isinstance(item, dict):
            continue
        author = item.get("author") or {}
        if isinstance(author, dict):
            name = normalize_whitespace(str(author.get("display_name") or ""))
            if name:
                authors.append(name)
    return unique_preserve_order(authors)


def openalex_abstract(work: dict[str, Any]) -> str:
    abstract = normalize_whitespace(str(work.get("abstract") or ""))
    if abstract:
        return abstract
    return abstract_from_inverted_index(work.get("abstract_inverted_index"))


def abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: dict[int, str] = {}
    for word, raw_indexes in index.items():
        if not isinstance(raw_indexes, list):
            continue
        for raw_index in raw_indexes:
            try:
                positions[int(raw_index)] = str(word)
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    return normalize_whitespace(" ".join(positions[idx] for idx in sorted(positions)))


def first_openalex_pdf_url(work: dict[str, Any]) -> str:
    for location in [work.get("best_oa_location"), work.get("primary_location")]:
        if isinstance(location, dict):
            pdf_url = normalize_whitespace(str(location.get("pdf_url") or ""))
            if pdf_url:
                return pdf_url
    locations = work.get("locations", [])
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict):
                pdf_url = normalize_whitespace(str(location.get("pdf_url") or ""))
                if pdf_url:
                    return pdf_url
    return ""


def openalex_location_source_name(location: dict[str, Any]) -> str:
    source = location.get("source") or {}
    if isinstance(source, dict):
        return normalize_whitespace(str(source.get("display_name") or ""))
    return ""


def extract_openalex_categories(work: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for key in ("topics", "concepts"):
        values = work.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                name = normalize_whitespace(str(value.get("display_name") or ""))
                if name:
                    categories.append(name)
    return unique_preserve_order(categories)


def first_nonempty(*values: str) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def dedupe_openalex_results(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for paper in papers:
        key = str(paper.get("id") or paper.get("doi") or normalize_title(paper.get("title")))
        if key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result
