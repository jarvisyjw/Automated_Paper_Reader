"""Fetch candidate papers for Codex-assisted daily literature review.

This script intentionally stops at candidate retrieval. It does not choose the
final Top 10 and does not write a Markdown report; Codex should do that after
reading and scoring the candidate pool.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fetch_arxiv import fetch_arxiv
from fetch_openalex import fetch_openalex
from fetch_openreview import fetch_openreview
from rank_papers import build_candidate_pool
from utils import (
    dedupe_papers,
    ensure_dirs,
    check_network_preflight,
    load_config,
    network_preflight_urls,
    NetworkPreflightError,
    normalize_title,
    parse_target_date,
    read_json,
    resolve_output_paths,
    setup_logger,
    split_csv,
    stable_source_key,
    write_json_atomic,
)


class FetchStageError(RuntimeError):
    """Raised when source fetches fail and writing empty candidates would be misleading."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch paper candidates for Codex review.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--date", default="today", help="'today' or YYYY-MM-DD")
    parser.add_argument(
        "--stage",
        default="fetch",
        choices=["fetch"],
        help="Only 'fetch' is supported; final reports are written by Codex, not this script.",
    )
    parser.add_argument("--lookback-days", type=int, default=None, help="Override source lookback days")
    parser.add_argument("--force", action="store_true", help="Overwrite existing candidate outputs")
    parser.add_argument("--sources", default=None, help="Comma-separated source list, e.g. arxiv,openreview,openalex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_pipeline(args)
    except (NetworkPreflightError, FetchStageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Candidates: {result['candidates_path']}")
    print(f"Raw: {result['raw_path']}")
    duplicate_check = result.get("duplicate_check") or {}
    if duplicate_check.get("status") == "duplicate_of_previous":
        print(
            "Duplicate candidate set: "
            f"{duplicate_check.get('duplicate_of_date')} "
            f"({duplicate_check.get('shared_count', 0)} unchanged ids)"
        )
        print("Action: write a no-new-batch note instead of repeating Top 10")
    elif result.get("recommended_action") == "write_no_new_batch_note":
        print("No new candidates for the target date.")
        print("Action: write a no-new-batch note instead of a Top 10 report")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "stage", "fetch") != "fetch":
        raise ValueError("Only --stage fetch is supported. Final reports must be written by Codex.")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    report_date = parse_target_date(args.date)
    paths = resolve_output_paths(config_path, config)
    ensure_dirs([paths["raw_dir"], paths["processed_dir"], paths["report_dir"], paths["log_dir"]])

    date_str = report_date.isoformat()
    raw_path = paths["raw_dir"] / f"{date_str}.json"
    candidates_path = paths["processed_dir"] / f"{date_str}_candidates.json"
    log_path = paths["log_dir"] / f"{date_str}.log"
    logger = setup_logger(log_path)
    enabled_sources = resolve_enabled_sources(args, config)

    if candidates_path.exists() and not args.force:
        logger.info("Candidate file already exists and --force was not set: %s", candidates_path)
        return {
            "skipped": True,
            "raw_path": str(raw_path),
            "candidates_path": str(candidates_path),
            "log_path": str(log_path),
        }

    run_network_preflight(config, enabled_sources, logger)

    logger.info("Starting fetch-only candidate pipeline for %s", date_str)
    research_profile = config.get("research_profile", {})

    source_counts: dict[str, int] = {}
    source_warnings: dict[str, list[str]] = {}
    all_papers: list[dict[str, Any]] = []

    if "arxiv" in enabled_sources:
        source_config = config.get("sources", {}).get("arxiv", {})
        lookback = args.lookback_days or int(source_config.get("lookback_days", 3))
        papers, warnings = fetch_arxiv(source_config, research_profile, report_date, lookback, logger)
        all_papers.extend(papers)
        source_counts["arxiv"] = len(papers)
        source_warnings["arxiv"] = warnings

    if "openreview" in enabled_sources:
        source_config = config.get("sources", {}).get("openreview", {})
        lookback = args.lookback_days or int(source_config.get("lookback_days", 3))
        papers, warnings = fetch_openreview(source_config, report_date, lookback, logger)
        all_papers.extend(papers)
        source_counts["openreview"] = len(papers)
        source_warnings["openreview"] = warnings

    if "openalex" in enabled_sources:
        source_config = config.get("sources", {}).get("openalex", {})
        lookback = args.lookback_days or int(source_config.get("lookback_days", 3))
        papers, warnings = fetch_openalex(source_config, research_profile, report_date, lookback, logger)
        all_papers.extend(papers)
        source_counts["openalex"] = len(papers)
        source_warnings["openalex"] = warnings

    deduped = dedupe_papers(all_papers)
    logger.info("Total papers fetched=%s, deduped=%s", len(all_papers), len(deduped))

    if not deduped and source_had_failures(source_warnings):
        existing_count = existing_candidate_count(candidates_path)
        if existing_count > 0:
            message = (
                "All source fetches failed or returned zero papers; "
                f"preserving existing candidate file with {existing_count} papers: {candidates_path}"
            )
            logger.error(message)
            raise FetchStageError(message)
        message = (
            "All enabled source fetches failed or returned zero papers with warnings; "
            "aborting before writing an empty candidate file. Check network/proxy and source warnings in "
            f"{log_path}"
        )
        logger.error(message)
        raise FetchStageError(message)

    candidate_limit = int(config.get("retrieval", {}).get("candidate_limit", 80))
    candidates = build_candidate_pool(
        deduped,
        research_profile,
        report_date,
        candidate_limit=candidate_limit,
    )
    duplicate_check = compare_with_previous_candidates(candidates, paths["processed_dir"], report_date)
    if duplicate_check.get("status") == "duplicate_of_previous":
        logger.warning(
            "Candidate set is identical to %s; downstream report should not repeat Top 10.",
            duplicate_check.get("duplicate_of_date"),
        )
    recommended_action = recommended_downstream_action(candidates, duplicate_check)

    raw_payload = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "fetch",
        "sources": list(enabled_sources),
        "source_counts": source_counts,
        "warnings": source_warnings,
        "paper_count": len(deduped),
        "candidate_count": len(candidates),
        "duplicate_check": duplicate_check,
        "recommended_action": recommended_action,
        "papers": deduped,
    }
    write_json_atomic(raw_path, raw_payload)

    write_json_atomic(candidates_path, candidates)
    logger.info("Candidate pool written with %s papers: %s", len(candidates), candidates_path)
    logger.info("Fetch stage complete; no report was generated by the script.")

    return {
        "skipped": False,
        "raw_path": str(raw_path),
        "candidates_path": str(candidates_path),
        "log_path": str(log_path),
        "candidate_count": len(candidates),
        "source_counts": source_counts,
        "duplicate_check": duplicate_check,
        "recommended_action": recommended_action,
    }


def resolve_enabled_sources(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.sources:
        requested = split_csv(args.sources)
    else:
        requested = [
            name
            for name, source_config in config.get("sources", {}).items()
            if source_config.get("enabled", True)
        ]
    supported = {"arxiv", "openreview", "openalex"}
    return [source for source in requested if source in supported]


def run_network_preflight(config: dict[str, Any], enabled_sources: list[str], logger: Any) -> None:
    urls = network_preflight_urls(config, enabled_sources)
    if not urls:
        logger.info("Network preflight skipped; no enabled network sources.")
        return

    timeout = int(config.get("network", {}).get("preflight_timeout_seconds", 20))
    retries = int(config.get("network", {}).get("preflight_retries", 1))
    logger.info("Running network preflight for %s", ", ".join(urls))
    try:
        check_network_preflight(urls, timeout=timeout, retries=retries)
    except NetworkPreflightError as exc:
        logger.error("Network preflight failed; aborting before fetch. %s", exc)
        raise
    logger.info("Network preflight passed.")


def source_had_failures(source_warnings: dict[str, list[str]]) -> bool:
    return any(bool(warnings) for warnings in source_warnings.values())


def recommended_downstream_action(candidates: list[dict[str, Any]], duplicate_check: dict[str, Any]) -> str:
    if not candidates:
        return "write_no_new_batch_note"
    if duplicate_check.get("recommended_action") == "write_no_new_batch_note":
        return "write_no_new_batch_note"
    return "score_and_write_report"


def existing_candidate_count(path: Path) -> int:
    try:
        data = read_json(path)
    except Exception:
        return 0
    return len(data) if isinstance(data, list) else 0


def compare_with_previous_candidates(
    candidates: list[dict[str, Any]],
    processed_dir: Path,
    report_date: date,
) -> dict[str, Any]:
    """Detect whether today's final candidate pool only repeats the previous batch."""

    current_ids = candidate_identities(candidates)
    if not current_ids:
        return {
            "status": "no_current_candidates",
            "current_count": 0,
            "previous_candidates_path": "",
            "recommended_action": "write_no_new_batch_note",
        }

    previous_path = find_previous_candidates_path(processed_dir, report_date)
    if previous_path is None:
        return {
            "status": "no_previous_candidate_file",
            "current_count": len(current_ids),
            "previous_candidates_path": "",
            "recommended_action": "score_and_write_report",
        }

    try:
        previous_data = read_json(previous_path)
    except Exception as exc:
        return {
            "status": "previous_candidate_read_error",
            "current_count": len(current_ids),
            "previous_candidates_path": str(previous_path),
            "error": str(exc),
            "recommended_action": "score_and_write_report",
        }

    previous_candidates = previous_data if isinstance(previous_data, list) else []
    previous_ids = candidate_identities(previous_candidates)
    previous_date = previous_candidate_date(previous_path)
    current_counter = Counter(current_ids)
    previous_counter = Counter(previous_ids)
    duplicate = current_counter == previous_counter and bool(previous_counter)
    added = list((current_counter - previous_counter).elements())
    removed = list((previous_counter - current_counter).elements())

    result = {
        "status": "duplicate_of_previous" if duplicate else "changed",
        "current_count": len(current_ids),
        "previous_count": len(previous_ids),
        "shared_count": sum((current_counter & previous_counter).values()),
        "new_count": len(added),
        "removed_count": len(removed),
        "same_order": current_ids == previous_ids,
        "previous_candidates_path": str(previous_path),
        "duplicate_of_date": previous_date.isoformat() if duplicate and previous_date else "",
        "recommended_action": "write_no_new_batch_note" if duplicate else "score_and_write_report",
    }
    if added:
        result["sample_new_ids"] = added[:10]
    if removed:
        result["sample_removed_ids"] = removed[:10]
    return result


def candidate_identities(candidates: list[dict[str, Any]]) -> list[str]:
    identities: list[str] = []
    for paper in candidates:
        identity = candidate_identity(paper)
        if identity:
            identities.append(identity)
    return identities


def candidate_identity(paper: dict[str, Any]) -> str:
    source_key = stable_source_key(paper)
    if source_key:
        return source_key
    title_key = normalize_title(str(paper.get("title", "")))
    return f"title:{title_key}" if title_key else ""


def find_previous_candidates_path(processed_dir: Path, report_date: date) -> Path | None:
    previous: list[tuple[date, Path]] = []
    for path in processed_dir.glob("*_candidates.json"):
        path_date = previous_candidate_date(path)
        if path_date and path_date < report_date:
            previous.append((path_date, path))
    if not previous:
        return None
    return max(previous, key=lambda item: item[0])[1]


def previous_candidate_date(path: Path) -> date | None:
    suffix = "_candidates.json"
    if not path.name.endswith(suffix):
        return None
    raw_date = path.name[: -len(suffix)]
    try:
        return datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        return None


if __name__ == "__main__":
    main()
