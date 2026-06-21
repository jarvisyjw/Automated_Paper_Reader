"""Generate scored report via OpenAI-compatible LLM API.

Reads the candidate pool produced by daily_papers.py, calls an OpenAI-compatible
LLM API to semantically score papers and produce a Markdown daily report, then
writes the results to data/processed/YYYY-MM-DD_scored.json and
reports/YYYY-MM-DD.md.

Environment variables:
    OPENAI_API_BASE       Base URL for the API (e.g. https://api.openai.com/v1)
    OPENAI_API_KEY        API key
    OPENAI_MODEL_NAME     Model name (e.g. gpt-4o, qwen-turbo)
    PAPER_DAILY_USE_ENV_PROXY=1  Opt-in for environment proxy (see utils.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from io import BytesIO
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from utils import (
    load_config,
    read_json,
    setup_logger,
    resolve_output_paths,
    should_use_env_proxy,
)

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_LLM_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_LLM_BATCH_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM-scored paper report.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--date", default="today", help="'today' or YYYY-MM-DD")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit candidates sent to the LLM; defaults to OPENAI_LLM_LIMIT or 20",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="LLM request timeout; defaults to OPENAI_TIMEOUT_SECONDS or 1800",
    )
    return parser.parse_args()


def get_env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val else default


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint and return the response text."""
    import requests as _requests

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    session = _requests.Session()
    session.trust_env = should_use_env_proxy()
    resp = session.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def build_system_prompt(config: dict[str, Any]) -> str:
    research = config.get("research_profile", {})
    return f"""You are a research assistant. Score academic papers and write a daily literature report.

Research profile:
- Background: {research.get('goal', '')}
- Focus topics: {', '.join(research.get('background_topics', []))}

Scoring dimensions (each 0-5):
- methodological_relevance: How well the paper's methods align with the research background
- inspiration_value: Whether the paper offers novel ideas
- transferability_to_my_research: Can the method transfer to our problems
- paper_quality: Experiment rigor, clarity, reproducibility
- novelty_timeliness: How new and timely the contribution is
- actionability: Can we implement or adapt something from this paper

Formula: final_score = 0.25*methodological_relevance + 0.25*inspiration_value + 0.20*transferability_to_my_research + 0.15*paper_quality + 0.10*novelty_timeliness + 0.05*actionability

Reading evidence values: "abstract-only", "paper page", or "pdf"
Use reading_evidence="pdf" only when PDF text evidence is provided for that paper.

Output ONLY valid JSON with no markdown code fence. Do not wrap the JSON in triple backticks."""


DEFAULT_REPORT_TEMPLATE = """Write a Markdown report with sections:
- Overview
- Top 10 Papers
- Trends of the day
- Potential research ideas
- Papers worth adding to related work

For each selected top paper, include motivation, core idea, experiments,
strengths, limitations, relevance to the research profile, actionable idea,
evidence level, and URL.
"""

DEFAULT_PAPER_TEMPLATE = """For each selected top paper, include:
- Motivation
- Core technical idea
- Method details
- Experiments and evidence
- Strengths
- Limitations and risks
- Relevance to the research profile
- Actionable follow-up
"""


def resolve_config_path(config_path: str | Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent / path


def resolve_report_template_path(config_path: str | Path, config: dict[str, Any]) -> Path | None:
    report_config = config.get("report", {})
    raw_path = str(report_config.get("template_path", "")).strip()
    if not raw_path:
        return None
    return resolve_config_path(config_path, raw_path)


def resolve_paper_template_path(config_path: str | Path, config: dict[str, Any]) -> Path | None:
    report_config = config.get("report", {})
    raw_path = str(report_config.get("paper_template_path", "")).strip()
    if not raw_path:
        return None
    return resolve_config_path(config_path, raw_path)


def load_report_template(config_path: str | Path, config: dict[str, Any]) -> str:
    template_path = resolve_report_template_path(config_path, config)
    if template_path is None:
        return DEFAULT_REPORT_TEMPLATE
    if not template_path.exists():
        raise FileNotFoundError(f"Report template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()


def load_paper_template(config_path: str | Path, config: dict[str, Any]) -> str:
    template_path = resolve_paper_template_path(config_path, config)
    if template_path is None:
        return DEFAULT_PAPER_TEMPLATE
    if not template_path.exists():
        raise FileNotFoundError(f"Per-paper template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()


def pdf_reading_config(config: dict[str, Any]) -> dict[str, Any]:
    pdf_config = config.get("report", {}).get("pdf_reading", {})
    return {
        "enabled": bool(pdf_config.get("enabled", False)),
        "max_papers": int(pdf_config.get("max_papers", DEFAULT_LLM_LIMIT)),
        "max_pages": int(pdf_config.get("max_pages", 8)),
        "max_chars_per_paper": int(pdf_config.get("max_chars_per_paper", 10000)),
        "max_total_chars": int(pdf_config.get("max_total_chars", 90000)),
        "request_timeout_seconds": int(pdf_config.get("request_timeout_seconds", 60)),
    }


def llm_batch_size(config: dict[str, Any]) -> int:
    report_config = config.get("report", {})
    return max(1, int(report_config.get("llm_batch_size", DEFAULT_LLM_BATCH_SIZE)))


def translation_config(config: dict[str, Any]) -> dict[str, Any]:
    translation = config.get("report", {}).get("translation", {})
    return {
        "enabled": bool(translation.get("enabled", False)),
        "target_language": str(translation.get("target_language", "zh-CN")),
    }


def extract_pdf_text(pdf_bytes: bytes, max_pages: int, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[str] = []
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
        current = normalize_prompt_text("\n\n".join(chunks))
        if len(current) >= max_chars:
            return current[:max_chars]
    return normalize_prompt_text("\n\n".join(chunks))[:max_chars]


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.split())


def fetch_pdf_bytes(pdf_url: str, timeout_seconds: int) -> bytes:
    import requests as _requests

    session = _requests.Session()
    session.trust_env = should_use_env_proxy()
    response = session.get(pdf_url, timeout=timeout_seconds, headers={"User-Agent": "paper-daily-llm/0.1"})
    response.raise_for_status()
    return response.content


def enrich_candidates_with_pdf_text(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    pdf_config = pdf_reading_config(config)
    if not pdf_config["enabled"]:
        logger.info("PDF reading disabled.")
        return candidates

    max_papers = max(0, int(pdf_config["max_papers"]))
    if max_papers <= 0:
        logger.info("PDF reading skipped because max_papers <= 0.")
        return candidates

    enriched: list[dict[str, Any]] = []
    remaining_total_chars = max(0, int(pdf_config["max_total_chars"]))
    effective_max_papers = min(max_papers, len(candidates))
    logger.info(
        "Reading PDF text for up to %d papers, max_pages=%d, max_chars_per_paper=%d, max_total_chars=%d.",
        effective_max_papers,
        pdf_config["max_pages"],
        pdf_config["max_chars_per_paper"],
        remaining_total_chars,
    )
    for idx, paper in enumerate(candidates):
        updated = dict(paper)
        if idx >= max_papers:
            enriched.append(updated)
            continue
        if remaining_total_chars <= 0:
            updated["_pdf_read_warning"] = "Skipped because total PDF text budget was exhausted."
            enriched.append(updated)
            continue

        pdf_url = str(updated.get("pdf_url") or "").strip()
        if not pdf_url:
            updated["_pdf_read_warning"] = "No PDF URL available."
            enriched.append(updated)
            continue

        try:
            pdf_bytes = fetch_pdf_bytes(pdf_url, int(pdf_config["request_timeout_seconds"]))
            max_chars_for_this_paper = min(int(pdf_config["max_chars_per_paper"]), remaining_total_chars)
            pdf_text = extract_pdf_text(
                pdf_bytes,
                max_pages=int(pdf_config["max_pages"]),
                max_chars=max_chars_for_this_paper,
            )
            if pdf_text:
                updated["_pdf_text_excerpt"] = pdf_text
                updated["_reading_evidence_hint"] = "pdf"
                remaining_total_chars -= len(pdf_text)
                logger.info("Extracted PDF text for %s (%d chars).", updated.get("id", pdf_url), len(pdf_text))
            else:
                updated["_pdf_read_warning"] = "PDF downloaded, but no extractable text was found."
        except Exception as exc:
            updated["_pdf_read_warning"] = f"PDF read failed: {exc}"
            logger.warning("PDF read failed for %s: %s", updated.get("id", pdf_url), exc)
        enriched.append(updated)
    return enriched


def strip_prompt_only_fields(paper: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in paper.items() if not key.startswith("_pdf_") and key != "_reading_evidence_hint"}


def translate_report_markdown(
    report_markdown: str,
    api_base: str,
    api_key: str,
    model: str,
    target_language: str,
    timeout_seconds: int,
) -> str:
    system_prompt = (
        "You are a careful academic translator. Translate Markdown reports while preserving "
        "all Markdown structure, heading levels, bullet structure, URLs, arXiv IDs, code spans, "
        "numbers, equations, model names, paper titles, and citation-like strings unless a natural "
        "Chinese translation is clearly appropriate. Return only the translated Markdown."
    )
    user_prompt = (
        f"Translate the following Markdown report into {target_language}. "
        "Keep the report title format, links, and all Markdown formatting intact.\n\n"
        "----- BEGIN MARKDOWN -----\n"
        f"{report_markdown}\n"
        "----- END MARKDOWN -----"
    )
    translated = call_llm(
        api_base,
        api_key,
        model,
        system_prompt,
        user_prompt,
        temperature=0.1,
        timeout_seconds=timeout_seconds,
    )
    return strip_markdown_fence(translated)


def build_user_prompt(
    candidates: list[dict[str, Any]],
    report_date: str,
    report_template: str,
    paper_template: str,
) -> str:
    rendered_report_template = report_template.replace("{report_date}", report_date)
    papers_text = []
    for i, p in enumerate(candidates, 1):
        papers_text.append(
            f"### Paper {i}\n"
            f"**Title:** {p.get('title', '')}\n"
            f"**Authors:** {', '.join(p.get('authors', []))}\n"
            f"**Abstract:** {p.get('abstract', '')}\n"
            f"**URL:** {p.get('url', '')}\n"
            f"**Categories:** {', '.join(p.get('categories', []))}\n"
            f"**Keyword matches:** {', '.join(p.get('matched_keywords', []))}\n"
            f"**Retrieval reason:** {p.get('retrieval_reason', '')}\n"
            f"**Reading evidence hint:** {p.get('_reading_evidence_hint', 'abstract-only')}\n"
            f"**PDF read warning:** {p.get('_pdf_read_warning', '')}\n"
            f"**PDF text excerpt:** {p.get('_pdf_text_excerpt', '')}"
        )
    return (
        f"Score the following {len(candidates)} papers for the date {report_date}.\n\n"
        "Select the top 10 papers and score ALL papers. For each paper provide "
        "semantic_scores, selected_for_deep_read (true if selected for deeper reading), "
        "selected_for_top10 (true if in top 10), reading_evidence, and semantic_rank. "
        "Set reading_evidence to pdf when a PDF text excerpt is present; otherwise use abstract-only.\n\n"
        "Then write the Markdown report according to this report template:\n\n"
        "----- BEGIN REPORT TEMPLATE -----\n"
        f"{rendered_report_template}\n"
        "----- END REPORT TEMPLATE -----\n\n"
        "For each selected paper inside the report, follow this per-paper summary template:\n\n"
        "----- BEGIN PER-PAPER TEMPLATE -----\n"
        f"{paper_template}\n"
        "----- END PER-PAPER TEMPLATE -----\n\n"
        "Respond with a JSON object matching this schema:\n"
        "{\n"
        '  "date": "...",\n'
        '  "status": "ok",\n'
        "  \"papers\": [\n"
        "    {\n"
        '      "id": "...",\n'
        '      "title": "...",\n'
        '      "semantic_scores": {\n'
        '        "methodological_relevance": 0,\n'
        '        "inspiration_value": 0,\n'
        '        "transferability_to_my_research": 0,\n'
        '        "paper_quality": 0,\n'
        '        "novelty_timeliness": 0,\n'
        '        "actionability": 0,\n'
        '        "final_score": 0\n'
        "      },\n"
        '      "selected_for_deep_read": false,\n'
        '      "selected_for_top10": false,\n'
        '      "reading_evidence": "abstract-only",\n'
        '      "semantic_rank": 0\n'
        "    }\n"
        "  ],\n"
        '  "report_markdown": "...full markdown report text..."\n'
        "}\n\n"
        "Here are the papers:\n\n" + "\n\n".join(papers_text)
    )


def build_scoring_prompt(
    candidates: list[dict[str, Any]],
    report_date: str,
    batch_index: int,
    batch_count: int,
) -> str:
    papers_text = []
    for i, p in enumerate(candidates, 1):
        papers_text.append(
            f"### Paper {i}\n"
            f"**ID:** {p.get('id', '')}\n"
            f"**Title:** {p.get('title', '')}\n"
            f"**Authors:** {', '.join(p.get('authors', []))}\n"
            f"**Abstract:** {p.get('abstract', '')}\n"
            f"**URL:** {p.get('url', '')}\n"
            f"**Categories:** {', '.join(p.get('categories', []))}\n"
            f"**Keyword matches:** {', '.join(p.get('matched_keywords', []))}\n"
            f"**Retrieval reason:** {p.get('retrieval_reason', '')}\n"
            f"**Reading evidence hint:** {p.get('_reading_evidence_hint', 'abstract-only')}\n"
            f"**PDF read warning:** {p.get('_pdf_read_warning', '')}\n"
            f"**PDF text excerpt:** {p.get('_pdf_text_excerpt', '')}"
        )
    return (
        f"Score batch {batch_index}/{batch_count} for the date {report_date}. "
        "Score every paper independently against the research profile. Do not force a fixed number "
        "of papers to be selected inside this batch; global top papers will be chosen after all batches.\n\n"
        "For each paper, provide semantic_scores, selected_for_deep_read, reading_evidence, "
        "a concise evidence_summary, and a concise relevance_rationale. "
        "Set reading_evidence to pdf when a PDF text excerpt is present; otherwise use abstract-only.\n\n"
        "Respond with a JSON object matching this schema:\n"
        "{\n"
        '  "date": "...",\n'
        '  "status": "ok",\n'
        "  \"papers\": [\n"
        "    {\n"
        '      "id": "...",\n'
        '      "title": "...",\n'
        '      "semantic_scores": {\n'
        '        "methodological_relevance": 0,\n'
        '        "inspiration_value": 0,\n'
        '        "transferability_to_my_research": 0,\n'
        '        "paper_quality": 0,\n'
        '        "novelty_timeliness": 0,\n'
        '        "actionability": 0,\n'
        '        "final_score": 0\n'
        "      },\n"
        '      "selected_for_deep_read": false,\n'
        '      "reading_evidence": "abstract-only",\n'
        '      "evidence_summary": "...",\n'
        '      "relevance_rationale": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Here are the papers:\n\n" + "\n\n".join(papers_text)
    )


def build_report_prompt(
    scored_candidates: list[dict[str, Any]],
    report_date: str,
    report_template: str,
    paper_template: str,
) -> str:
    rendered_report_template = report_template.replace("{report_date}", report_date)
    compact_papers = []
    for p in scored_candidates:
        scores = p.get("semantic_scores", {})
        compact_papers.append(
            {
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "abstract": p.get("abstract", ""),
                "url": p.get("url", ""),
                "categories": p.get("categories", []),
                "matched_keywords": p.get("matched_keywords", []),
                "semantic_scores": scores,
                "semantic_rank": p.get("semantic_rank", 0),
                "selected_for_top10": p.get("selected_for_top10", False),
                "selected_for_deep_read": p.get("selected_for_deep_read", False),
                "reading_evidence": p.get("reading_evidence", "abstract-only"),
                "evidence_summary": p.get("evidence_summary", ""),
                "relevance_rationale": p.get("relevance_rationale", ""),
            }
        )
    return (
        f"Write the final Markdown daily paper report for {report_date} using the globally ranked papers below. "
        "Use the ranking and scores as authoritative. Focus detailed per-paper summaries on papers with "
        "selected_for_top10=true.\n\n"
        "Report template:\n\n"
        "----- BEGIN REPORT TEMPLATE -----\n"
        f"{rendered_report_template}\n"
        "----- END REPORT TEMPLATE -----\n\n"
        "Per-paper summary template:\n\n"
        "----- BEGIN PER-PAPER TEMPLATE -----\n"
        f"{paper_template}\n"
        "----- END PER-PAPER TEMPLATE -----\n\n"
        "Return only the Markdown report. Do not wrap it in a JSON object or code fence.\n\n"
        "Globally scored papers:\n\n"
        f"{json.dumps(compact_papers, ensure_ascii=False, indent=2)}"
    )


def parse_llm_json(llm_output: str, logger: logging.Logger, context: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(llm_output)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s LLM response as JSON: %s", context, exc)
        logger.error("Raw LLM output (first 500 chars):\n%s", cleaned[:500])
        raise
    if not isinstance(result, dict):
        raise ValueError(f"{context} LLM response is not a JSON object.")
    return result


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def final_score(paper: dict[str, Any]) -> float:
    scores = paper.get("semantic_scores", {})
    try:
        return float(scores.get("final_score", 0))
    except (TypeError, ValueError):
        return 0.0


def merge_and_rank_candidates(
    candidates: list[dict[str, Any]],
    scored_papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {p.get("id"): strip_prompt_only_fields(p) for p in candidates}
    merged_by_id: dict[str, dict[str, Any]] = {}
    for scored in scored_papers:
        paper_id = scored.get("id", "")
        if not paper_id:
            continue
        original = candidate_by_id.get(paper_id, {})
        merged_by_id[paper_id] = {**original, **scored}

    for candidate in candidates:
        paper_id = candidate.get("id", "")
        if paper_id and paper_id not in merged_by_id:
            merged_by_id[paper_id] = strip_prompt_only_fields(candidate)

    ranked = sorted(merged_by_id.values(), key=final_score, reverse=True)
    for rank, paper in enumerate(ranked, 1):
        paper["semantic_rank"] = rank
        paper["selected_for_top10"] = rank <= 10
        paper["selected_for_deep_read"] = bool(paper.get("selected_for_deep_read") or rank <= 10)
    return ranked


def write_no_new_batch_note(paths: dict[str, Path], target_date: date, raw_data: dict[str, Any], logger: logging.Logger) -> None:
    dup = raw_data.get("duplicate_check", {})
    note = (
        f"# {target_date} — No New Candidate Batch\n\n"
        f"Today's candidate pool is identical to or empty compared to the most recent previous pool.\n\n"
        f"- Candidate count: {len(raw_data.get('candidates', raw_data.get('papers', [])))}\n"
        f"- Duplicate status: {dup.get('status', 'unknown')}\n"
        f"- Recommended action: {dup.get('recommended_action', 'unknown')}\n\n"
        f"This usually means arXiv's recent-list has not yet updated for today, "
        f"or there were no new announcement batches matching the configured sources.\n"
    )
    report_path = paths["report_dir"] / f"{target_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(note, encoding="utf-8")
    logger.info("Wrote no-new-batch note to %s", report_path)


def main() -> None:
    args = parse_args()
    target_date = date.today() if args.date == "today" else date.fromisoformat(args.date)

    config = load_config(args.config)
    paths = resolve_output_paths(args.config, config)
    log_path = paths["log_dir"] / f"{target_date}_llm.log"
    logger = setup_logger(log_path)

    api_base = get_env("OPENAI_API_BASE", DEFAULT_API_BASE)
    api_key = get_env("OPENAI_API_KEY")
    model = get_env("OPENAI_MODEL_NAME", "gpt-4o")
    timeout_seconds = args.timeout_seconds or int(
        get_env("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)) or str(DEFAULT_TIMEOUT_SECONDS)
    )

    if not api_key:
        logger.error("OPENAI_API_KEY is not set. Aborting LLM report generation.")
        sys.exit(1)

    candidate_path = paths["processed_dir"] / f"{target_date}_candidates.json"
    if not candidate_path.exists():
        logger.error("Candidate file not found: %s. Run fetch stage first.", candidate_path)
        sys.exit(1)

    candidates = read_json(candidate_path)
    if not isinstance(candidates, list):
        logger.error("Candidate file is not a list: %s", candidate_path)
        sys.exit(1)
    original_candidate_count = len(candidates)
    llm_limit = args.limit
    if llm_limit is None:
        llm_limit = int(get_env("OPENAI_LLM_LIMIT", str(DEFAULT_LLM_LIMIT)) or str(DEFAULT_LLM_LIMIT))
    if llm_limit <= 0:
        logger.error("--limit/OPENAI_LLM_LIMIT must be a positive integer.")
        sys.exit(1)
    if len(candidates) > llm_limit:
        candidates = candidates[:llm_limit]
        logger.info("Limited LLM input candidates from %d to %d.", original_candidate_count, len(candidates))

    raw_path = paths["raw_dir"] / f"{target_date}.json"
    raw_data = read_json(raw_path) if raw_path.exists() else {}

    dup = raw_data.get("duplicate_check", {})
    rec_action = dup.get("recommended_action", raw_data.get("recommended_action", ""))
    dup_status = dup.get("status", "")

    if (
        dup_status == "duplicate_of_previous"
        or rec_action == "write_no_new_batch_note"
        or not candidates
    ):
        logger.info("No new candidate batch. Writing short note.")
        write_no_new_batch_note(paths, target_date, {
            "candidates": candidates,
            "papers": raw_data.get("papers", []),
            "duplicate_check": dup,
            "recommended_action": rec_action,
        }, logger)
        return

    batch_size = llm_batch_size(config)
    candidate_batches = batched(candidates, batch_size)
    logger.info(
        "Calling LLM API to score %d papers in %d batch(es) of up to %d with timeout=%ss per call...",
        len(candidates),
        len(candidate_batches),
        batch_size,
        timeout_seconds,
    )

    try:
        report_template = load_report_template(args.config, config)
        paper_template = load_paper_template(args.config, config)
    except Exception as exc:
        logger.error("Failed to load report template: %s", exc)
        sys.exit(1)

    system_prompt = build_system_prompt(config)
    scored_papers: list[dict[str, Any]] = []
    for batch_index, candidate_batch in enumerate(candidate_batches, 1):
        logger.info(
            "Scoring LLM batch %d/%d with %d paper(s)...",
            batch_index,
            len(candidate_batches),
            len(candidate_batch),
        )
        candidates_for_prompt = enrich_candidates_with_pdf_text(candidate_batch, config, logger)
        user_prompt = build_scoring_prompt(
            candidates_for_prompt,
            str(target_date),
            batch_index,
            len(candidate_batches),
        )
        try:
            llm_output = call_llm(api_base, api_key, model, system_prompt, user_prompt, timeout_seconds=timeout_seconds)
        except Exception as exc:
            logger.error("LLM API call failed for scoring batch %d/%d: %s", batch_index, len(candidate_batches), exc)
            sys.exit(1)

        try:
            batch_result = parse_llm_json(llm_output, logger, f"batch {batch_index}/{len(candidate_batches)}")
        except Exception:
            sys.exit(1)
        batch_papers = batch_result.get("papers", [])
        if not isinstance(batch_papers, list):
            logger.error("Batch %d/%d response does not contain a papers list.", batch_index, len(candidate_batches))
            sys.exit(1)
        scored_papers.extend(batch_papers)

    scored_candidates = merge_and_rank_candidates(candidates, scored_papers)

    logger.info("Generating final Markdown report from %d globally ranked paper(s)...", len(scored_candidates))
    report_prompt = build_report_prompt(scored_candidates, str(target_date), report_template, paper_template)
    try:
        report_markdown = call_llm(
            api_base,
            api_key,
            model,
            system_prompt,
            report_prompt,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.error("LLM API call failed for final report generation: %s", exc)
        sys.exit(1)

    report_markdown = strip_markdown_fence(report_markdown)

    scored_path = paths["processed_dir"] / f"{target_date}_scored.json"
    scored_output = {
        "date": str(target_date),
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(candidate_path),
        "candidate_count": len(candidates),
        "deep_read_ids": [p["id"] for p in scored_candidates if p.get("selected_for_deep_read")],
        "top10_ids": [p["id"] for p in scored_candidates if p.get("selected_for_top10")],
        "papers": scored_candidates,
    }

    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8") as f:
        json.dump(scored_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved scores to %s", scored_path)

    report_path = paths["report_dir"] / f"{target_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_markdown, encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    translation = translation_config(config)
    translated_path: Path | None = None
    if translation["enabled"]:
        logger.info("Translating report to %s...", translation["target_language"])
        try:
            translated_markdown = translate_report_markdown(
                report_markdown,
                api_base,
                api_key,
                model,
                translation["target_language"],
                timeout_seconds,
            )
        except Exception as exc:
            logger.error("Report translation failed: %s", exc)
            sys.exit(1)
        translated_path = paths["report_dir"] / "zh" / f"{target_date}.md"
        translated_path.parent.mkdir(parents=True, exist_ok=True)
        translated_path.write_text(translated_markdown, encoding="utf-8")
        logger.info("Saved translated report to %s", translated_path)

    print(f"\nSummary:")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Scores:     {scored_path}")
    print(f"  Report:     {report_path}")
    if translated_path:
        print(f"  Report zh:  {translated_path}")


if __name__ == "__main__":
    main()
