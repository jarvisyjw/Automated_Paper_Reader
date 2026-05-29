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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from utils import (
    load_config,
    read_json,
    setup_logger,
    resolve_output_paths,
)

DEFAULT_API_BASE = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM-scored paper report.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--date", default="today", help="'today' or YYYY-MM-DD")
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
    resp = _requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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

Output ONLY valid JSON with no markdown code fence. Do not wrap the JSON in triple backticks."""


def build_user_prompt(candidates: list[dict[str, Any]], report_date: str) -> str:
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
            f"**Retrieval reason:** {p.get('retrieval_reason', '')}"
        )
    return (
        f"Score the following {len(candidates)} papers for the date {report_date}.\n\n"
        "Select the top 10 papers and score ALL papers. For each paper provide "
        "semantic_scores, selected_for_deep_read (true if selected for deeper reading), "
        "selected_for_top10 (true if in top 10), reading_evidence, and semantic_rank.\n\n"
        "Then write a Markdown report with sections: Overview, Top 10 Papers, "
        "Trends of the day, Potential research ideas, Papers worth adding to related work.\n\n"
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
    logger = setup_logger(config, target_date)

    api_base = get_env("OPENAI_API_BASE", DEFAULT_API_BASE)
    api_key = get_env("OPENAI_API_KEY")
    model = get_env("OPENAI_MODEL_NAME", "gpt-4o")

    if not api_key:
        logger.error("OPENAI_API_KEY is not set. Aborting LLM report generation.")
        sys.exit(1)

    candidate_path = paths["processed_dir"] / f"{target_date}_candidates.json"
    if not candidate_path.exists():
        logger.error("Candidate file not found: %s. Run fetch stage first.", candidate_path)
        sys.exit(1)

    candidates = read_json(str(candidate_path))
    if not isinstance(candidates, list):
        logger.error("Candidate file is not a list: %s", candidate_path)
        sys.exit(1)

    raw_path = paths["raw_dir"] / f"{target_date}.json"
    raw_data = read_json(str(raw_path)) if raw_path.exists() else {}

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

    logger.info("Calling LLM API to score %d papers...", len(candidates))

    system_prompt = build_system_prompt(config)
    user_prompt = build_user_prompt(candidates, str(target_date))

    try:
        llm_output = call_llm(api_base, api_key, model, system_prompt, user_prompt)
    except Exception as exc:
        logger.error("LLM API call failed: %s", exc)
        sys.exit(1)

    llm_output = llm_output.strip()
    # Strip markdown code fences if LLM wraps the JSON in ```json ... ```
    if llm_output.startswith("```"):
        first_newline = llm_output.find("\n")
        if first_newline != -1:
            llm_output = llm_output[first_newline + 1:]
        if llm_output.endswith("```"):
            llm_output = llm_output[:-3]
        llm_output = llm_output.strip()

    try:
        result = json.loads(llm_output)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM response as JSON: %s", exc)
        logger.error("Raw LLM output (first 500 chars):\n%s", llm_output[:500])
        sys.exit(1)

    papers_out = result.get("papers", [])
    report_markdown = result.get("report_markdown", "# Daily Report\n\nNo report content generated.")

    scored_candidates = []
    candidate_by_id = {p.get("id"): p for p in candidates}

    for scored in papers_out:
        paper_id = scored.get("id", "")
        original = candidate_by_id.get(paper_id, {})
        merged = {**original, **scored}
        scored_candidates.append(merged)

    scored_path = paths["processed_dir"] / f"{target_date}_scored.json"
    scored_output = {
        "date": str(target_date),
        "status": result.get("status", "ok"),
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

    print(f"\nSummary:")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Scores:     {scored_path}")
    print(f"  Report:     {report_path}")


if __name__ == "__main__":
    main()
