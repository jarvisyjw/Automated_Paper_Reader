from pathlib import Path

import pytest

from scripts.llm_report import (
    build_user_prompt,
    llm_batch_size,
    load_paper_template,
    load_report_template,
    merge_and_rank_candidates,
    pdf_reading_config,
    strip_prompt_only_fields,
    strip_markdown_fence,
    translation_config,
)


def test_load_report_template_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    template_path = tmp_path / "custom.md"
    config_path.write_text("report:\n  template_path: custom.md\n", encoding="utf-8")
    template_path.write_text("# Custom Report\n\n- Motivation", encoding="utf-8")

    template = load_report_template(config_path, {"report": {"template_path": "custom.md"}})

    assert template == "# Custom Report\n\n- Motivation"


def test_load_report_template_default_when_unconfigured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    template = load_report_template(config_path, {})

    assert "Top 10 Papers" in template
    assert "motivation" in template


def test_load_report_template_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("report:\n  template_path: missing.md\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_report_template(config_path, {"report": {"template_path": "missing.md"}})


def test_load_paper_template_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    template_path = tmp_path / "paper.md"
    config_path.write_text("report:\n  paper_template_path: paper.md\n", encoding="utf-8")
    template_path.write_text("## Per Paper\n\n- Experiments", encoding="utf-8")

    template = load_paper_template(config_path, {"report": {"paper_template_path": "paper.md"}})

    assert template == "## Per Paper\n\n- Experiments"


def test_load_paper_template_default_when_unconfigured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    template = load_paper_template(config_path, {})

    assert "Motivation" in template
    assert "Actionable follow-up" in template


def test_load_paper_template_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("report:\n  paper_template_path: missing.md\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_paper_template(config_path, {"report": {"paper_template_path": "missing.md"}})


def test_build_user_prompt_includes_template() -> None:
    prompt = build_user_prompt(
        [
            {
                "title": "Example Paper",
                "authors": ["A. Author"],
                "abstract": "An abstract.",
                "url": "https://example.com",
                "categories": ["cs.LG"],
                "matched_keywords": ["calibration"],
                "retrieval_reason": "matched calibration",
            }
        ],
        "2026-06-01",
        "## Custom Section\n\n- Motivation",
        "## Per Paper\n\n- Experiments",
    )

    assert "----- BEGIN REPORT TEMPLATE -----" in prompt
    assert "----- BEGIN PER-PAPER TEMPLATE -----" in prompt
    assert "## Custom Section" in prompt
    assert "## Per Paper" in prompt
    assert "Example Paper" in prompt


def test_build_user_prompt_includes_pdf_evidence() -> None:
    prompt = build_user_prompt(
        [
            {
                "title": "PDF Paper",
                "authors": [],
                "abstract": "Short abstract.",
                "url": "https://example.com",
                "categories": [],
                "matched_keywords": [],
                "retrieval_reason": "test",
                "_reading_evidence_hint": "pdf",
                "_pdf_text_excerpt": "This is extracted full-text evidence from the PDF.",
            }
        ],
        "2026-06-08",
        "report template",
        "paper template",
    )

    assert "Reading evidence hint:** pdf" in prompt
    assert "This is extracted full-text evidence from the PDF." in prompt


def test_pdf_reading_config_defaults_to_disabled() -> None:
    config = pdf_reading_config({})

    assert config["enabled"] is False
    assert config["max_papers"] == 20
    assert config["max_total_chars"] == 90000


def test_llm_batch_size_defaults_to_five() -> None:
    assert llm_batch_size({}) == 5
    assert llm_batch_size({"report": {"llm_batch_size": 3}}) == 3
    assert llm_batch_size({"report": {"llm_batch_size": 0}}) == 1


def test_merge_and_rank_candidates_assigns_global_top10() -> None:
    candidates = [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]
    scored = [
        {"id": "a", "semantic_scores": {"final_score": 2.0}},
        {"id": "b", "semantic_scores": {"final_score": 4.0}},
    ]

    ranked = merge_and_rank_candidates(candidates, scored)

    assert [paper["id"] for paper in ranked] == ["b", "a"]
    assert [paper["semantic_rank"] for paper in ranked] == [1, 2]
    assert all(paper["selected_for_top10"] for paper in ranked)


def test_strip_prompt_only_fields_removes_pdf_text() -> None:
    paper = {
        "id": "paper-1",
        "title": "Keep me",
        "_pdf_text_excerpt": "large text",
        "_pdf_read_warning": "warning",
        "_reading_evidence_hint": "pdf",
    }

    stripped = strip_prompt_only_fields(paper)

    assert stripped == {"id": "paper-1", "title": "Keep me"}


def test_translation_config_defaults_to_disabled() -> None:
    config = translation_config({})

    assert config == {"enabled": False, "target_language": "zh-CN"}


def test_translation_config_reads_report_settings() -> None:
    config = translation_config({"report": {"translation": {"enabled": True, "target_language": "zh-Hans"}}})

    assert config == {"enabled": True, "target_language": "zh-Hans"}


def test_strip_markdown_fence() -> None:
    assert strip_markdown_fence("```markdown\n# Title\n```") == "# Title"
    assert strip_markdown_fence("plain") == "plain"
