from pathlib import Path

import pytest

from scripts.llm_report import build_user_prompt, load_paper_template, load_report_template


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
