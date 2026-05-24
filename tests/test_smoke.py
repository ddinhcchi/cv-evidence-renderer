"""Smoke tests — verify package imports and CLI registers."""

from __future__ import annotations

from typer.testing import CliRunner


def test_package_imports() -> None:
    import cv_evidence_renderer

    assert cv_evidence_renderer.__version__


def test_public_api_surface() -> None:
    from cv_evidence_renderer import EvidenceRecorder, render_from_jsonl

    assert callable(EvidenceRecorder)
    assert callable(render_from_jsonl)


def test_cli_version() -> None:
    from cv_evidence_renderer.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_help() -> None:
    from cv_evidence_renderer.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "evidence" in result.stdout.lower()
