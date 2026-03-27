"""Tests for CLI interface."""
import pytest
from typer.testing import CliRunner
from veritas.cli.main import app

runner = CliRunner()

def test_cli_check_rejects_empty():
    result = runner.invoke(app, ["check", ""])
    assert result.exit_code != 0

def test_cli_check_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["check", "Test claim"])
    assert result.exit_code != 0

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout.lower() or "veritas" in result.stdout.lower()

def test_cli_check_help():
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.stdout
    assert "--json" in result.stdout
    assert "--domain" in result.stdout
