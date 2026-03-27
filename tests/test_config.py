"""Tests for Veritas configuration."""

import os

import pytest

from veritas.core.config import Config, VeritasConfigError


def test_config_defaults():
    config = Config()
    assert config.model == "claude-sonnet-4-6"
    assert config.challenge_round is True
    assert config.max_search_results == 5
    assert config.timeout_seconds == 30
    assert config.verbose is False


def test_config_custom_values():
    config = Config(
        model="claude-opus-4-6",
        search_provider="tavily",
        challenge_round=False,
        timeout_seconds=60,
    )
    assert config.model == "claude-opus-4-6"
    assert config.search_provider == "tavily"
    assert config.challenge_round is False
    assert config.timeout_seconds == 60


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key-456")
    monkeypatch.setenv("VERITAS_MODEL", "claude-opus-4-6")
    config = Config()
    assert config.anthropic_api_key == "test-key-123"
    assert config.search_api_key == "brave-key-456"
    assert config.model == "claude-opus-4-6"


def test_config_tavily_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key-789")
    config = Config()
    assert config.search_api_key == "tavily-key-789"


def test_config_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("VERITAS_MODEL", "claude-opus-4-6")
    config = Config(model="claude-sonnet-4-6")
    assert config.model == "claude-sonnet-4-6"


def test_veritas_config_error():
    err = VeritasConfigError("Missing API key")
    assert str(err) == "Missing API key"
    assert isinstance(err, Exception)


def test_config_validate_raises_on_missing_anthropic_key():
    config = Config()
    with pytest.raises(VeritasConfigError, match="ANTHROPIC_API_KEY"):
        config.validate()


def test_config_validate_passes_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = Config()
    config.validate()
