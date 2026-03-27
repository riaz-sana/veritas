"""Tests for enterprise features: tiered models, confidence routing, caching, domains."""

import json
import os
import tempfile

import pytest

from veritas.core.cache import VerdictCache
from veritas.core.config import AgentModels, Config
from veritas.core.result import Verdict, VerificationResult


# ── AgentModels ──────────────────────────────────────────────────────

def test_agent_models_default():
    models = AgentModels.default("claude-sonnet-4-6")
    assert models.logic == "claude-sonnet-4-6"
    assert models.source == "claude-sonnet-4-6"
    assert models.adversary == "claude-sonnet-4-6"
    assert models.calibration == "claude-sonnet-4-6"
    assert models.synthesiser == "claude-sonnet-4-6"


def test_agent_models_economy():
    models = AgentModels.economy()
    assert models.logic == "claude-haiku-4-5"
    assert models.source == "claude-sonnet-4-6"
    assert models.adversary == "claude-sonnet-4-6"
    assert models.calibration == "claude-haiku-4-5"
    assert models.synthesiser == "claude-haiku-4-5"


def test_config_default_creates_agent_models():
    config = Config()
    assert config.agent_models is not None
    assert config.agent_models.logic == config.model


def test_config_economy_tier():
    config = Config(agent_models=AgentModels.economy())
    assert config.agent_models.logic == "claude-haiku-4-5"
    assert config.agent_models.source == "claude-sonnet-4-6"


# ── Confidence Routing ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confidence_routing_skips_verification(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from veritas import verify

    config = Config(confidence_routing=True, confidence_threshold=0.8)
    result = await verify(
        "Test claim",
        config=config,
        source_confidence=0.95,
    )
    assert result.verdict == Verdict.VERIFIED
    assert result.metadata.get("skipped") is True
    assert result.metadata.get("reason") == "confidence_routing"


@pytest.mark.asyncio
async def test_confidence_routing_verifies_when_below_threshold(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch
    from veritas import verify

    mock_result = VerificationResult(
        verdict=Verdict.REFUTED, confidence=0.9, summary="Wrong",
        failure_modes=[], evidence=[], contested=False,
        challenge_round=None, metadata={},
    )
    config = Config(confidence_routing=True, confidence_threshold=0.8)

    with patch("veritas.core.verify.VerificationRunner") as MockRunner:
        MockRunner.return_value.run = AsyncMock(return_value=mock_result)
        result = await verify("Test claim", config=config, source_confidence=0.5)

    assert result.verdict == Verdict.REFUTED
    assert result.metadata.get("skipped") is not True


@pytest.mark.asyncio
async def test_confidence_routing_disabled_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch
    from veritas import verify

    mock_result = VerificationResult(
        verdict=Verdict.VERIFIED, confidence=0.9, summary="OK",
        failure_modes=[], evidence=[], contested=False,
        challenge_round=None, metadata={},
    )

    with patch("veritas.core.verify.VerificationRunner") as MockRunner:
        MockRunner.return_value.run = AsyncMock(return_value=mock_result)
        # Even with high source confidence, routing is off by default
        result = await verify("Test claim", source_confidence=0.99)

    assert result.metadata.get("skipped") is not True


# ── Caching ──────────────────────────────────────────────────────────

def _make_result(verdict=Verdict.VERIFIED, confidence=0.9):
    return VerificationResult(
        verdict=verdict, confidence=confidence, summary="Test",
        failure_modes=[], evidence=[], contested=False,
        challenge_round=None, metadata={},
    )


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        assert cache.get("unknown claim") is None


def test_cache_put_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        result = _make_result()
        cache.put("test claim", None, None, result)
        cached = cache.get("test claim")
        assert cached is not None
        assert cached.verdict == Verdict.VERIFIED
        assert cached.metadata.get("cache_hit") is True


def test_cache_respects_context():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        cache.put("claim", "context A", None, _make_result(Verdict.VERIFIED))
        cache.put("claim", "context B", None, _make_result(Verdict.REFUTED))
        assert cache.get("claim", "context A").verdict == Verdict.VERIFIED
        assert cache.get("claim", "context B").verdict == Verdict.REFUTED


def test_cache_respects_domain():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        cache.put("claim", None, "medical", _make_result(Verdict.PARTIAL))
        assert cache.get("claim", None, "medical").verdict == Verdict.PARTIAL
        assert cache.get("claim", None, "legal") is None


def test_cache_ttl_expiry():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db", ttl_seconds=0)  # Expire immediately
        cache.put("claim", None, None, _make_result())
        assert cache.get("claim") is None  # Expired


def test_cache_clear():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        cache.put("claim1", None, None, _make_result())
        cache.put("claim2", None, None, _make_result())
        removed = cache.clear()
        assert removed == 2
        assert cache.get("claim1") is None


def test_cache_stats():
    with tempfile.TemporaryDirectory() as tmp:
        cache = VerdictCache(db_path=f"{tmp}/test.db")
        cache.put("claim1", None, None, _make_result())
        cache.put("claim2", None, None, _make_result())
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2


# ── Domain Extensions ────────────────────────────────────────────────

def test_domain_extension_returns_empty_for_none():
    from veritas.agents.domains import get_domain_extension
    assert get_domain_extension("logic", None) == ""


def test_domain_extension_returns_empty_for_unknown():
    from veritas.agents.domains import get_domain_extension
    assert get_domain_extension("logic", "underwater_basket_weaving") == ""


def test_domain_extension_code_for_logic():
    from veritas.agents.domains import get_domain_extension
    ext = get_domain_extension("logic", "code")
    assert "spec" in ext.lower()
    assert "edge case" in ext.lower()


def test_domain_extension_schema_for_source():
    from veritas.agents.domains import get_domain_extension
    ext = get_domain_extension("source", "schema")
    assert "column" in ext.lower() or "table" in ext.lower()


def test_domain_extension_medical_for_adversary():
    from veritas.agents.domains import get_domain_extension
    ext = get_domain_extension("adversary", "medical")
    assert "dosage" in ext.lower() or "dangerous" in ext.lower()


def test_base_agent_uses_domain_extension():
    from veritas.agents.logic import LogicVerifier

    class MockProvider:
        async def generate(self, prompt, system=""): return ""

    agent = LogicVerifier(provider=MockProvider())
    base_prompt = agent.get_system_prompt(None)
    code_prompt = agent.get_system_prompt("code")
    assert len(code_prompt) > len(base_prompt)
    assert "spec" in code_prompt.lower()
