"""Tests for the public verify() API."""
import pytest
from veritas import VerificationResult, Verdict, verify
from veritas.core.config import VeritasConfigError

def test_verify_rejects_empty_claim():
    with pytest.raises(ValueError, match="claim"):
        import asyncio
        asyncio.run(verify(""))

def test_verify_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(VeritasConfigError):
        import asyncio
        asyncio.run(verify("test claim"))

@pytest.mark.asyncio
async def test_verify_returns_verification_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch
    mock_result = VerificationResult(
        verdict=Verdict.VERIFIED, confidence=0.9, summary="Test passed.",
        failure_modes=[], evidence=[], contested=False, challenge_round=None, metadata={},
    )
    with patch("veritas.core.verify.VerificationRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=mock_result)
        result = await verify("Water boils at 100C")
    assert isinstance(result, VerificationResult)
    assert result.verdict == Verdict.VERIFIED

@pytest.mark.asyncio
async def test_verify_passes_all_options(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch
    mock_result = VerificationResult(
        verdict=Verdict.VERIFIED, confidence=0.9, summary="OK",
        failure_modes=[], evidence=[], contested=False, challenge_round=None, metadata={},
    )
    with patch("veritas.core.verify.VerificationRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=mock_result)
        result = await verify("Test claim", context="Some context", domain="technical", references=["doc.pdf"], model="claude-opus-4-6")
        instance.run.assert_called_once_with(claim="Test claim", context="Some context", domain="technical", references=["doc.pdf"])
