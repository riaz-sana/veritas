"""Tests for Pre-Action Verification (Agentic AI)."""

import json
import pytest

from veritas.agentic.verification import (
    ActionBlockedError,
    ActionNeedsReviewError,
    ActionRisk,
    ActionVerdict,
    ActionVerificationResult,
    before_action,
    verify_action,
    verify_plan,
)
from veritas.core.config import Config


# ── Data Model Tests ─────────────────────────────────────────────────

def test_action_verdict_enum():
    assert ActionVerdict.APPROVED == "approved"
    assert ActionVerdict.BLOCKED == "blocked"
    assert ActionVerdict.NEEDS_HUMAN_REVIEW == "needs_human_review"


def test_action_risk_creation():
    risk = ActionRisk(
        category="data_loss",
        severity="critical",
        description="DELETE without WHERE clause",
        mitigation="Add WHERE clause to limit scope",
    )
    assert risk.severity == "critical"


def test_action_result_approved():
    result = ActionVerificationResult(
        verdict=ActionVerdict.APPROVED,
        confidence=0.95,
        reasoning="Action is correct",
        risks=[],
        action="send_email",
        parameters={"to": "user@example.com"},
        agent_reasoning="User requested email",
    )
    assert result.approved is True
    assert result.blocked is False


def test_action_result_blocked():
    result = ActionVerificationResult(
        verdict=ActionVerdict.BLOCKED,
        confidence=0.9,
        reasoning="Wrong recipient",
        risks=[ActionRisk("incorrect_target", "high", "Email to wrong person", "Verify recipient")],
        action="send_email",
        parameters={"to": "wrong@example.com"},
        agent_reasoning="",
    )
    assert result.approved is False
    assert result.blocked is True
    assert len(result.risks) == 1


def test_action_result_str():
    result = ActionVerificationResult(
        verdict=ActionVerdict.BLOCKED,
        confidence=0.9,
        reasoning="Wrong recipient",
        risks=[],
        action="send_email",
        parameters={},
        agent_reasoning="",
    )
    text = str(result)
    assert "BLOCKED" in text


def test_action_result_to_dict():
    result = ActionVerificationResult(
        verdict=ActionVerdict.APPROVED,
        confidence=0.95,
        reasoning="OK",
        risks=[],
        action="test",
        parameters={"x": 1},
        agent_reasoning="",
    )
    d = result.to_dict()
    assert d["verdict"] == "approved"
    assert d["approved"] is True
    json.dumps(d)  # serializable


def test_action_result_report():
    result = ActionVerificationResult(
        verdict=ActionVerdict.APPROVED_WITH_WARNINGS,
        confidence=0.7,
        reasoning="Proceed with caution",
        risks=[ActionRisk("irreversible", "medium", "Cannot undo", "Add confirmation")],
        action="delete_user",
        parameters={"user_id": "123"},
        agent_reasoning="User requested deletion",
    )
    report = result.report()
    assert "delete_user" in report
    assert "irreversible" in report
    assert "MEDIUM" in report


# ── verify_action Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_action_approved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "verdict": "approved",
        "confidence": 0.95,
        "reasoning": "Action is correct and safe",
        "risks": [],
    })

    with patch("veritas.agentic.verification.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        result = await verify_action(
            action="send_email",
            parameters={"to": "user@example.com", "subject": "Hello"},
            reasoning="User requested this email",
            goal="Send a welcome email to new user",
        )

    assert result.verdict == ActionVerdict.APPROVED
    assert result.approved is True


@pytest.mark.asyncio
async def test_verify_action_blocked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "verdict": "blocked",
        "confidence": 0.9,
        "reasoning": "Amount exceeds authorized limit",
        "risks": [{"category": "scope_exceeded", "severity": "critical", "description": "Transfer of $1M exceeds $10K limit", "mitigation": "Require manager approval"}],
    })

    with patch("veritas.agentic.verification.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        result = await verify_action(
            action="transfer_funds",
            parameters={"amount": 1000000, "to_account": "XXX"},
            reasoning="Agent decided to transfer funds",
        )

    assert result.verdict == ActionVerdict.BLOCKED
    assert result.blocked is True
    assert len(result.risks) == 1
    assert result.risks[0].severity == "critical"


# ── verify_plan Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_plan_approved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "verdict": "approved",
        "confidence": 0.85,
        "reasoning": "Plan is well-structured",
        "step_analysis": [
            {"step": 1, "action": "fetch data", "verdict": "ok", "concern": ""},
            {"step": 2, "action": "process", "verdict": "ok", "concern": ""},
        ],
        "risks": [],
        "missing_steps": [],
        "unnecessary_steps": [],
    })

    with patch("veritas.agentic.verification.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        result = await verify_plan(
            goal="Process customer refund",
            steps=["Fetch order details", "Validate refund eligibility", "Process refund", "Send confirmation email"],
        )

    assert result.verdict == ActionVerdict.APPROVED
    assert result.approved is True


# ── @before_action Decorator Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_before_action_decorator_approved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "verdict": "approved",
        "confidence": 0.95,
        "reasoning": "OK",
        "risks": [],
    })

    call_log = []

    @before_action
    async def send_notification(user_id: str, message: str):
        call_log.append({"user_id": user_id, "message": message})
        return "sent"

    with patch("veritas.agentic.verification.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        result = await send_notification("user123", "Hello!")

    assert result == "sent"
    assert len(call_log) == 1


@pytest.mark.asyncio
async def test_before_action_decorator_blocked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "verdict": "blocked",
        "confidence": 0.9,
        "reasoning": "Dangerous action",
        "risks": [{"category": "data_loss", "severity": "critical", "description": "Will delete all data", "mitigation": "Don't"}],
    })

    @before_action
    async def delete_everything(confirm: bool = False):
        return "deleted"

    with patch("veritas.agentic.verification.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        with pytest.raises(ActionBlockedError) as exc_info:
            await delete_everything(confirm=True)

    assert "blocked" in str(exc_info.value).lower()
    assert exc_info.value.verification_result.blocked is True


@pytest.mark.asyncio
async def test_before_action_has_veritas_flag():
    @before_action
    async def some_action():
        pass

    assert hasattr(some_action, "_veritas_enabled")
    assert some_action._veritas_enabled is True
