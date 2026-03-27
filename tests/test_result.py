"""Tests for Veritas data models."""

import json

from veritas.core.result import (
    AgentFinding,
    ChallengeResult,
    FailureMode,
    FailureModeType,
    Verdict,
    VerificationResult,
)


def test_verdict_enum_values():
    assert Verdict.VERIFIED == "VERIFIED"
    assert Verdict.PARTIAL == "PARTIAL"
    assert Verdict.UNCERTAIN == "UNCERTAIN"
    assert Verdict.DISPUTED == "DISPUTED"
    assert Verdict.REFUTED == "REFUTED"


def test_failure_mode_type_enum_values():
    assert FailureModeType.FACTUAL_ERROR == "factual_error"
    assert FailureModeType.LOGICAL_INCONSISTENCY == "logical_inconsistency"
    assert FailureModeType.UNSUPPORTED_INFERENCE == "unsupported_inference"
    assert FailureModeType.TEMPORAL_ERROR == "temporal_error"
    assert FailureModeType.SCOPE_ERROR == "scope_error"
    assert FailureModeType.SOURCE_CONFLICT == "source_conflict"


def test_failure_mode_creation():
    fm = FailureMode(
        type=FailureModeType.FACTUAL_ERROR,
        detail="Released in 2006, actually 2007",
        agent="source_verifier",
    )
    assert fm.type == FailureModeType.FACTUAL_ERROR
    assert fm.agent == "source_verifier"


def test_agent_finding_creation():
    finding = AgentFinding(
        agent="logic_verifier",
        finding="consistent",
        confidence=0.9,
        details=[{"type": "no_issues", "description": "Claim is internally consistent"}],
    )
    assert finding.agent == "logic_verifier"
    assert finding.confidence == 0.9
    assert finding.sources == []
    assert finding.reasoning == ""


def test_agent_finding_with_sources():
    finding = AgentFinding(
        agent="source_verifier",
        finding="supported",
        confidence=0.85,
        details=[],
        sources=["https://example.com/article"],
        reasoning="Found corroborating source",
    )
    assert len(finding.sources) == 1


def test_challenge_result_creation():
    cr = ChallengeResult(
        contested_points=["Date of release"],
        adversary_finding=AgentFinding(
            agent="adversary",
            finding="counterexample_found",
            confidence=0.8,
            details=[{"type": "factual_error", "description": "Wrong year"}],
        ),
        resolution="Adversary confirmed the date is incorrect",
    )
    assert len(cr.contested_points) == 1


def test_verification_result_creation():
    result = VerificationResult(
        verdict=Verdict.REFUTED,
        confidence=0.91,
        summary="The first iPhone was released June 2007, not 2006.",
        failure_modes=[
            FailureMode(
                type=FailureModeType.FACTUAL_ERROR,
                detail="Wrong release year",
                agent="source_verifier",
            )
        ],
        evidence=[
            AgentFinding(
                agent="source_verifier",
                finding="contradiction",
                confidence=0.95,
                details=[],
                sources=["https://en.wikipedia.org/wiki/IPhone"],
            )
        ],
        contested=False,
        challenge_round=None,
        metadata={"duration_ms": 4200, "agents_used": 5, "model": "claude-sonnet-4-6"},
    )
    assert result.verdict == Verdict.REFUTED
    assert result.confidence == 0.91
    assert len(result.failure_modes) == 1
    assert result.contested is False


def test_verification_result_str():
    result = VerificationResult(
        verdict=Verdict.REFUTED,
        confidence=0.91,
        summary="The first iPhone was released June 2007, not 2006.",
        failure_modes=[],
        evidence=[],
        contested=False,
        challenge_round=None,
        metadata={},
    )
    text = str(result)
    assert "REFUTED" in text
    assert "0.91" in text
    assert "2007" in text


def test_verification_result_to_dict():
    result = VerificationResult(
        verdict=Verdict.VERIFIED,
        confidence=0.95,
        summary="Claim is accurate.",
        failure_modes=[],
        evidence=[],
        contested=False,
        challenge_round=None,
        metadata={"duration_ms": 1000},
    )
    d = result.to_dict()
    assert d["verdict"] == "VERIFIED"
    assert d["confidence"] == 0.95
    json.dumps(d)


def test_verification_result_report():
    result = VerificationResult(
        verdict=Verdict.PARTIAL,
        confidence=0.6,
        summary="Some parts verified.",
        failure_modes=[
            FailureMode(
                type=FailureModeType.TEMPORAL_ERROR,
                detail="Data from 2020, claim about 2026",
                agent="source_verifier",
            )
        ],
        evidence=[
            AgentFinding(
                agent="logic_verifier",
                finding="consistent",
                confidence=0.9,
                details=[],
            )
        ],
        contested=False,
        challenge_round=None,
        metadata={},
    )
    report = result.report()
    assert "PARTIAL" in report
    assert "temporal_error" in report
    assert "logic_verifier" in report
