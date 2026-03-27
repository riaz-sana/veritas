"""Tests for synthesiser agent."""
import json
import pytest
from veritas.agents.synthesiser import Synthesiser
from veritas.core.result import AgentFinding, VerificationResult, Verdict

class MockProvider:
    def __init__(self, response: str):
        self.response = response
    async def generate(self, prompt: str, system: str = "") -> str:
        return self.response

def _make_finding(agent: str, finding: str, confidence: float, details: list | None = None) -> AgentFinding:
    return AgentFinding(agent=agent, finding=finding, confidence=confidence, details=details or [])

def test_synthesiser_creation():
    agent = Synthesiser(provider=MockProvider(""))
    assert agent.name == "synthesiser"

@pytest.mark.asyncio
async def test_synthesiser_all_agree_verified():
    llm_response = json.dumps({"verdict": "VERIFIED", "confidence": 0.92, "summary": "All agents agree.", "failure_modes": [], "contested": False})
    agent = Synthesiser(provider=MockProvider(llm_response))
    findings = [
        _make_finding("logic_verifier", "consistent", 0.95),
        _make_finding("source_verifier", "supported", 0.9),
        _make_finding("adversary", "no_counterexample", 0.85),
        _make_finding("calibration", "well_calibrated", 0.9),
    ]
    result = await agent.synthesise(claim="Water boils at 100C at sea level", findings=findings)
    assert result.verdict == Verdict.VERIFIED
    assert result.confidence > 0.8
    assert result.contested is False

@pytest.mark.asyncio
async def test_synthesiser_refuted():
    llm_response = json.dumps({"verdict": "REFUTED", "confidence": 0.91, "summary": "iPhone released 2007, not 2006.", "failure_modes": [{"type": "factual_error", "detail": "Wrong year", "agent": "source_verifier"}], "contested": False})
    agent = Synthesiser(provider=MockProvider(llm_response))
    findings = [
        _make_finding("logic_verifier", "consistent", 0.9),
        _make_finding("source_verifier", "contradiction", 0.95, [{"type": "factual_error", "description": "Wrong year"}]),
        _make_finding("adversary", "counterexample_found", 0.9),
        _make_finding("calibration", "overconfident", 0.7),
    ]
    result = await agent.synthesise(claim="iPhone released in 2006", findings=findings)
    assert result.verdict == Verdict.REFUTED
    assert len(result.failure_modes) > 0

@pytest.mark.asyncio
async def test_synthesiser_detects_conflict():
    llm_response = json.dumps({"verdict": "DISPUTED", "confidence": 0.5, "summary": "Agents disagree.", "failure_modes": [], "contested": True})
    agent = Synthesiser(provider=MockProvider(llm_response))
    findings = [
        _make_finding("logic_verifier", "consistent", 0.9),
        _make_finding("source_verifier", "supported", 0.8),
        _make_finding("adversary", "counterexample_found", 0.85),
        _make_finding("calibration", "overconfident", 0.6),
    ]
    result = await agent.synthesise(claim="Contested claim", findings=findings)
    assert result.contested is True

@pytest.mark.asyncio
async def test_synthesiser_handles_missing_agents():
    llm_response = json.dumps({"verdict": "UNCERTAIN", "confidence": 0.3, "summary": "Insufficient data.", "failure_modes": [], "contested": False})
    agent = Synthesiser(provider=MockProvider(llm_response))
    findings = [_make_finding("logic_verifier", "consistent", 0.9)]
    result = await agent.synthesise(claim="Some claim", findings=findings)
    assert result.verdict == Verdict.UNCERTAIN
    assert "agents_used" in result.metadata

@pytest.mark.asyncio
async def test_synthesiser_handles_malformed_llm_response():
    agent = Synthesiser(provider=MockProvider("not json at all"))
    findings = [_make_finding("logic_verifier", "consistent", 0.9)]
    result = await agent.synthesise(claim="test", findings=findings)
    assert result.verdict == Verdict.UNCERTAIN
    assert result.confidence == 0.0
