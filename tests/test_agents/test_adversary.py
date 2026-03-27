"""Tests for adversary agent."""
import json
import pytest
from veritas.agents.adversary import Adversary
from veritas.core.result import AgentFinding

class MockProvider:
    def __init__(self, response: str):
        self.response = response
    async def generate(self, prompt: str, system: str = "") -> str:
        return self.response

def test_adversary_system_prompt():
    agent = Adversary(provider=MockProvider(""))
    assert "counterexample" in agent.system_prompt.lower() or "disprove" in agent.system_prompt.lower()

def test_adversary_parse_no_counterexample():
    agent = Adversary(provider=MockProvider(""))
    response = json.dumps({"finding": "no_counterexample", "confidence": 0.8, "details": [], "reasoning": "Could not disprove"})
    finding = agent.parse_response(response)
    assert finding.finding == "no_counterexample"

def test_adversary_parse_counterexample_found():
    agent = Adversary(provider=MockProvider(""))
    response = json.dumps({"finding": "counterexample_found", "confidence": 0.9, "details": [{"type": "factual_error", "description": "If true, X follows, but X is false"}], "reasoning": "Reductio ad absurdum"})
    finding = agent.parse_response(response)
    assert finding.finding == "counterexample_found"
    assert len(finding.details) == 1

def test_adversary_parse_malformed():
    agent = Adversary(provider=MockProvider(""))
    finding = agent.parse_response("garbage")
    assert finding.finding == "parse_error"

@pytest.mark.asyncio
async def test_adversary_verify():
    response = json.dumps({"finding": "counterexample_found", "confidence": 0.85, "details": [{"type": "scope_error", "description": "Claim too broad"}], "reasoning": "Not all cases fit"})
    agent = Adversary(provider=MockProvider(response))
    finding = await agent.verify(claim="All birds can fly", context=None, domain=None, references=[])
    assert finding.finding == "counterexample_found"

@pytest.mark.asyncio
async def test_adversary_challenge():
    response = json.dumps({"finding": "counterexample_found", "confidence": 0.9, "details": [{"type": "factual_error", "description": "Date is wrong"}], "reasoning": "Evidence confirms discrepancy"})
    agent = Adversary(provider=MockProvider(response))
    finding = await agent.challenge(claim="iPhone released in 2006", contested_points=["Release date conflicts"], agent_findings=[])
    assert finding.finding == "counterexample_found"
