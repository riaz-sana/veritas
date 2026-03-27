"""Tests for logic verifier agent."""
import json
import pytest
from veritas.agents.logic import LogicVerifier
from veritas.core.result import AgentFinding

class MockProvider:
    def __init__(self, response: str):
        self.response = response
    async def generate(self, prompt: str, system: str = "") -> str:
        return self.response

def test_logic_verifier_system_prompt():
    agent = LogicVerifier(provider=MockProvider(""))
    assert "logic" in agent.system_prompt.lower()
    assert "consistency" in agent.system_prompt.lower()

def test_logic_verifier_parse_consistent():
    agent = LogicVerifier(provider=MockProvider(""))
    response = json.dumps({"finding": "consistent", "confidence": 0.9, "details": []})
    finding = agent.parse_response(response)
    assert finding.agent == "logic_verifier"
    assert finding.finding == "consistent"
    assert finding.confidence == 0.9

def test_logic_verifier_parse_inconsistency():
    agent = LogicVerifier(provider=MockProvider(""))
    response = json.dumps({"finding": "inconsistency", "confidence": 0.85, "details": [{"type": "logical_inconsistency", "description": "Premise A contradicts conclusion B"}]})
    finding = agent.parse_response(response)
    assert finding.finding == "inconsistency"
    assert len(finding.details) == 1

def test_logic_verifier_parse_malformed_response():
    agent = LogicVerifier(provider=MockProvider(""))
    finding = agent.parse_response("This is not JSON at all")
    assert finding.agent == "logic_verifier"
    assert finding.finding == "parse_error"
    assert finding.confidence == 0.0

@pytest.mark.asyncio
async def test_logic_verifier_verify():
    response = json.dumps({"finding": "consistent", "confidence": 0.95, "details": []})
    agent = LogicVerifier(provider=MockProvider(response))
    finding = await agent.verify(claim="Water boils at 100C at sea level", context=None, domain="scientific", references=[])
    assert finding.agent == "logic_verifier"
    assert finding.finding == "consistent"
