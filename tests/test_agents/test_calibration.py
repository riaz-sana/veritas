"""Tests for calibration agent."""
import json
import pytest
from veritas.agents.calibration import CalibrationAgent
from veritas.core.result import AgentFinding

class MockProvider:
    def __init__(self, response: str):
        self.response = response
    async def generate(self, prompt: str, system: str = "") -> str:
        return self.response

def test_calibration_agent_system_prompt():
    agent = CalibrationAgent(provider=MockProvider(""))
    assert "confidence" in agent.system_prompt.lower()
    assert "calibrat" in agent.system_prompt.lower()

def test_calibration_parse_well_calibrated():
    agent = CalibrationAgent(provider=MockProvider(""))
    response = json.dumps({"finding": "well_calibrated", "confidence": 0.85, "details": [], "reasoning": "Claim specificity matches expected confidence"})
    finding = agent.parse_response(response)
    assert finding.finding == "well_calibrated"

def test_calibration_parse_overconfident():
    agent = CalibrationAgent(provider=MockProvider(""))
    response = json.dumps({"finding": "overconfident", "confidence": 0.4, "details": [{"type": "unsupported_inference", "description": "Stated with certainty but weak evidence"}], "reasoning": "Hedging language absent"})
    finding = agent.parse_response(response)
    assert finding.finding == "overconfident"
    assert finding.confidence == 0.4

@pytest.mark.asyncio
async def test_calibration_verify():
    response = json.dumps({"finding": "well_calibrated", "confidence": 0.9, "details": [], "reasoning": "Simple factual claim"})
    agent = CalibrationAgent(provider=MockProvider(response))
    finding = await agent.verify(claim="Water boils at 100C at sea level", context=None, domain="scientific", references=[])
    assert finding.agent == "calibration"
    assert finding.finding == "well_calibrated"
