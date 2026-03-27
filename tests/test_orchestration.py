"""Tests for orchestration layer."""
import json
import pytest
from veritas.core.config import Config
from veritas.core.result import AgentFinding, Verdict
from veritas.orchestration.runner import VerificationRunner
from veritas.providers.base import SearchResult

class MockProvider:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.call_count = 0
    async def generate(self, prompt: str, system: str = "") -> str:
        self.call_count += 1
        sys_lower = system.lower()
        # Check most specific keys first to avoid false matches
        if "synthe" in sys_lower and "synthe" in self.responses:
            return self.responses["synthe"]
        for key, response in self.responses.items():
            if key in sys_lower:
                return response
        return json.dumps({"finding": "consistent", "confidence": 0.5, "details": []})

class MockSearchProvider:
    async def search(self, query: str, num_results: int = 5):
        return [SearchResult(title="Test", url="https://test.com", snippet="Test result")]

def _mock_responses():
    return {
        "logic": json.dumps({"finding": "consistent", "confidence": 0.9, "details": []}),
        "source": json.dumps({"finding": "supported", "confidence": 0.85, "details": [], "sources": ["https://test.com"], "reasoning": "Found support"}),
        "adversary": json.dumps({"finding": "no_counterexample", "confidence": 0.8, "details": [], "reasoning": "Could not disprove"}),
        "calibrat": json.dumps({"finding": "well_calibrated", "confidence": 0.85, "details": [], "reasoning": "Appropriate confidence"}),
        "synthe": json.dumps({"verdict": "VERIFIED", "confidence": 0.88, "summary": "Claim is verified.", "failure_modes": [], "contested": False}),
    }

@pytest.mark.asyncio
async def test_runner_runs_all_agents():
    provider = MockProvider(_mock_responses())
    runner = VerificationRunner(llm_provider=provider, search_provider=MockSearchProvider(), config=Config())
    result = await runner.run(claim="Water boils at 100C at sea level", context=None, domain="scientific", references=[])
    assert result.verdict == Verdict.VERIFIED
    assert len(result.evidence) == 4
    assert provider.call_count == 5

@pytest.mark.asyncio
async def test_runner_no_challenge_when_disabled():
    responses = _mock_responses()
    responses["synthe"] = json.dumps({"verdict": "DISPUTED", "confidence": 0.5, "summary": "Agents disagree.", "failure_modes": [], "contested": True})
    runner = VerificationRunner(llm_provider=MockProvider(responses), search_provider=MockSearchProvider(), config=Config(challenge_round=False))
    result = await runner.run(claim="test", context=None, domain=None, references=[])
    assert result.challenge_round is None
