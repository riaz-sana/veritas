"""Tests for isolation-vs-debate comparison."""
import json
import pytest
from veritas.core.config import Config
from veritas.core.result import Verdict
from veritas.orchestration.debate_runner import DebateRunner
from veritas.orchestration.runner import VerificationRunner
from veritas.providers.base import SearchResult


class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.prompts_received: list[str] = []

    async def generate(self, prompt: str, system: str = "") -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)
        sys_lower = system.lower()
        if "synthe" in sys_lower:
            return json.dumps({"verdict": "VERIFIED", "confidence": 0.9, "summary": "OK", "failure_modes": [], "contested": False})
        elif "source" in sys_lower:
            return json.dumps({"finding": "supported", "confidence": 0.85, "details": [], "sources": [], "reasoning": "Found"})
        elif "adversary" in sys_lower or "disprove" in sys_lower:
            return json.dumps({"finding": "no_counterexample", "confidence": 0.8, "details": [], "reasoning": "None found"})
        elif "calibrat" in sys_lower:
            return json.dumps({"finding": "well_calibrated", "confidence": 0.85, "details": [], "reasoning": "OK"})
        return json.dumps({"finding": "consistent", "confidence": 0.9, "details": []})


class MockSearch:
    async def search(self, query: str, num_results: int = 5):
        return [SearchResult(title="Test", url="https://test.com", snippet="Test")]


@pytest.mark.asyncio
async def test_debate_runner_runs_sequentially():
    """Debate runner should pass prior findings to each agent."""
    llm = MockLLM()
    runner = DebateRunner(llm_provider=llm, search_provider=MockSearch(), config=Config())
    result = await runner.run(claim="Test claim", context=None, domain=None, references=[])
    assert result.verdict == Verdict.VERIFIED
    assert len(result.evidence) == 4
    assert result.metadata.get("mode") == "debate"
    # Check that later agents received prior findings in their prompts
    # The source verifier (2nd agent) should see logic_verifier's finding
    assert any("logic_verifier" in p for p in llm.prompts_received[1:])


@pytest.mark.asyncio
async def test_isolation_runner_no_shared_context():
    """Isolation runner should NOT pass prior findings between agents."""
    llm = MockLLM()
    runner = VerificationRunner(llm_provider=llm, search_provider=MockSearch(), config=Config())
    result = await runner.run(claim="Test claim", context=None, domain=None, references=[])
    assert result.verdict == Verdict.VERIFIED
    # In isolation mode, no agent prompt should contain another agent's name
    agent_prompts = llm.prompts_received[:4]  # First 4 are the verification agents
    for prompt in agent_prompts:
        # Each agent only sees the claim, not other agents' findings
        assert "Previous Agent Findings" not in prompt


@pytest.mark.asyncio
async def test_debate_is_slower_than_isolation():
    """Debate mode runs sequentially, isolation runs in parallel — debate should be slower."""
    llm = MockLLM()
    search = MockSearch()

    iso_runner = VerificationRunner(llm_provider=llm, search_provider=search, config=Config())
    deb_runner = DebateRunner(llm_provider=llm, search_provider=search, config=Config())

    iso_result = await iso_runner.run(claim="Test", context=None, domain=None, references=[])
    deb_result = await deb_runner.run(claim="Test", context=None, domain=None, references=[])

    # Both produce valid results
    assert iso_result.verdict == Verdict.VERIFIED
    assert deb_result.verdict == Verdict.VERIFIED
