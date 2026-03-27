"""Integration tests for the full Veritas pipeline."""
import json
import pytest
from veritas import Config, VerificationResult, Verdict, FailureMode, FailureModeType, AgentFinding, verify
from veritas.orchestration.runner import VerificationRunner
from veritas.providers.base import SearchResult

class MockLLM:
    async def generate(self, prompt: str, system: str = "") -> str:
        sys_lower = system.lower()
        if "synthe" in sys_lower:
            return json.dumps({"verdict": "REFUTED", "confidence": 0.91, "summary": "The first iPhone was released June 2007, not 2006.", "failure_modes": [{"type": "factual_error", "detail": "Wrong release year", "agent": "source_verifier"}], "contested": False})
        elif "logic" in sys_lower:
            return json.dumps({"finding": "consistent", "confidence": 0.9, "details": []})
        elif "source" in sys_lower:
            return json.dumps({"finding": "contradiction", "confidence": 0.9, "details": [{"type": "factual_error", "description": "iPhone released 2007, not 2006"}], "sources": ["https://en.wikipedia.org/wiki/IPhone"], "reasoning": "Wikipedia confirms 2007 release"})
        elif "adversary" in sys_lower or "disprove" in sys_lower:
            return json.dumps({"finding": "counterexample_found", "confidence": 0.85, "details": [{"type": "factual_error", "description": "2006 is incorrect"}], "reasoning": "Multiple sources confirm 2007"})
        elif "calibrat" in sys_lower:
            return json.dumps({"finding": "overconfident", "confidence": 0.7, "details": [], "reasoning": "Claim states a specific date with certainty"})
        return json.dumps({"finding": "consistent", "confidence": 0.5, "details": []})

class MockSearch:
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        return [SearchResult(title="iPhone - Wikipedia", url="https://en.wikipedia.org/wiki/IPhone", snippet="The iPhone was first released on June 29, 2007.")]

@pytest.mark.asyncio
async def test_full_pipeline_refuted():
    runner = VerificationRunner(llm_provider=MockLLM(), search_provider=MockSearch(), config=Config())
    result = await runner.run(claim="The first iPhone was released in 2006", context=None, domain="technical", references=[])
    assert isinstance(result, VerificationResult)
    assert result.verdict == Verdict.REFUTED
    assert result.confidence > 0.8
    assert len(result.evidence) == 4
    assert len(result.failure_modes) == 1
    assert result.failure_modes[0].type == FailureModeType.FACTUAL_ERROR
    assert "2007" in result.summary
    text = str(result)
    assert "REFUTED" in text
    d = result.to_dict()
    assert d["verdict"] == "REFUTED"
    json.dumps(d)
    report = result.report()
    assert "factual_error" in report
    assert "source_verifier" in report

@pytest.mark.asyncio
async def test_full_pipeline_verified():
    class VerifiedMockLLM:
        async def generate(self, prompt: str, system: str = "") -> str:
            sys_lower = system.lower()
            if "synthe" in sys_lower:
                return json.dumps({"verdict": "VERIFIED", "confidence": 0.95, "summary": "Claim is accurate.", "failure_modes": [], "contested": False})
            elif "source" in sys_lower:
                return json.dumps({"finding": "supported", "confidence": 0.9, "details": [], "sources": [], "reasoning": ""})
            elif "adversary" in sys_lower or "disprove" in sys_lower:
                return json.dumps({"finding": "no_counterexample", "confidence": 0.9, "details": [], "reasoning": ""})
            elif "calibrat" in sys_lower:
                return json.dumps({"finding": "well_calibrated", "confidence": 0.9, "details": [], "reasoning": ""})
            return json.dumps({"finding": "consistent", "confidence": 0.9, "details": []})

    runner = VerificationRunner(llm_provider=VerifiedMockLLM(), search_provider=MockSearch(), config=Config())
    result = await runner.run(claim="Water boils at 100 degrees Celsius at sea level", context=None, domain="scientific", references=[])
    assert result.verdict == Verdict.VERIFIED
    assert result.confidence > 0.9
    assert len(result.failure_modes) == 0
