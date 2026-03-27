"""Tests for source verifier agent."""
import json
import pytest
from veritas.agents.source import SourceVerifier
from veritas.core.result import AgentFinding
from veritas.providers.base import SearchResult

class MockProvider:
    def __init__(self, response: str):
        self.response = response
    async def generate(self, prompt: str, system: str = "") -> str:
        return self.response

class MockSearchProvider:
    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results or []
        self.queries: list[str] = []
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return self.results

def test_source_verifier_system_prompt():
    agent = SourceVerifier(provider=MockProvider(""), search_provider=MockSearchProvider())
    assert "source" in agent.system_prompt.lower() or "factual" in agent.system_prompt.lower()

def test_source_verifier_parse_supported():
    agent = SourceVerifier(provider=MockProvider(""), search_provider=MockSearchProvider())
    response = json.dumps({"finding": "supported", "confidence": 0.9, "details": [], "sources": ["https://example.com"], "reasoning": "Found corroborating evidence"})
    finding = agent.parse_response(response)
    assert finding.finding == "supported"
    assert len(finding.sources) == 1

def test_source_verifier_parse_contradiction():
    agent = SourceVerifier(provider=MockProvider(""), search_provider=MockSearchProvider())
    response = json.dumps({"finding": "contradiction", "confidence": 0.85, "details": [{"type": "factual_error", "description": "Wrong date"}], "sources": ["https://example.com/correct"], "reasoning": "Source says 2007, claim says 2006"})
    finding = agent.parse_response(response)
    assert finding.finding == "contradiction"
    assert finding.reasoning == "Source says 2007, claim says 2006"

def test_source_verifier_parse_malformed():
    agent = SourceVerifier(provider=MockProvider(""), search_provider=MockSearchProvider())
    finding = agent.parse_response("not json")
    assert finding.finding == "parse_error"

@pytest.mark.asyncio
async def test_source_verifier_builds_search_context():
    search_results = [SearchResult(title="iPhone History", url="https://example.com/iphone", snippet="The iPhone was released June 29, 2007")]
    llm_response = json.dumps({"finding": "contradiction", "confidence": 0.9, "details": [{"type": "factual_error", "description": "Wrong year"}], "sources": ["https://example.com/iphone"], "reasoning": "2007 not 2006"})
    search = MockSearchProvider(search_results)
    agent = SourceVerifier(provider=MockProvider(llm_response), search_provider=search)
    finding = await agent.verify(claim="The iPhone was released in 2006", context=None, domain=None, references=[])
    assert finding.finding == "contradiction"
    assert len(search.queries) >= 1
