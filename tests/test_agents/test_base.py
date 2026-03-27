"""Tests for base agent interface."""

import pytest

from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider


class MockProvider:
    def __init__(self, response: str = "mock response"):
        self.response = response
        self.calls: list[dict] = []

    async def generate(self, prompt: str, system: str = "") -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.response


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent(name="test", provider=MockProvider())


class ConcreteAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    def parse_response(self, response: str) -> AgentFinding:
        return AgentFinding(
            agent=self.name,
            finding="test",
            confidence=1.0,
            details=[],
        )


def test_concrete_agent_creation():
    provider = MockProvider()
    agent = ConcreteAgent(name="test_agent", provider=provider)
    assert agent.name == "test_agent"


@pytest.mark.asyncio
async def test_concrete_agent_verify():
    provider = MockProvider(response='{"finding": "consistent"}')
    agent = ConcreteAgent(name="test_agent", provider=provider)
    finding = await agent.verify(
        claim="test claim",
        context=None,
        domain=None,
        references=[],
    )
    assert finding.agent == "test_agent"
    assert finding.finding == "test"
    assert len(provider.calls) == 1
