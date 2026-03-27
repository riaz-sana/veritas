"""Base agent interface for verification agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider


class BaseAgent(ABC):
    def __init__(self, name: str, provider: LLMProvider):
        self.name = name
        self.provider = provider

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def parse_response(self, response: str) -> AgentFinding: ...

    def build_prompt(
        self,
        claim: str,
        context: str | None,
        domain: str | None,
        references: list[str],
    ) -> str:
        parts = [f"## Claim to Verify\n{claim}"]
        if context:
            parts.append(f"\n## Context\n{context}")
        if domain:
            parts.append(f"\n## Domain\n{domain}")
        if references:
            parts.append(f"\n## References\n" + "\n".join(f"- {r}" for r in references))
        return "\n".join(parts)

    async def verify(
        self,
        claim: str,
        context: str | None,
        domain: str | None,
        references: list[str],
    ) -> AgentFinding:
        prompt = self.build_prompt(claim, context, domain, references)
        response = await self.provider.generate(prompt, system=self.system_prompt)
        return self.parse_response(response)
