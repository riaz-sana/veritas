"""Base agent interface for verification agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from veritas.agents.domains import get_domain_extension
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

    def _agent_type(self) -> str:
        """Return agent type for domain extension lookup."""
        type_map = {
            "logic_verifier": "logic",
            "source_verifier": "source",
            "adversary": "adversary",
            "calibration": "calibration",
        }
        return type_map.get(self.name, "")

    def get_system_prompt(self, domain: str | None = None) -> str:
        """Get system prompt with domain-specific extensions."""
        base = self.system_prompt
        extension = get_domain_extension(self._agent_type(), domain)
        if extension:
            return base + "\n" + extension
        return base

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
        response = await self.provider.generate(prompt, system=self.get_system_prompt(domain))
        return self.parse_response(response)
