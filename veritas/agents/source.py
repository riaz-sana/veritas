"""Source Verifier agent — factual cross-reference via web search."""
from __future__ import annotations
import json
from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider, SearchProvider

class SourceVerifier(BaseAgent):
    def __init__(self, provider: LLMProvider, search_provider: SearchProvider):
        super().__init__(name="source_verifier", provider=provider)
        self.search_provider = search_provider

    @property
    def system_prompt(self) -> str:
        return """You are a source verification agent. Your job is to check factual claims against provided search results and references.

You must respond with ONLY a JSON object in this exact format:
{
  "finding": "supported" | "contradiction" | "insufficient_info",
  "confidence": <float 0.0-1.0>,
  "details": [{"type": "factual_error" | "temporal_error" | "source_conflict", "description": "<specific description>"}],
  "sources": ["<url1>", "<url2>"],
  "reasoning": "<step-by-step reasoning>"
}

Rules:
- Compare EVERY factual element against search results
- Cite specific sources
- If sources disagree, note source_conflict
- If no relevant sources, report insufficient_info"""

    async def verify(self, claim: str, context: str | None, domain: str | None, references: list[str]) -> AgentFinding:
        try:
            search_results = await self.search_provider.search(claim)
        except Exception:
            search_results = []
        search_context = "\n\n".join(f"**{r.title}** ({r.url})\n{r.snippet}" for r in search_results) if search_results else "No search results available."
        prompt_parts = [self.build_prompt(claim, context, domain, references)]
        prompt_parts.append(f"\n## Search Results\n{search_context}")
        full_prompt = "\n".join(prompt_parts)
        response = await self.provider.generate(full_prompt, system=self.system_prompt)
        return self.parse_response(response)

    def parse_response(self, response: str) -> AgentFinding:
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return AgentFinding(
                agent=self.name, finding=data.get("finding", "parse_error"),
                confidence=float(data.get("confidence", 0.0)),
                details=data.get("details", []),
                sources=data.get("sources", []),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return AgentFinding(
                agent=self.name, finding="parse_error", confidence=0.0,
                details=[{"type": "error", "description": f"Failed to parse: {response[:200]}"}],
            )
