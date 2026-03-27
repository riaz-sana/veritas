"""Adversary agent — constructs counterexamples and challenges claims."""
from __future__ import annotations
import json
from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider

class Adversary(BaseAgent):
    def __init__(self, provider: LLMProvider):
        super().__init__(name="adversary", provider=provider)

    @property
    def system_prompt(self) -> str:
        return """You are an adversary verification agent. Your ONLY job is to try to disprove the claim. Construct counterexamples, find edge cases, attempt to show the claim is false.

You must respond with ONLY a JSON object:
{
  "finding": "counterexample_found" | "no_counterexample" | "insufficient_info",
  "confidence": <float 0.0-1.0>,
  "details": [{"type": "factual_error" | "logical_inconsistency" | "scope_error" | "temporal_error" | "unsupported_inference", "description": "<specific counterexample>"}],
  "reasoning": "<step-by-step adversarial reasoning>"
}

Rules:
- Default stance is SKEPTICISM
- Try reductio ad absurdum
- Look for edge cases, exceptions, boundary conditions
- Check for overgeneralizations and unwarranted causal claims
- If you genuinely cannot find a counterexample, say so
- Do NOT be contrarian for its own sake"""

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
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return AgentFinding(
                agent=self.name, finding="parse_error", confidence=0.0,
                details=[{"type": "error", "description": f"Failed to parse: {response[:200]}"}],
            )

    async def challenge(self, claim: str, contested_points: list[str], agent_findings: list[AgentFinding]) -> AgentFinding:
        findings_text = "\n".join(f"- {f.agent}: {f.finding} (confidence: {f.confidence})" for f in agent_findings)
        prompt = (
            f"## Claim\n{claim}\n\n"
            f"## Contested Points\n" + "\n".join(f"- {p}" for p in contested_points) + "\n\n"
            f"## Agent Findings\n{findings_text}\n\n"
            "Focus your adversarial analysis specifically on the contested points above."
        )
        response = await self.provider.generate(prompt, system=self.system_prompt)
        return self.parse_response(response)
