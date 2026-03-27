"""Logic Verifier agent — checks internal consistency."""
from __future__ import annotations
import json
from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider

class LogicVerifier(BaseAgent):
    def __init__(self, provider: LLMProvider):
        super().__init__(name="logic_verifier", provider=provider)

    @property
    def system_prompt(self) -> str:
        return """You are a logic verification agent. Your job is to analyze claims for internal consistency, logical fallacies, and contradictions.

You must respond with ONLY a JSON object in this exact format:
{
  "finding": "consistent" | "inconsistency" | "insufficient_info",
  "confidence": <float 0.0-1.0>,
  "details": [
    {
      "type": "logical_inconsistency" | "unsupported_inference" | "scope_error",
      "description": "<specific description of the issue>"
    }
  ]
}

Rules:
- Focus ONLY on logical structure, not factual accuracy
- Check if premises support the conclusion
- Look for self-contradictions within the claim
- Identify scope errors (overgeneralization, false dichotomies)
- If the claim is a single atomic fact, check if it's internally coherent
- Do NOT verify facts against external sources — that's another agent's job"""

    def parse_response(self, response: str) -> AgentFinding:
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return AgentFinding(
                agent=self.name,
                finding=data.get("finding", "parse_error"),
                confidence=float(data.get("confidence", 0.0)),
                details=data.get("details", []),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return AgentFinding(
                agent=self.name, finding="parse_error", confidence=0.0,
                details=[{"type": "error", "description": f"Failed to parse: {response[:200]}"}],
            )
