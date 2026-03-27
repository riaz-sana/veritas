"""Calibration agent — audits confidence vs evidence alignment."""
from __future__ import annotations
import json
from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding
from veritas.providers.base import LLMProvider

class CalibrationAgent(BaseAgent):
    def __init__(self, provider: LLMProvider):
        super().__init__(name="calibration", provider=provider)

    @property
    def system_prompt(self) -> str:
        return """You are a calibration verification agent. Your job is to assess whether a claim's apparent confidence level is justified by available evidence.

You must respond with ONLY a JSON object:
{
  "finding": "well_calibrated" | "overconfident" | "underconfident" | "insufficient_info",
  "confidence": <float 0.0-1.0>,
  "details": [{"type": "unsupported_inference" | "scope_error", "description": "<calibration concern>"}],
  "reasoning": "<analysis of confidence vs evidence>"
}

Rules:
- Absolute statements ("always", "never", "all") signal high confidence
- Compare claim confidence to verifiability
- Hedged claims ("usually", "often") are appropriately uncertain
- Specific numbers/dates signal high precision — are they justified?
- "overconfident" = sounds more certain than evidence warrants
- "underconfident" = hedges unnecessarily on established facts"""

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
