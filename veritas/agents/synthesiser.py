"""Synthesiser agent — aggregates all findings into a structured verdict."""
from __future__ import annotations
import json
import time
from veritas.agents.base import BaseAgent
from veritas.core.result import AgentFinding, FailureMode, FailureModeType, Verdict, VerificationResult
from veritas.providers.base import LLMProvider

_FAILURE_TYPE_MAP = {v.value: v for v in FailureModeType}

class Synthesiser(BaseAgent):
    def __init__(self, provider: LLMProvider):
        super().__init__(name="synthesiser", provider=provider)

    @property
    def system_prompt(self) -> str:
        return """You are a synthesis agent. You receive findings from multiple independent verification agents and produce a final verdict.

You must respond with ONLY a JSON object:
{
  "verdict": "VERIFIED" | "PARTIAL" | "UNCERTAIN" | "DISPUTED" | "REFUTED",
  "confidence": <float 0.0-1.0>,
  "summary": "<one-sentence summary>",
  "failure_modes": [{"type": "<failure_type>", "detail": "<description>", "agent": "<agent>"}],
  "contested": <true if agents significantly disagree>
}

Verdict rules:
- VERIFIED: All agents agree claim is supported
- PARTIAL: Some parts verified, others not
- UNCERTAIN: Insufficient evidence
- DISPUTED: Agents significantly disagree
- REFUTED: Clear evidence contradicts claim

failure_mode types: factual_error, logical_inconsistency, unsupported_inference, temporal_error, scope_error, source_conflict"""

    def parse_response(self, response: str) -> AgentFinding:
        return AgentFinding(agent=self.name, finding="", confidence=0.0, details=[])

    async def synthesise(self, claim: str, findings: list[AgentFinding]) -> VerificationResult:
        start = time.monotonic()
        findings_text = "\n\n".join(
            f"### {f.agent}\n- Finding: {f.finding}\n- Confidence: {f.confidence}\n- Details: {json.dumps(f.details)}\n- Reasoning: {f.reasoning}"
            for f in findings
        )
        prompt = f"## Claim\n{claim}\n\n## Agent Findings\n{findings_text}\n\nSynthesise these findings into a single verdict."
        response = await self.provider.generate(prompt, system=self.system_prompt)
        duration_ms = int((time.monotonic() - start) * 1000)

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            failure_modes = []
            for fm in data.get("failure_modes", []):
                fm_type = _FAILURE_TYPE_MAP.get(fm.get("type", ""))
                if fm_type:
                    failure_modes.append(FailureMode(type=fm_type, detail=fm.get("detail", ""), agent=fm.get("agent", "unknown")))
            return VerificationResult(
                verdict=Verdict(data["verdict"]), confidence=float(data.get("confidence", 0.0)),
                summary=data.get("summary", ""), failure_modes=failure_modes, evidence=findings,
                contested=data.get("contested", False), challenge_round=None,
                metadata={"duration_ms": duration_ms, "agents_used": len(findings), "model": "synthesiser"},
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return VerificationResult(
                verdict=Verdict.UNCERTAIN, confidence=0.0, summary="Failed to synthesise.",
                failure_modes=[], evidence=findings, contested=False, challenge_round=None,
                metadata={"duration_ms": duration_ms, "agents_used": len(findings), "error": f"Parse error: {response[:200]}"},
            )
