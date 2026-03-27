"""Veritas data models for verification results."""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNCERTAIN = "UNCERTAIN"
    DISPUTED = "DISPUTED"
    REFUTED = "REFUTED"


class FailureModeType(str, Enum):
    FACTUAL_ERROR = "factual_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    TEMPORAL_ERROR = "temporal_error"
    SCOPE_ERROR = "scope_error"
    SOURCE_CONFLICT = "source_conflict"


class FailureMode(BaseModel):
    type: FailureModeType
    detail: str
    agent: str


class AgentFinding(BaseModel):
    agent: str
    finding: str
    confidence: float
    details: list[dict]
    sources: list[str] = []
    reasoning: str = ""


class ChallengeResult(BaseModel):
    contested_points: list[str]
    adversary_finding: AgentFinding
    resolution: str


class VerificationResult(BaseModel):
    verdict: Verdict
    confidence: float
    summary: str
    failure_modes: list[FailureMode]
    evidence: list[AgentFinding]
    contested: bool
    challenge_round: ChallengeResult | None
    metadata: dict

    def __str__(self) -> str:
        return f"{self.verdict.value} ({self.confidence:.2f}) — {self.summary}"

    def to_dict(self) -> dict:
        return json.loads(self.model_dump_json())

    def report(self) -> str:
        lines = [
            f"# Verification Report",
            f"",
            f"**Verdict:** {self.verdict.value}",
            f"**Confidence:** {self.confidence:.2f}",
            f"**Summary:** {self.summary}",
            f"",
        ]
        if self.failure_modes:
            lines.append("## Failure Modes")
            lines.append("")
            for fm in self.failure_modes:
                lines.append(f"- **{fm.type.value}** ({fm.agent}): {fm.detail}")
            lines.append("")
        if self.evidence:
            lines.append("## Evidence")
            lines.append("")
            for ev in self.evidence:
                lines.append(f"### {ev.agent}")
                lines.append(f"- Finding: {ev.finding}")
                lines.append(f"- Confidence: {ev.confidence:.2f}")
                if ev.reasoning:
                    lines.append(f"- Reasoning: {ev.reasoning}")
                if ev.sources:
                    lines.append(f"- Sources: {', '.join(ev.sources)}")
                lines.append("")
        if self.contested and self.challenge_round:
            lines.append("## Challenge Round")
            lines.append("")
            lines.append(f"Contested points: {', '.join(self.challenge_round.contested_points)}")
            lines.append(f"Resolution: {self.challenge_round.resolution}")
            lines.append("")
        return "\n".join(lines)
