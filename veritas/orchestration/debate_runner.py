"""Debate-mode runner — agents share context (control group for comparison)."""
from __future__ import annotations
import asyncio
import time
import json
from veritas.agents.adversary import Adversary
from veritas.agents.calibration import CalibrationAgent
from veritas.agents.logic import LogicVerifier
from veritas.agents.source import SourceVerifier
from veritas.agents.synthesiser import Synthesiser
from veritas.core.config import Config
from veritas.core.result import AgentFinding, VerificationResult
from veritas.providers.base import LLMProvider, SearchProvider


class DebateRunner:
    """Runs verification with shared context — agents see previous findings.

    This is the CONTROL GROUP for the isolation-vs-debate experiment.
    Unlike VerificationRunner (isolated), agents here run sequentially
    and each sees all previous agents' findings in their prompt.
    This mirrors the standard multi-agent debate approach (Du et al., ICML 2024).
    """

    def __init__(self, llm_provider: LLMProvider, search_provider: SearchProvider, config: Config):
        self.config = config
        self.logic = LogicVerifier(provider=llm_provider)
        self.source = SourceVerifier(provider=llm_provider, search_provider=search_provider)
        self.adversary = Adversary(provider=llm_provider)
        self.calibration = CalibrationAgent(provider=llm_provider)
        self.synthesiser = Synthesiser(provider=llm_provider)

    def _format_prior_findings(self, findings: list[AgentFinding]) -> str:
        if not findings:
            return ""
        lines = ["\n## Previous Agent Findings (shared context)"]
        for f in findings:
            lines.append(f"\n### {f.agent}")
            lines.append(f"- Finding: {f.finding}")
            lines.append(f"- Confidence: {f.confidence}")
            if f.details:
                lines.append(f"- Details: {json.dumps(f.details)}")
            if f.reasoning:
                lines.append(f"- Reasoning: {f.reasoning}")
        return "\n".join(lines)

    async def _verify_with_context(self, agent, claim, context, domain, references, prior_findings):
        """Run an agent with prior findings injected into its prompt."""
        prior_text = self._format_prior_findings(prior_findings)
        augmented_context = (context or "") + prior_text
        return await agent.verify(claim, augmented_context, domain, references)

    async def run(self, claim: str, context: str | None, domain: str | None, references: list[str]) -> VerificationResult:
        start = time.monotonic()
        findings: list[AgentFinding] = []

        # Sequential: each agent sees all previous findings (shared context)
        logic_finding = await self._verify_with_context(
            self.logic, claim, context, domain, references, findings)
        findings.append(logic_finding)

        source_finding = await self._verify_with_context(
            self.source, claim, context, domain, references, findings)
        findings.append(source_finding)

        adversary_finding = await self._verify_with_context(
            self.adversary, claim, context, domain, references, findings)
        findings.append(adversary_finding)

        calibration_finding = await self._verify_with_context(
            self.calibration, claim, context, domain, references, findings)
        findings.append(calibration_finding)

        # Synthesise
        result = await self.synthesiser.synthesise(claim=claim, findings=findings)

        duration_ms = int((time.monotonic() - start) * 1000)
        result.metadata["total_duration_ms"] = duration_ms
        result.metadata["model"] = self.config.model
        result.metadata["mode"] = "debate"
        return result
