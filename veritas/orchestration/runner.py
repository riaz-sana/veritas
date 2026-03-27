"""Verification runner — orchestrates all agents."""
from __future__ import annotations
import asyncio
import time
from veritas.agents.adversary import Adversary
from veritas.agents.calibration import CalibrationAgent
from veritas.agents.logic import LogicVerifier
from veritas.agents.source import SourceVerifier
from veritas.agents.synthesiser import Synthesiser
from veritas.core.config import Config
from veritas.core.result import VerificationResult
from veritas.orchestration.challenge import run_challenge_round
from veritas.providers.base import LLMProvider, SearchProvider

class VerificationRunner:
    def __init__(self, llm_provider: LLMProvider, search_provider: SearchProvider, config: Config):
        self.config = config
        self.logic = LogicVerifier(provider=llm_provider)
        self.source = SourceVerifier(provider=llm_provider, search_provider=search_provider)
        self.adversary = Adversary(provider=llm_provider)
        self.calibration = CalibrationAgent(provider=llm_provider)
        self.synthesiser = Synthesiser(provider=llm_provider)

    async def run(self, claim: str, context: str | None, domain: str | None, references: list[str]) -> VerificationResult:
        start = time.monotonic()
        findings = await asyncio.gather(
            self.logic.verify(claim, context, domain, references),
            self.source.verify(claim, context, domain, references),
            self.adversary.verify(claim, context, domain, references),
            self.calibration.verify(claim, context, domain, references),
        )
        findings_list = list(findings)
        result = await self.synthesiser.synthesise(claim=claim, findings=findings_list)
        if result.contested and self.config.challenge_round:
            result = await run_challenge_round(claim=claim, initial_result=result, adversary=self.adversary, synthesiser=self.synthesiser)
        duration_ms = int((time.monotonic() - start) * 1000)
        result.metadata["total_duration_ms"] = duration_ms
        result.metadata["model"] = self.config.model
        return result
