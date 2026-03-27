"""Verification runner — orchestrates all agents with tiered model support."""
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
from veritas.providers.claude import ClaudeProvider


class VerificationRunner:
    """Orchestrates parallel agent verification with optional tiered models."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        search_provider: SearchProvider,
        config: Config,
    ):
        self.config = config

        # Tiered models: each agent can use a different model
        agent_models = config.agent_models
        if agent_models and isinstance(llm_provider, ClaudeProvider):
            api_key = config.anthropic_api_key
            logic_provider = ClaudeProvider(model=agent_models.logic, api_key=api_key)
            source_provider = ClaudeProvider(model=agent_models.source, api_key=api_key)
            adversary_provider = ClaudeProvider(model=agent_models.adversary, api_key=api_key)
            calibration_provider = ClaudeProvider(model=agent_models.calibration, api_key=api_key)
            synthesiser_provider = ClaudeProvider(model=agent_models.synthesiser, api_key=api_key)
        else:
            # All agents share the same provider
            logic_provider = source_provider = adversary_provider = llm_provider
            calibration_provider = synthesiser_provider = llm_provider

        self.logic = LogicVerifier(provider=logic_provider)
        self.source = SourceVerifier(provider=source_provider, search_provider=search_provider)
        self.adversary = Adversary(provider=adversary_provider)
        self.calibration = CalibrationAgent(provider=calibration_provider)
        self.synthesiser = Synthesiser(provider=synthesiser_provider)

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
        if self.config.agent_models:
            result.metadata["agent_models"] = {
                "logic": self.config.agent_models.logic,
                "source": self.config.agent_models.source,
                "adversary": self.config.agent_models.adversary,
                "calibration": self.config.agent_models.calibration,
                "synthesiser": self.config.agent_models.synthesiser,
            }
        return result
