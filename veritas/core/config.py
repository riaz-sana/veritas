"""Veritas configuration and error handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class VeritasConfigError(Exception):
    """Raised when Veritas configuration is invalid."""


@dataclass
class AgentModels:
    """Per-agent model overrides for cost optimization.

    Use cheaper/faster models for simple agents, expensive ones for critical agents.

    Default tier:   all agents use the main model (Sonnet)
    Economy tier:   logic/calibration/synthesiser use Haiku, source/adversary use Sonnet
    Custom:         set each agent model individually
    """
    logic: str = ""
    source: str = ""
    adversary: str = ""
    calibration: str = ""
    synthesiser: str = ""

    @classmethod
    def default(cls, model: str) -> AgentModels:
        """All agents use the same model."""
        return cls(logic=model, source=model, adversary=model, calibration=model, synthesiser=model)

    @classmethod
    def economy(cls, main_model: str = "claude-sonnet-4-6", cheap_model: str = "claude-haiku-4-5") -> AgentModels:
        """Cheap models for simple agents, expensive for critical ones. ~60% cost reduction."""
        return cls(
            logic=cheap_model,
            source=main_model,
            adversary=main_model,
            calibration=cheap_model,
            synthesiser=cheap_model,
        )


@dataclass
class Config:
    """Veritas configuration with enterprise features."""

    # LLM settings
    model: str = ""
    agent_models: AgentModels | None = None
    search_provider: str = "brave"
    anthropic_api_key: str = ""
    search_api_key: str = ""

    # Verification behavior
    challenge_round: bool = True
    max_search_results: int = 5
    timeout_seconds: int = 30
    verbose: bool = False

    # Enterprise: confidence routing
    confidence_routing: bool = False
    confidence_threshold: float = 0.8  # Skip verification if source confidence >= this

    # Enterprise: caching
    cache_enabled: bool = False
    cache_path: str = ""  # SQLite path, defaults to .veritas/cache.db
    cache_ttl_seconds: int = 3600  # 1 hour default

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get("VERITAS_MODEL", "claude-sonnet-4-6")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.search_api_key:
            self.search_api_key = os.environ.get(
                "BRAVE_API_KEY", os.environ.get("TAVILY_API_KEY", "")
            )
        if not self.cache_path:
            self.cache_path = os.environ.get("VERITAS_CACHE_PATH", ".veritas/cache.db")
        if self.agent_models is None:
            self.agent_models = AgentModels.default(self.model)

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.anthropic_api_key:
            raise VeritasConfigError(
                "ANTHROPIC_API_KEY is required. Set it as an environment variable "
                "or pass anthropic_api_key to Config()."
            )
