"""Veritas configuration and error handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class VeritasConfigError(Exception):
    """Raised when Veritas configuration is invalid."""


@dataclass
class Config:
    model: str = ""
    search_provider: str = "brave"
    anthropic_api_key: str = ""
    search_api_key: str = ""
    challenge_round: bool = True
    max_search_results: int = 5
    timeout_seconds: int = 30
    verbose: bool = False

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get("VERITAS_MODEL", "claude-sonnet-4-6")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.search_api_key:
            self.search_api_key = os.environ.get(
                "BRAVE_API_KEY", os.environ.get("TAVILY_API_KEY", "")
            )

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise VeritasConfigError(
                "ANTHROPIC_API_KEY is required. Set it as an environment variable "
                "or pass anthropic_api_key to Config()."
            )
