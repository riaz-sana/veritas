"""Veritas — Adversarial parallel verification of AI outputs."""
__version__ = "0.1.0"

from veritas.core.config import AgentModels, Config, VeritasConfigError
from veritas.core.result import (
    AgentFinding, ChallengeResult, FailureMode, FailureModeType, Verdict, VerificationResult,
)
from veritas.core.verify import verify

__all__ = [
    "verify", "Config", "AgentModels", "VeritasConfigError", "Verdict", "VerificationResult",
    "FailureMode", "FailureModeType", "AgentFinding", "ChallengeResult",
]
