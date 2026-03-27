"""Veritas — Adversarial parallel verification of AI outputs."""
__version__ = "0.1.0"

from veritas.core.config import AgentModels, Config, VeritasConfigError
from veritas.core.result import (
    AgentFinding, ChallengeResult, FailureMode, FailureModeType, Verdict, VerificationResult,
)
from veritas.core.verify import verify

# RAG Diagnostics (Option B)
from veritas.diagnostics.rag import diagnose_rag, RAGDiagnosis, RAGDiagnosticResult

# Pre-Action Verification (Option A)
from veritas.agentic.verification import (
    before_action, verify_action, verify_plan,
    ActionVerdict, ActionVerificationResult, ActionRisk,
    ActionBlockedError, ActionNeedsReviewError,
)

__all__ = [
    # Core
    "verify", "Config", "AgentModels", "VeritasConfigError",
    "Verdict", "VerificationResult", "FailureMode", "FailureModeType",
    "AgentFinding", "ChallengeResult",
    # RAG Diagnostics
    "diagnose_rag", "RAGDiagnosis", "RAGDiagnosticResult",
    # Pre-Action Verification
    "before_action", "verify_action", "verify_plan",
    "ActionVerdict", "ActionVerificationResult", "ActionRisk",
    "ActionBlockedError", "ActionNeedsReviewError",
]
