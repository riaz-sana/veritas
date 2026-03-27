"""RAG Diagnostic Engine — root-cause diagnosis of RAG pipeline failures.

Existing tools say "your answer is wrong." Veritas tells you WHERE it failed and WHY.

Separates 5 failure modes:
- RETRIEVAL_MISS: relevant documents exist but weren't retrieved
- RETRIEVAL_NOISE: irrelevant documents were retrieved, misleading the LLM
- GENERATION_HALLUCINATION: LLM fabricated facts not in retrieved documents
- GENERATION_CONTRADICTION: LLM contradicted facts IN the retrieved documents
- KNOWLEDGE_GAP: the answer isn't in the knowledge base at all

Usage:
    from veritas.diagnostics.rag import diagnose_rag

    result = await diagnose_rag(
        query="What is our refund policy?",
        retrieved_docs=["Policy doc 1...", "Policy doc 2..."],
        generated_answer="Our refund window is 90 days...",
    )

    result.diagnosis        # RAGDiagnosis.GENERATION_HALLUCINATION
    result.root_cause       # "Answer states '90 days' but docs say '30 days' (policy.md)"
    result.fix_suggestion   # "Generation is unfaithful. Consider constraining output..."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum

from veritas.core.config import Config
from veritas.core.result import AgentFinding, FailureMode, Verdict, VerificationResult
from veritas.providers.base import LLMProvider
from veritas.providers.claude import ClaudeProvider


class RAGDiagnosis(str, Enum):
    """Root-cause categories for RAG pipeline failures."""
    FAITHFUL = "faithful"
    RETRIEVAL_MISS = "retrieval_miss"
    RETRIEVAL_NOISE = "retrieval_noise"
    GENERATION_HALLUCINATION = "generation_hallucination"
    GENERATION_CONTRADICTION = "generation_contradiction"
    KNOWLEDGE_GAP = "knowledge_gap"
    PARTIAL_RETRIEVAL = "partial_retrieval"


@dataclass
class RAGDiagnosticResult:
    """Complete diagnostic result for a RAG pipeline output."""

    diagnosis: RAGDiagnosis
    root_cause: str
    fix_suggestion: str

    # Scores for each pipeline stage (0.0 = bad, 1.0 = good)
    retrieval_relevance: float    # Are retrieved docs relevant to the query?
    generation_fidelity: float    # Does the answer stick to retrieved docs?
    answer_completeness: float    # Does the answer fully address the query?
    knowledge_coverage: float     # Is the answer in the knowledge base?

    # Evidence
    verification_result: VerificationResult | None = None
    retrieval_analysis: dict = field(default_factory=dict)
    generation_analysis: dict = field(default_factory=dict)

    # Metadata
    duration_ms: int = 0

    def __str__(self) -> str:
        return (
            f"{self.diagnosis.value} — {self.root_cause}\n"
            f"  Retrieval relevance:  {self.retrieval_relevance:.0%}\n"
            f"  Generation fidelity:  {self.generation_fidelity:.0%}\n"
            f"  Answer completeness:  {self.answer_completeness:.0%}\n"
            f"  Knowledge coverage:   {self.knowledge_coverage:.0%}\n"
            f"  Fix: {self.fix_suggestion}"
        )

    def to_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis.value,
            "root_cause": self.root_cause,
            "fix_suggestion": self.fix_suggestion,
            "retrieval_relevance": round(self.retrieval_relevance, 3),
            "generation_fidelity": round(self.generation_fidelity, 3),
            "answer_completeness": round(self.answer_completeness, 3),
            "knowledge_coverage": round(self.knowledge_coverage, 3),
            "retrieval_analysis": self.retrieval_analysis,
            "generation_analysis": self.generation_analysis,
            "duration_ms": self.duration_ms,
        }

    def report(self) -> str:
        lines = [
            "# RAG Diagnostic Report",
            "",
            f"**Diagnosis:** {self.diagnosis.value}",
            f"**Root Cause:** {self.root_cause}",
            f"**Fix Suggestion:** {self.fix_suggestion}",
            "",
            "## Pipeline Stage Scores",
            "",
            f"| Stage | Score | Status |",
            f"|-------|-------|--------|",
            f"| Retrieval Relevance | {self.retrieval_relevance:.0%} | {'OK' if self.retrieval_relevance > 0.7 else 'ISSUE'} |",
            f"| Generation Fidelity | {self.generation_fidelity:.0%} | {'OK' if self.generation_fidelity > 0.7 else 'ISSUE'} |",
            f"| Answer Completeness | {self.answer_completeness:.0%} | {'OK' if self.answer_completeness > 0.7 else 'ISSUE'} |",
            f"| Knowledge Coverage | {self.knowledge_coverage:.0%} | {'OK' if self.knowledge_coverage > 0.7 else 'ISSUE'} |",
            "",
        ]
        if self.retrieval_analysis:
            lines.append("## Retrieval Analysis")
            lines.append("")
            for key, val in self.retrieval_analysis.items():
                lines.append(f"- **{key}:** {val}")
            lines.append("")
        if self.generation_analysis:
            lines.append("## Generation Analysis")
            lines.append("")
            for key, val in self.generation_analysis.items():
                lines.append(f"- **{key}:** {val}")
            lines.append("")
        return "\n".join(lines)


_DIAGNOSE_SYSTEM_PROMPT = """You are a RAG pipeline diagnostic engine. You analyze WHY a RAG system produced a given answer.

You receive:
- The user's query
- The documents that were retrieved
- The generated answer

Your job is to determine the ROOT CAUSE of any issues by analyzing each pipeline stage independently.

You must respond with ONLY a JSON object:
{
  "retrieval_relevance": <float 0.0-1.0 — are retrieved docs relevant to the query?>,
  "generation_fidelity": <float 0.0-1.0 — does the answer stick to facts in the docs?>,
  "answer_completeness": <float 0.0-1.0 — does the answer fully address the query?>,
  "knowledge_coverage": <float 0.0-1.0 — could the answer be derived from the docs?>,
  "diagnosis": "faithful" | "retrieval_miss" | "retrieval_noise" | "generation_hallucination" | "generation_contradiction" | "knowledge_gap" | "partial_retrieval",
  "root_cause": "<specific explanation of what went wrong and where>",
  "fix_suggestion": "<actionable suggestion for the team>",
  "retrieval_analysis": {
    "relevant_docs": <count of docs actually relevant to query>,
    "total_docs": <total retrieved docs>,
    "missing_topics": "<topics the query asks about but docs don't cover>",
    "noise_docs": "<docs that are irrelevant to the query>"
  },
  "generation_analysis": {
    "fabricated_claims": ["<list of claims in the answer NOT supported by docs>"],
    "contradicted_claims": ["<list of claims that CONTRADICT the docs>"],
    "supported_claims": ["<list of claims properly grounded in docs>"],
    "source_mapping": "<which doc supports which claim>"
  }
}

Diagnosis rules:
- "faithful": Answer is accurate and grounded in the retrieved documents
- "retrieval_miss": Query asks about X but retrieved docs don't contain X (retrieval failure)
- "retrieval_noise": Retrieved docs are mostly irrelevant, confusing the LLM
- "generation_hallucination": Docs are fine but LLM added facts not in them
- "generation_contradiction": LLM stated something that contradicts the docs
- "knowledge_gap": The answer to the query genuinely doesn't exist in the docs
- "partial_retrieval": Some relevant docs retrieved, but key ones missing

Be SPECIFIC in root_cause — cite exact text from the docs and answer.
Be ACTIONABLE in fix_suggestion — tell the team exactly what to change."""


async def diagnose_rag(
    query: str,
    retrieved_docs: list[str],
    generated_answer: str,
    config: Config | None = None,
) -> RAGDiagnosticResult:
    """Diagnose root cause of a RAG pipeline output.

    Args:
        query: The user's original query.
        retrieved_docs: List of document strings that were retrieved.
        generated_answer: The answer the RAG pipeline produced.
        config: Optional Veritas config.

    Returns:
        RAGDiagnosticResult with diagnosis, scores, root cause, and fix suggestion.
    """
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    # Build the diagnostic prompt
    docs_text = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)
    )
    prompt = (
        f"## User Query\n{query}\n\n"
        f"## Retrieved Documents ({len(retrieved_docs)} total)\n{docs_text}\n\n"
        f"## Generated Answer\n{generated_answer}\n\n"
        "Analyze the RAG pipeline and diagnose the root cause of any issues."
    )

    response = await provider.generate(prompt, system=_DIAGNOSE_SYSTEM_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    # Parse response
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        return RAGDiagnosticResult(
            diagnosis=RAGDiagnosis(data.get("diagnosis", "faithful")),
            root_cause=data.get("root_cause", ""),
            fix_suggestion=data.get("fix_suggestion", ""),
            retrieval_relevance=float(data.get("retrieval_relevance", 0.0)),
            generation_fidelity=float(data.get("generation_fidelity", 0.0)),
            answer_completeness=float(data.get("answer_completeness", 0.0)),
            knowledge_coverage=float(data.get("knowledge_coverage", 0.0)),
            retrieval_analysis=data.get("retrieval_analysis", {}),
            generation_analysis=data.get("generation_analysis", {}),
            duration_ms=duration_ms,
        )
    except (json.JSONDecodeError, ValueError) as e:
        return RAGDiagnosticResult(
            diagnosis=RAGDiagnosis.FAITHFUL,
            root_cause=f"Diagnostic parse error: {e}",
            fix_suggestion="Re-run diagnosis",
            retrieval_relevance=0.0,
            generation_fidelity=0.0,
            answer_completeness=0.0,
            knowledge_coverage=0.0,
            duration_ms=duration_ms,
        )
