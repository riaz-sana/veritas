"""RAG Diagnostic Engine — multi-agent root-cause diagnosis of RAG failures.

Unlike existing tools (RAGAS, DeepEval) that give you a faithfulness SCORE,
this engine uses 4 specialized agents in isolation to diagnose the ROOT CAUSE
of a RAG pipeline failure — separating retrieval problems from generation
problems from knowledge gaps.

Architecture:
                     query + docs + answer
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Retrieval   │ │  Generation  │ │  Coverage    │
    │  Auditor     │ │  Auditor     │ │  Auditor     │
    │              │ │              │ │              │
    │ Are the docs │ │ Is the answer│ │ Could the    │
    │ relevant to  │ │ faithful to  │ │ answer be    │
    │ the query?   │ │ the docs?    │ │ derived from │
    │              │ │              │ │ the docs?    │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                 ┌──────────────────┐
                 │   Diagnostic     │
                 │   Synthesiser    │
                 │                  │
                 │ Combines stage   │
                 │ analyses into    │
                 │ root cause +     │
                 │ fix suggestion   │
                 └──────────────────┘

Each agent runs in isolation — the Retrieval Auditor doesn't know what the
Generation Auditor found, preventing confirmation bias in the diagnosis.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum

from veritas.core.config import Config
from veritas.providers.claude import ClaudeProvider
from veritas.providers.base import LLMProvider


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
class ClaimMapping:
    """Maps a single claim in the answer to its source (or lack thereof)."""
    claim: str
    grounded: bool
    source_doc_index: int | None = None  # Which doc supports this, if any
    source_quote: str = ""               # Exact quote from doc
    issue: str = ""                      # What's wrong, if not grounded


@dataclass
class RAGDiagnosticResult:
    """Complete diagnostic result for a RAG pipeline output."""

    diagnosis: RAGDiagnosis
    root_cause: str
    fix_suggestion: str

    # Per-stage scores (0.0 = bad, 1.0 = good)
    retrieval_relevance: float
    generation_fidelity: float
    answer_completeness: float
    knowledge_coverage: float

    # Detailed evidence from each auditor
    claim_mappings: list[ClaimMapping] = field(default_factory=list)
    retrieval_analysis: dict = field(default_factory=dict)
    generation_analysis: dict = field(default_factory=dict)
    coverage_analysis: dict = field(default_factory=dict)

    # Raw auditor findings
    auditor_findings: dict = field(default_factory=dict)

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
            "claim_mappings": [
                {"claim": cm.claim, "grounded": cm.grounded, "source_doc": cm.source_doc_index,
                 "source_quote": cm.source_quote, "issue": cm.issue}
                for cm in self.claim_mappings
            ],
            "retrieval_analysis": self.retrieval_analysis,
            "generation_analysis": self.generation_analysis,
            "coverage_analysis": self.coverage_analysis,
            "duration_ms": self.duration_ms,
        }

    def report(self) -> str:
        lines = [
            "# RAG Diagnostic Report",
            "",
            f"**Diagnosis:** {self.diagnosis.value}",
            f"**Root Cause:** {self.root_cause}",
            f"**Fix:** {self.fix_suggestion}",
            "",
            "## Pipeline Stage Scores",
            "",
            "| Stage | Score | Status |",
            "|-------|-------|--------|",
            f"| Retrieval Relevance | {self.retrieval_relevance:.0%} | {'OK' if self.retrieval_relevance > 0.7 else 'PROBLEM'} |",
            f"| Generation Fidelity | {self.generation_fidelity:.0%} | {'OK' if self.generation_fidelity > 0.7 else 'PROBLEM'} |",
            f"| Answer Completeness | {self.answer_completeness:.0%} | {'OK' if self.answer_completeness > 0.7 else 'PROBLEM'} |",
            f"| Knowledge Coverage  | {self.knowledge_coverage:.0%} | {'OK' if self.knowledge_coverage > 0.7 else 'PROBLEM'} |",
            "",
        ]
        if self.claim_mappings:
            lines.append("## Claim-Level Analysis")
            lines.append("")
            for cm in self.claim_mappings:
                status = "GROUNDED" if cm.grounded else "UNGROUNDED"
                lines.append(f"- [{status}] \"{cm.claim}\"")
                if cm.grounded and cm.source_quote:
                    lines.append(f"  Source (doc {cm.source_doc_index}): \"{cm.source_quote}\"")
                elif not cm.grounded and cm.issue:
                    lines.append(f"  Issue: {cm.issue}")
            lines.append("")
        return "\n".join(lines)


# ── Auditor Prompts ──────────────────────────────────────────────────

_RETRIEVAL_AUDITOR_PROMPT = """You are a Retrieval Auditor. You ONLY analyze whether the retrieved documents are relevant to the user's query. You know NOTHING about the generated answer — you haven't seen it.

Given a query and retrieved documents, analyze:
1. How many documents are relevant to the query?
2. Are there obvious topics the query asks about that NO document covers?
3. Are any documents completely irrelevant (noise)?
4. Would a human reading these documents have enough information to answer the query?

Respond with ONLY a JSON object:
{
  "relevance_score": <float 0.0-1.0>,
  "relevant_doc_indices": [<0-indexed list of relevant doc indices>],
  "irrelevant_doc_indices": [<0-indexed list of irrelevant doc indices>],
  "missing_topics": ["<topics the query asks about but docs don't cover>"],
  "could_answer_from_docs": <true if docs contain enough info to answer>,
  "reasoning": "<step-by-step analysis>"
}

Be SPECIFIC — cite exact parts of docs and query."""

_GENERATION_AUDITOR_PROMPT = """You are a Generation Auditor. You ONLY analyze whether the generated answer is faithful to the retrieved documents. You do NOT assess whether the answer is correct in general — only whether it's grounded in the provided documents.

For EACH distinct claim in the answer, determine:
1. Is this claim directly supported by a specific passage in the documents?
2. If supported, which document and what exact quote supports it?
3. If NOT supported, is it a fabrication (not in docs at all) or a contradiction (opposite of what docs say)?

Respond with ONLY a JSON object:
{
  "fidelity_score": <float 0.0-1.0>,
  "claim_analysis": [
    {
      "claim": "<extracted claim from the answer>",
      "grounded": <true/false>,
      "source_doc_index": <0-indexed doc index or null>,
      "source_quote": "<exact supporting quote from doc, or empty>",
      "issue_type": "none" | "fabrication" | "contradiction" | "exaggeration" | "oversimplification",
      "issue_detail": "<specific description of the problem>"
    }
  ],
  "fabricated_claims": ["<claims with no basis in docs>"],
  "contradicted_claims": ["<claims that oppose what docs say>"],
  "reasoning": "<step-by-step analysis>"
}

Be PRECISE — quote exact text from both the answer and the documents."""

_COVERAGE_AUDITOR_PROMPT = """You are a Coverage Auditor. You analyze whether the retrieved documents COULD contain the answer to the query, and whether the answer fully addresses what was asked.

You assess two things:
1. Knowledge coverage: Do the documents contain the information needed to answer the query?
2. Answer completeness: Does the answer address all aspects of the query?

Respond with ONLY a JSON object:
{
  "knowledge_coverage_score": <float 0.0-1.0 — could the query be answered from these docs?>,
  "answer_completeness_score": <float 0.0-1.0 — does the answer address all parts of the query?>,
  "query_aspects": ["<list of distinct things the query asks about>"],
  "covered_aspects": ["<aspects that docs cover>"],
  "uncovered_aspects": ["<aspects that docs DON'T cover>"],
  "answered_aspects": ["<aspects the answer addresses>"],
  "missed_aspects": ["<aspects the query asks about that the answer ignores>"],
  "reasoning": "<step-by-step analysis>"
}"""

_DIAGNOSTIC_SYNTHESISER_PROMPT = """You are a Diagnostic Synthesiser. You receive independent analyses from three auditors who each examined a different aspect of a RAG pipeline. They did NOT see each other's work.

Your job: combine their findings into a single ROOT CAUSE diagnosis.

Auditor findings:
- Retrieval Auditor: analyzed whether retrieved docs match the query
- Generation Auditor: analyzed whether the answer is faithful to the docs
- Coverage Auditor: analyzed whether docs could answer the query

Diagnosis rules:
- "faithful": All three auditors report positive scores. No issues found.
- "retrieval_miss": Retrieval Auditor reports low relevance AND Coverage Auditor says docs can't answer the query. The problem is retrieval, not generation.
- "retrieval_noise": Retrieval Auditor reports many irrelevant docs. LLM was confused by noise.
- "generation_hallucination": Retrieval Auditor says docs are relevant, but Generation Auditor found fabricated claims. LLM added facts not in docs.
- "generation_contradiction": Retrieval Auditor says docs are relevant, but Generation Auditor found contradicted claims. LLM stated the opposite of what docs say.
- "knowledge_gap": Coverage Auditor says docs genuinely don't contain the needed info. This isn't a retrieval failure — the knowledge isn't in the KB at all.
- "partial_retrieval": Some relevant docs retrieved but key ones missing. Coverage shows gaps.

Respond with ONLY a JSON object:
{
  "diagnosis": "<one of the diagnosis categories above>",
  "root_cause": "<specific, actionable explanation citing evidence from auditors>",
  "fix_suggestion": "<concrete suggestion for the engineering team>",
  "confidence": <float 0.0-1.0>,
  "stage_scores": {
    "retrieval_relevance": <from retrieval auditor>,
    "generation_fidelity": <from generation auditor>,
    "answer_completeness": <from coverage auditor>,
    "knowledge_coverage": <from coverage auditor>
  }
}

The fix_suggestion must be SPECIFIC and ACTIONABLE:
- Bad: "Improve retrieval"
- Good: "Documents about [specific topic] are missing from the knowledge base. Add documentation covering [X, Y, Z] to the corpus."
- Good: "Retrieval is correct but the LLM fabricates [specific fact]. Consider adding a system prompt constraint: 'Only state facts explicitly present in the provided documents.'"
"""


# ── Engine ───────────────────────────────────────────────────────────

async def diagnose_rag(
    query: str,
    retrieved_docs: list[str],
    generated_answer: str,
    config: Config | None = None,
) -> RAGDiagnosticResult:
    """Diagnose root cause of a RAG pipeline output using multi-agent analysis.

    Runs 3 auditors in parallel (isolation), then synthesises diagnosis.
    Each auditor sees different information to prevent confirmation bias:
    - Retrieval Auditor sees: query + docs (NOT the answer)
    - Generation Auditor sees: docs + answer (analyzes faithfulness)
    - Coverage Auditor sees: query + docs + answer (analyzes completeness)
    """
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    docs_text = "\n\n---\n\n".join(
        f"[Document {i}]\n{doc}" for i, doc in enumerate(retrieved_docs)
    )

    # Build isolated prompts — each auditor gets different context
    retrieval_prompt = (
        f"## User Query\n{query}\n\n"
        f"## Retrieved Documents ({len(retrieved_docs)} total)\n{docs_text}"
    )

    generation_prompt = (
        f"## Retrieved Documents ({len(retrieved_docs)} total)\n{docs_text}\n\n"
        f"## Generated Answer\n{generated_answer}"
    )

    coverage_prompt = (
        f"## User Query\n{query}\n\n"
        f"## Retrieved Documents ({len(retrieved_docs)} total)\n{docs_text}\n\n"
        f"## Generated Answer\n{generated_answer}"
    )

    # Run all 3 auditors in parallel (isolated — they don't see each other's work)
    retrieval_task = provider.generate(retrieval_prompt, system=_RETRIEVAL_AUDITOR_PROMPT)
    generation_task = provider.generate(generation_prompt, system=_GENERATION_AUDITOR_PROMPT)
    coverage_task = provider.generate(coverage_prompt, system=_COVERAGE_AUDITOR_PROMPT)

    retrieval_raw, generation_raw, coverage_raw = await asyncio.gather(
        retrieval_task, generation_task, coverage_task
    )

    # Parse auditor responses
    retrieval_data = _parse_json(retrieval_raw)
    generation_data = _parse_json(generation_raw)
    coverage_data = _parse_json(coverage_raw)

    # Build claim mappings from generation auditor
    claim_mappings = []
    for ca in generation_data.get("claim_analysis", []):
        claim_mappings.append(ClaimMapping(
            claim=ca.get("claim", ""),
            grounded=ca.get("grounded", False),
            source_doc_index=ca.get("source_doc_index"),
            source_quote=ca.get("source_quote", ""),
            issue=ca.get("issue_detail", ""),
        ))

    # Synthesise — pass all three auditor findings to the synthesiser
    synth_prompt = (
        f"## Retrieval Auditor Findings\n```json\n{json.dumps(retrieval_data, indent=2)}\n```\n\n"
        f"## Generation Auditor Findings\n```json\n{json.dumps(generation_data, indent=2)}\n```\n\n"
        f"## Coverage Auditor Findings\n```json\n{json.dumps(coverage_data, indent=2)}\n```"
    )

    synth_raw = await provider.generate(synth_prompt, system=_DIAGNOSTIC_SYNTHESISER_PROMPT)
    synth_data = _parse_json(synth_raw)

    duration_ms = int((time.monotonic() - start) * 1000)

    # Extract stage scores from synthesiser or fallback to individual auditors
    stage_scores = synth_data.get("stage_scores", {})

    return RAGDiagnosticResult(
        diagnosis=RAGDiagnosis(synth_data.get("diagnosis", "faithful")),
        root_cause=synth_data.get("root_cause", ""),
        fix_suggestion=synth_data.get("fix_suggestion", ""),
        retrieval_relevance=float(stage_scores.get(
            "retrieval_relevance", retrieval_data.get("relevance_score", 0.0)
        )),
        generation_fidelity=float(stage_scores.get(
            "generation_fidelity", generation_data.get("fidelity_score", 0.0)
        )),
        answer_completeness=float(stage_scores.get(
            "answer_completeness", coverage_data.get("answer_completeness_score", 0.0)
        )),
        knowledge_coverage=float(stage_scores.get(
            "knowledge_coverage", coverage_data.get("knowledge_coverage_score", 0.0)
        )),
        claim_mappings=claim_mappings,
        retrieval_analysis=retrieval_data,
        generation_analysis=generation_data,
        coverage_analysis=coverage_data,
        auditor_findings={
            "retrieval": retrieval_data,
            "generation": generation_data,
            "coverage": coverage_data,
            "synthesis": synth_data,
        },
        duration_ms=duration_ms,
    )


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling code blocks."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"error": f"Failed to parse: {text[:200]}"}
