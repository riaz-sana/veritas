"""Tests for RAG Diagnostic Engine."""

import json
import pytest

from veritas.diagnostics.rag import RAGDiagnosis, RAGDiagnosticResult, diagnose_rag
from veritas.core.config import Config


def test_rag_diagnosis_enum():
    assert RAGDiagnosis.FAITHFUL == "faithful"
    assert RAGDiagnosis.GENERATION_HALLUCINATION == "generation_hallucination"
    assert RAGDiagnosis.RETRIEVAL_MISS == "retrieval_miss"
    assert RAGDiagnosis.KNOWLEDGE_GAP == "knowledge_gap"


def test_rag_diagnostic_result_str():
    result = RAGDiagnosticResult(
        diagnosis=RAGDiagnosis.GENERATION_HALLUCINATION,
        root_cause="Answer says 90 days but docs say 30 days",
        fix_suggestion="Constrain generation to retrieved context",
        retrieval_relevance=0.9,
        generation_fidelity=0.2,
        answer_completeness=0.8,
        knowledge_coverage=1.0,
    )
    text = str(result)
    assert "generation_hallucination" in text
    assert "90 days" in text


def test_rag_diagnostic_result_to_dict():
    result = RAGDiagnosticResult(
        diagnosis=RAGDiagnosis.RETRIEVAL_MISS,
        root_cause="Relevant docs not retrieved",
        fix_suggestion="Improve chunking strategy",
        retrieval_relevance=0.1,
        generation_fidelity=0.5,
        answer_completeness=0.3,
        knowledge_coverage=0.8,
    )
    d = result.to_dict()
    assert d["diagnosis"] == "retrieval_miss"
    assert d["retrieval_relevance"] == 0.1
    json.dumps(d)  # Must be serializable


def test_rag_diagnostic_result_report():
    result = RAGDiagnosticResult(
        diagnosis=RAGDiagnosis.GENERATION_CONTRADICTION,
        root_cause="LLM contradicted the source",
        fix_suggestion="Use stricter grounding",
        retrieval_relevance=0.9,
        generation_fidelity=0.1,
        answer_completeness=0.7,
        knowledge_coverage=1.0,
        retrieval_analysis={"relevant_docs": 3, "total_docs": 3},
        generation_analysis={"fabricated_claims": ["claim X"]},
    )
    report = result.report()
    assert "generation_contradiction" in report.lower() or "Generation" in report
    assert "PROBLEM" in report  # generation_fidelity is 0.1


@pytest.mark.asyncio
async def test_diagnose_rag_runs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    # Multi-agent: 3 auditor responses + 1 synthesiser response
    call_count = [0]
    responses = [
        # Retrieval auditor
        json.dumps({"relevance_score": 0.9, "relevant_doc_indices": [0], "irrelevant_doc_indices": [], "missing_topics": [], "could_answer_from_docs": True, "reasoning": "Docs are relevant"}),
        # Generation auditor
        json.dumps({"fidelity_score": 0.2, "claim_analysis": [{"claim": "90-day refund", "grounded": False, "source_doc_index": None, "source_quote": "", "issue_type": "fabrication", "issue_detail": "Docs say 30 days not 90"}], "fabricated_claims": ["90-day refund"], "contradicted_claims": [], "reasoning": "LLM fabricated"}),
        # Coverage auditor
        json.dumps({"knowledge_coverage_score": 1.0, "answer_completeness_score": 0.8, "query_aspects": ["refund policy"], "covered_aspects": ["refund policy"], "uncovered_aspects": [], "answered_aspects": ["refund policy"], "missed_aspects": [], "reasoning": "Docs cover it"}),
        # Synthesiser
        json.dumps({"diagnosis": "generation_hallucination", "root_cause": "Answer adds facts not in documents", "fix_suggestion": "Constrain output", "confidence": 0.9, "stage_scores": {"retrieval_relevance": 0.9, "generation_fidelity": 0.2, "answer_completeness": 0.8, "knowledge_coverage": 1.0}}),
    ]

    async def mock_generate(prompt, system=""):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    with patch("veritas.diagnostics.rag.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(side_effect=mock_generate)
        result = await diagnose_rag(
            query="What is our refund policy?",
            retrieved_docs=["Refund window is 30 days..."],
            generated_answer="Our refund window is 90 days.",
        )

    assert result.diagnosis == RAGDiagnosis.GENERATION_HALLUCINATION
    assert result.retrieval_relevance == 0.9
    assert result.generation_fidelity == 0.2
    assert len(result.claim_mappings) >= 1
    assert result.claim_mappings[0].grounded is False
