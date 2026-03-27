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
    assert "ISSUE" in report  # generation_fidelity is 0.1


@pytest.mark.asyncio
async def test_diagnose_rag_runs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from unittest.mock import AsyncMock, patch

    mock_response = json.dumps({
        "retrieval_relevance": 0.9,
        "generation_fidelity": 0.2,
        "answer_completeness": 0.8,
        "knowledge_coverage": 1.0,
        "diagnosis": "generation_hallucination",
        "root_cause": "Answer adds facts not in documents",
        "fix_suggestion": "Constrain output to retrieved context",
        "retrieval_analysis": {"relevant_docs": 2, "total_docs": 2},
        "generation_analysis": {"fabricated_claims": ["90-day refund"]},
    })

    with patch("veritas.diagnostics.rag.ClaudeProvider") as MockProvider:
        MockProvider.return_value.generate = AsyncMock(return_value=mock_response)
        result = await diagnose_rag(
            query="What is our refund policy?",
            retrieved_docs=["Refund window is 30 days..."],
            generated_answer="Our refund window is 90 days.",
        )

    assert result.diagnosis == RAGDiagnosis.GENERATION_HALLUCINATION
    assert result.retrieval_relevance == 0.9
    assert result.generation_fidelity == 0.2
    assert "90" in result.root_cause or "facts" in result.root_cause
