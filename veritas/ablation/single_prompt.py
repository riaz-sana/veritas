"""Single-prompt baselines for ablation study.

These are the STRONGEST POSSIBLE single-prompt implementations of
diagnose_rag and verify_action. They get the same information as the
multi-agent versions but in ONE LLM call instead of 4-5.

The ablation question: does splitting into multiple agents produce
materially better results than one well-crafted prompt?
"""

from __future__ import annotations

import json
import time

from veritas.core.config import Config
from veritas.providers.claude import ClaudeProvider


# ── Single-Prompt RAG Diagnosis ──────────────────────────────────────

_SINGLE_RAG_PROMPT = """You are analyzing a RAG (Retrieval-Augmented Generation) pipeline output. Given a user query, retrieved documents, and the generated answer, perform a COMPLETE diagnostic analysis.

Analyze ALL of the following in one pass:

1. RETRIEVAL QUALITY: Are the retrieved documents relevant to the query? Are any irrelevant (noise)? Are there topics the query asks about that no document covers?

2. GENERATION FIDELITY: For EACH distinct claim in the answer, is it:
   - Directly supported by a specific passage in a document? (cite the doc and quote)
   - Fabricated? (not in any document)
   - Contradicting a document? (opposite of what a doc says)

3. KNOWLEDGE COVERAGE: Could the query be answered from these documents at all? Does the answer address all aspects of the query?

4. ROOT CAUSE: What is the primary failure mode?
   - "faithful": No issues
   - "retrieval_miss": Docs don't cover the query topic
   - "retrieval_noise": Irrelevant docs confused the LLM
   - "generation_hallucination": LLM added facts not in docs
   - "generation_contradiction": LLM stated opposite of docs
   - "knowledge_gap": Answer isn't in the KB at all
   - "partial_retrieval": Some relevant docs missing

Respond with ONLY JSON:
{
  "retrieval_relevance": <float 0.0-1.0>,
  "generation_fidelity": <float 0.0-1.0>,
  "answer_completeness": <float 0.0-1.0>,
  "knowledge_coverage": <float 0.0-1.0>,
  "diagnosis": "<root cause category>",
  "root_cause": "<specific explanation citing evidence>",
  "fix_suggestion": "<actionable suggestion>",
  "claim_analysis": [
    {
      "claim": "<claim from answer>",
      "grounded": <true/false>,
      "source_doc_index": <int or null>,
      "source_quote": "<exact quote or empty>",
      "issue": "<what's wrong if not grounded>"
    }
  ],
  "retrieval_issues": ["<specific retrieval problems>"],
  "generation_issues": ["<specific generation problems>"]
}

Be SPECIFIC — cite exact text from both the answer and documents."""


async def single_prompt_diagnose_rag(
    query: str,
    retrieved_docs: list[str],
    generated_answer: str,
    config: Config | None = None,
) -> dict:
    """Single-prompt RAG diagnosis baseline."""
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    docs_text = "\n\n---\n\n".join(
        f"[Document {i}]\n{doc}" for i, doc in enumerate(retrieved_docs)
    )
    prompt = (
        f"## User Query\n{query}\n\n"
        f"## Retrieved Documents ({len(retrieved_docs)} total)\n{docs_text}\n\n"
        f"## Generated Answer\n{generated_answer}"
    )

    response = await provider.generate(prompt, system=_SINGLE_RAG_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    data = _parse_json(response)
    data["duration_ms"] = duration_ms
    data["method"] = "single_prompt"
    data["llm_calls"] = 1
    return data


# ── Single-Prompt Action Verification ────────────────────────────────

_SINGLE_ACTION_PROMPT = """You are verifying whether an AI agent's planned action is correct and safe BEFORE it executes. Analyze ALL of the following in one pass:

1. REASONING: Is the logic behind choosing this action sound? Any logical fallacies or unstated assumptions?

2. PARAMETERS: Are the parameter values correct for the stated goal? Any missing, wrong, or suspicious values?

3. RISKS: What could go wrong? Is this irreversible? Security concerns? Compliance issues? Data loss potential?

4. SCOPE: Does the action match the goal? Doing too much? Too little? Is there a simpler alternative?

Respond with ONLY JSON:
{
  "verdict": "approved" | "approved_with_warnings" | "blocked" | "needs_human_review",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<overall assessment>",
  "reasoning_analysis": {
    "verdict": "sound" | "flawed" | "uncertain",
    "concerns": ["<logical issues>"],
    "unstated_assumptions": ["<assumptions>"]
  },
  "parameter_analysis": [
    {
      "param": "<name>",
      "value": "<value>",
      "status": "ok" | "wrong" | "suspicious" | "missing",
      "issue": "<what's wrong>"
    }
  ],
  "risks": [
    {
      "category": "data_loss" | "incorrect_target" | "irreversible" | "security" | "compliance" | "scope_exceeded",
      "severity": "critical" | "high" | "medium" | "low",
      "description": "<specific risk>",
      "mitigation": "<how to address>"
    }
  ],
  "scope_analysis": {
    "matches_goal": <true/false>,
    "excess": ["<things action does beyond goal>"],
    "gaps": ["<things goal needs but action doesn't do>"]
  }
}"""


async def single_prompt_verify_action(
    action: str,
    parameters: dict | None = None,
    reasoning: str = "",
    goal: str = "",
    context: str = "",
    config: Config | None = None,
) -> dict:
    """Single-prompt action verification baseline."""
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    params = parameters or {}
    prompt_parts = [
        f"## Action: {action}",
        f"## Parameters:\n```json\n{json.dumps(params, indent=2, default=str)}\n```",
    ]
    if reasoning:
        prompt_parts.append(f"## Agent's Reasoning: {reasoning}")
    if goal:
        prompt_parts.append(f"## Goal: {goal}")
    if context:
        prompt_parts.append(f"## Context: {context}")

    prompt = "\n\n".join(prompt_parts)
    response = await provider.generate(prompt, system=_SINGLE_ACTION_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    data = _parse_json(response)
    data["duration_ms"] = duration_ms
    data["method"] = "single_prompt"
    data["llm_calls"] = 1
    return data


def _parse_json(text: str) -> dict:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"error": f"Failed to parse: {text[:200]}"}
