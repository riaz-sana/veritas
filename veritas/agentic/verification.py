"""Pre-Action Verification for Agentic AI — multi-agent architecture.

Verifies that an agent's planned action is correct BEFORE it executes.
Uses 4 specialized verification agents in parallel isolation:

                    action + params + reasoning
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Reasoning   │ │  Parameter   │ │    Risk      │
    │  Verifier    │ │  Verifier    │ │  Assessor    │
    │              │ │              │ │              │
    │ Is the logic │ │ Are params   │ │ What could   │
    │ behind this  │ │ correct for  │ │ go wrong?    │
    │ action sound?│ │ the goal?    │ │ Irreversible?│
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           │    ┌───────────────────┐        │
           │    │  Scope Verifier   │        │
           │    │                   │        │
           │    │ Does the action   │        │
           │    │ match the stated  │        │
           │    │ goal? Too much?   │        │
           │    │ Too little?       │        │
           │    └────────┬──────────┘        │
           │             │                   │
           └─────────────┼───────────────────┘
                         ▼
              ┌──────────────────┐
              │  Action          │
              │  Synthesiser     │
              │                  │
              │  Combines into   │
              │  verdict + risks │
              └──────────────────┘

Three integration patterns:
1. @before_action decorator — automatic verification
2. verify_action() — explicit gate before execution
3. verify_plan() — verify multi-step plan before any step runs
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from veritas.core.config import Config
from veritas.core.result import AgentFinding, FailureMode, FailureModeType
from veritas.providers.claude import ClaudeProvider
from veritas.providers.base import LLMProvider


class ActionVerdict(str, Enum):
    """Verdict for a planned action."""
    APPROVED = "approved"
    APPROVED_WITH_WARNINGS = "approved_with_warnings"
    BLOCKED = "blocked"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass
class ActionRisk:
    """A specific risk identified in a planned action."""
    category: str   # "data_loss", "incorrect_target", "logic_error", "scope_exceeded", "missing_validation", "irreversible", "security", "compliance"
    severity: str   # "critical", "high", "medium", "low"
    description: str
    mitigation: str


@dataclass
class ActionVerificationResult:
    """Result of multi-agent pre-action verification."""

    verdict: ActionVerdict
    confidence: float
    reasoning: str
    risks: list[ActionRisk]

    # What was verified
    action: str
    parameters: dict
    agent_reasoning: str

    # Evidence from each verifier
    verifier_findings: dict = field(default_factory=dict)

    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict in (ActionVerdict.APPROVED, ActionVerdict.APPROVED_WITH_WARNINGS)

    @property
    def blocked(self) -> bool:
        return self.verdict == ActionVerdict.BLOCKED

    def __str__(self) -> str:
        icon = {"approved": "APPROVED", "approved_with_warnings": "WARNING", "blocked": "BLOCKED", "needs_human_review": "REVIEW"}
        status = icon.get(self.verdict.value, "?")
        risks_text = f" ({len(self.risks)} risks)" if self.risks else ""
        return f"{status} ({self.confidence:.2f}) — {self.reasoning}{risks_text}"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risks": [{"category": r.category, "severity": r.severity, "description": r.description, "mitigation": r.mitigation} for r in self.risks],
            "action": self.action,
            "parameters": self.parameters,
            "approved": self.approved,
            "verifier_findings": self.verifier_findings,
            "duration_ms": self.duration_ms,
        }

    def report(self) -> str:
        lines = [
            f"# Action Verification: {self.action}",
            "",
            f"**Verdict:** {self.verdict.value}",
            f"**Confidence:** {self.confidence:.2f}",
            f"**Reasoning:** {self.reasoning}",
            "",
        ]
        if self.risks:
            lines.append("## Risks Identified")
            lines.append("")
            for r in self.risks:
                lines.append(f"### [{r.severity.upper()}] {r.category}")
                lines.append(f"- **Issue:** {r.description}")
                lines.append(f"- **Mitigation:** {r.mitigation}")
                lines.append("")
        if self.verifier_findings:
            lines.append("## Verifier Details")
            lines.append("")
            for name, finding in self.verifier_findings.items():
                lines.append(f"### {name}")
                if isinstance(finding, dict):
                    lines.append(f"- Verdict: {finding.get('verdict', 'unknown')}")
                    if finding.get('concerns'):
                        for c in finding['concerns']:
                            lines.append(f"  - {c}")
                lines.append("")
        return "\n".join(lines)


# ── Verifier Prompts ─────────────────────────────────────────────────

_REASONING_VERIFIER_PROMPT = """You are a Reasoning Verifier. You ONLY analyze whether the LOGIC behind a planned action is sound. You do NOT assess parameters or risks — other agents handle those.

Given an action, the agent's reasoning for choosing it, and the original goal, determine:
1. Does the reasoning logically lead to this action?
2. Are there logical fallacies in the reasoning?
3. Are there unstated assumptions that might be wrong?
4. Is the reasoning internally consistent?

Respond with ONLY JSON:
{
  "verdict": "sound" | "flawed" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "concerns": ["<specific logical issues found>"],
  "unstated_assumptions": ["<assumptions the reasoning relies on but doesn't state>"],
  "reasoning": "<your step-by-step analysis>"
}"""

_PARAMETER_VERIFIER_PROMPT = """You are a Parameter Verifier. You ONLY analyze whether the action's parameters are correct for the stated goal. You do NOT assess the reasoning or risks — other agents handle those.

Given an action, its parameters, and the original goal, determine:
1. Do the parameter values match what the goal requires?
2. Are any required parameters missing?
3. Are any parameter values obviously wrong (wrong type, out of range, wrong entity)?
4. Do the parameters contain anything that contradicts the stated goal?

Respond with ONLY JSON:
{
  "verdict": "correct" | "incorrect" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "param_analysis": [
    {
      "param": "<parameter name>",
      "value": "<parameter value>",
      "status": "ok" | "wrong" | "suspicious" | "missing",
      "issue": "<what's wrong, if anything>"
    }
  ],
  "missing_params": ["<parameters that should be present but aren't>"],
  "reasoning": "<your step-by-step analysis>"
}"""

_RISK_ASSESSOR_PROMPT = """You are a Risk Assessor. You ONLY analyze what could go wrong if this action executes. You do NOT assess whether the action is logically correct — other agents handle that.

Given an action and its parameters, identify:
1. Is this action irreversible? (deletes, sends, payments, publications)
2. Could this action cause data loss or corruption?
3. Are there security concerns? (injection, unauthorized access, data exposure)
4. Are there compliance concerns? (PII, financial regulations, rate limits)
5. What's the blast radius if something goes wrong?

Respond with ONLY JSON:
{
  "risk_level": "none" | "low" | "medium" | "high" | "critical",
  "is_irreversible": <true/false>,
  "risks": [
    {
      "category": "data_loss" | "incorrect_target" | "irreversible" | "security" | "compliance" | "rate_limit" | "cascade_failure",
      "severity": "critical" | "high" | "medium" | "low",
      "description": "<specific risk>",
      "mitigation": "<how to reduce this risk>",
      "likelihood": "certain" | "likely" | "possible" | "unlikely"
    }
  ],
  "requires_confirmation": <true if human should confirm before execution>,
  "reasoning": "<your risk analysis>"
}"""

_SCOPE_VERIFIER_PROMPT = """You are a Scope Verifier. You ONLY analyze whether the action matches the stated goal — not doing too much, not doing too little. You do NOT assess correctness or risk — other agents handle those.

Given an action, its parameters, and the original goal, determine:
1. Does this action actually achieve the goal?
2. Does the action do MORE than what was asked? (scope creep)
3. Does the action do LESS than what was needed? (incomplete)
4. Is there a simpler action that would achieve the same goal?

Respond with ONLY JSON:
{
  "verdict": "matches_goal" | "exceeds_goal" | "insufficient" | "wrong_goal",
  "confidence": <float 0.0-1.0>,
  "scope_analysis": {
    "goal_requirements": ["<what the goal asks for>"],
    "action_effects": ["<what the action will actually do>"],
    "excess": ["<things the action does that weren't asked for>"],
    "gaps": ["<things the goal needs that the action doesn't do>"]
  },
  "simpler_alternative": "<simpler way to achieve the goal, or null>",
  "reasoning": "<your analysis>"
}"""

_ACTION_SYNTHESISER_PROMPT = """You are an Action Synthesiser. You receive independent analyses from four verification agents who each examined a different aspect of a planned action. They did NOT see each other's work.

Combine their findings into a final verdict:
- Reasoning Verifier: analyzed if the logic is sound
- Parameter Verifier: analyzed if parameters are correct
- Risk Assessor: identified what could go wrong
- Scope Verifier: analyzed if action matches the goal

Verdict rules:
- "approved": All verifiers report positive. No critical/high risks. Logic sound, params correct, scope matches.
- "approved_with_warnings": Generally positive but has medium risks or minor concerns. Safe to proceed with awareness.
- "blocked": ANY critical issue — flawed reasoning, wrong parameters, critical risk, or wrong goal. Do NOT execute.
- "needs_human_review": Verifiers disagree or are uncertain. Human judgment needed.

Respond with ONLY JSON:
{
  "verdict": "approved" | "approved_with_warnings" | "blocked" | "needs_human_review",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<synthesis of all verifier findings>",
  "risks": [<consolidated list of risks from risk assessor, only medium+ severity>]
}"""


_PLAN_VERIFIER_PROMPT = """You are a Plan Verifier. An AI agent has created a multi-step plan. Analyze the ENTIRE plan for correctness, ordering, completeness, and risks.

Check:
1. Are steps in the right order? (dependencies respected)
2. Are any steps missing? (validation, error handling, cleanup, rollback)
3. Are any steps unnecessary? (scope creep)
4. Could any step fail and leave the system in a bad state?
5. Are irreversible steps placed AFTER confirmation/validation steps?
6. Does the plan actually achieve the stated goal?
7. What's the blast radius if step N fails midway?

Respond with ONLY JSON:
{
  "verdict": "approved" | "approved_with_warnings" | "blocked" | "needs_human_review",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<overall assessment>",
  "step_analysis": [
    {
      "step": <number>,
      "action": "<description>",
      "verdict": "ok" | "warning" | "blocked",
      "concern": "<issue if any>",
      "depends_on": [<step numbers this depends on>],
      "reversible": <true/false>
    }
  ],
  "risks": [
    {
      "category": "ordering" | "missing_step" | "scope_exceeded" | "irreversible" | "cascade_failure" | "incomplete_cleanup",
      "severity": "critical" | "high" | "medium" | "low",
      "description": "<specific risk>",
      "mitigation": "<how to fix>"
    }
  ],
  "missing_steps": ["<steps that should be in the plan but aren't>"],
  "unnecessary_steps": ["<steps that shouldn't be in the plan>"],
  "failure_scenario": "<what happens if step N fails midway through execution>"
}"""


# ── Engine ───────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"error": f"Failed to parse: {text[:200]}"}


async def verify_action(
    action: str,
    parameters: dict | None = None,
    reasoning: str = "",
    goal: str = "",
    context: str = "",
    config: Config | None = None,
) -> ActionVerificationResult:
    """Verify a planned action using 4 agents in parallel isolation.

    Each verifier independently analyzes a different aspect:
    - Reasoning Verifier: is the logic sound?
    - Parameter Verifier: are params correct?
    - Risk Assessor: what could go wrong?
    - Scope Verifier: does the action match the goal?
    """
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    params = parameters or {}
    params_json = json.dumps(params, indent=2, default=str)

    # Build prompts for each verifier — different context for isolation
    reasoning_prompt = (
        f"## Action: {action}\n"
        f"## Agent's Reasoning: {reasoning}\n"
        f"## Goal: {goal}\n"
        + (f"## Context: {context}" if context else "")
    )

    param_prompt = (
        f"## Action: {action}\n"
        f"## Parameters:\n```json\n{params_json}\n```\n"
        f"## Goal: {goal}"
    )

    risk_prompt = (
        f"## Action: {action}\n"
        f"## Parameters:\n```json\n{params_json}\n```\n"
        + (f"## Context: {context}" if context else "")
    )

    scope_prompt = (
        f"## Action: {action}\n"
        f"## Parameters:\n```json\n{params_json}\n```\n"
        f"## Goal: {goal}\n"
        f"## Agent's Reasoning: {reasoning}"
    )

    # Run all 4 verifiers in parallel (isolated)
    reasoning_raw, param_raw, risk_raw, scope_raw = await asyncio.gather(
        provider.generate(reasoning_prompt, system=_REASONING_VERIFIER_PROMPT),
        provider.generate(param_prompt, system=_PARAMETER_VERIFIER_PROMPT),
        provider.generate(risk_prompt, system=_RISK_ASSESSOR_PROMPT),
        provider.generate(scope_prompt, system=_SCOPE_VERIFIER_PROMPT),
    )

    reasoning_data = _parse_json(reasoning_raw)
    param_data = _parse_json(param_raw)
    risk_data = _parse_json(risk_raw)
    scope_data = _parse_json(scope_raw)

    # Synthesise
    synth_prompt = (
        f"## Reasoning Verifier\n```json\n{json.dumps(reasoning_data, indent=2)}\n```\n\n"
        f"## Parameter Verifier\n```json\n{json.dumps(param_data, indent=2)}\n```\n\n"
        f"## Risk Assessor\n```json\n{json.dumps(risk_data, indent=2)}\n```\n\n"
        f"## Scope Verifier\n```json\n{json.dumps(scope_data, indent=2)}\n```"
    )

    synth_raw = await provider.generate(synth_prompt, system=_ACTION_SYNTHESISER_PROMPT)
    synth_data = _parse_json(synth_raw)

    duration_ms = int((time.monotonic() - start) * 1000)

    # Build risks from synthesiser + risk assessor
    risks = []
    for r in synth_data.get("risks", []) + risk_data.get("risks", []):
        if isinstance(r, dict):
            risks.append(ActionRisk(
                category=r.get("category", "logic_error"),
                severity=r.get("severity", "medium"),
                description=r.get("description", ""),
                mitigation=r.get("mitigation", ""),
            ))
    # Deduplicate by description
    seen = set()
    unique_risks = []
    for r in risks:
        if r.description not in seen:
            seen.add(r.description)
            unique_risks.append(r)

    return ActionVerificationResult(
        verdict=ActionVerdict(synth_data.get("verdict", "needs_human_review")),
        confidence=float(synth_data.get("confidence", 0.0)),
        reasoning=synth_data.get("reasoning", ""),
        risks=unique_risks,
        action=action,
        parameters=params,
        agent_reasoning=reasoning,
        verifier_findings={
            "reasoning": reasoning_data,
            "parameters": param_data,
            "risk": risk_data,
            "scope": scope_data,
            "synthesis": synth_data,
        },
        duration_ms=duration_ms,
    )


async def verify_plan(
    goal: str,
    steps: list[str] | list[dict],
    context: str = "",
    config: Config | None = None,
) -> ActionVerificationResult:
    """Verify a multi-step plan before any step executes."""
    if config is None:
        config = Config()
    config.validate()

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    if steps and isinstance(steps[0], str):
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    else:
        steps_text = "\n".join(
            f"{i+1}. {s.get('action', s)} — params: {json.dumps(s.get('params', {}), default=str)}"
            for i, s in enumerate(steps)
        )

    prompt = f"## Goal\n{goal}\n\n## Planned Steps\n{steps_text}"
    if context:
        prompt += f"\n\n## Context\n{context}"

    response = await provider.generate(prompt, system=_PLAN_VERIFIER_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    data = _parse_json(response)

    risks = [
        ActionRisk(
            category=r.get("category", "logic_error"),
            severity=r.get("severity", "medium"),
            description=r.get("description", ""),
            mitigation=r.get("mitigation", ""),
        )
        for r in data.get("risks", [])
        if isinstance(r, dict)
    ]

    return ActionVerificationResult(
        verdict=ActionVerdict(data.get("verdict", "needs_human_review")),
        confidence=float(data.get("confidence", 0.0)),
        reasoning=data.get("reasoning", ""),
        risks=risks,
        action="plan_verification",
        parameters={"goal": goal, "num_steps": len(steps)},
        agent_reasoning="",
        verifier_findings={"plan_analysis": data},
        duration_ms=duration_ms,
        metadata={
            "step_analysis": data.get("step_analysis", []),
            "missing_steps": data.get("missing_steps", []),
            "unnecessary_steps": data.get("unnecessary_steps", []),
            "failure_scenario": data.get("failure_scenario", ""),
        },
    )


def before_action(
    func: Callable | None = None,
    *,
    goal: str = "",
    context: str = "",
    block_on_failure: bool = True,
    config: Config | None = None,
):
    """Decorator that verifies an action using multi-agent verification before execution.

    Usage:
        @before_action
        async def send_email(to: str, subject: str, body: str):
            ...

        @before_action(goal="Process refund", block_on_failure=True)
        async def process_refund(order_id: str, amount: float):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            reasoning = params.pop("_reasoning", "")
            action_goal = params.pop("_goal", goal)

            result = await verify_action(
                action=fn.__name__,
                parameters=params,
                reasoning=reasoning,
                goal=action_goal,
                context=context,
                config=config,
            )

            if result.blocked and block_on_failure:
                raise ActionBlockedError(
                    f"Action '{fn.__name__}' blocked by Veritas: {result.reasoning}",
                    verification_result=result,
                )

            if result.verdict == ActionVerdict.NEEDS_HUMAN_REVIEW and block_on_failure:
                raise ActionNeedsReviewError(
                    f"Action '{fn.__name__}' needs human review: {result.reasoning}",
                    verification_result=result,
                )

            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)

        wrapper._veritas_enabled = True
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class ActionBlockedError(Exception):
    """Raised when Veritas blocks an action."""
    def __init__(self, message: str, verification_result: ActionVerificationResult):
        super().__init__(message)
        self.verification_result = verification_result


class ActionNeedsReviewError(Exception):
    """Raised when Veritas requires human review for an action."""
    def __init__(self, message: str, verification_result: ActionVerificationResult):
        super().__init__(message)
        self.verification_result = verification_result
