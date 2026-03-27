"""Pre-Action Verification for Agentic AI.

Verifies that an agent's planned action is correct BEFORE it executes.
This is the gap: agents take actions (API calls, database writes, emails)
based on reasoning that might be wrong. Nobody checks before execution.

Three integration patterns:

1. Decorator — wrap action functions:
    @before_action
    async def send_email(to, subject, body):
        ...

2. Explicit gate — verify before acting:
    decision = await verify_action(action="send_email", reasoning="...", context="...")
    if decision.approved:
        send_email(...)

3. Pipeline step — insert into agentic workflow:
    plan = agent.plan(task)
    verified_plan = await verify_plan(plan)
    agent.execute(verified_plan)
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


class ActionVerdict(str, Enum):
    """Verdict for a planned action."""
    APPROVED = "approved"           # Action is safe and correct
    APPROVED_WITH_WARNINGS = "approved_with_warnings"  # Proceed but note concerns
    BLOCKED = "blocked"             # Do NOT execute — reasoning is flawed
    NEEDS_HUMAN_REVIEW = "needs_human_review"  # Uncertain — escalate


@dataclass
class ActionRisk:
    """A specific risk identified in a planned action."""
    category: str   # "data_loss", "incorrect_target", "logic_error", "scope_exceeded", "missing_validation", "irreversible"
    severity: str   # "critical", "high", "medium", "low"
    description: str
    mitigation: str


@dataclass
class ActionVerificationResult:
    """Result of pre-action verification."""

    verdict: ActionVerdict
    confidence: float
    reasoning: str
    risks: list[ActionRisk]

    # What was verified
    action: str
    parameters: dict
    agent_reasoning: str

    # Metadata
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
        lines.append("## Parameters Verified")
        lines.append("")
        for k, v in self.parameters.items():
            lines.append(f"- **{k}:** {v}")
        return "\n".join(lines)


_ACTION_VERIFY_SYSTEM_PROMPT = """You are a pre-action verification agent. An AI agent is about to execute an action. Your job is to verify the action is correct and safe BEFORE it executes.

You receive:
- The action to be taken (function name, API call, etc.)
- The parameters/arguments
- The agent's reasoning for why it chose this action
- Optional: the original task/goal

You must respond with ONLY a JSON object:
{
  "verdict": "approved" | "approved_with_warnings" | "blocked" | "needs_human_review",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<why you reached this verdict>",
  "risks": [
    {
      "category": "data_loss" | "incorrect_target" | "logic_error" | "scope_exceeded" | "missing_validation" | "irreversible" | "security" | "compliance",
      "severity": "critical" | "high" | "medium" | "low",
      "description": "<specific risk>",
      "mitigation": "<how to address it>"
    }
  ]
}

Verdict rules:
- "approved": Action is correct, parameters are valid, reasoning is sound
- "approved_with_warnings": Action is probably correct but has minor concerns
- "blocked": Action should NOT execute — reasoning is flawed, parameters are wrong, or risks are too high
- "needs_human_review": Cannot determine correctness — escalate to human

Check for:
1. Do the parameters match what the reasoning describes?
2. Is the action appropriate for the stated goal?
3. Are there irreversible consequences (deletes, sends, payments)?
4. Are there edge cases the agent missed?
5. Does the action scope match the task scope (not doing too much or too little)?
6. Are required validations present (auth, permissions, rate limits)?
7. Could this action cause data loss or corruption?
8. Are there security or compliance concerns?"""


_PLAN_VERIFY_SYSTEM_PROMPT = """You are a plan verification agent. An AI agent has created a multi-step plan. Your job is to verify the ENTIRE PLAN is correct before any step executes.

You receive:
- The original task/goal
- The planned steps (in order)
- Optional: context and constraints

You must respond with ONLY a JSON object:
{
  "verdict": "approved" | "approved_with_warnings" | "blocked" | "needs_human_review",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<overall assessment>",
  "step_analysis": [
    {
      "step": <step number>,
      "action": "<step description>",
      "verdict": "ok" | "warning" | "blocked",
      "concern": "<issue if any>"
    }
  ],
  "risks": [
    {
      "category": "data_loss" | "incorrect_target" | "logic_error" | "scope_exceeded" | "missing_validation" | "irreversible" | "ordering" | "dependency",
      "severity": "critical" | "high" | "medium" | "low",
      "description": "<specific risk>",
      "mitigation": "<how to address it>"
    }
  ],
  "missing_steps": ["<any steps that should be in the plan but aren't>"],
  "unnecessary_steps": ["<any steps that shouldn't be in the plan>"]
}

Check for:
1. Are steps in the right order? (dependencies respected)
2. Are any steps missing? (validation, error handling, cleanup)
3. Are any steps unnecessary? (scope creep)
4. Could any step fail and leave the system in a bad state?
5. Are irreversible steps placed after confirmation/validation steps?
6. Does the plan actually achieve the stated goal?"""


async def verify_action(
    action: str,
    parameters: dict | None = None,
    reasoning: str = "",
    goal: str = "",
    context: str = "",
    config: Config | None = None,
) -> ActionVerificationResult:
    """Verify a planned action before execution.

    Args:
        action: Name of the action (function name, API endpoint, etc.)
        parameters: Action parameters/arguments
        reasoning: The agent's reasoning for choosing this action
        goal: The original task/goal
        context: Additional context
        config: Veritas config
    """
    if config is None:
        config = Config()
    config.validate()

    

    start = time.monotonic()
    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    params = parameters or {}
    prompt_parts = [
        f"## Action\n{action}",
        f"\n## Parameters\n```json\n{json.dumps(params, indent=2, default=str)}\n```",
    ]
    if reasoning:
        prompt_parts.append(f"\n## Agent's Reasoning\n{reasoning}")
    if goal:
        prompt_parts.append(f"\n## Original Goal\n{goal}")
    if context:
        prompt_parts.append(f"\n## Context\n{context}")

    prompt = "\n".join(prompt_parts)
    response = await provider.generate(prompt, system=_ACTION_VERIFY_SYSTEM_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        risks = [
            ActionRisk(
                category=r.get("category", "logic_error"),
                severity=r.get("severity", "medium"),
                description=r.get("description", ""),
                mitigation=r.get("mitigation", ""),
            )
            for r in data.get("risks", [])
        ]

        return ActionVerificationResult(
            verdict=ActionVerdict(data.get("verdict", "needs_human_review")),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            risks=risks,
            action=action,
            parameters=params,
            agent_reasoning=reasoning,
            duration_ms=duration_ms,
        )
    except (json.JSONDecodeError, ValueError):
        return ActionVerificationResult(
            verdict=ActionVerdict.NEEDS_HUMAN_REVIEW,
            confidence=0.0,
            reasoning=f"Verification parse error",
            risks=[],
            action=action,
            parameters=params,
            agent_reasoning=reasoning,
            duration_ms=duration_ms,
        )


async def verify_plan(
    goal: str,
    steps: list[str] | list[dict],
    context: str = "",
    config: Config | None = None,
) -> ActionVerificationResult:
    """Verify a multi-step plan before any step executes.

    Args:
        goal: The original task/goal
        steps: List of planned steps (strings or dicts with 'action' and 'params')
        context: Additional context
        config: Veritas config
    """
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

    response = await provider.generate(prompt, system=_PLAN_VERIFY_SYSTEM_PROMPT)
    duration_ms = int((time.monotonic() - start) * 1000)

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        risks = [
            ActionRisk(
                category=r.get("category", "logic_error"),
                severity=r.get("severity", "medium"),
                description=r.get("description", ""),
                mitigation=r.get("mitigation", ""),
            )
            for r in data.get("risks", [])
        ]

        return ActionVerificationResult(
            verdict=ActionVerdict(data.get("verdict", "needs_human_review")),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            risks=risks,
            action="plan_verification",
            parameters={"goal": goal, "num_steps": len(steps)},
            agent_reasoning="",
            duration_ms=duration_ms,
            metadata={
                "step_analysis": data.get("step_analysis", []),
                "missing_steps": data.get("missing_steps", []),
                "unnecessary_steps": data.get("unnecessary_steps", []),
            },
        )
    except (json.JSONDecodeError, ValueError):
        return ActionVerificationResult(
            verdict=ActionVerdict.NEEDS_HUMAN_REVIEW,
            confidence=0.0,
            reasoning="Verification parse error",
            risks=[],
            action="plan_verification",
            parameters={"goal": goal, "num_steps": len(steps)},
            agent_reasoning="",
            duration_ms=duration_ms,
        )


def before_action(
    func: Callable | None = None,
    *,
    goal: str = "",
    context: str = "",
    block_on_failure: bool = True,
    config: Config | None = None,
):
    """Decorator that verifies an action before execution.

    Usage:
        @before_action
        async def send_email(to: str, subject: str, body: str):
            ...

        @before_action(goal="Process refund", block_on_failure=True)
        async def process_refund(order_id: str, amount: float):
            ...

    When the decorated function is called:
    1. Veritas verifies the action + parameters are correct
    2. If APPROVED: function executes normally
    3. If BLOCKED: raises ActionBlockedError (if block_on_failure=True) or returns verification result
    4. If NEEDS_HUMAN_REVIEW: raises ActionNeedsReviewError
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Build parameters dict from function signature
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            # Get reasoning from kwargs if provided
            reasoning = params.pop("_reasoning", "")
            action_goal = params.pop("_goal", goal)

            # Verify the action
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

            # Execute the action
            if inspect.iscoroutinefunction(fn):
                action_result = await fn(*args, **kwargs)
            else:
                action_result = fn(*args, **kwargs)

            return action_result

        # Attach verification result to the wrapper for inspection
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
