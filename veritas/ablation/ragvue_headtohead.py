"""Head-to-head: Veritas (information asymmetry) vs RAGVUE-style (full context).

The question: does giving each auditor DIFFERENT information produce
better claim-level analysis than giving one evaluator EVERYTHING?

RAGVUE approach (replicated): Single LLM call with answer + contexts → claim extraction + verification
Veritas approach: 3 auditors with information asymmetry → synthesiser

We test on the same claims with known ground truth and score both on
claim-level accuracy: did it correctly identify which claims are grounded
and which are hallucinated?
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from veritas.core.config import Config
from veritas.providers.claude import ClaudeProvider


# ── Test Cases with Ground Truth ─────────────────────────────────────

@dataclass
class GroundTruthCase:
    name: str
    question: str
    contexts: list[str]
    answer: str
    # Ground truth: each claim and whether it's actually grounded
    claims_truth: list[dict]  # [{"claim": "...", "grounded": true/false, "reason": "..."}]


CASES = [
    GroundTruthCase(
        name="Refund policy — multiple fabrications",
        question="What is our refund policy?",
        contexts=[
            "Company Return Policy: Customers may return unused items within 30 days of purchase with original receipt. Electronics have a 15-day return window. Sale items are final sale. Refunds are processed to the original payment method within 5-7 business days."
        ],
        answer="Our refund policy allows returns within 90 days for all items including sale items. Refunds are processed instantly to any payment method of your choice.",
        claims_truth=[
            {"claim": "returns within 90 days", "grounded": False, "reason": "Doc says 30 days"},
            {"claim": "all items including sale items", "grounded": False, "reason": "Doc says sale items are final sale"},
            {"claim": "refunds processed instantly", "grounded": False, "reason": "Doc says 5-7 business days"},
            {"claim": "any payment method of your choice", "grounded": False, "reason": "Doc says original payment method"},
        ],
    ),
    GroundTruthCase(
        name="Financial report — mixed correct and wrong",
        question="What was Q3 revenue?",
        contexts=[
            "Q3 2025 Financial Results: Acme Corp reported revenue of $4.2 billion, up 12% year-over-year. Net income was $340 million. Operating margin improved to 18.5%."
        ],
        answer="Acme Corp's Q3 2025 revenue was $4.5 billion, up 15% year-over-year. Net income was $340 million with an operating margin of 18.5%.",
        claims_truth=[
            {"claim": "revenue $4.5 billion", "grounded": False, "reason": "Doc says $4.2 billion"},
            {"claim": "up 15% year-over-year", "grounded": False, "reason": "Doc says 12%"},
            {"claim": "net income $340 million", "grounded": True, "reason": "Matches doc exactly"},
            {"claim": "operating margin 18.5%", "grounded": True, "reason": "Matches doc exactly"},
        ],
    ),
    GroundTruthCase(
        name="Faithful answer — all correct",
        question="How many employees does the company have?",
        contexts=[
            "Company Overview: Acme Corp has 15,000 employees across 23 countries. Founded in 1995 by Jane Smith."
        ],
        answer="Acme Corp has 15,000 employees working across 23 countries.",
        claims_truth=[
            {"claim": "15,000 employees", "grounded": True, "reason": "Matches doc"},
            {"claim": "23 countries", "grounded": True, "reason": "Matches doc"},
        ],
    ),
    GroundTruthCase(
        name="Drug dosage — critical medical error",
        question="What is the recommended dosage for Metformin?",
        contexts=[
            "Metformin prescribing information: Initial dose 500mg twice daily or 850mg once daily with meals. Maximum dose 2550mg/day in adults. Not recommended for patients with eGFR below 30."
        ],
        answer="The recommended starting dose of Metformin is 500mg three times daily. Maximum dose is 3000mg per day. It is safe for all patients regardless of kidney function.",
        claims_truth=[
            {"claim": "500mg three times daily", "grounded": False, "reason": "Doc says twice daily or 850mg once daily"},
            {"claim": "maximum 3000mg per day", "grounded": False, "reason": "Doc says 2550mg/day"},
            {"claim": "safe for all patients regardless of kidney function", "grounded": False, "reason": "Doc explicitly says not recommended for eGFR below 30"},
        ],
    ),
    GroundTruthCase(
        name="API docs — subtle parameter change",
        question="How does the users API work?",
        contexts=[
            "The /users endpoint supports GET (list, paginated, max 100/page), POST (create, requires email+name), DELETE (soft-delete, admin only). Rate limit: 1000 req/min. Auth: Bearer token."
        ],
        answer="The /users endpoint supports GET for listing (max 100/page), POST for creating users (requires email and name), and DELETE for permanently removing users (any authenticated user). Rate limit is 1000 requests per minute.",
        claims_truth=[
            {"claim": "GET max 100/page", "grounded": True, "reason": "Matches"},
            {"claim": "POST requires email and name", "grounded": True, "reason": "Matches"},
            {"claim": "DELETE permanently removing", "grounded": False, "reason": "Doc says soft-delete"},
            {"claim": "any authenticated user can delete", "grounded": False, "reason": "Doc says admin only"},
            {"claim": "1000 requests per minute", "grounded": True, "reason": "Matches"},
        ],
    ),
    GroundTruthCase(
        name="Meeting notes — scope expansion",
        question="What was decided in the Q2 planning meeting?",
        contexts=[
            "Q2 Planning: Marketing budget $2.3M. Three campaigns: summer sale (June), back-to-school (Aug), brand awareness (ongoing). Hire 2 content writers. Next review April 15."
        ],
        answer="The Q2 marketing budget was set at $2.3M for five major campaigns spanning all of Q2 and Q3. The team will hire an entire creative department of 8 people to support the expanded initiative.",
        claims_truth=[
            {"claim": "budget $2.3M", "grounded": True, "reason": "Matches"},
            {"claim": "five major campaigns", "grounded": False, "reason": "Doc says three campaigns"},
            {"claim": "spanning all of Q2 and Q3", "grounded": False, "reason": "Not stated in doc"},
            {"claim": "hire entire creative department of 8", "grounded": False, "reason": "Doc says 2 content writers"},
        ],
    ),
    GroundTruthCase(
        name="Research paper — subtle number changes",
        question="What were the study results?",
        contexts=[
            "The study examined 500 participants over 12 months. Treatment group (n=250) showed 23% symptom reduction vs 8% placebo. Side effects: headache 12%, nausea 7%. p=0.003."
        ],
        answer="A study of 500 participants over 12 months found a 23% reduction in symptoms for the treatment group versus 8% for placebo (p=0.003). Common side effects included headache (15%) and nausea (10%).",
        claims_truth=[
            {"claim": "500 participants over 12 months", "grounded": True, "reason": "Matches"},
            {"claim": "23% symptom reduction", "grounded": True, "reason": "Matches"},
            {"claim": "8% placebo", "grounded": True, "reason": "Matches"},
            {"claim": "p=0.003", "grounded": True, "reason": "Matches"},
            {"claim": "headache 15%", "grounded": False, "reason": "Doc says 12%"},
            {"claim": "nausea 10%", "grounded": False, "reason": "Doc says 7%"},
        ],
    ),
    GroundTruthCase(
        name="Legal policy — added nonexistent benefit",
        question="What PTO do employees get?",
        contexts=[
            "PTO Policy: 15 days/year for first 3 years. 20 days after 3 years. 25 days after 7 years. Max 10 days carryover. 2 weeks advance notice required."
        ],
        answer="Employees start with 15 days PTO annually, increasing to 20 days after 3 years and 25 days after 7 years. Up to 10 days carry over. Additionally, employees receive 5 mental health days per year that don't count against PTO.",
        claims_truth=[
            {"claim": "15 days annually to start", "grounded": True, "reason": "Matches"},
            {"claim": "20 days after 3 years", "grounded": True, "reason": "Matches"},
            {"claim": "25 days after 7 years", "grounded": True, "reason": "Matches"},
            {"claim": "10 days carryover", "grounded": True, "reason": "Matches"},
            {"claim": "5 mental health days per year", "grounded": False, "reason": "Completely fabricated — not in doc"},
        ],
    ),
]


# ── RAGVUE-Style Evaluator (single pass, full context) ───────────────

_RAGVUE_STYLE_PROMPT = """You are a strict factual evaluation agent.
Determine exactly which parts of the ANSWER are grounded in the CONTEXTS.

1. Extract every factual claim from the ANSWER.
2. For each claim, check if it is supported by the CONTEXTS using strict rules:
   - All entities, numbers, dates must match exactly
   - If only part of a claim is supported, mark it NOT supported
   - Support must come ONLY from the CONTEXTS
3. Classify each claim as "supported" or "hallucinated"

Return JSON ONLY:
{
  "claims": [
    {"claim": "<extracted claim>", "grounded": true/false, "evidence": "<context quote or empty>", "reason": "<why>"}
  ]
}"""


# ── Veritas-Style Evaluator (information asymmetry) ──────────────────

_GENERATION_AUDITOR_PROMPT = """You are a Generation Auditor. You ONLY check whether claims in an ANSWER are supported by CONTEXTS.

For EACH factual claim, determine:
1. Is it directly supported by a specific passage?
2. If yes, cite the exact quote
3. If no, is it fabricated (not in docs) or contradicts docs?

ALL entities, numbers, dates must match EXACTLY. Partial support = not supported.

Return JSON:
{
  "claims": [
    {"claim": "<claim>", "grounded": true/false, "evidence": "<exact quote or empty>", "reason": "<why>"}
  ]
}"""

_RETRIEVAL_AUDITOR_PROMPT = """You are a Retrieval Auditor. Given a QUESTION and CONTEXTS, assess retrieval quality.
You have NOT seen the answer — you don't know what was generated.

Determine:
1. Are the contexts relevant to the question?
2. Do the contexts contain enough information to fully answer the question?
3. What specific topics does the question ask about that contexts cover/miss?

Return JSON:
{
  "relevance_score": <0.0-1.0>,
  "can_answer": true/false,
  "covered_topics": ["<topics contexts cover>"],
  "missing_topics": ["<topics question asks about but contexts don't cover>"]
}"""


# ── Scoring ──────────────────────────────────────────────────────────

@dataclass
class ClaimResult:
    claim_text: str
    predicted_grounded: bool
    actual_grounded: bool
    correct: bool
    evidence: str = ""
    reason: str = ""


@dataclass
class CaseResult:
    case_name: str
    method: str
    total_claims_in_truth: int
    claims_found: int
    correct_classifications: int
    false_positives: int   # said grounded when actually hallucinated
    false_negatives: int   # said hallucinated when actually grounded
    claim_details: list[ClaimResult] = field(default_factory=list)
    duration_ms: int = 0
    llm_calls: int = 0

    @property
    def claim_accuracy(self) -> float:
        return self.correct_classifications / self.total_claims_in_truth if self.total_claims_in_truth else 0


def _match_claims(predicted_claims: list[dict], ground_truth: list[dict]) -> list[ClaimResult]:
    """Match predicted claims to ground truth claims by text similarity."""
    results = []
    for gt in ground_truth:
        gt_text = gt["claim"].lower()
        gt_grounded = gt["grounded"]

        # Find best matching predicted claim
        best_match = None
        best_score = 0
        for pc in predicted_claims:
            pc_text = (pc.get("claim", "") or "").lower()
            # Simple keyword overlap
            gt_words = set(gt_text.split())
            pc_words = set(pc_text.split())
            if not gt_words:
                continue
            overlap = len(gt_words & pc_words) / len(gt_words)
            if overlap > best_score:
                best_score = overlap
                best_match = pc

        if best_match and best_score > 0.3:
            predicted_grounded = best_match.get("grounded", True)
            results.append(ClaimResult(
                claim_text=gt["claim"],
                predicted_grounded=predicted_grounded,
                actual_grounded=gt_grounded,
                correct=predicted_grounded == gt_grounded,
                evidence=best_match.get("evidence", ""),
                reason=best_match.get("reason", ""),
            ))
        else:
            # Claim not found — count as wrong (missed claim)
            results.append(ClaimResult(
                claim_text=gt["claim"],
                predicted_grounded=True,  # Assume grounded if not found (optimistic)
                actual_grounded=gt_grounded,
                correct=gt_grounded,  # Only correct if it was actually grounded
                reason="Claim not identified by evaluator",
            ))

    return results


def _parse_json(text: str) -> dict:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"claims": []}


# ── Runner ───────────────────────────────────────────────────────────

async def run_headtohead(config: Config | None = None) -> dict:
    """Run RAGVUE-style vs Veritas-style on all cases."""
    if config is None:
        config = Config()
    config.validate()

    provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)

    ragvue_results = []
    veritas_results = []

    for i, case in enumerate(CASES):
        print(f"\n[{i+1}/{len(CASES)}] {case.name}", flush=True)
        contexts_text = "\n\n".join(f"[{j+1}] {c}" for j, c in enumerate(case.contexts))

        # ── RAGVUE-style: single pass, full context ──
        ragvue_prompt = f"QUESTION: {case.question}\n\nCONTEXTS:\n{contexts_text}\n\nANSWER: {case.answer}"
        t0 = time.monotonic()
        ragvue_raw = await provider.generate(ragvue_prompt, system=_RAGVUE_STYLE_PROMPT)
        ragvue_ms = int((time.monotonic() - t0) * 1000)
        ragvue_data = _parse_json(ragvue_raw)
        ragvue_claims = ragvue_data.get("claims", [])
        ragvue_matched = _match_claims(ragvue_claims, case.claims_truth)

        ragvue_correct = sum(1 for r in ragvue_matched if r.correct)
        ragvue_fp = sum(1 for r in ragvue_matched if r.predicted_grounded and not r.actual_grounded)
        ragvue_fn = sum(1 for r in ragvue_matched if not r.predicted_grounded and r.actual_grounded)

        ragvue_results.append(CaseResult(
            case_name=case.name, method="ragvue_style",
            total_claims_in_truth=len(case.claims_truth),
            claims_found=len(ragvue_claims),
            correct_classifications=ragvue_correct,
            false_positives=ragvue_fp, false_negatives=ragvue_fn,
            claim_details=ragvue_matched, duration_ms=ragvue_ms, llm_calls=1,
        ))

        # ── Veritas-style: information asymmetry ──
        # Retrieval auditor sees question + contexts (NOT answer)
        ret_prompt = f"QUESTION: {case.question}\n\nCONTEXTS:\n{contexts_text}"
        # Generation auditor sees contexts + answer (NOT question)
        gen_prompt = f"CONTEXTS:\n{contexts_text}\n\nANSWER: {case.answer}"

        t1 = time.monotonic()
        ret_raw, gen_raw = await asyncio.gather(
            provider.generate(ret_prompt, system=_RETRIEVAL_AUDITOR_PROMPT),
            provider.generate(gen_prompt, system=_GENERATION_AUDITOR_PROMPT),
        )
        veritas_ms = int((time.monotonic() - t1) * 1000)
        gen_data = _parse_json(gen_raw)
        veritas_claims = gen_data.get("claims", [])
        veritas_matched = _match_claims(veritas_claims, case.claims_truth)

        veritas_correct = sum(1 for r in veritas_matched if r.correct)
        veritas_fp = sum(1 for r in veritas_matched if r.predicted_grounded and not r.actual_grounded)
        veritas_fn = sum(1 for r in veritas_matched if not r.predicted_grounded and r.actual_grounded)

        veritas_results.append(CaseResult(
            case_name=case.name, method="veritas_asymmetry",
            total_claims_in_truth=len(case.claims_truth),
            claims_found=len(veritas_claims),
            correct_classifications=veritas_correct,
            false_positives=veritas_fp, false_negatives=veritas_fn,
            claim_details=veritas_matched, duration_ms=veritas_ms, llm_calls=2,
        ))

        rv_acc = ragvue_correct / len(case.claims_truth) if case.claims_truth else 0
        vt_acc = veritas_correct / len(case.claims_truth) if case.claims_truth else 0
        winner = "VERITAS" if vt_acc > rv_acc else "RAGVUE" if rv_acc > vt_acc else "TIE"
        print(f"  RAGVUE: {ragvue_correct}/{len(case.claims_truth)} ({rv_acc:.0%}) | Veritas: {veritas_correct}/{len(case.claims_truth)} ({vt_acc:.0%}) → {winner} | RV:{ragvue_ms}ms VT:{veritas_ms}ms", flush=True)

    # ── Summary ──
    total_truth = sum(len(c.claims_truth) for c in CASES)
    rv_total_correct = sum(r.correct_classifications for r in ragvue_results)
    vt_total_correct = sum(r.correct_classifications for r in veritas_results)
    rv_total_fp = sum(r.false_positives for r in ragvue_results)
    vt_total_fp = sum(r.false_positives for r in veritas_results)
    rv_total_fn = sum(r.false_negatives for r in ragvue_results)
    vt_total_fn = sum(r.false_negatives for r in veritas_results)
    rv_total_ms = sum(r.duration_ms for r in ragvue_results)
    vt_total_ms = sum(r.duration_ms for r in veritas_results)

    summary = {
        "total_cases": len(CASES),
        "total_claims": total_truth,
        "ragvue_style": {
            "correct": rv_total_correct, "accuracy": rv_total_correct / total_truth,
            "false_positives": rv_total_fp, "false_negatives": rv_total_fn,
            "duration_ms": rv_total_ms, "llm_calls": len(CASES),
        },
        "veritas_asymmetry": {
            "correct": vt_total_correct, "accuracy": vt_total_correct / total_truth,
            "false_positives": vt_total_fp, "false_negatives": vt_total_fn,
            "duration_ms": vt_total_ms, "llm_calls": len(CASES) * 2,
        },
    }

    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: RAGVUE-style vs Veritas Information Asymmetry")
    print(f"{'='*60}")
    print(f"Cases: {len(CASES)} | Claims: {total_truth}")
    print(f"")
    print(f"| Metric | RAGVUE-style | Veritas (asymmetry) | Delta |")
    print(f"|--------|-------------|--------------------:|------:|")
    print(f"| Claim accuracy | {rv_total_correct}/{total_truth} ({rv_total_correct/total_truth:.1%}) | {vt_total_correct}/{total_truth} ({vt_total_correct/total_truth:.1%}) | {(vt_total_correct-rv_total_correct)/total_truth:+.1%} |")
    print(f"| False positives | {rv_total_fp} | {vt_total_fp} | {vt_total_fp-rv_total_fp:+d} |")
    print(f"| False negatives | {rv_total_fn} | {vt_total_fn} | {vt_total_fn-rv_total_fn:+d} |")
    print(f"| Duration | {rv_total_ms}ms | {vt_total_ms}ms | {vt_total_ms/rv_total_ms:.1f}x |")
    print(f"| LLM calls | {len(CASES)} | {len(CASES)*2} | 2x |")
    print(f"")

    rv_wins = sum(1 for r, v in zip(ragvue_results, veritas_results) if r.claim_accuracy > v.claim_accuracy)
    vt_wins = sum(1 for r, v in zip(ragvue_results, veritas_results) if v.claim_accuracy > r.claim_accuracy)
    ties = len(CASES) - rv_wins - vt_wins
    print(f"Per-case: RAGVUE wins {rv_wins} | Veritas wins {vt_wins} | Ties {ties}")

    winner = "VERITAS" if vt_total_correct > rv_total_correct else "RAGVUE" if rv_total_correct > vt_total_correct else "TIE"
    print(f"\nOverall winner on claim accuracy: **{winner}**")

    return summary
