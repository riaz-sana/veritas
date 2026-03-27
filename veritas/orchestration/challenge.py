"""Challenge round logic for contested verifications."""
from __future__ import annotations
from veritas.agents.adversary import Adversary
from veritas.agents.synthesiser import Synthesiser
from veritas.core.result import AgentFinding, ChallengeResult, VerificationResult

async def run_challenge_round(claim: str, initial_result: VerificationResult, adversary: Adversary, synthesiser: Synthesiser) -> VerificationResult:
    contested_points = _identify_contested_points(initial_result.evidence)
    if not contested_points:
        return initial_result
    challenge_finding = await adversary.challenge(claim=claim, contested_points=contested_points, agent_findings=initial_result.evidence)
    all_findings = list(initial_result.evidence) + [challenge_finding]
    final_result = await synthesiser.synthesise(claim=claim, findings=all_findings)
    final_result.challenge_round = ChallengeResult(
        contested_points=contested_points, adversary_finding=challenge_finding,
        resolution=f"Challenge round completed. Final verdict: {final_result.verdict.value}",
    )
    return final_result

def _identify_contested_points(findings: list[AgentFinding]) -> list[str]:
    contested = []
    positive = {"supported", "consistent", "no_counterexample", "well_calibrated"}
    negative = {"contradiction", "inconsistency", "counterexample_found", "overconfident"}
    has_positive = any(f.finding in positive for f in findings)
    has_negative = any(f.finding in negative for f in findings)
    if has_positive and has_negative:
        for f in findings:
            if f.finding in negative:
                for detail in f.details:
                    desc = detail.get("description", "")
                    if desc:
                        contested.append(f"{f.agent}: {desc}")
                if not f.details:
                    contested.append(f"{f.agent} reports {f.finding}")
    return contested
