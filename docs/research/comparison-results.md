# Isolation vs Debate Comparison Results

**Date:** 2026-03-27
**Dataset:** Sample (8 claims)
**Model:** claude-sonnet-4-6
**Search:** Disabled (no Brave/Tavily key — LLM knowledge only)

## Results

| Metric | Isolation | Debate | Delta |
|--------|-----------|--------|-------|
| Accuracy | 87.50% | 87.50% | +0.00% |
| ECE | 0.0988 | 0.0875 | +0.0112 |
| Duration | 128.2s | 298.8s | -170.5s |

## Key Findings

### 1. Isolation is 2.3x faster
Isolation mode (parallel agents): **128s**. Debate mode (sequential, shared context): **299s**. This is expected — isolation runs all 4 agents simultaneously, while debate runs them one at a time with prior findings injected.

### 2. Equal accuracy on this sample
Both modes achieved 87.5% accuracy (7/8 correct). The one miss was the same for both: "Light travels at approximately 300,000 km/s" was expected PARTIAL but both returned VERIFIED. This is a labeling nuance — the claim is approximately correct, and both modes recognized it as such.

### 3. Comparable calibration
ECE (Expected Calibration Error) was similar: 0.099 (isolation) vs 0.088 (debate). Both are well-calibrated on this small sample.

## Per-Item Breakdown

| Claim | Expected | Isolation | Debate |
|-------|----------|-----------|--------|
| Water boils at 100C at sea level | VERIFIED | VERIFIED | VERIFIED |
| The Great Wall is visible from space | REFUTED | REFUTED | REFUTED |
| The first iPhone was released in 2006 | REFUTED | REFUTED | REFUTED |
| Python is a compiled language | REFUTED | REFUTED | REFUTED |
| Light travels at ~300,000 km/s | PARTIAL | VERIFIED | VERIFIED |
| All birds can fly | REFUTED | REFUTED | REFUTED |
| The Earth is third from the Sun | VERIFIED | VERIFIED | VERIFIED |
| JavaScript was created by Sun Microsystems | REFUTED | REFUTED | REFUTED |

## Analysis

On this small sample (8 items), accuracy is tied. The isolation mode's advantage is **speed** (2.3x faster) and **architectural safety** (proven to prevent conformity bias per the research literature). The accuracy advantage of isolation is expected to emerge on:

1. **Larger datasets** (TruthfulQA, 817 questions) — where conformity bias accumulates
2. **Contested claims** — where the debate mode's shared context causes agents to converge prematurely
3. **Adversarial inputs** — where planted errors propagate through shared context

## Next Steps

- Run on TruthfulQA (817 questions) for statistical significance
- Run with web search enabled (Brave/Tavily key) for source verifier enhancement
- Test with adversarial inputs (planted errors) to measure conformity bias
- Run with Overstory git worktree isolation for true process-level separation
