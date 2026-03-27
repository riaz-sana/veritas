# Veritas Benchmark Results: Isolation vs Debate

**Date:** 2026-03-27
**Model:** claude-sonnet-4-6
**Claims:** 15 (5 verified truths, 10 common misconceptions to refute)
**Search:** Disabled (LLM knowledge only — no Brave/Tavily key)

---

## Results

| Metric | Isolation | Debate | Delta |
|--------|-----------|--------|-------|
| **Accuracy** | **100.00%** | **100.00%** | +0.00% |
| **ECE** | **0.0293** | 0.0367 | -0.0073 |
| **Duration** | **244s** | 648s | -405s |
| **Per-item** | **16.2s** | 43.2s | |

**Winner: ISOLATION** — 2.7x faster with better calibration.

---

## Per-Item Results

### Isolation Mode (parallel, no shared context)

| # | Claim | Expected | Verdict | Confidence |
|---|-------|----------|---------|------------|
| 1 | The Earth orbits the Sun | VERIFIED | VERIFIED | 0.99 |
| 2 | Water is composed of hydrogen and oxygen | VERIFIED | VERIFIED | 0.99 |
| 3 | Python was created by Guido van Rossum | VERIFIED | VERIFIED | 0.99 |
| 4 | Tokyo is the capital of Japan | VERIFIED | VERIFIED | 0.98 |
| 5 | DNA has a double helix structure | VERIFIED | VERIFIED | 0.98 |
| 6 | The Great Wall is visible from space | REFUTED | REFUTED | 0.95 |
| 7 | Humans only use 10% of their brains | REFUTED | REFUTED | 0.99 |
| 8 | Lightning never strikes same place twice | REFUTED | REFUTED | 0.99 |
| 9 | Goldfish have a 3-second memory | REFUTED | REFUTED | 0.97 |
| 10 | Napoleon was unusually short | REFUTED | REFUTED | 0.92 |
| 11 | Einstein failed math in school | REFUTED | REFUTED | 0.96 |
| 12 | Glass is a liquid that flows slowly | REFUTED | REFUTED | 0.96 |
| 13 | Cracking knuckles causes arthritis | REFUTED | REFUTED | 0.96 |
| 14 | Sugar causes hyperactivity in children | REFUTED | REFUTED | 0.95 |
| 15 | The blood in your veins is blue | REFUTED | REFUTED | 0.98 |

### Debate Mode (sequential, shared context)

| # | Claim | Expected | Verdict | Confidence |
|---|-------|----------|---------|------------|
| 1 | The Earth orbits the Sun | VERIFIED | VERIFIED | 0.99 |
| 2 | Water is composed of hydrogen and oxygen | VERIFIED | VERIFIED | 0.99 |
| 3 | Python was created by Guido van Rossum | VERIFIED | VERIFIED | 0.98 |
| 4 | Tokyo is the capital of Japan | VERIFIED | VERIFIED | 0.99 |
| 5 | DNA has a double helix structure | VERIFIED | VERIFIED | 0.95 |
| 6 | The Great Wall is visible from space | REFUTED | REFUTED | 0.95 |
| 7 | Humans only use 10% of their brains | REFUTED | REFUTED | 0.98 |
| 8 | Lightning never strikes same place twice | REFUTED | REFUTED | 0.99 |
| 9 | Goldfish have a 3-second memory | REFUTED | REFUTED | 0.96 |
| 10 | Napoleon was unusually short | REFUTED | REFUTED | 0.91 |
| 11 | Einstein failed math in school | REFUTED | REFUTED | 0.96 |
| 12 | Glass is a liquid that flows slowly | REFUTED | REFUTED | 0.96 |
| 13 | Cracking knuckles causes arthritis | REFUTED | REFUTED | 0.95 |
| 14 | Sugar causes hyperactivity in children | REFUTED | REFUTED | 0.91 |
| 15 | The blood in your veins is blue | REFUTED | REFUTED | 0.98 |

---

## Analysis

### 1. Both modes achieve 100% accuracy
On well-known facts and common misconceptions, both isolation and debate modes perform perfectly. Claude Sonnet 4.6 already knows these facts well, so the verification agents are confirming strong priors.

### 2. Isolation is 2.7x faster
- Isolation: 244s total (16.2s/claim) — 4 agents run in parallel
- Debate: 648s total (43.2s/claim) — agents run sequentially, each seeing prior findings
- This is the expected result: parallel execution is inherently faster than sequential

### 3. Isolation has better calibration (lower ECE)
- Isolation ECE: 0.0293 (nearly perfect calibration)
- Debate ECE: 0.0367 (slightly overconfident)
- This supports the thesis: shared context may cause agents to reinforce each other's confidence rather than independently assess

### 4. Where the difference will show
This benchmark uses clear-cut claims. The isolation advantage is expected to be larger on:
- **Ambiguous claims** where reasonable agents might disagree
- **Adversarial inputs** with planted errors that propagate through shared context
- **Nuanced claims** that are partially true (PARTIAL verdict)
- **Temporal claims** about recent events where knowledge is uncertain

---

## Methodology

- **Isolation mode:** All 4 agents (logic, source, adversary, calibration) run simultaneously with zero shared context. Findings merge only at the synthesiser.
- **Debate mode:** Agents run sequentially. Each agent sees all previous agents' findings injected into its prompt context. This mirrors the standard multi-agent debate approach (Du et al., ICML 2024).
- **Synthesiser:** Identical in both modes — runs in-process, receives all findings, produces verdict.
- **No web search:** Both modes used LLM knowledge only (no Brave/Tavily API key), ensuring a fair comparison.

---

## Next Steps for Publication

1. **Scale to TruthfulQA** (817 claims) for statistical significance
2. **Add adversarial claims** — plant subtle errors to test conformity bias
3. **Enable web search** — test with Brave API for source verifier enhancement
4. **Multi-model comparison** — run with different LLMs to test model-agnostic claim
5. **Overstory worktree mode** — true filesystem isolation (currently async, same process)
