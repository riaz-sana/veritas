# Veritas Research Findings — Complete Record

**Project:** Veritas — Multi-agent verification for AI outputs
**Date:** 2026-03-28
**Duration:** ~24 hours of research and development
**Model used for all experiments:** Claude Sonnet 4.6
**Total API spend:** ~$50-80 estimated across all benchmarks

---

## Executive Summary

We set out to build a groundbreaking multi-agent verification system based on the hypothesis that **agent isolation prevents conformity bias and produces better verification**. After extensive testing, the core hypothesis was **disproven** for the specific application of RAG evaluation. However, the research produced several valuable findings about multi-agent verification, practical tooling, and identified genuinely unoccupied territory for future work.

### Key Conclusions

1. **Information asymmetry does NOT improve RAG claim verification.** Full-context single-pass evaluation (RAGVUE-style) outperforms isolated multi-agent evaluation on bias-triggering cases (97.1% vs 91.4% claim accuracy).

2. **Multi-agent IS more thorough than single-prompt** for open-ended analysis. The ablation study showed +1.6 completeness, +1.0 specificity over single-prompt — but this thoroughness doesn't translate to better accuracy on claim-level tasks.

3. **The practical tool works well.** `verify()`, `diagnose_rag()`, and `verify_action()` all produce useful, actionable results. The engineering is solid — 110 tests, clean API, multiple distribution channels.

4. **The novelty claims we started with were wrong.** RAGVUE, RAG-X, and Superagent exist as direct competitors. The Agent-as-a-Judge paradigm is well-established. We weren't first.

5. **Genuinely unoccupied territory exists in code verification** — executable property testing for Python/JS from natural language specs. Axiom does Lean, Qodo does LLM judgment, nobody does executable verification for Python.

---

## What We Built

### Core Library (veritas/)
- `verify(claim, context, domain)` — 5-agent parallel claim verification
- `diagnose_rag(query, docs, answer)` — 3-auditor RAG diagnostic engine
- `verify_action()` / `@before_action` / `verify_plan()` — pre-action verification for agentic AI
- CLI: `veritas check`, `veritas shell`, `veritas benchmark`
- Claude Code skill (`/verify`)
- MCP server for any AI tool
- Enterprise features: tiered models, caching, confidence routing, domain-specific prompts

### Test Suite
- 110 unit tests passing
- Real API integration tests with Claude Sonnet 4.6

### Distribution
- pip-installable from private git
- Claude Code skill
- MCP server
- Usage guide covering RAG, agentic, CI/CD, production middleware patterns

---

## All Experiments and Results

### Experiment 1: Isolation vs Debate — Simple Claims (30 claims)

**Hypothesis:** Isolated agents produce more accurate verdicts than shared-context debate.

**Dataset:** 30 claims — 10 verified truths, 20 common misconceptions to refute.

**Method:** Same model (Sonnet 4.6), same claims. Isolation mode runs 4 agents in parallel with no shared context. Debate mode runs agents sequentially, each seeing prior findings.

**Results:**

| Metric | Isolation | Debate |
|--------|-----------|--------|
| Accuracy | 96.67% | 93.33% |
| ECE | 0.069 | 0.089 |
| Duration | 501s | 1,240s |

**Conclusion:** Small accuracy edge for isolation (+3.3%), better calibration, 2.5x faster. But sample too small for significance, and claims too straightforward.

---

### Experiment 2: Adversarial Claims (50 claims)

**Hypothesis:** Planted subtle errors would propagate through shared-context debate but get caught by isolated agents.

**Dataset:** 50 claims — 20 subtle factual errors (off-by-one year/number), 15 scope errors (overgeneralizations), 15 planted confident errors (wrong claims stated with certainty).

**Results:**

| Metric | Isolation | Debate |
|--------|-----------|--------|
| Detection rate | 100% | 100% |
| Calibration (ECE) | 0.029 | 0.037 |
| Duration | 835s | 2,231s |

**Conclusion:** Both modes caught everything. Claude Sonnet 4.6 is too knowledgeable for these claims. The conformity bias we predicted didn't materialize because the model's knowledge is strong. Isolation is 2.7x faster.

---

### Experiment 3: FaithBench — Hard Hallucination Detection (50 samples)

**Hypothesis:** Veritas isolation mode will outperform debate on the hardest hallucination detection benchmark (NAACL 2025).

**Dataset:** 50 samples from FaithBench — human-annotated summarization hallucinations from 10 LLMs, curated from cases where GPT-4o-as-judge disagreed with humans.

**Results:**

| Metric | Isolation | Debate | Published SOTA (o3-mini) |
|--------|-----------|--------|--------------------------|
| Balanced Accuracy | **58.0%** | 48.0% | ~58% |
| Precision | **60.0%** | 48.4% | — |
| F1 | 53.3% | 53.6% | ~55% |
| Duration | 805s | 1,940s | — |

**Conclusion:** Isolation significantly outperforms debate on balanced accuracy (+10%). Matches o3-mini's published SOTA. But F1 is tied — isolation trades recall for precision. Debate mode over-flags faithful content (more false positives).

---

### Experiment 4: RAG Grounding (25 doc-answer pairs)

**Hypothesis:** Veritas correctly identifies unfaithful RAG outputs with claim-level precision.

**Dataset:** 25 synthetic doc-answer pairs — 12 faithful, 13 hallucinated across 5 error types (wrong_number, entity_swap, fabricated_fact, unsupported_claim, scope_expansion).

**Results:**

| Metric | Isolation | Debate |
|--------|-----------|--------|
| F1 | **89.7%** | 81.3% |
| Precision | **81.3%** | 68.4% |
| Recall | 100% | 100% |
| False Positives | **3** | 6 |

**Conclusion:** Both modes catch every hallucination (100% recall). Isolation has half the false positives — debate's shared context makes agents overly suspicious. This was the strongest result for isolation.

---

### Experiment 5: Ablation — Multi-Agent vs Single-Prompt (9 cases)

**Hypothesis:** Multiple specialized agents produce better analysis than one comprehensive prompt.

**Dataset:** 9 test cases with ground truth — 5 RAG diagnostic + 4 action verification.

**Method:** Blind evaluation by a separate LLM judge (randomized presentation order to prevent position bias). Same model for both approaches.

**Results:**

| Dimension | Multi-Agent | Single-Prompt | Delta |
|-----------|-------------|---------------|-------|
| accuracy | 9.1 | 9.1 | Tie |
| specificity | **9.4** | 8.4 | +1.0 |
| completeness | **9.7** | 8.1 | +1.6 |
| claim_coverage | **9.8** | 9.1 | +0.7 |
| overall | **9.3** | 8.6 | +0.7 |

Multi-agent wins 7/9, ties 2/9. Cost: 4.4x more. Speed: 1.7x slower.

**Conclusion:** Both get the core diagnosis RIGHT (tied accuracy). Multi-agent produces significantly more thorough analysis — more complete, more specific, better claim coverage. The architecture justifies itself for thoroughness but not for accuracy.

**Important caveat:** Evaluation used LLM-as-judge (Claude evaluating Claude's outputs). This introduces potential systematic bias.

---

### Experiment 6: RAGVUE Head-to-Head — Standard Claims (8 cases, 33 claims)

**Hypothesis:** Our information asymmetry design (agents seeing different context) produces better claim-level accuracy than RAGVUE's full-context approach.

**Dataset:** 8 RAG cases with manually labeled ground truth claims.

**Method:**
- RAGVUE-style: single LLM call, sees question + contexts + answer
- Veritas asymmetry: Retrieval Auditor sees question + contexts (NOT answer), Generation Auditor sees contexts + answer

**Results:**

| Metric | RAGVUE-style | Veritas (asymmetry) |
|--------|-------------|---------------------|
| Claim accuracy | 33/33 (100%) | 33/33 (100%) |
| False positives | 0 | 0 |
| LLM calls | 8 | 16 |

**Conclusion:** Tied on clear-cut cases. Both approaches score perfectly when claims are unambiguous. Information asymmetry adds cost without improving accuracy.

---

### Experiment 7: RAGVUE Head-to-Head — Bias-Triggering Cases (6 cases, 35 claims)

**Hypothesis:** On cases specifically designed to trigger confirmation bias (plausible answers from wrong docs, confident extrapolation from vague docs, real quotes interwoven with fabrications, same jargon with opposite conclusions, correct numbers from wrong time period), information asymmetry will prevent the evaluator from being anchored by the plausible answer.

**Dataset:** 6 adversarial cases designed to exploit specific confirmation bias patterns:

1. **Plausible answer, wrong product docs** — answer describes Lisinopril side effects (medically accurate) but docs are about Losartan. Tests if evaluator notices the drug name mismatch.
2. **Confident extrapolation from vague docs** — docs are noncommittal ("discussed," "expressed interest"), answer states definitive recommendation. Tests if answer's confidence anchors evaluator.
3. **Real quotes interwoven with fabrications** — alternates between exact doc quotes and made-up facts. Tests if real quotes make fabrications seem grounded.
4. **Same jargon, opposite conclusions** — security audit doc says HIGH RISK, answer says strong security. Uses same terms (/search, SQL injection, /admin) but opposite conclusions. Tests if shared terminology causes evaluator to miss the reversal.
5. **Correct numbers, wrong time period** — Q2 numbers presented as Q3 data. All numbers are real but from wrong quarter. Tests temporal attribution detection.
6. **Real framework, fabricated details** — real policy framework with fabricated escalation paths, tooling, and review steps added. Tests if real framework makes fabrications seem plausible.

**Results:**

| Metric | RAGVUE-style (full context) | Veritas (asymmetry) |
|--------|---------------------------|---------------------|
| Claim accuracy | **34/35 (97.1%)** | 32/35 (91.4%) |
| False positives | **1** | 3 |
| Per-case wins | **2** | 0 |

**Per-case breakdown:**

| Case | RAGVUE | Veritas | Winner |
|------|--------|---------|--------|
| Wrong product docs | 7/7 | 6/7 | RAGVUE |
| Confident extrapolation | 3/4 | 3/4 | TIE |
| Interwoven fabrications | 7/7 | 7/7 | TIE |
| Opposite conclusions | 5/5 | 5/5 | TIE |
| Wrong time period | 5/5 | 4/5 | RAGVUE |
| Fabricated details | 7/7 | 7/7 | TIE |

**What Veritas got wrong that RAGVUE got right:**
- Case 1: The Generation Auditor (seeing docs + answer, no question) didn't catch that "dizziness" was from Losartan's profile being applied to Lisinopril. The RAGVUE evaluator, seeing the question "What are the side effects of Lisinopril?", correctly identified the drug name mismatch.
- Case 5: The auditor missed that the Q2 data was being attributed to Q3. Without seeing the question asking about Q3, the temporal mismatch wasn't apparent from docs + answer alone.

**Why this happened:** The Generation Auditor doesn't see the question. For cases where the groundedness violation depends on the RELATIONSHIP BETWEEN QUESTION AND ANSWER (not just answer vs docs), missing the question is a handicap, not an advantage.

**Conclusion: Information asymmetry HURTS claim-level accuracy on bias-triggering cases.** Full-context evaluation (RAGVUE-style) is more accurate because it has more signal to work with. The confirmation bias we feared didn't materialize — Claude Sonnet 4.6 is robust enough to evaluate correctly even when the answer is deceptively plausible.

---

## What We Proved

1. **Multi-agent produces more THOROUGH analysis than single-prompt** (ablation: +1.6 completeness, +1.0 specificity). For high-stakes reviews where you need exhaustive evidence, multiple agents find more.

2. **Multi-agent does NOT produce more ACCURATE claim-level verdicts** (all head-to-heads: tied or worse). For binary "is this grounded?" decisions, a single well-crafted prompt is sufficient.

3. **Information asymmetry hurts for RAG evaluation.** Removing the question from the Generation Auditor's context removes signal needed to assess temporal attribution and entity-context relationships.

4. **Isolation is consistently 2-3x faster than debate.** Parallel execution always beats sequential shared-context.

5. **Isolation has fewer false positives than debate on grounding tasks.** Debate's shared context makes agents overly suspicious.

6. **Current models (Sonnet 4.6) are robust against the confirmation biases we hypothesized.** The model correctly sees through plausible-sounding fabrications even with full context.

## What We Disproved

1. **"Information asymmetry prevents confirmation bias in evaluation"** — Disproven. Full context is better. The bias doesn't manifest at current model capability levels.

2. **"Nobody does RAG root-cause diagnosis"** — Wrong. RAGVUE (Jan 2026), RAG-X (March 2026), RAGXplain all do this.

3. **"Nobody does pre-action verification"** — Wrong. Superagent's Safety Agent (Dec 2025) does this with policy-based approach.

4. **"Our isolation architecture is novel"** — Partially wrong. Agent-as-a-Judge (ICML 2025) is a recognized paradigm. Multi-agent evaluation is well-studied.

## What Remains Genuinely Novel

1. **The ablation study data itself** — a controlled comparison of multi-agent vs single-prompt with blind evaluation. Few have published this.

2. **The bias-triggering test methodology** — 6 specific bias patterns for testing RAG evaluators. This test suite is useful for anyone evaluating evaluation tools.

3. **The three-in-one library** — no single tool combines claim verification + RAG diagnostics + action verification in one pip install.

4. **The practical code** — `verify()`, `diagnose_rag()`, `@before_action` with 110 tests, CLI, MCP server, skill. Well-engineered regardless of novelty claims.

---

## Competitive Landscape (as of 2026-03-28)

### RAG Evaluation

| Tool | Approach | Claim-Level? | Root Cause? | Pip? |
|------|----------|-------------|-------------|------|
| **RAGAS** | Score-based metrics | No | No | Yes |
| **RAGVUE** | Single-pass LLM, 18 metrics, claim-level | Yes | Yes | Yes |
| **RAG-X** | Independent retriever/generator evaluation | Partial | Yes | Research code |
| **RAGXplain** | Metrics → actionable guidance | No | Partial | Research code |
| **Galileo Luna** | Fine-tuned small model, ms inference | No | No | SaaS |
| **Veritas** | Multi-agent (proven no better than single-pass) | Yes | Yes | Yes |

### Action Verification

| Tool | Approach | LLM-based? | Open Source? |
|------|----------|-----------|-------------|
| **Superagent Safety Agent** | Declarative policy engine | No (rules) | Yes |
| **Veritas** | Multi-agent LLM reasoning | Yes | Yes |
| **Databricks DASF v3.0** | Framework/guidelines | N/A | Guidelines only |

### Code Verification (the gap)

| Tool | Languages | Approach | Accessible? |
|------|-----------|----------|-------------|
| **Axiom** ($1.6B) | Lean | Formal proofs | API |
| **DeepSeek-Prover** | Lean | Theorem proving | Open weights |
| **Qodo** | All | LLM judgment (not executable) | Production |
| **Nobody** | **Python/JS** | **Executable property testing** | **Gap** |

---

## Recommended Next Steps

### 1. Adopt RAGVUE-style for RAG diagnostics
Switch `diagnose_rag()` from multi-agent to single-pass with full context. It's more accurate, half the cost, and proven.

### 2. Ship Veritas as a practical tool, not a research contribution
The three-in-one library (`verify` + `diagnose_rag` + `verify_action`) with CLI, skill, and MCP server is genuinely useful. Compete on integration and convenience, not on architectural novelty.

### 3. Build Veridex (code verification) as the real novel contribution
Executable property testing for Python/JS from natural language specs. Nobody does this. Axiom does Lean, Qodo does LLM judgment. The gap: spec → Hypothesis tests → execute → report.

### 4. Publish the ablation methodology
The bias-triggering test cases and multi-agent vs single-prompt ablation methodology are useful contributions to the evaluation community, even though our hypothesis was disproven.

---

## Files and Data

### Benchmark Results (JSON)
- `docs/research/benchmarks/adversarial-results.json` — 50-claim adversarial benchmark
- `docs/research/benchmarks/faithbench-results.json` — 50-sample FaithBench
- `docs/research/benchmarks/rag-grounding-results.json` — 25-item RAG grounding
- `docs/research/ablation-results.json` — 9-case multi-agent vs single-prompt ablation
- `docs/research/ragvue-headtohead-results.json` — 8-case standard head-to-head
- `docs/research/bias-headtohead-results.json` — 6-case bias-triggering head-to-head

### Methodology Documents
- `docs/research/benchmarks/methodology.md` — evaluation design principles
- `docs/research/benchmarks/adversarial-dataset.md` — adversarial claim design rationale
- `docs/research/ablation-study.md` — multi-agent vs single-prompt study design
- `docs/research/honest-assessment-march-2026.md` — competitive landscape with sources
- `docs/research/next-moves.md` — researched next steps

### Code
- `veritas/ablation/single_prompt.py` — single-prompt baselines
- `veritas/ablation/runner.py` — ablation study runner with blind evaluation
- `veritas/ablation/ragvue_headtohead.py` — standard head-to-head
- `veritas/ablation/bias_cases.py` — bias-triggering head-to-head

---

## Acknowledgments

This research was conducted entirely within a single Claude Code session using Claude Opus 4.6 as the development assistant and Claude Sonnet 4.6 for all benchmark experiments. Overstory multi-agent orchestration framework was used for project structure. All API calls used the Anthropic SDK.
