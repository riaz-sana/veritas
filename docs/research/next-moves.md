# Next Moves — Researched and Validated

**Date:** 2026-03-28
**Constraint:** Every claim below was verified with web search. No assumptions.

---

## Option 1: Prove Information Asymmetry Beats RAGVUE

### The Research Gap

RAGVUE (Jan 2026) evaluates with full context — all evaluators see everything. Our design gives each auditor DIFFERENT information (Retrieval Auditor doesn't see the answer). The question: **does information asymmetry produce better diagnoses?**

Academic support: Research confirms "adding more LLM agents without principled interaction design can amplify biases" and that "information isolation mechanisms" are needed to prevent bias ([Emergent Bias in Multi-Agent Systems, 2025](https://arxiv.org/pdf/2512.16433)). But nobody has tested information asymmetry specifically for RAG evaluation.

### How to Solve the Speed/Cost Problem

Our multi-agent approach is 4x more expensive and 1.7x slower. Three solutions:

**1. Fine-tuned small models for cheap agents (proven approach)**
Recent work shows fine-tuned small models match LLM performance for hallucination detection:
- HaluAgent: open-source small LLM matches GPT-4 on hallucination benchmarks
- Galileo Luna: fine-tuned small models, millisecond inference
- Transformer-CRF classifiers outperform fine-tuned LLM baselines for token-level detection
- A lightweight detector can be trained on one GPU

**Action:** Fine-tune a small model (e.g. Qwen 2.5 1.5B or Phi-3 mini) for each auditor role. Retrieval Auditor doesn't need Sonnet — it's checking document relevance. Use Sonnet only for the Synthesiser.

**2. Speculative parallel execution**
Run all agents truly in parallel with early termination. If 3/4 agents agree strongly, skip waiting for the 4th. Research shows "predict-verify paradigm" at the action level can overlap computation.

**3. Ship fast mode as default, thorough mode as option**
Our ablation proved single-prompt gets accuracy right (9.1/10). Ship that as default. Multi-agent is the premium mode when thoroughness matters.

### The Head-to-Head Test Plan

1. Take RAGVUE's own evaluation dataset (or use FaithBench)
2. Run RAGVUE on it (their published metrics)
3. Run Veritas (full context, like RAGVUE does) on it
4. Run Veritas (information asymmetry, our design) on it
5. Compare claim-level accuracy, false positives, false negatives

If asymmetry wins → publishable result: "Information asymmetry in multi-agent RAG evaluation reduces confirmation bias"
If asymmetry loses → we adopt RAGVUE's approach and compete on integration

**Estimated effort:** 1-2 days
**Estimated cost:** ~$30-50 API calls

---

## Option 3: Proof-Carrying Code Verification

### Latest Landscape (searched today)

**Who's doing this:**

| Player | What They Do | Accessible? | Languages |
|--------|-------------|------------|-----------|
| **Axiom** ($200M Series A, March 2026) | Formal proofs in Lean via AxiomProver | API (axle.axiommath.ai) | Lean only |
| **DeepSeek-Prover-V2** | Lean theorem proving | Open weights | Lean only |
| **Harmonic (Aristotle)** | Lean proof writing | Startup, limited access | Lean |
| **Logical Intelligence** | Lean proofs | Startup | Lean |
| **MCTS + Verification** | Monte Carlo tree search for verified code | GitHub prototype | Dafny, Coq, Lean, Rust |
| **Qodo** | AI code review before merge | Production tool | All languages |

### The Real Gap

**Every formal verification tool works in Lean or Dafny.** Nobody works in Python or JavaScript — the languages 90% of enterprise code is written in.

**Axiom is $1.6B valued** — they're clearly onto something. But they prove things in Lean. Enterprise developers don't write Lean. The gap is:

> "AI generates Python code. How do I KNOW it's correct before deploying it?"

Current answer: run tests, do code review, hope for the best. AI-generated code has **37% more high-severity vulnerabilities** than human-written code. There's an estimated **40% quality deficit** where more AI code enters pipelines than reviewers can validate.

### What Would Be Groundbreaking

**A tool that generates lightweight verification evidence for Python/JavaScript code — not full Lean proofs, but property-based assertions + counterexample search + invariant checking.**

```python
from veritas import verify_code

result = verify_code(
    code=generated_function,
    spec="Sort a list of integers in ascending order",
)

result.verified_properties    # ["returns sorted list", "preserves all elements", "handles empty input"]
result.counterexamples        # [] (none found)
result.invariants_checked     # ["output length == input length", "output[i] <= output[i+1]"]
result.edge_cases_tested      # ["empty list", "single element", "already sorted", "reverse sorted", "duplicates"]
result.vulnerabilities        # ["no integer overflow risk for Python"]
result.verdict                # VERIFIED (all properties hold)
```

This is NOT full formal verification (that's Axiom's territory). This is **practical verification** that:
1. Generates property-based tests automatically from a spec
2. Runs counterexample search (fuzzing) against those properties
3. Checks invariants via static analysis
4. Reports edge cases and vulnerabilities
5. Gives a confidence verdict

**Why this is novel:**
- Axiom/DeepSeek work in Lean → we work in Python/JS
- Qodo does AI code review (LLM judgment) → we do property verification (executable checks)
- Nobody combines: spec → properties → counterexample search → invariant checking → verdict

**Why this is hard:**
- Property generation from natural language specs is an open problem
- Static analysis of Python is limited (dynamic typing)
- Counterexample search (fuzzing) needs runtime execution
- "Verified" is a strong claim — we need to be careful about what we promise

### What's Actually Feasible

**Level 1 (buildable now):** Multi-agent code review with our architecture. Logic agent checks spec compliance, Source agent checks for known vulnerability patterns, Adversary generates edge case inputs, Calibration assesses confidence. This is what our `domain="code"` already does, but done properly with information asymmetry.

**Level 2 (buildable in weeks):** Property-based test generation. Given a spec, generate Hypothesis-style property tests and actually RUN them. This is concrete verification — executable proof, not LLM judgment.

**Level 3 (research project):** Automatic invariant inference and lightweight formal methods (SMT solvers, symbolic execution). This approaches real formal verification for a subset of Python.

**Recommendation:** Build Level 1 + Level 2 together. That gives us: multi-agent code review (our architecture) + executable property tests (concrete proof). Nobody offers this combination.

---

## Recommended Sequence

### Week 1: Prove Information Asymmetry (Option 1)
- Head-to-head against RAGVUE on same dataset
- If it wins: publish the finding, it's our core differentiator
- If it loses: adopt their approach, focus on other differentiators
- Simultaneously: ship `Config(mode="fast")` for production speed

### Week 2-3: Build Code Verification Level 1+2 (Option 3)
- Multi-agent code review with information asymmetry
- Property-based test generation from specs (executable, not LLM judgment)
- Counterexample search via Hypothesis/fuzzing
- This is genuinely unoccupied territory for Python/JS

### After: Iterate Based on Real Usage
- Ship to colleagues
- Get feedback on what actually matters
- Double down on what works
