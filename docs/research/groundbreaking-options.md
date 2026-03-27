# What Would Make Veritas Groundbreaking — Deep Analysis

**Date:** 2026-03-27
**Purpose:** Honest assessment of where the real opportunities are, based on deep research into the current state of AI verification.

---

## The Landscape as of March 2026

### What's Been Solved
- Basic hallucination detection (SelfCheckGPT, SAFE, ChainPoll — all work ~55-60% F1)
- RAG faithfulness checking (RAGAS, DeepEval, TruLens — mature ecosystem)
- Token-level detection probes (INSIDE, HaluGate — read hidden states)

### What Has NOT Been Solved

1. **The Circular Problem** — Using LLMs to check LLMs is fundamentally circular. If the checker has the same biases as the generator, it misses the same errors. No multi-agent approach (including ours) escapes this.

2. **High-Confidence Hallucinations** — Models sometimes hallucinate with high certainty. Entropy-based methods fail because the probability distribution is sharply peaked around the WRONG answer. This is the hardest category.

3. **RAG Entanglement** — When a RAG system fails, nobody can tell if the retriever surfaced wrong docs OR the LLM hallucinated despite correct context. 40-60% of RAG deployments fail to reach production because of this.

4. **Real-Time Verification at Production Scale** — Every verification tool adds 5-30 seconds of latency. The only real-time approach (HaluGate) achieves only 59% F1 on token-level detection.

5. **Verification of Actions, Not Just Text** — Agentic AI takes actions (API calls, database writes, tool use). Nobody verifies the ACTION is correct before execution. "Verification-aware planning" is a new concept with only one paper (VeriMAP, Oct 2025).

6. **Knowledge Base Drift** — Enterprise documents change. Verification systems don't track when their own reference knowledge becomes outdated.

---

## Three Groundbreaking Options (Ranked)

### Option A: Verification-Aware Agentic Pipeline (HIGHEST IMPACT)

**The gap:** AI agents are being deployed in enterprise (2026 is the year of agentic AI). They take actions — calling APIs, writing to databases, sending emails, making decisions. NOBODY is systematically verifying these actions before they execute.

Current guardrails are rule-based (Lakera, NeMo) — they block bad prompts but don't verify that the agent's REASONING is correct before it acts.

**What we'd build:** A verification layer that sits between an agent's decision and its execution:

```python
@veritas.before_action
async def execute_trade(symbol: str, amount: float):
    # Veritas automatically verifies:
    # 1. Is the reasoning behind this trade sound?
    # 2. Do the numbers match the data the agent was given?
    # 3. Does this action align with the stated goal?
    # 4. Are there edge cases or risks the agent missed?
    ...
```

**Why it's groundbreaking:**
- VeriMAP (Oct 2025) is the only paper on this, and it's a framework, not a tool
- The 2026 "Agents at Work" playbook calls this the #1 unsolved problem
- EU AI Act (2026 enforcement) requires verifiable AI decision-making
- Nobody has built a pip-installable pre-action verification library
- Market timing is perfect — every enterprise is deploying agents right now

**What makes it different from what we have:**
- Current Veritas verifies TEXT after generation
- This would verify ACTIONS before execution
- The agents would check: is this the right action given the goal and data?
- Integrates at the tool-call level, not the output level

**Risk:** Hard to build well. Agent actions are diverse (API calls, code execution, data mutations). Designing a universal verification interface is challenging.

---

### Option B: RAG Diagnostic Engine (HIGHEST PRACTICAL VALUE)

**The gap:** 40-60% of RAG deployments fail to reach production. The #1 reason: when the RAG system gives a wrong answer, teams can't diagnose WHY.

Was it:
- Bad retrieval? (Wrong documents surfaced)
- Bad generation? (LLM hallucinated despite correct documents)
- Bad chunking? (Documents were split poorly)
- Missing knowledge? (The answer isn't in the knowledge base at all)

No tool answers this question. RAGAS measures faithfulness but doesn't diagnose root cause. Existing eval tools tell you WHAT failed, not WHERE in the pipeline it failed.

**What we'd build:**

```python
from veritas import diagnose_rag

result = diagnose_rag(
    query="What is our refund policy?",
    retrieved_docs=docs,
    generated_answer=answer,
    knowledge_base_path="./docs/",  # optional: check if answer exists in KB
)

result.diagnosis
# RAGDiagnosis.GENERATION_HALLUCINATION
# "The answer adds a '90-day window' not present in any retrieved document.
#  Retrieved documents mention '30-day window' (policy.md:L42).
#  Root cause: generation hallucination, not retrieval failure."

result.retrieval_quality   # 0.85 — retrieved docs are relevant
result.generation_fidelity # 0.3  — answer diverges from retrieved docs
result.knowledge_coverage  # 1.0  — answer IS in the knowledge base
result.fix_suggestion      # "Generation is unfaithful to retrieved context. Consider: ..."
```

**Why it's groundbreaking:**
- 40-60% RAG failure rate in enterprise is a MASSIVE market pain
- No tool provides root-cause diagnosis (retrieval vs generation vs chunking vs missing knowledge)
- Our RAG grounding benchmark showed 89.7% F1 — we're already strong here
- This builds on Veritas's existing strength (grounded verification with context)
- Directly actionable — tells teams what to fix, not just that something failed

**What makes it different from what we have:**
- Current Veritas says "REFUTED" — the answer is wrong
- This would say "REFUTED because the LLM hallucinated despite correct retrieval at line 42 of policy.md"
- It separates the pipeline into stages and verifies EACH stage independently
- Provides fix suggestions, not just error detection

**Risk:** Lower. This is a natural extension of what we've built. The verification agents already check claims against context.

---

### Option C: Proof-Carrying Verification for Code Generation (MOST NOVEL)

**The gap:** LLM-generated code is everywhere. Copilot, Claude Code, Cursor. But there's no verification layer between "code generated" and "code deployed." Vericoding (Sep 2025) showed LLMs can generate formal proofs for ~82% of Dafny programs, but nobody has made this accessible.

**What we'd build:** A verification layer that generates lightweight proofs alongside code verification:

```python
from veritas import verify_code

result = verify_code(
    code=generated_function,
    spec="Function should sort a list in O(n log n) and handle empty lists",
    language="python",
)

result.verdict          # Verdict.PARTIAL
result.proof_status     # "2/3 properties verified"
result.verified_properties   # ["handles empty list", "returns sorted output"]
result.unverified_properties # ["O(n log n) complexity — could not verify"]
result.counterexample        # "Input [3, 1, 2] returns [1, 2, 3] ✓ but worst-case analysis inconclusive"
```

**Why it's groundbreaking:**
- Martin Kleppmann (Dec 2025): "AI will make formal verification go mainstream"
- Vericoding success rates went from 68% to 96% in Dafny in one year
- Nobody has made this accessible as a pip-installable tool
- Combines our adversarial approach (counterexample finding) with formal methods
- Directly addresses the "vibe coding" safety concern

**What makes it different from what we have:**
- Current `domain="code"` checks code against spec with LLM reasoning
- This would actually attempt to PROVE properties mathematically
- The adversary agent would generate counterexamples via fuzzing, not just LLM reasoning
- Could integrate with existing formal verification tools (Dafny, Lean, Z3)

**Risk:** Highest. Formal verification is hard. Making it accessible is harder. The gap between "82% in Dafny" and "works for arbitrary Python" is enormous.

---

## My Recommendation

**Option B (RAG Diagnostic Engine) first, with elements of Option A.**

Here's why:

1. **Market timing:** RAG is WHERE enterprises are RIGHT NOW. Agentic is coming but RAG is the current pain.

2. **Builds on strength:** Our RAG grounding benchmark showed 89.7% F1. We're already good at this.

3. **Unique angle:** Nobody diagnoses root cause. RAGAS says "faithfulness: 0.3." We'd say "generation hallucinated at line 42 despite correct retrieval."

4. **Practical impact:** Telling a team "your chunking is too coarse" or "your retrieval is missing documents about X" is 10x more valuable than "your answer is wrong."

5. **Natural path to Option A:** Once we can diagnose RAG pipelines, extending to agentic pipelines (verify the reasoning before action) is a natural step.

6. **Publishable:** "Root-cause diagnosis of RAG failures with multi-agent verification" is a real paper. Nobody has done this.

The groundbreaking claim becomes:

> "Existing RAG evaluation tools tell you THAT your pipeline failed. Veritas tells you WHERE it failed and WHY — separating retrieval errors from generation hallucinations from missing knowledge, with actionable fix suggestions."

That's a tool teams would actually pay for.
