# Veritas

### Can you trust your AI's output?

LLMs hallucinate. RAG pipelines fabricate facts not in the documents. AI agents take wrong actions based on flawed reasoning. In 2026, the best hallucination detectors still only score [58% accuracy](https://github.com/vectara/FaithBench). AI-generated code has [37% more vulnerabilities](https://checkmarx.com) than human-written code. And [40-60% of enterprise RAG deployments fail](https://www.stackai.com/blog/rag-limitations) to reach production because nobody can diagnose why.

**Veritas is a research project exploring multi-agent verification for AI outputs** — with real benchmarks, tested hypotheses, and documented findings on what works and what doesn't.

```bash
pip install git+https://github.com/riaz-sana/veritas.git
```

---

## What It Does

Three tools. One interface. Every AI architecture.

```python
from veritas import verify, diagnose_rag, before_action

# 1. Verify any claim
result = await verify("The first iPhone was released in 2006")
# REFUTED (0.98) — Released June 2007, not 2006.

# 2. Diagnose WHY a RAG pipeline failed
result = await diagnose_rag(
    query="What is our refund policy?",
    retrieved_docs=["Policy: 30-day return window..."],
    generated_answer="Our refund window is 90 days.",
)
# generation_contradiction — Answer says '90 days' but doc says '30 days'
# Retrieval: 85% relevant  ← docs are fine
# Generation:  0% faithful  ← THIS is where it broke
# 3 ungrounded claims identified with source quotes

# 3. Verify agent actions BEFORE execution
@before_action
async def transfer_funds(account: str, amount: float):
    ...
# BLOCKED (0.99) — Amount $500K is 100x the $5K invoice. 12 risks identified.
```

---

## The Research Question

> **Does using multiple isolated AI agents produce better verification than a single well-crafted prompt?**

We ran 7 experiments to find out. Some confirmed our hypotheses. Some didn't.

### What We Proved

| Finding | Evidence |
|---------|----------|
| Multi-agent is **more thorough** than single-prompt | +1.6 completeness, +1.0 specificity in [blind evaluation](docs/research/ablation-study.md) (9 cases) |
| Multi-agent does NOT improve **binary accuracy** | Both score 9.1/10 on getting the core diagnosis right |
| Isolation is **2-3x faster** than shared-context debate | Consistent across [all benchmarks](docs/research/FINDINGS.md) |
| Isolation produces **fewer false positives** on RAG tasks | 3 vs 6 false alarms on [25 grounding tests](docs/research/benchmarks/rag-grounding-results.json) |

### What We Disproved

| Hypothesis | Result |
|-----------|--------|
| "Information asymmetry prevents confirmation bias" | Full-context evaluation [outperforms](docs/research/bias-headtohead-results.json) isolated agents 97.1% vs 91.4% on bias-triggering cases |

Full experiment details, raw data, and methodology: **[docs/research/FINDINGS.md](docs/research/FINDINGS.md)**

---

## Benchmark Results

### Ablation: Multi-Agent vs Single-Prompt

9 test cases (5 RAG + 4 action verification). Blind LLM judge, randomized order.

| Dimension | Multi-Agent | Single-Prompt | Winner |
|-----------|:-----------:|:-------------:|--------|
| Accuracy | 9.1 | 9.1 | Tie |
| Completeness | **9.7** | 8.1 | Multi-Agent (+1.6) |
| Specificity | **9.4** | 8.4 | Multi-Agent (+1.0) |
| Overall | **9.3** | 8.6 | Multi-Agent (+0.7) |
| Cost | 4.4x | 1x | Single-Prompt |
| Speed | 22.6s | 13.7s | Single-Prompt |

**Takeaway:** Multi-agent finds more issues and cites better evidence. Single-prompt gets the verdict right at 1/4 the cost. Choose based on whether you need thoroughness or speed.

### FaithBench (NAACL 2025 — hardest hallucination benchmark)

| Metric | Veritas | Published SOTA (o3-mini) |
|--------|:-------:|:------------------------:|
| Balanced Accuracy | **58%** | 58% |

### RAG Grounding (25 doc-answer pairs)

| Metric | Isolation Mode | Debate Mode |
|--------|:--------------:|:-----------:|
| F1 | **89.7%** | 81.3% |
| Precision | **81.3%** | 68.4% |
| Recall | 100% | 100% |

### RAGVUE Head-to-Head (bias-triggering cases)

| Metric | Full-Context (RAGVUE-style) | Isolated Agents (Veritas) |
|--------|:---------------------------:|:-------------------------:|
| Claim Accuracy | **97.1%** | 91.4% |
| False Positives | **1** | 3 |

**Honest conclusion:** Full-context single-pass evaluation beats our isolated multi-agent approach for claim-level accuracy. We adopted this finding.

---

## How It Works

```
                      ┌─────────────────┐
                      │  Input (claim,   │
                      │  docs, action)   │
                      └────────┬────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                   │
            ▼                  ▼                   ▼
  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
  │   Agent 1      │ │   Agent 2      │ │   Agent 3      │
  │                │ │                │ │                │
  │ Logic /        │ │ Facts /        │ │ Adversary /    │
  │ Retrieval      │ │ Generation     │ │ Risk           │
  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
          │                  │                   │
          └──────────────────┼───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │    Synthesiser      │
                  │                     │
                  │ Verdict + evidence  │
                  │ + failure modes     │
                  └─────────────────────┘
```

Each tool uses specialized agents:

| Tool | Agents | What they check |
|------|--------|----------------|
| `verify()` | Logic, Source, Adversary, Calibration | Consistency, facts, counterexamples, confidence |
| `diagnose_rag()` | Retrieval, Generation, Coverage | Docs relevant? Answer faithful? KB has the info? |
| `verify_action()` | Reasoning, Parameters, Risk, Scope | Logic sound? Params correct? Risks? Matches goal? |

---

## Real Test Output

### RAG Diagnostics — caught 3 fabricated claims with source evidence

```
Input:  "Our refund window is 90 days for all items. Refunds processed instantly."
Source: "30-day return window. Sale items final sale. 5-7 business days."

Diagnosis: generation_contradiction
Retrieval:  85% relevant  ← docs are correct
Generation:  0% faithful  ← LLM ignored the documents

Claims:
  [UNGROUNDED] "90 days" → doc says 30 days
  [UNGROUNDED] "all items including sale" → doc says sale items final sale
  [UNGROUNDED] "processed instantly" → doc says 5-7 business days

Fix: Add system prompt constraint to only use facts from documents.
```

### Action Verification — blocked a $500K fraud-pattern transfer

```
Action:  transfer_funds($500,000 → unknown_external_789)
Goal:    Pay vendor invoice #INV-2025-001 for $5,000

BLOCKED (0.99 confidence)

Risks identified (12):
  [CRITICAL] Amount is 100x the invoice ($500K vs $5K)
  [CRITICAL] Recipient 'unknown_external_789' is unverified
  [CRITICAL] Pattern matches Business Email Compromise fraud
  [CRITICAL] $500K triggers mandatory AML reporting
  [HIGH]     Wire transfer is irreversible
```

---

## Install & Use

```bash
pip install git+https://github.com/riaz-sana/veritas.git
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Python:**
```python
from veritas import verify, diagnose_rag, verify_action, before_action
```

**CLI:**
```bash
veritas check "Any claim"
veritas check "..." --verbose --json
veritas shell
```

**Claude Code:** `/verify The RAG says our policy is 90 days`

**MCP Server** (Claude Desktop, Cursor, any AI tool):
```json
{"mcpServers": {"veritas": {"command": "python", "args": ["-m", "veritas.mcp_server"]}}}
```

**Works with everything:** LangChain, LlamaIndex, CrewAI, AutoGen, FastAPI, CI/CD, batch eval. Same `verify(claim, context)` interface for all. See [docs/USAGE.md](docs/USAGE.md).

---

## Enterprise Features

```python
from veritas import Config, AgentModels

# Economy mode — Haiku for simple agents, Sonnet for critical (~60% cheaper)
config = Config(agent_models=AgentModels.economy())

# Caching — zero cost on repeat queries
config = Config(cache_enabled=True)

# Confidence routing — skip verification for high-confidence outputs
config = Config(confidence_routing=True, confidence_threshold=0.8)

# Domain-specific — code, schema, medical, legal, financial, scientific
result = await verify(claim=generated_code, context=spec, domain="code")
```

---

## Verdicts & Failure Modes

| Verdict | Meaning |
|---------|---------|
| `VERIFIED` | Evidence supports the claim |
| `PARTIAL` | Some parts correct, some not |
| `UNCERTAIN` | Insufficient evidence |
| `DISPUTED` | Conflicting evidence |
| `REFUTED` | Evidence contradicts the claim |

When something fails, Veritas classifies WHY:

| Type | What it means |
|------|--------------|
| `factual_error` | A fact is wrong |
| `logical_inconsistency` | Reasoning contradicts itself |
| `unsupported_inference` | Claim exceeds the evidence |
| `temporal_error` | Information is outdated |
| `scope_error` | Overgeneralization |
| `source_conflict` | Sources disagree |

---

## Project Structure

```
veritas/
  core/           # verify(), config, cache, data models
  agents/         # 5 verification agents + domain prompts
  diagnostics/    # RAG diagnostic engine
  agentic/        # Pre-action verification + @before_action
  orchestration/  # Parallel runner + challenge round
  providers/      # Claude, Brave Search, Tavily
  cli/            # check, shell, benchmark commands
  ablation/       # Multi-agent vs single-prompt comparison code
  benchmarks/     # FaithBench, RAG grounding, adversarial datasets
  mcp_server.py   # MCP server for any AI tool
skills/verify/    # Claude Code skill
docs/
  research/       # All experiment data, findings, methodology
  USAGE.md        # Integration patterns
  ENTERPRISE-REALITY.md  # Honest deployment assessment
```

110 tests. Python 3.10+. MIT License.

---

## Research Documents

| Document | What's in it |
|----------|-------------|
| **[FINDINGS.md](docs/research/FINDINGS.md)** | All 7 experiments — what we proved, what we disproved, raw data |
| [Ablation Study](docs/research/ablation-study.md) | Multi-agent vs single-prompt — methodology, 9 test cases, blind evaluation |
| [Honest Assessment](docs/research/honest-assessment-march-2026.md) | Competitive landscape — RAGVUE, Superagent, Galileo Luna, who does what |
| [Enterprise Reality](docs/ENTERPRISE-REALITY.md) | Where Veritas works, where it doesn't, cost/latency analysis |
| [Benchmark Methodology](docs/research/benchmarks/methodology.md) | Why each benchmark, dataset design, evaluation principles |
| [Groundbreaking Options](docs/research/groundbreaking-options.md) | Analysis of genuinely unoccupied territory (code verification) |

### Raw Benchmark Data (JSON)

| Dataset | Samples | Key Result |
|---------|---------|------------|
| [Ablation](docs/research/ablation-results.json) | 9 cases | MA 9.3 vs SP 8.6 overall |
| [FaithBench](docs/research/benchmarks/faithbench-results.json) | 50 | 58% balanced accuracy |
| [RAG Grounding](docs/research/benchmarks/rag-grounding-results.json) | 25 | 89.7% F1 |
| [Adversarial](docs/research/benchmarks/adversarial-results.json) | 50 | 100% detection (too easy) |
| [RAGVUE H2H](docs/research/ragvue-headtohead-results.json) | 33 claims | Tied at 100% |
| [Bias H2H](docs/research/bias-headtohead-results.json) | 35 claims | RAGVUE 97.1% vs Veritas 91.4% |

---

## References

### Papers
- Du et al. "[Improving Factuality through Multiagent Debate](https://arxiv.org/abs/2305.14325)" — ICML 2024
- "[Emergent social conventions and collective bias in LLM populations](https://www.science.org/doi/10.1126/sciadv.adu9368)" — Science Advances 2025
- "[Cross-Context Verification](https://arxiv.org/abs/2603.21454)" — 2026
- "[RAGVUE](https://arxiv.org/abs/2601.04196)" — 2026
- "[Agent-as-a-Judge](https://arxiv.org/abs/2601.05111)" — ICML 2025
- "[Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0)" — Nature 2024
- "[FaithBench](https://github.com/vectara/FaithBench)" — NAACL 2025
- Amazon "[Enhancing LLM-as-a-Judge via Multi-Agent Collaboration](https://assets.amazon.science/48/5d/20927f094559a4465916e28f41b5/enhancing-llm-as-a-judge-via-multi-agent-collaboration.pdf)"

### Competing Tools
- [RAGVUE](https://github.com/KeerthanaMurugaraj/RAGVue) — claim-level RAG evaluation
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation metrics
- [Superagent](https://github.com/superagent-ai/superagent) — agentic AI safety
- [Galileo Luna](https://galileo.ai) — fast hallucination detection
- [Axiom](https://axiommath.ai) — formal verification for Lean

---

## License

MIT
