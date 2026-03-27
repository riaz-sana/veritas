# Veritas

**Multi-agent verification for AI outputs.** Three tools in one library.

```
pip install git+ssh://git@github.com/yourorg/veritas.git
```

---

## The Three Tools

Veritas does three things. The interface is always the same: pass what you want to check, get a structured result.

```
┌─────────────────────────────────────────────────────────┐
│                        VERITAS                          │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   verify()   │  │ diagnose_rag │  │ verify_action │  │
│  │              │  │     ()       │  │     ()        │  │
│  │ Check any    │  │ WHY did the  │  │ Is this agent │  │
│  │ claim for    │  │ RAG pipeline │  │ action safe   │  │
│  │ accuracy     │  │ fail?        │  │ to execute?   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  Same architecture: agents in isolation → synthesiser   │
└─────────────────────────────────────────────────────────┘
```

### 1. Verify Claims

```python
from veritas import verify

result = await verify("The first iPhone was released in 2006")
# REFUTED (0.98) — The first iPhone was released June 2007, not 2006.
```

### 2. Diagnose RAG Failures

```python
from veritas import diagnose_rag

result = await diagnose_rag(
    query="What is our refund policy?",
    retrieved_docs=["Policy: 30-day return window..."],
    generated_answer="Our refund window is 90 days.",
)
# generation_contradiction — Answer says '90 days' but doc says '30 days'
#   Retrieval relevance:  85%    ← Docs are fine
#   Generation fidelity:   0%    ← THIS is where it broke
#   3 ungrounded claims identified with source evidence
```

### 3. Verify Agent Actions Before Execution

```python
from veritas import before_action

@before_action
async def transfer_funds(account: str, amount: float):
    ...

# Veritas checks reasoning, parameters, risks, and scope
# BLOCKED (0.99) — Amount $500K is 100x the $5K invoice. 12 risks identified.
```

---

## How It Works

All three tools use the same architecture: **specialized agents analyze in parallel isolation, then a synthesiser combines their findings.**

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
  │   (isolated)   │ │   (isolated)   │ │   (isolated)   │
  │                │ │                │ │                │
  │ Sees different │ │ Sees different │ │ Sees different │
  │ information    │ │ information    │ │ information    │
  │ than Agent 2   │ │ than Agent 1   │ │ than Agent 1   │
  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
          │                  │                   │
          └──────────────────┼───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │    Synthesiser      │
                  │                     │
                  │ Combines independent│
                  │ findings into       │
                  │ verdict + evidence  │
                  └─────────────────────┘
```

**Why isolation matters:** When agents share context, they reinforce each other's biases. When they're isolated, they catch different things. Our ablation study proved this: multi-agent finds +1.6 more issues (completeness) and cites +1.0 more specific evidence than a single prompt.

### What each tool runs:

| Tool | Agents | What each checks |
|------|--------|-----------------|
| `verify()` | Logic, Source, Adversary, Calibration, Synthesiser | Consistency, facts, counterexamples, confidence |
| `diagnose_rag()` | Retrieval Auditor, Generation Auditor, Coverage Auditor, Synthesiser | Docs relevant? Answer faithful? KB has the info? |
| `verify_action()` | Reasoning, Parameters, Risk, Scope, Synthesiser | Logic sound? Params correct? What could go wrong? Matches goal? |

---

## Quick Start

```bash
# Install
pip install git+ssh://git@github.com/yourorg/veritas.git

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Try it
veritas check "The Great Wall is visible from space"
```

### Python

```python
from veritas import verify, diagnose_rag, before_action, Verdict

# Verify a claim
result = await verify("claim", context="optional docs", domain="technical")

# Diagnose a RAG failure
diag = await diagnose_rag(query, retrieved_docs, generated_answer)

# Gate an agent action
@before_action
async def dangerous_action(param):
    ...
```

### CLI

```bash
veritas check "claim"              # Quick check
veritas check "..." --verbose      # Full evidence chain
veritas check "..." --json         # JSON output
cat output.txt | veritas check --stdin
veritas shell                      # Interactive mode
```

### Claude Code Skill

```
/verify The RAG pipeline says our refund window is 90 days
```

### MCP Server (any AI tool)

```json
{
  "mcpServers": {
    "veritas": {
      "command": "python",
      "args": ["-m", "veritas.mcp_server"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

---

## Integration — Same Pattern Everywhere

```python
# RAG: verify answer against retrieved docs
result = await verify(claim=rag_answer, context="\n".join(docs))

# Or use diagnose_rag for root-cause analysis
diag = await diagnose_rag(query, docs, answer)
if diag.diagnosis != RAGDiagnosis.FAITHFUL:
    print(f"Root cause: {diag.root_cause}")
    print(f"Fix: {diag.fix_suggestion}")

# Agentic: verify before acting
result = await verify_action(action="send_email", parameters={...}, goal="...")
if not result.approved:
    print(f"Blocked: {result.reasoning}")

# Or use the decorator
@before_action(goal="Process refund")
async def process_refund(order_id, amount):
    ...

# Batch evaluation
results = [await verify(claim=o) for o in model_outputs]

# CI/CD
veritas check "$(cat ai_output.txt)" --json
```

---

## Verdicts & Failure Modes

| Verdict | Meaning | What to do |
|---------|---------|------------|
| `VERIFIED` | Evidence supports the claim | Safe to use |
| `PARTIAL` | Some parts correct, some not | Check failure modes |
| `UNCERTAIN` | Not enough evidence | Human review |
| `DISPUTED` | Conflicting evidence | Human review |
| `REFUTED` | Evidence contradicts the claim | Don't use — check why |

When something is wrong, Veritas tells you the TYPE:

| Failure Mode | Example |
|-------------|---------|
| `factual_error` | "Released in 2006" → actually 2007 |
| `logical_inconsistency` | Premises don't support conclusion |
| `unsupported_inference` | Correlation stated as causation |
| `temporal_error` | Using 2020 data for a 2026 claim |
| `scope_error` | "All X do Y" when only some do |
| `source_conflict` | Two sources disagree |

---

## Enterprise Features

```python
from veritas import Config, AgentModels

# Economy mode — 60% cheaper (Haiku for simple agents, Sonnet for critical ones)
config = Config(agent_models=AgentModels.economy())

# Caching — zero cost on repeat queries
config = Config(cache_enabled=True, cache_ttl_seconds=3600)

# Confidence routing — skip verification for high-confidence outputs
config = Config(confidence_routing=True, confidence_threshold=0.8)
result = await verify("claim", config=config, source_confidence=0.95)
# Returns instantly — skipped

# Domain-specific prompts (code, schema, medical, legal, financial, scientific)
result = await verify(claim=generated_code, context=spec, domain="code")
```

---

## Benchmark Results

### Ablation: Multi-Agent vs Single-Prompt (9 cases, blind evaluation)

| Metric | Multi-Agent | Single-Prompt | Winner |
|--------|-------------|---------------|--------|
| Accuracy | 9.1 | 9.1 | Tie |
| Completeness | **9.7** | 8.1 | **MA (+1.6)** |
| Specificity | **9.4** | 8.4 | **MA (+1.0)** |
| Overall | **9.3** | 8.6 | **MA (+0.7)** |
| Cost | 4.4x | 1x | SP cheaper |

Multi-agent wins 7/9 cases. Same accuracy, but significantly more thorough.

### RAG Grounding (25 items)

| Metric | Result |
|--------|--------|
| F1 | **89.7%** |
| Precision | 81.3% |
| Recall | 100% |

### FaithBench (NAACL 2025, 50 samples)

| Metric | Result | Published SOTA |
|--------|--------|---------------|
| Balanced Accuracy | **58%** | 58% (o3-mini) |

---

## Docs

| Document | Description |
|----------|-------------|
| [Usage Guide](docs/USAGE.md) | Complete integration patterns for every architecture |
| [Enterprise Reality](docs/ENTERPRISE-REALITY.md) | Honest assessment of where it works and doesn't |
| [Ablation Study](docs/research/ablation-study.md) | Multi-agent vs single-prompt proof |
| [Project Status](docs/research/project-status-report.md) | Full status with strengths and weaknesses |

---

## Performance

| Metric | Value |
|--------|-------|
| Time per verify() | ~15-20s (multi-agent) / ~5s (single-prompt) |
| Time per diagnose_rag() | ~12-20s |
| Time per verify_action() | ~20-25s |
| Cost per call | ~$0.08 (multi-agent) / ~$0.02 (single-prompt) |
| Tests | 110 passing |

---

## Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."       # Required
export BRAVE_API_KEY="..."                   # Optional — enables web search
```

---

## License

MIT
