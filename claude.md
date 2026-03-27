# Veritas

## What This Is
A multi-agent verification library for AI outputs. Three capabilities:

1. **`verify(claim, context)`** — verify any claim or AI output
2. **`diagnose_rag(query, docs, answer)`** — diagnose WHY a RAG pipeline failed
3. **`verify_action()` / `@before_action`** — verify agent actions before execution

## Architecture
All three use the same pattern: specialized agents in parallel isolation → synthesiser.
Each agent sees different information (information asymmetry) to prevent confirmation bias.

## Key Files
- `veritas/core/verify.py` — main entry point
- `veritas/diagnostics/rag.py` — RAG diagnostic engine (3 auditors + synthesiser)
- `veritas/agentic/verification.py` — pre-action verification (4 verifiers + synthesiser)
- `veritas/agents/` — the 5 core verification agents
- `veritas/core/config.py` — config with enterprise features (tiered models, caching, routing)

## Branches
- `main` — shipping code (library, CLI, skill, MCP, tests, docs)
- `research` — everything above + benchmarks, raw results, methodology, ablation study

## Tests
```bash
.venv/bin/python -m pytest tests/ -q  # 110 tests
```

## Proven Results
- Ablation: multi-agent beats single-prompt 7/9 cases (+1.6 completeness, +1.0 specificity)
- RAG grounding: 89.7% F1 (isolation mode)
- FaithBench: 58% balanced accuracy (matches o3-mini SOTA)
