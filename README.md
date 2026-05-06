# PriAgent: Multi-Agent Android Privacy Compliance Auditor

Implementation of the AAAI 2026 paper **"PriAgent: A Collaborative Multi-Agent Framework for Auditing Android Privacy Compliance"**, enhanced with LangChain, LangGraph, and RAG.

## Overview

PriAgent automates Android privacy compliance auditing using a four-stage multi-agent pipeline. Given static analysis outputs (taint flows), decompiled app code, and a privacy policy, it identifies genuine privacy violations while filtering out false positives.

```
Static Analysis Flows
        │
        ▼
┌─────────────────┐
│  FlowShaper     │  Stage 1 — Aggregates equivalent flows, detects call-chain cycles,
│                 │            and applies LLM-based semantic deduplication
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LocusFinder    │  Stage 2 — Identifies critical methods (semantic loci) per flow
│                 │            using dynamic τ_loci = clamp(chain_len // 5, 3, 8)
└────────┬────────┘
         │ LangGraph Send API (parallel fan-out)
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Verifier│ │Verifier│  Stage 3 — RAG-powered verification with self-reflection loop
└────┬───┘ └────┬───┘           (re-examines low-confidence verdicts < 0.60)
     └────┬─────┘
          │ (fan-in)
          ▼
┌─────────────────┐
│ PolicyScanner   │  Stage 4a — Extracts structured claims from privacy policy text
└────────┬────────┘
         ▼
┌─────────────────┐
│ComplianceArbiter│  Stage 4b — Cross-examines TPs against policy; assigns risk tier
└─────────────────┘
```

## Key Technical Features

| Feature | Implementation |
|---|---|
| Multi-agent orchestration | LangGraph `StateGraph` with `MemorySaver` checkpointing |
| Parallel verification | LangGraph `Send` API — one branch per flow |
| RAG knowledge base | FAISS + HuggingFace embeddings, parent-child retrieval strategy |
| Structured LLM output | `llm.with_structured_output(PydanticModel)` — no brittle JSON parsing |
| Self-reflection loop | Second LLM pass for verdicts with confidence < 0.60 |
| Tool use | `@tool`-decorated functions for on-demand code retrieval and inspection |
| Cross-session memory | JSON-persisted violation patterns; surfaced as "pattern hints" in future runs |
| Code summarization | Compresses methods > 3000 tokens before LLM injection |

## Improvements Over the Paper

1. **Dynamic τ_loci** — adaptive semantic locus cap based on chain length, vs fixed τ=5
2. **Self-reflection** — low-confidence verdicts are re-examined from a devil's advocate perspective
3. **Semantic deduplication** — LLM judges whether flows with different packages are semantically equivalent (Pass 3, beyond syntactic grouping)
4. **Cross-session memory** — violation patterns accumulate across audits and prime future sessions
5. **Actionable recommendations** — each violation report includes developer remediation guidance
6. **Broader RAG coverage** — dual-source knowledge base: hand-annotated FP indicators + 27-class JSON knowledge base

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

**Run the demo** (built-in sample data, includes a real malware flow and several FP cases):
```bash
python main.py demo
```

**Audit a real app:**
```bash
python main.py analyze \
  --app-name "MyApp" \
  --flows flows.json \
  --policy policy.txt \
  --code-json decompiled.json
```

**Rebuild the RAG index** (needed after modifying `android_api_kb.py` or the JSON knowledge base):
```bash
python main.py build-kb
```

**Run unit tests** (no API key required):
```bash
python tests/test_core.py
```

## Project Structure

```
priagent/
├── agents/
│   ├── flow_shaper.py        # Stage 1: flow abstraction + deduplication
│   ├── locus_finder.py       # Stage 2: semantic locus identification
│   ├── semantic_verifier.py  # Stage 3: RAG-powered verification + reflection
│   ├── policy_scanner.py     # Stage 4a: privacy policy parsing
│   ├── compliance_arbiter.py # Stage 4b: compliance synthesis
│   └── code_summarizer.py    # Helper: compresses large code blocks
├── rag/
│   ├── android_api_kb.py     # Primary KB: 12 APIs with FP indicators
│   └── knowledge_base.py     # FAISS index + parent-child retrieval
├── workflow/
│   └── priagent_graph.py     # LangGraph StateGraph orchestration
├── memory/
│   └── audit_memory.py       # Session + cross-session memory
├── tools/
│   └── code_tools.py         # @tool functions for agent use
├── schemas/
│   └── data_models.py        # Pydantic v2 data models
├── data/
│   ├── sample_flows.json             # 5 sample taint flows (demo)
│   ├── sample_policy.txt             # Sample privacy policy (demo)
│   └── android_privacy_knowledge (RAG).json  # 27-class supplementary KB
└── tests/
    └── test_core.py          # Unit tests (no API key needed)
```

## Datasets

This implementation is compatible with the following public Android privacy/taint analysis benchmarks:

- **UBCBench** — Benchmark for Undeclared Behavior in Android apps
  - Repository: https://github.com/LinaQiu/UBCBench
  - Use: provides ground-truth flows for evaluating precision/recall of the audit pipeline

- **TaintBench** — Malware-based taint analysis benchmark
  - Website: https://taintbench.github.io/
  - Repository: https://github.com/TaintBench/TaintBench
  - Use: malicious app flows for testing true-positive detection rate

To use these datasets, export their taint flows in the `DataFlow` JSON format (see `data/sample_flows.json` for the schema) and pass them via `python main.py analyze --flows <path>`.

## Citation

```bibtex
@inproceedings{priagent2026,
  title     = {PriAgent: A Collaborative Multi-Agent Framework for Auditing Android Privacy Compliance},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```
