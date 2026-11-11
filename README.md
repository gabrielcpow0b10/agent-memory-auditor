# Agent Memory Auditor

A local-first NLP research project for detecting contradictions,
legitimate updates, and outdated information in conversational
agent memory.

## Overview

Agent Memory Auditor evaluates a new memory against an existing
memory history without automatically modifying stored information.

The research combines:
- Semantic retrieval using Sentence-BERT.
- Bidirectional natural language inference using DeBERTa-v3.
- Temporal change detection (planned).
- Explainable four-label memory classification (planned).

## Memory Labels

- `consistent`: Compatible or equivalent information.
- `new`: New information without a detected conflict.
- `updated`: A previous fact is replaced by a legitimate change.
- `contradictory`: Conflicting information without sufficient
  evidence of a legitimate update.

## Current Implementation

- Validated and typed memory schemas.
- Top-k semantic retrieval with cosine similarity.
- Bidirectional NLI scoring contracts.
- Synthetic retrieval demonstration.
- Automated unit tests.
- Documented NLI false-positive analysis.

Temporal reasoning and the final decision pipeline are still
under development.

## Quickstart

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,retrieval]"
python -m pytest -q
python examples/semantic_retrieval_demo.py
```

## Research Protocol

The planned evaluation uses 300 hand-checked synthetic memory
pairs, balanced across four labels.

- Development: 100 pairs.
- Held-out test: 200 pairs.
- RQ1: Retrieval effectiveness and computational cost.
- RQ2: Temporal updates versus contradictions.
- RQ3: Comparison against a local LLM judge.

The final dataset and benchmark results are not yet available.

## Known Limitations

Initial exploratory tests identified false-positive
contradictions in the DeBERTa-v3 NLI checkpoint.

See `docs/nli_diagnostic.md`.

## Privacy and Safety

- Local-first architecture.
- Synthetic data for research and demonstrations.
- Read-only auditing; no automatic memory modifications.
- No private agent memories or credentials in the repository.

## Project Status

Research prototype under active development.

Course: CAI 4304 — Natural Language Processing.
Florida International University.
