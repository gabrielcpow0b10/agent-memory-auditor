# NLI Diagnostic — November 7, 2026

## Environment

- Model: cross-encoder/nli-deberta-v3-base
- Device: CPU
- Python: 3.12.16
- Sentence-Transformers: 5.7.0
- PyTorch: 2.15.0

## Preliminary observations

Six synthetic sentence pairs were evaluated in both directions.

| Pair | Forward | Reverse | Expected |
|---|---|---|---|
| Dog / chess | contradiction | contradiction | neutral |
| Dog / cat | neutral | contradiction | neutral |
| Miami / programmer | neutral | neutral | neutral |
| Pizza / bicycle | neutral | contradiction | neutral |
| Miami / reside Miami | entailment | entailment | entailment |
| No coffee / daily coffee | contradiction | contradiction | contradiction |

Directional predictions: 8 correct out of 12.

This is an exploratory diagnostic, not a benchmark result.

## Observed failure modes

1. False contradictions between compatible memories.
2. Different predictions when premise and hypothesis are reversed.
3. High softmax confidence on incorrect predictions.

## Research implications

The proposed bidirectional contradiction rule may amplify
false positives. This should be tested on the development
set before any thresholds or policies are modified.

The held-out test set must not be used for tuning.

## Next actions

- Implement traceable bidirectional NLI inference.
- Add unit tests with mocked model outputs.
- Build and review the 20-pair annotation pilot.
- Evaluate alternative decision policies on development data.
- Preserve the original Stage 2 method as a baseline.
