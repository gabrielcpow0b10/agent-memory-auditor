# Pipeline Baseline — Error Analysis

Date: 2026-11-16
Status: Development pilot, not held-out evaluation

## Experimental setup

- Retrieval: sentence-transformers/all-MiniLM-L6-v2
- NLI: cross-encoder/nli-deberta-v3-base
- Device: CPU
- Pilot: 20 synthetic manually annotated pairs
- Retrieval similarity threshold: 0.35

## Pilot results

- Overall: 18/20 correct (90%)
- Consistent: 5/5
- New: 4/5
- Updated: 5/5
- Contradictory: 4/5

## Failure N001

Expected: new
Predicted: contradictory

Old: I have a dog.
New: I also have a cat.

Similarity: 0.680973
Forward: neutral (0.9981)
Reverse: contradiction (0.9988)
Temporal update evidence: false

Root cause:
A false reverse-direction NLI contradiction
was accepted by the baseline decision policy.

## Failure X003

Expected: contradictory
Predicted: new

Old: I have never lived outside Canada.
New: I lived in Germany in 2014.

Similarity: 0.287917
Retrieval threshold: 0.35
Forward contradiction: 0.9999
Reverse contradiction: 0.9999
Temporal update evidence: false

Root cause:
Retrieval filtering excluded the conflicting
memory before NLI could contribute to the
pipeline's final decision.

## Research implications

The baseline has two distinct failure modes:
1. False contradictions from directional NLI errors.
2. Missed contradictions caused by retrieval filtering.

Both must be evaluated separately.

Preserve the original Stage 2 baseline.
Compare alternative policies on development data only.
Do not tune on the held-out test set.

## Faculty feedback alignment

Stage 1 feedback requires:
- Mapping three NLI outputs to four memory labels.
- Defining the 300-pair annotation scheme.
- Defining and preserving a held-out split.

## Limitations

The 20-pair pilot is not an independent benchmark.
The observed 90% accuracy is preliminary.
