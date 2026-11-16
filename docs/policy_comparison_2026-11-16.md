# Policy Comparison — November 16, 2026

## Objective

Evaluate retrieval and bidirectional NLI decision
policies without modifying the original Stage 2 baseline.

## Experimental setup

Dataset: 20 manually annotated synthetic memory pairs.
Split: development pilot.
Encoder: sentence-transformers/all-MiniLM-L6-v2.
NLI: cross-encoder/nli-deberta-v3-base.
Device: CPU.

## Policies

| Policy | Similarity threshold | Contradiction rule |
|---|---|---|
| A | 0.35 | Either direction |
| B | 0.25 | Either direction |
| C | 0.35 | Both directions |
| D | 0.25 | Both directions |

## Results

| Policy | Correct | Errors |
|---|---|---|
| A | 18/20 | N001, X003 |
| B | 18/20 | N001, N003 |
| C | 19/20 | X003 |
| D | 19/20 | N003 |

## Failure analysis

### N001 — False directional contradiction

Similarity: 0.680973
Forward contradiction: 0.001864
Reverse contradiction: 0.998831

Bidirectional consensus eliminates this false positive.

### N003 — False bidirectional contradiction

Similarity: 0.296889
Forward contradiction: 0.999953
Reverse contradiction: 0.998763

Both directions falsely predict contradiction
between two compatible hobbies.

### X003 — Missed historical contradiction

Similarity: 0.287917
Forward contradiction: 0.999949
Reverse contradiction: 0.999950

The baseline retrieval threshold excludes this
genuine contradiction.

## Key finding

No single minimum similarity threshold can recover
X003 while excluding N003:

similarity(X003) < similarity(N003).

Bidirectional NLI consensus alone is insufficient
because N003 is a false contradiction in both directions.

## Research implications

A potential future approach is to examine whether
both memories make incompatible claims about the
same attribute before declaring a contradiction.

This is a research hypothesis, not a validated solution.

## Limitations

- Only 20 development examples were evaluated.
- Policies were explored using these same examples.
- The observed 95% is not an unbiased performance estimate.
- No held-out test examples were used.
- The baseline remains unchanged.

## Next steps

1. Preserve baseline and comparison artifacts.
2. Expand development data to 100 balanced pairs.
3. Compare policies using reproducible metrics.
4. Select and freeze the final decision policy.
5. Evaluate once on the 200 held-out pairs.
6. Report per-class precision, recall, F1 and
   confusion matrices.
