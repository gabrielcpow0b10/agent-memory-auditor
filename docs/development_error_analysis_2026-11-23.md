# Development Error Analysis — November 23, 2026

## Experiment

Dataset: 100 synthetic development candidates.

Models:
- Sentence-BERT: all-MiniLM-L6-v2
- NLI: cross-encoder/nli-deberta-v3-base

All four decision policies were evaluated without
changing the archived Stage 2 baseline.

## Results

| Policy | Correct | Accuracy |
|---|---:|---:|
| A_baseline | 80/100 | 80% |
| B_retrieval | 76/100 | 76% |
| C_consensus | 85/100 | 85% |
| D_combined | 83/100 | 83% |

Policy C achieved the highest observed development
accuracy, improving on A by five percentage points.

## Policy C: Error Categories

### False NLI contradictions (7)

N006, N009, N014, N016, N018, N022, N023

Both NLI directions assigned high contradiction
scores to compatible statements.

### Temporal changes not recognized (4)

U007, U013, U016, U018

The system recognized contradictions but did not
establish sufficient temporal update evidence.

### Updates without NLI consensus (2)

U010, U025

Temporal evidence was detected, but the NLI
consensus rule was not satisfied.

### Retrieval false negative (1)

X003

The similarity score of 0.287917 fell below
the baseline retrieval threshold of 0.35.

### Consistency false negative (1)

C022

The system returned NEW instead of CONSISTENT
despite high semantic similarity.

## Research Interpretation

The errors show that retrieval, NLI, and temporal
reasoning contribute distinct failure modes.

The best-performing development policy is C,
but it is not yet a final independent result.

## Scientific Limitations

- The development dataset consists of 100 synthetic memory pairs.
- Final semantic annotation review remains pending.
- Development data was used to explore policies.
- Reported accuracy is exploratory, not held-out.
- No results from the 200-pair test split are used.

## Next Steps

1. Complete semantic annotation review.
2. Evaluate targeted fixes using development data.
3. Freeze the selected decision policy.
4. Prepare the independent 200-pair test split.
5. Run held-out evaluation once after freezing.
6. Report precision, recall, F1 and confusion matrix.
7. Complete RQ1-RQ3 experiments and Stage 3 slides.

The original Stage 2 baseline must be preserved.
