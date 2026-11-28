# Agent Memory Auditor — Held-Out Protocol v0.1

## Research Objective

Evaluate the frozen C_consensus policy on 200 unseen
memory pairs without modifying the selected classifier.

## Dataset Design

Total: 300 pairs.

Development: 100 pairs.
Held-out: 200 pairs.

Held-out distribution:

- consistent: 50
- new: 50
- updated: 50
- contradictory: 50

Final distribution: 75 pairs per label.

## Independence Requirements

1. No shared persona IDs across splits.
2. No identical or reversed memory pairs.
3. No overlapping paraphrase families.
4. Use distinct synthetic scenarios.
5. Keep proposed labels separate from predictions.
6. Complete annotation review before evaluation.
7. Do not tune on held-out examples.

## Annotation Rules

The four gold labels are:

- consistent
- new
- updated
- contradictory

Temporal markers:

- explicit
- implicit
- none

Use semantic reasoning and temporal evidence.

All examples must retain their documented
synthetic provenance.

DNLI provenance must only be claimed for examples
actually obtained and adapted from DNLI.

## Frozen Model

Policy: C_consensus

Configuration:
docs/policy_freeze_v0_1.json

Development result: 85/100.

The original Stage 2 baseline remains unchanged.

## Final Evaluation

After annotation review:

1. Verify 200 gold-labeled pairs.
2. Check split integrity.
3. Run the frozen policy.
4. Compute accuracy.
5. Compute per-class precision and recall.
6. Compute macro and weighted F1.
7. Generate a confusion matrix.
8. Document errors and limitations.

Do not tune the frozen policy using test results.

## Scientific Limitations

Synthetic examples do not establish performance
on real conversational memory.

Single-reviewer approval does not establish
inter-annotator agreement.

Cohen's kappa must not be reported without
a genuine repeated annotation experiment.

## Status

Protocol defined.
Held-out dataset construction pending.
Independent evaluation not yet performed.
