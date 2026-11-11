# Agent Memory Auditor — Annotation Guide v0.1

## Purpose

This document defines the gold-label annotation protocol
for synthetic conversational agent memory pairs.

Labels are assigned by semantic interpretation, never
by copying predictions from an NLP model.

## Labels

### consistent
The incoming memory expresses information equivalent
to, or supported by, an existing memory.

### new
The incoming memory contributes compatible information
without replacing or contradicting an earlier fact.

### updated
The incoming memory replaces an earlier fact, and
there is identifiable evidence of a legitimate change.

### contradictory
The two memories contain incompatible claims without
sufficient evidence of a legitimate update.

## Temporal markers

- explicit: An overt temporal/change expression such as
  "now", "no longer", "moved to", or "switched to".
- implicit: A later timestamp combined with a change verb,
  such as "started", indicating a replacement.
- none: No relevant temporal change expression.

A temporal word alone does not prove an update.
A later timestamp alone does not prove an update.

## Annotation procedure

1. Read both memories and their timestamps.
2. Identify the facts or attributes being compared.
3. Determine whether both statements are compatible.
4. Check whether a previous fact is being replaced.
5. Examine the evidence of temporal change.
6. Assign exactly one of the four labels.
7. Record the marker type, provenance, and rationale.
8. Flag ambiguous examples for manual review.

## Pilot dataset

- 20 manually authored synthetic pairs.
- Five examples per memory label.
- Split identifier: pilot.
- Source: synthetic_manual.
- No real personal memory data.

The pilot is development material only.

## Final research dataset

Target: 300 manually reviewed pairs.

- consistent: 75
- new: 75
- updated: 75
- contradictory: 75

Development split: 100 pairs.
Held-out split: 200 pairs.

Separate related personas and paraphrase families to
avoid information leakage between the two splits.

The final corpus should contain both DNLI-derived and
manually authored synthetic examples.

## Quality control

A sample of 60 final pairs will be annotated again
after one week.

Agreement will be measured using Cohen's kappa.

Ambiguous cases must be reviewed or replaced.

The held-out set must never be used to tune
thresholds, marker rules, or model prompts.

## Known failure modes

- False contradictions between compatible facts.
- NLI disagreement when sentence order is reversed.
- Temporal expressions without genuine updates.
- Confusing additions of information with replacements.
- Overconfident model predictions.

## Status

Version 0.1 — Pilot annotation protocol.
Not yet a validated final benchmark.
