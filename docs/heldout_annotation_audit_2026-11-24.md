# Agent Memory Auditor
## Held-Out Annotation and Split Audit

Date: 2026-11-24

## 1. Dataset Composition

- Development pairs: 100
- Held-out pairs: 200
- Total corpus: 300
- Held-out pairs per semantic label: 50
- Total pairs per semantic label: 75

## 2. Annotation Review

Researcher and annotation reviewer: Gabriel Cruz.

All 200 held-out pairs were individually reviewed and approved by the researcher.

All 200 held-out candidates received an approved
AI-assisted annotation review using opaque review IDs.

Original pair IDs, proposed labels, marker annotations,
and rationales were hidden in the review interface.

The review did not use classifier predictions.

Review results:

- Approved: 200
- Flagged: 0
- Pending: 0
- Semantic label agreement: 200/200 (100%)
- Temporal marker agreement: 195/200 (97.5%)

These figures describe agreement with the original
draft annotations, not independent inter-annotator
agreement or classifier performance.

AI assistance was involved in both corpus construction
and annotation review. Therefore, the review cannot
be described as independent human annotation.

## 3. Temporal Marker Adjudication

Five updated examples were initially reviewed as
explicit instead of implicit:

- R018 / HU008
- R063 / HU007
- R083 / HU009
- R089 / HU010
- R146 / HU006

All five contain the verb started and describe an
inferred replacement of an earlier exclusive state.

Annotation Guide v0.1 identifies started combined
with temporal evidence of replacement as implicit.

The original review is preserved. A separate
adjudication CSV records all five corrections.

Final marker interpretation:

- Explicit: overt change expressions or direct
  temporal cues, including temporal distractors.
- Implicit: inferred replacement through started
  or temporal scope constructions such as since,
  as of, effective, and beginning in.
- None: no relevant temporal marker.

Starting an additional compatible activity may
contain an explicit temporal cue without constituting
an update.

These clarifications concern gold annotations only.
The frozen classifier and temporal logic are unchanged.

Expected final marker distribution:

| Label | Explicit | Implicit | None |
|---|---:|---:|---:|
| consistent | 0 | 0 | 50 |
| new | 7 | 1 | 42 |
| updated | 25 | 25 | 0 |
| contradictory | 8 | 0 | 42 |
| Total | 40 | 26 | 134 |

The held-out corpus contains 16 temporal distractors
outside the updated class.

## 4. Cross-Split Integrity

The lexical screening compared all 20,000 possible
development and held-out pair combinations.

Normalization used case folding and word extraction.

SequenceMatcher similarity was calculated in both
direct and reversed memory order.

Results:

- Shared persona IDs: 0
- Identical or reversed normalized pairs: 0
- Cross-split pairs sharing identical memories: 0
- Maximum lexical similarity: 0.6246

The 15 highest-scoring comparisons were inspected.

No equivalent complete memory pairs were identified
among these candidates.

Some conceptual and structural similarities remain.

Notable candidates:

- N010 / HN016: coffee-related scenarios.
- C020 / HX038: motorcycle-related scenarios.
- N017 / HN048: bicycle ownership and rented transport.

The last comparison presents notable scenario and
template overlap despite different transportation facts.

Lexical screening cannot establish the complete
absence of semantic leakage or shared paraphrase
families across the full corpus.

## 5. Targeted Ambiguity Review

The following annotations are retained under ordinary
language interpretations:

- R010: consistent. Recording versus measuring.
- R056: consistent. Course format versus personal pace.
- R084: updated. A fixed setting can later change.
- R087: updated. Paper envelopes and archival sleeves
  may overlap as storage categories.
- R139: contradictory. Interpretation focuses on the
  content of the reported assertion.
- R153: consistent. The original entrance statement
  supports the existence of a step-free entrance.
- R171: consistent. Equivalent costs despite awkward
  wording of the annual fee.
- R197: consistent. Ordinary interpretation of the
  planted tree's location.

R087 presents a particularly notable ambiguity.

The primary evaluation will include all 200 pairs.

A secondary sensitivity analysis will exclude these
eight predetermined review IDs, leaving 192 pairs.

This exclusion set was defined before held-out
classifier predictions were generated.

The secondary analysis will not replace the primary
evaluation or guide policy selection.

## 6. Scientific Limitations

- The corpus is synthetic.
- Corpus construction and annotation involved AI.
- The annotation review was not independently
  conducted by multiple human annotators.
- The planned repeated annotation of 60 examples
  after one week has not been conducted.
- Cohen's kappa is not reported.
- No verified DNLI-derived examples are claimed.
- Shared linguistic templates and conceptual
  overlap may remain across splits.
- Results cannot establish performance on real
  conversational agent memory.

## 7. Experimental Independence

C_consensus was selected using development data
before held-out evaluation.

Development accuracy: 85/100.

Held-out predictions have not been generated.

No classifier parameters, thresholds, temporal
rules, or model prompts were modified during this
annotation audit.

The final gold dataset and model revisions must
be verified and frozen before evaluation.

## 8. Audit Status

- Structural split validation: PASS
- Annotation completeness: PASS
- Annotation mapping integrity: PASS
- Temporal marker adjudication: COMPLETE
- Lexical overlap screening: COMPLETE
- Targeted ambiguity review: COMPLETE
- Final dataset freeze: PENDING
- Held-out evaluation: NOT EXECUTED
