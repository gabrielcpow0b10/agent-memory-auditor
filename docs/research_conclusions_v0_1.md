# Agent Memory Auditor
## Experimental Findings and Research Conclusions

Author: Gabriel Cruz Castro
Institution: Florida International University
Course: CAI 4304 — Natural Language Processing
Date: November 25, 2026

## 1. Research Objective

This project investigates whether a local NLP pipeline
can identify consistent, new, updated, and contradictory
information in conversational agent memory.

The implemented approach combines Sentence-BERT,
cosine similarity, bidirectional natural language
inference using DeBERTa-v3, and temporal change rules.

The four-label mapping extends the standard three-class
NLI task by distinguishing legitimate updates from
contradictions.

## 2. Experimental Design

Two experimental resources were used:

- Dialogue NLI controlled histories for RQ1.
- A human-reviewed synthetic corpus of 300 memory pairs
  for RQ2 and RQ3.

The synthetic corpus contains 100 development pairs
and 200 held-out pairs, balanced across four labels.

The NLP policy and LLM prompt were frozen before their
respective held-out inference runs.

Macro F1 is the primary classification metric.
Accuracy, precision, recall, confusion matrices,
latency, and statistical comparisons complement it.

## 3. RQ1 — Semantic Retrieval

Does semantic retrieval improve conflict detection
compared with recency and exhaustive retrieval,
and at what computational cost?

The experiment used 60 Dialogue NLI source cases,
540 controlled histories, and 2,160 method evaluations.

| Method | Macro F1 | Conflict retrieval recall | Target conflict recall |
|---|---:|---:|---:|
| Recency | 0.3021 | 33.33% | 26.67% |
| Exhaustive | 0.1767 | 100.00% | 80.00% |
| Semantic | 0.3876 | 60.00% | 50.00% |
| Oracle | 0.6827 | 100.00% | 80.00% |

Semantic retrieval required 2,700 NLI pair comparisons,
compared with 30,600 for exhaustive retrieval.

This represents a 91.18% reduction in pair comparisons.

Mean latency was 197.00 ms for semantic retrieval
and 1,931.90 ms for exhaustive retrieval.

Semantic retrieval therefore achieved approximately
9.81 times lower mean latency.

The paired cluster bootstrap estimated a macro-F1
improvement of 0.0855 over recency, with a 95%
confidence interval of [0.0002, 0.1697].

However, semantic conflict retrieval recall decreased
from 85% for 20 memories to 35% for 100 memories.

The compatible-case false alarm rate remained high
at 70.83% for semantic retrieval.

Conclusion: semantic retrieval improves efficiency
and performance relative to recency, but does not
achieve recall close to the oracle. The original
expectation is only partially supported.

## 4. RQ2 — Temporal Reasoning

Do temporal change signals distinguish legitimate
updates from contradictions better than simple
resolution policies?

| Policy | Accuracy | Macro F1 |
|---|---:|---:|
| C_consensus | 70.50% | 0.6488 |
| Most Recent Wins | 63.50% | 0.5652 |
| Never Update | 68.00% | 0.5969 |

The implemented temporal policy achieved the highest
observed accuracy and macro F1.

However, McNemar's test did not establish a statistically
significant accuracy difference:

- Most Recent Wins: p = 0.1456.
- Never Update: p = 0.0625.

The pipeline correctly identified only 5 of 50 updates,
while correctly identifying 47 of 50 contradictions.

The error analysis identified 45 update errors among
the 59 total classification errors.

Only 8 of 50 genuine updates were recognized as
temporal update candidates.

Conclusion: temporal rules provide an observed
improvement over the simple baselines, but the
statistical evidence is insufficient to establish
superiority at the 0.05 significance level.

Limited temporal rule coverage remains a major
weakness of the implemented system.

## 5. RQ3 — Local LLM Judge

How does the NLP pipeline compare with a local LLM
in classification performance and processing latency?

The experiment compared the frozen NLP pipeline
against Qwen3 8B, quantized using Q4_K_M and executed
locally through Ollama.

Both systems evaluated the same 200 memory pairs.

| Metric | NLP Pipeline | Qwen3 8B |
|---|---:|---:|
| Accuracy | 70.50% | 89.50% |
| Macro F1 | 0.6488 | 0.8980 |
| Updated recall | 10.00% | 98.00% |
| Contradictory recall | 94.00% | 80.00% |
| Mean latency | 39.52 ms | 1,016.02 ms |

The LLM achieved 38 additional correct classifications.

Its macro-F1 improvement was 0.2491, with a paired
95% bootstrap confidence interval of [0.1816, 0.3132].

McNemar's test produced p = 0.00001484.

The NLP pipeline achieved approximately 25.71 times
lower mean latency.

Qwen3 correctly classified 49 of 50 updates but
misclassified 10 contradictions as updates.

The NLP pipeline classified no genuine contradictions
as updates in this particular evaluation.

Conclusion: the original RQ3 expectation is supported
within this experimental setting. The LLM distinguishes
updates more effectively but requires substantially
greater processing time.

The NLP pipeline retains an advantage in contradiction
recall and processing latency.

## 6. Sensitivity Analysis

Eight ambiguous examples were identified before
the original held-out evaluation.

Excluding these examples produced a secondary
evaluation containing 192 pairs.

| Metric | NLP Pipeline | Qwen3 8B |
|---|---:|---:|
| Accuracy | 70.83% | 89.58% |
| Macro F1 | 0.6549 | 0.8987 |

The main findings remained similar.

The complete 200-pair evaluation remains the
primary reported result.

## 7. NLP Concepts and Academic Foundation

The implementation connects the research design
to the following NLP concepts:

- Semantic sentence representations.
- Cosine similarity.
- Natural language inference.
- Entailment, neutrality, and contradiction.
- Premise-hypothesis directionality.
- Regular expressions and temporal markers.
- Classification and confusion matrices.

The Stage 2 paper provides the literature foundation,
including Sentence-BERT, DeBERTa-v3, Dialogue NLI,
and research on long-term conversational memory.

The statistical evaluation uses McNemar's test
and bootstrap confidence intervals as specified
in the Stage 2 evaluation plan.

## 8. Methodological Deviations and Limitations

The following differences from the original
research plan must be acknowledged:

1. The 300-pair corpus is entirely synthetic.
   The Stage 2 plan proposed 150 DNLI-derived and
   150 hand-written pairs. In the final implementation,
   20 pilot pairs were written manually and 280 pairs
   were drafted with AI assistance. All 300 pairs were
   individually reviewed by the researcher.
   Dialogue NLI was used separately for RQ1.

2. The synthetic corpus was individually reviewed and approved by Gabriel Cruz Castro.
   Independent annotation agreement was not established.
   Dataset provenance and review procedures are documented
   in the held-out annotation audit.

3. The planned repeated annotation of 60 pairs
   was not completed. Cohen's kappa is not reported.

4. The selected NLP policy requires contradiction
   agreement in both NLI directions, rather than
   accepting contradiction from either direction.

5. Policy selection was performed on development.
   The held-out results were not used to modify
   the selected NLP policy.

6. The held-out pair evaluator computes NLI before
   applying the similarity decision gate. Its
   latency does not demonstrate early-exit savings.

7. RQ1 evaluates three-class DNLI conflict detection,
   not the complete four-class temporal classifier.

8. RQ1 distractors were not independently verified
   as semantically neutral to each incoming memory.
   Some may introduce additional conflicts.

9. RQ1 inference was resumed after an SSH interruption.
   Saved results were preserved, but latency measurements
   were collected across two execution sessions.

10. RQ3 compares one known candidate pair at a time.
    It does not evaluate full-history LLM retrieval.

11. NLP inference ran on CPU, while Qwen3 was observed
    using GPU acceleration on the same Apple M4 system.

12. The NLP held-out results were already available
    before the LLM configuration was prepared.

13. The experiments are limited to English and
    controlled synthetic or persona-based memories.

14. Lexical split screening cannot establish the
    complete absence of semantic or template overlap.

15. Latency and NLI comparison counts are computational
    cost proxies. Energy consumption and monetary
    costs were not directly measured.

16. Confidence intervals and significance tests should
    be interpreted within the experimental corpus.
    Multiple-comparison adjustments were not applied.

### Supplementary Post-Evaluation Findings

The selected NLP policy achieved 85% accuracy on
development and 70.5% on held-out data, indicating
a 14.5 percentage-point generalization gap.

Limited temporal rule coverage is a major observed
failure mode.

Single-memory classifiers achieved 50.00% and
52.50% accuracy, suggesting potential lexical
shortcuts in the synthetic corpus.

RQ3 compares complete system configurations rather
than isolating the capabilities of individual models.

These supplementary findings are documented in
[Final Scientific Review](final_scientific_review_2026-11-27.md).

## 9. Reproducibility

The project includes frozen datasets, model identifiers,
configuration manifests, archived predictions,
statistical evaluation scripts, and figure generation.

At the experimental milestone:

- 138 automated tests passed.
- Six statistical analyses were independently
  recalculated in an isolated temporary workspace.
- All six reproduced JSON outputs were byte-identical.
- All 13 archived result files were preserved.
- Eight scientific figures were generated.

This establishes statistical reproducibility from
archived predictions in the recorded environment.

It does not establish cross-platform reproducibility
of every model inference.

## 10. Overall Conclusion

Agent Memory Auditor demonstrates that a specialized
local NLP architecture can perform memory auditing
with low processing latency.

Semantic retrieval reduces the number of expensive
NLI comparisons, but retrieval quality decreases
as the memory history grows.

Temporal rules help distinguish memory relationships,
but their limited coverage prevents reliable
identification of legitimate updates.

The local LLM provides stronger temporal classification,
while the specialized NLP pipeline remains substantially
faster and achieves higher contradiction recall.

These results motivate a possible hybrid architecture
that uses lightweight NLP components for routine cases
and a local LLM for difficult temporal decisions.

Such a hybrid system remains future work and has not
been experimentally validated in this project.

## 11. Supporting Evidence

- results/heldout/metrics_v0_1.json
- results/heldout/error_diagnostics_v0_1.json
- results/heldout/rq2_comparison_v0_1.json
- results/rq1/metrics_v0_1.json
- results/rq1/statistics_v0_1.json
- results/rq3/metrics_v0_1.json
- docs/heldout_annotation_audit_2026-11-24.md
- docs/policy_freeze_v0_1.json
- docs/rq3_freeze_v0_1.json
- docs/figures/manifest_v0_1.json

## Annotation Review Clarification

The researcher's responsibilities and personal annotation review are clarified in the [annotation review clarification](annotation_review_clarification.md).
