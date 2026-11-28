# Agent Memory Auditor — Final Scientific Review

Author: Gabriel Cruz Castro
Institution: Florida International University
Course: CAI 4304 — Natural Language Processing
Date: November 27, 2026

## 1. Development and Held-Out Performance

The selected NLP policy achieved 85% accuracy on
development data and 70.5% on the held-out corpus.

This represents a decrease of 14.5 percentage points.

Of the 59 held-out classification errors, 45 involved
legitimate memory updates.

The temporal module detected only 8 of 50 actual
updates as candidates for replacement.

These findings indicate limited temporal rule coverage
and suggest sensitivity to differences in vocabulary
between development and held-out examples.

The available evidence does not establish that lexical
overfitting alone explains the entire performance gap.

## 2. Single-Memory Diagnostic

An exploratory diagnostic was conducted after the
primary held-out evaluation.

Two classifiers were trained using TF-IDF unigrams
and bigrams with logistic regression.

Each classifier received only one memory from each
pair. Both were trained on the 100 development
examples and evaluated on the 200 held-out examples.

| Input | Accuracy | Macro F1 |
|---|---:|---:|
| Random reference | 25.00% | — |
| Old memory only | 50.00% | 0.4947 |
| New memory only | 52.50% | 0.5300 |

The results indicate that individual memories contain
lexical information associated with the target labels.

These patterns may provide classification shortcuts
that do not require complete relational reasoning.

The extent to which the original systems exploit
these patterns has not been established.

Reproduction command:

    python -m scripts.diagnose_single_memory

## 3. RQ3 Comparison Limitations

The local LLM receives natural-language definitions
of the four memory labels and temporal decision
instructions.

The specialized NLP pipeline implements its decisions
through fixed numerical thresholds, bidirectional NLI,
and a limited set of temporal expressions.

Consequently, RQ3 compares complete system
configurations rather than isolating model capability.

The performance difference cannot be attributed
exclusively to the underlying models.

## 4. Experimental Integrity

This review is supplementary and exploratory.

No original datasets, predictions, frozen prompts,
classification policies, or primary experimental
results were modified.

The original RQ1, RQ2, and RQ3 results remain
the primary findings of this project.
