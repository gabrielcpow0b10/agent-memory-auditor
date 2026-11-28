"""Post-evaluation diagnostic for lexical shortcuts."""

import csv
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


def load_csv(filename):
    with Path(filename).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    dev = load_csv(
        "datasets/development/annotation_review_v0_1.csv"
    )
    test = load_csv(
        "datasets/heldout/heldout_inputs_v0_1.csv"
    )
    gold_rows = load_csv(
        "datasets/heldout/heldout_gold_v0_1.csv"
    )

    gold = {
        row["review_id"]: row["label"]
        for row in gold_rows
    }

    assert len(dev) == 100
    assert len(test) == len(gold_rows) == len(gold) == 200
    assert len({row["review_id"] for row in test}) == 200
    assert set(gold) == {row["review_id"] for row in test}

    y_train = [row["reviewed_label"] for row in dev]
    y_test = [gold[row["review_id"]] for row in test]

    assert len(Counter(y_train)) == 4
    assert set(Counter(y_train).values()) == {25}
    assert len(Counter(y_test)) == 4
    assert set(Counter(y_test).values()) == {50}

    print("=== SINGLE-MEMORY DIAGNOSTIC ===")
    print("Development:", len(dev))
    print("Held-out:", len(test))
    print("Random reference: 25.00%")

    for field in ("old_memory", "new_memory"):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=4304,
            ),
        )

        x_train = [row[field] for row in dev]
        x_test = [row[field] for row in test]

        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        correct = sum(
            a == b for a, b in zip(y_test, predicted, strict=True)
        )

        print(f"\n=== {field.upper()} ONLY ===")
        print(f"Correct: {correct}/200")
        print(f"Accuracy: {accuracy_score(y_test, predicted):.2%}")
        print(
            f"Macro F1: "
            f"{f1_score(y_test, predicted, average='macro'):.4f}"
        )

    print("\nEXPLORATORY DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
