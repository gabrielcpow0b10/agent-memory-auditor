"""Interactive blind annotation of held-out memory pairs."""

import argparse
import csv
from collections import Counter
from pathlib import Path

PATH = Path("datasets/heldout/annotation_review_v0_1.csv")

LABELS = {
    "c": "consistent",
    "n": "new",
    "u": "updated",
    "x": "contradictory",
}

MARKERS = {
    "e": "explicit",
    "i": "implicit",
    "n": "none",
}


def load():
    with PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)

    assert fields is not None
    assert len(rows) == 200

    assert len({r["review_id"] for r in rows}) == 200

    return rows, fields


def save(rows, fields):
    temporary = PATH.with_suffix(".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(PATH)


def report(rows):
    counts = Counter(r["review_status"] for r in rows)

    print("\n=== HELD-OUT REVIEW STATUS ===")
    print("Approved:", counts["approved"])
    print("Flagged:", counts["flagged"])
    print("Pending:", counts["pending"])
    print("Total:", len(rows))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flagged",
        action="store_true",
        help="Revisit flagged annotations",
    )

    args = parser.parse_args()

    rows, fields = load()

    target = "flagged" if args.flagged else "pending"

    try:
        for row in rows:
            if row["review_status"] != target:
                continue

            print("\n" + "=" * 65)
            print("REVIEW:", row["review_id"])
            print("OLD:", row["old_memory"])
            print("NEW:", row["new_memory"])
            print("OLD TIME:", row["old_timestamp"])
            print("NEW TIME:", row["new_timestamp"])

            while True:
                print("\nLABEL:")
                print("c = consistent")
                print("n = new")
                print("u = updated")
                print("x = contradictory")
                print("f = flag")
                print("s = skip")
                print("q = quit")

                choice = input("Select: ").strip().lower()

                if choice == "q":
                    return

                if choice == "s":
                    break

                if choice == "f":
                    note = input("Reason for ambiguity: ").strip()

                    if not note:
                        print("A reason is required.")
                        continue

                    row["reviewed_label"] = ""
                    row["reviewed_marker"] = ""
                    row["review_status"] = "flagged"
                    row["review_notes"] = note

                    save(rows, fields)
                    break

                if choice not in LABELS:
                    print("Invalid label.")
                    continue

                print("\nTEMPORAL MARKER:")
                print("e = explicit")
                print("i = implicit")
                print("n = none")
                print("q = quit")

                marker = input("Select marker: ").strip().lower()

                if marker == "q":
                    return

                if marker not in MARKERS:
                    print("Invalid marker.")
                    continue

                note = input("Review note (optional): ").strip()

                row["reviewed_label"] = LABELS[choice]
                row["reviewed_marker"] = MARKERS[marker]
                row["review_status"] = "approved"
                row["review_notes"] = note

                save(rows, fields)
                break

            counts = Counter(r["review_status"] for r in rows)

            print(
                "PROGRESS:",
                counts["approved"],
                "/ 200 approved",
            )

    except (KeyboardInterrupt, EOFError):
        print("\nReview interrupted safely.")

    finally:
        report(rows)
        print("Saved:", PATH)


if __name__ == "__main__":
    main()
