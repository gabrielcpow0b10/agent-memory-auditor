"""Interactive review of synthetic development annotations."""

import csv
from collections import Counter
from pathlib import Path

PATH = Path("datasets/development/annotation_review_v0_1.csv")

LABELS = {"consistent", "new", "updated", "contradictory"}
MARKERS = {"explicit", "implicit", "none"}

PRIORITY = (
    "X005",
    "C024",
    "U007",
    "U018",
    "U025",
    "N010",
    "N011",
)


def save(rows, fields):
    temp = PATH.with_suffix(".csv.tmp")

    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(PATH)


def main():
    with PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)

    assert fields is not None
    assert len(rows) == 100

    priority_rows = [
        row for pair_id in PRIORITY for row in rows if row["pair_id"] == pair_id
    ]

    other_rows = [row for row in rows if row["pair_id"] not in PRIORITY]

    for row in priority_rows + other_rows:
        if row["review_status"] == "approved":
            continue

        print("\n" + "=" * 55)
        print("PAIR:", row["pair_id"])
        print("OLD:", row["old_memory"])
        print("NEW:", row["new_memory"])
        print("PROPOSED:", row["proposed_label"])
        print("MARKER:", row["proposed_marker"])
        print("RATIONALE:", row["rationale"])

        try:
            choice = (
                input("\nEnter=approve | e=edit | f=flag | s=skip | q=quit: ")
                .strip()
                .lower()
            )
        except EOFError:
            break

        if choice == "q":
            break

        if choice == "s":
            continue

        if choice == "":
            row["reviewed_label"] = row["proposed_label"]
            row["reviewed_marker"] = row["proposed_marker"]
            row["review_status"] = "approved"
            row["review_notes"] = "Manual confirmation"

        elif choice == "e":
            label = input("Correct label: ").strip().lower()
            marker = input("Correct marker: ").strip().lower()
            note = input("Reason for correction: ").strip()

            if label not in LABELS or marker not in MARKERS or not note:
                print("Invalid correction; nothing changed.")
                continue

            row["reviewed_label"] = label
            row["reviewed_marker"] = marker
            row["review_status"] = "approved"
            row["review_notes"] = note

        elif choice == "f":
            note = input("Why is this ambiguous? ").strip()

            if not note:
                print("A reason is required.")
                continue

            row["review_status"] = "flagged"
            row["reviewed_label"] = ""
            row["reviewed_marker"] = ""
            row["review_notes"] = note

        else:
            print("Unknown command; nothing changed.")
            continue

        save(rows, fields)

        counts = Counter(r["review_status"] for r in rows)
        print("PROGRESS:", dict(counts))

    counts = Counter(r["review_status"] for r in rows)

    print("\n=== ANNOTATION REVIEW STATUS ===")
    print("Approved:", counts["approved"])
    print("Flagged:", counts["flagged"])
    print("Pending:", counts["pending"])
    print("Saved:", PATH)


if __name__ == "__main__":
    main()
