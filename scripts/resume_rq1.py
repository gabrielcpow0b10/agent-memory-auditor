"""Resume an interrupted RQ1 benchmark without repeating completed cases."""

import csv
import os
from tempfile import NamedTemporaryFile

from scripts.evaluate_rq1 import (
    DESIGN,
    METHODS,
    OUTPUT,
    POSITIONS,
    SIZES,
    build_history,
    evaluate_history,
    load_models,
    validate_design,
)
from scripts.predict_heldout import sha256, verify_environment

DESIGN_HASH = "a117f5bdc42ff6de8f2f19034c5e9e3de284df35ba11b2194527d4daa2750efc"


def main():
    partial = OUTPUT.with_suffix(".partial.csv")

    assert partial.exists()
    assert not OUTPUT.exists()
    assert sha256(DESIGN) == DESIGN_HASH

    cases = validate_design()
    revisions = verify_environment()

    original = partial.read_bytes()

    assert original.endswith(b"\n")

    with partial.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        fields = reader.fieldnames
        rows = list(reader)

    expected = [
        (
            case["case_id"],
            str(size),
            position,
            case["label"],
            method,
        )
        for case in cases
        for size in SIZES
        for position in POSITIONS
        for method in METHODS
    ]

    saved = [
        (
            row["case_id"],
            row["history_size"],
            row["position"],
            row["expected"],
            row["method"],
        )
        for row in rows
    ]

    assert saved == expected[: len(rows)]
    assert len(rows) % 36 == 0

    for row in rows:
        assert None not in row
        assert all(value is not None for value in row.values())
        assert row["predicted"] in {"positive", "negative", "neutral"}

        count = {
            "recency": 5,
            "exhaustive": int(row["history_size"]),
            "semantic": 5,
            "oracle": 1,
        }[row["method"]]

        assert int(row["nli_pair_comparisons"]) == count
        assert int(row["nli_directional_inputs"]) == 2 * count

    completed = len(rows) // 36

    assert 0 < completed < len(cases)

    with NamedTemporaryFile(
        prefix="rq1_recovery_",
        suffix=".csv",
        dir=DESIGN.parent,
        delete=False,
    ) as backup:
        backup.write(original)
        print("Backup:", backup.name)

    print("Recovery validation: PASS", flush=True)
    print("Recovered cases:", completed, flush=True)
    print("Recovered evaluations:", len(rows), flush=True)

    encoder, model = load_models(revisions)

    print("Frozen models loaded: PASS", flush=True)

    total = len(rows)

    assert partial.read_bytes() == original

    with partial.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )

        for index, case in enumerate(cases[completed:], start=completed + 1):
            batch = []

            for size in SIZES:
                for position in POSITIONS:
                    history, incoming = build_history(case, size, position)

                    for method in METHODS:
                        row = {
                            "case_id": case["case_id"],
                            "history_size": size,
                            "position": position,
                            "expected": case["label"],
                        }

                        row.update(
                            evaluate_history(
                                history,
                                incoming,
                                encoder,
                                model,
                                method,
                            )
                        )

                        batch.append(row)

            assert len(batch) == 36

            writer.writerows(batch)
            file.flush()
            os.fsync(file.fileno())

            total += len(batch)

            print(
                f"Cases: {index}/60 | Evaluations: {total}/2160",
                flush=True,
            )

    assert total == 2160

    assert partial.read_bytes().startswith(original)

    assert not OUTPUT.exists()

    partial.rename(OUTPUT)

    print("Saved:", OUTPUT)
    print("SHA-256:", sha256(OUTPUT))
    print("RQ1 RECOVERY AND BENCHMARK: PASS")


if __name__ == "__main__":
    main()
