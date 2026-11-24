"""Structural validation for held-out Batch 02."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

BATCH01 = DATA / "heldout/batch_01_draft.csv"
BATCH02 = DATA / "heldout/batch_02_draft.csv"

LABELS = {
    "consistent": "HC",
    "new": "HN",
    "updated": "HU",
    "contradictory": "HX",
}


def load(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def pair_key(row):
    return tuple(
        sorted(
            " ".join(text.casefold().split())
            for text in (
                row["old_memory"],
                row["new_memory"],
            )
        )
    )


def test_batch02_balance_and_ids():
    rows = load(BATCH02)

    assert len(rows) == 40

    assert Counter(r["label"] for r in rows) == {label: 10 for label in LABELS}

    expected_ids = {
        f"{prefix}{i:03d}" for prefix in LABELS.values() for i in range(11, 21)
    }

    assert {r["pair_id"] for r in rows} == expected_ids


def test_batch02_metadata():
    rows = load(BATCH02)

    for row in rows:
        assert row["source"] == "synthetic_ai_draft"
        assert row["split"] == "held_out_candidate"
        assert row["rationale"].strip()

        old = datetime.fromisoformat(row["old_timestamp"])
        new = datetime.fromisoformat(row["new_timestamp"])

        assert old.utcoffset() is not None
        assert new.utcoffset() is not None
        assert old < new

        if row["label"] != "updated":
            assert row["marker_type"] == "none"

    markers = Counter(r["marker_type"] for r in rows if r["label"] == "updated")

    assert markers == {
        "explicit": 5,
        "implicit": 5,
    }


def test_heldout_development_separation():
    paths = [
        DATA / "pilot/annotation_pilot_v0_1.csv",
        *sorted((DATA / "development").glob("batch_*_draft.csv")),
    ]

    development = []

    for path in paths:
        development.extend(load(path))

    heldout = load(BATCH01) + load(BATCH02)
    combined = development + heldout

    assert len(development) == 100
    assert len(heldout) == 80
    assert len(combined) == 180

    assert len({r["pair_id"] for r in combined}) == 180

    assert len({r["persona_id"] for r in combined}) == 180

    assert len({pair_key(r) for r in combined}) == 180


def test_frozen_policy_integrity():
    path = ROOT / "docs/policy_freeze_v0_1.json"

    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["selected_policy"] == "C_consensus"
    assert manifest["held_out_used_for_selection"] is False

    for name, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

        assert actual == expected, name
