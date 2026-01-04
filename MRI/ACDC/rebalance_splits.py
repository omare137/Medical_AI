#!/usr/bin/env python3
"""
Utility to rebuild class-balanced train/val splits for the ACDC dataset.

Requirements:
    - processed/meta.csv already exists
    - existing splits.json provides the current test IDs (will be preserved)
    - exactly 20 labelled patients per class (0-4) within patient001-100

Output:
    - processed/splits.json overwritten with balanced train/val/test lists
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


VAL_PER_CLASS = 4
TRAIN_PER_CLASS = 16
RANDOM_SEED = 42


def get_processed_dir() -> Path:
    external = Path("/Volumes/Crucial X6/medical_ai_extra/processed")
    if external.exists():
        return external
    base_dir = Path(__file__).resolve().parent / "database" / "processed"
    return base_dir


def patient_num(patient_id: str) -> int:
    return int(patient_id.replace("patient", ""))


def main() -> None:
    processed_dir = get_processed_dir()
    meta_csv = processed_dir / "meta.csv"
    splits_json = processed_dir / "splits.json"

    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing meta.csv at {meta_csv}")
    if not splits_json.exists():
        raise FileNotFoundError(f"Missing splits.json at {splits_json}")

    df = pd.read_csv(meta_csv)
    with splits_json.open("r", encoding="utf-8") as f:
        existing_splits = json.load(f)

    rng = random.Random(RANDOM_SEED)

    label_groups: dict[int, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        pid = row["patient_id"]
        label = int(row["label"])
        if patient_num(pid) <= 100:
            label_groups[label].append(pid)

    train_ids: list[str] = []
    val_ids: list[str] = []

    for label in sorted(label_groups.keys()):
        ids = label_groups[label]
        if len(ids) < VAL_PER_CLASS + TRAIN_PER_CLASS:
            raise ValueError(
                f"Not enough samples for label {label}: expected at least "
                f"{VAL_PER_CLASS + TRAIN_PER_CLASS}, found {len(ids)}"
            )
        rng.shuffle(ids)
        val_ids.extend(ids[:VAL_PER_CLASS])
        train_ids.extend(ids[VAL_PER_CLASS : VAL_PER_CLASS + TRAIN_PER_CLASS])

    # Preserve existing test IDs (those beyond patient100)
    test_ids = existing_splits.get("test")
    if test_ids is None:
        test_ids = [
            pid for pid in df["patient_id"] if patient_num(pid) > 100
        ]

    balanced_splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(balanced_splits, f, indent=2)

    print("Balanced splits saved to", splits_json)
    print("Train size:", len(balanced_splits["train"]))
    print("Val size  :", len(balanced_splits["val"]))
    print("Test size :", len(balanced_splits["test"]))

    def label_counts(ids: list[str]) -> Counter:
        subset = df[df["patient_id"].isin(ids)]
        return Counter(subset["label"].astype(int))

    print("Label counts:")
    print("  Train:", label_counts(balanced_splits["train"]))
    print("  Val  :", label_counts(balanced_splits["val"]))
    print("  Test :", label_counts(balanced_splits["test"]))


if __name__ == "__main__":
    main()

