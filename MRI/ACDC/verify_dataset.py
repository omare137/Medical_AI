#!/usr/bin/env python3
"""
Quick sanity check for the processed ACDC dataset.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dataset import ACDCDataset


def get_processed_dir(base_dir: Path) -> Path:
    external = Path("/Volumes/Crucial X6/medical_ai_extra/processed")
    if external.exists():
        return external
    return base_dir / "database" / "processed"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = get_processed_dir(base_dir)
    meta_csv = processed_dir / "meta.csv"
    splits_json = processed_dir / "splits.json"

    def describe_split(name: str) -> ACDCDataset:
        ds = ACDCDataset(meta_csv, splits_json=splits_json, subset=name)
        counter = Counter(int(ds[i][1]) for i in range(len(ds)))
        print(f"{name.title()} split: {len(ds)} samples -> {dict(counter)}")
        return ds

    train_ds = describe_split("train")
    val_ds = describe_split("val")
    test_ds = describe_split("test")

    if len(train_ds) > 0:
        (image, metadata), label = train_ds[0]
        print("\nSample from train split:")
        print(f"Image shape: {tuple(image.shape)}")
        print(f"Metadata: {metadata.tolist()}")
        print(f"Label: {int(label)}")


if __name__ == "__main__":
    main()

