#!/usr/bin/env python3
"""
Utility to build a processed dataset from the ACDC training split.

Outputs:
    - processed/X_images/patientXXX.npy : normalized MRI volumes
    - processed/meta.csv                : metadata table
    - processed/splits.json             : static train/val/test ids
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import nibabel as nib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent / "database"
TRAIN_DIR = BASE_DIR / "training"
TEST_DIR = BASE_DIR / "testing"

# Route processed artefacts to the external storage (falls back to local if missing)
EXTRA_STORAGE_ROOT = Path("/Volumes/Crucial X6/medical_ai_extra")
if EXTRA_STORAGE_ROOT.exists():
    PROCESSED_DIR = EXTRA_STORAGE_ROOT / "processed"
else:
    PROCESSED_DIR = BASE_DIR / "processed"

X_IMAGES_DIR = PROCESSED_DIR / "X_images"
META_CSV_PATH = PROCESSED_DIR / "meta.csv"
SPLITS_JSON_PATH = PROCESSED_DIR / "splits.json"

LABEL_MAP: Dict[str, int] = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4,
}


@dataclass
class PatientRecord:
    patient_id: str
    filepath: str
    height: float | None
    weight: float | None
    ed: int | None
    es: int | None
    label: int

    def as_row(self) -> Dict[str, str | float | int | None]:
        return {
            "patient_id": self.patient_id,
            "filepath": self.filepath,
            "height": self.height,
            "weight": self.weight,
            "ED": self.ed,
            "ES": self.es,
            "label": self.label,
        }


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    mean = float(volume.mean())
    std = float(volume.std())
    if std == 0:
        return volume - mean
    return (volume - mean) / std


def parse_info_cfg(info_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with info_path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            data[key.strip().lower()] = value.strip()
    return data


def safe_cast(value: str | None, cast_type):
    if value is None:
        return None
    try:
        return cast_type(value)
    except (TypeError, ValueError):
        return None


def iter_patient_dirs(directory: Path) -> Iterable[Path]:
    pattern = re.compile(r"^patient\d{3}$")
    for path in sorted(directory.iterdir()):
        if path.is_dir() and pattern.match(path.name):
            yield path


def iter_all_patient_dirs() -> Iterable[Path]:
    for directory in (TRAIN_DIR, TEST_DIR):
        if directory.exists():
            yield from iter_patient_dirs(directory)


def build_processed_dataset() -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)
    X_IMAGES_DIR.mkdir(exist_ok=True)

    records: List[PatientRecord] = []

    for patient_dir in iter_all_patient_dirs():
        patient_id = patient_dir.name
        nii_path = patient_dir / f"{patient_id}_4d.nii.gz"
        if not nii_path.is_file():
            raise FileNotFoundError(f"Missing MRI volume for {patient_id}: {nii_path}")

        info_cfg = patient_dir / "Info.cfg"
        if not info_cfg.is_file():
            raise FileNotFoundError(f"Missing Info.cfg for {patient_id}")

        print(f"Processing {patient_id}")

        volume = nib.load(str(nii_path)).get_fdata()
        volume = normalize_volume(volume)

        npy_path = X_IMAGES_DIR / f"{patient_id}.npy"
        np.save(npy_path, volume)

        info = parse_info_cfg(info_cfg)
        group = info.get("group")
        if group is None or group not in LABEL_MAP:
            raise ValueError(f"Unknown or missing Group for {patient_id}: {group}")

        record = PatientRecord(
            patient_id=patient_id,
            filepath=str(npy_path.resolve()),
            height=safe_cast(info.get("height"), float),
            weight=safe_cast(info.get("weight"), float),
            ed=safe_cast(info.get("ed"), int),
            es=safe_cast(info.get("es"), int),
            label=LABEL_MAP[group],
        )
        records.append(record)

    write_meta_csv(records)
    write_splits(records)
    print(f"Saved {len(records)} patient entries.")


def write_meta_csv(records: Iterable[PatientRecord]) -> None:
    fieldnames = ["patient_id", "filepath", "height", "weight", "ED", "ES", "label"]
    with META_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def write_splits(records: Iterable[PatientRecord]) -> None:
    def patient_num(patient_id: str) -> int:
        return int(patient_id.replace("patient", ""))

    train_ids = []
    val_ids = []
    test_ids: List[str] = []

    for record in records:
        num = patient_num(record.patient_id)
        if 1 <= num <= 80:
            train_ids.append(record.patient_id)
        elif 81 <= num <= 100:
            val_ids.append(record.patient_id)
        else:
            test_ids.append(record.patient_id)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with SPLITS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


if __name__ == "__main__":
    build_processed_dataset()

