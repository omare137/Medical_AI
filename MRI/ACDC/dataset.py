"""
PyTorch Dataset + DataLoader helpers for the processed ACDC dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MetaTransform = Optional[Callable[[torch.Tensor], torch.Tensor]]
ImageTransform = Optional[Callable[[torch.Tensor], torch.Tensor]]


class ACDCDataset(Dataset):
    """
    Returns ((image_tensor, metadata_tensor), label_tensor) tuples.
    """

    def __init__(
        self,
        meta_csv: str | Path,
        splits_json: str | Path | None = None,
        subset: str | None = None,
        image_transform: ImageTransform = None,
        metadata_transform: MetaTransform = None,
        label_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        metadata_fields: Sequence[str] = ("height", "weight", "ED", "ES"),
    ) -> None:
        self.meta_csv = Path(meta_csv)
        self.splits_json = Path(splits_json) if splits_json else None
        self.subset = subset
        self.image_transform = image_transform
        self.metadata_transform = metadata_transform
        self.label_transform = label_transform
        self.metadata_fields = metadata_fields

        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict]:
        entries: List[Dict] = []

        with self.meta_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)

        if self.splits_json and self.subset:
            with self.splits_json.open("r", encoding="utf-8") as f:
                splits = json.load(f)
            allowed_ids = set(splits.get(self.subset, []))
            entries = [row for row in entries if row["patient_id"] in allowed_ids]

        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        entry = self.entries[idx]
        image = self._load_image(entry["filepath"])
        metadata = self._load_metadata(entry)
        label = torch.tensor(int(entry["label"]), dtype=torch.long)

        if self.image_transform:
            image = self.image_transform(image)
        if self.metadata_transform:
            metadata = self.metadata_transform(metadata)
        if self.label_transform:
            label = self.label_transform(label)

        return (image, metadata), label

    @staticmethod
    def _load_image(path: str) -> torch.Tensor:
        array = np.load(path)
        tensor = torch.from_numpy(array.astype(np.float32))
        return tensor

    def _load_metadata(self, entry: Dict) -> torch.Tensor:
        values: List[float] = []
        for key in self.metadata_fields:
            raw_value = entry.get(key)
            if raw_value in (None, "", "None"):
                values.append(float("nan"))
            else:
                values.append(float(raw_value))

        tensor = torch.tensor(values, dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

