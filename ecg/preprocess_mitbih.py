"""Preprocess MIT-BIH Arrhythmia ECG signals into beat-centric datasets.

This script pairs signal CSV files with their matching annotation text files,
extracts fixed-width beat windows around each annotated sample, and saves the
stacked segments (X) alongside their rhythm labels (y) as NumPy arrays.

Example
-------
python preprocess_mitbih.py --data-dir ./mitbih --output-dir ./processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IGNORED_ANNOTATION_TYPES = {"/", "~", "Q", "?"}


def find_sample_column(columns: Iterable[str]) -> Optional[str]:
    """Return the name of the sample index column if present."""
    for col in columns:
        lowered = col.lower()
        if "sample" in lowered and ("#" in lowered or "sample" == lowered.strip()):
            return col
    return None


def find_type_column(columns: Iterable[str]) -> Optional[str]:
    """Return the column containing the beat label."""
    for col in columns:
        if col.strip().lower() == "type":
            return col
    return None


def load_signal(signal_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the ECG signal and return sample numbers, signal matrix, and lead names."""
    df = pd.read_csv(signal_path)
    df.columns = [col.strip().strip("\"'") for col in df.columns]

    sample_col = find_sample_column(df.columns)
    if sample_col is None:
        raise ValueError(f"Could not find a sample column in {signal_path}")

    lead_cols = [col for col in df.columns if col != sample_col]
    if not lead_cols:
        raise ValueError(f"No lead columns detected in {signal_path}")

    sample_numbers = pd.to_numeric(df[sample_col], errors="coerce")
    if sample_numbers.isna().any():
        raise ValueError(f"Sample column in {signal_path} contains non-numeric entries.")

    signals = df[lead_cols].apply(pd.to_numeric, errors="coerce")
    if signals.isna().any().any():
        raise ValueError(f"Lead columns in {signal_path} contain non-numeric entries.")

    return sample_numbers.to_numpy(dtype=np.int64), signals.to_numpy(dtype=np.float32), lead_cols


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load annotations, ensuring the presence of sample index and type columns."""
    candidate_args = [
        dict(sep=r"\s+", engine="python", header=0),
        dict(sep=r"\s+", engine="python", header=None),
        dict(sep="\t", engine="python", header=0),
        dict(sep="\t", engine="python", header=None),
    ]

    annotations: Optional[pd.DataFrame] = None
    for kwargs in candidate_args:
        try:
            df = pd.read_csv(annotation_path, **kwargs)
            df.columns = [str(col).strip().strip("\"'") for col in df.columns]
            if find_sample_column(df.columns) and find_type_column(df.columns):
                annotations = df
                break
        except Exception:
            continue

    if annotations is None:
        raise ValueError(f"Unable to parse annotation file: {annotation_path}")

    sample_col = find_sample_column(annotations.columns)
    type_col = find_type_column(annotations.columns)
    assert sample_col is not None
    assert type_col is not None

    annotations = annotations[[sample_col, type_col]].rename(
        columns={sample_col: "sample", type_col: "type"}
    )
    annotations["sample"] = pd.to_numeric(annotations["sample"], errors="coerce")
    annotations = annotations.dropna(subset=["sample", "type"])
    annotations["sample"] = annotations["sample"].astype(np.int64)
    annotations["type"] = annotations["type"].astype(str).str.strip()

    annotations = annotations[~annotations["type"].isin(IGNORED_ANNOTATION_TYPES)]
    annotations = annotations.reset_index(drop=True)
    return annotations


def extract_record_segments(
    signal_path: Path,
    annotation_path: Path,
    window_radius: int,
) -> Tuple[List[np.ndarray], List[str], Sequence[str]]:
    """Extract beat windows for a single record."""
    sample_numbers, signal_values, leads = load_signal(signal_path)
    annotations = load_annotations(annotation_path)

    sample_to_index = {int(sample): idx for idx, sample in enumerate(sample_numbers)}

    segments: List[np.ndarray] = []
    labels: List[str] = []

    total_len = signal_values.shape[0]
    window_size = 2 * window_radius + 1

    for _, row in annotations.iterrows():
        sample = int(row["sample"])
        label = row["type"]

        center_idx = sample_to_index.get(sample)
        if center_idx is None:
            continue

        start = center_idx - window_radius
        end = center_idx + window_radius + 1
        if start < 0 or end > total_len:
            continue

        window = signal_values[start:end]
        # Ensure we always output the expected window shape.
        if window.shape[0] == window_size:
            segments.append(window)
            labels.append(label)

    return segments, labels, leads


def _annotation_record_id(path: Path) -> Optional[str]:
    """Derive the record identifier from an annotation filename."""
    name = path.stem
    if name.endswith("-annotations"):
        return name[: -len("-annotations")]
    if name.endswith("annotations"):
        return name[: -len("annotations")]
    return None


def discover_record_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Match signal CSV files with their corresponding annotation files."""
    csv_files = sorted(data_dir.glob("*.csv"))
    annotation_files = list(data_dir.glob("*annotations.txt"))

    annotation_lookup = {}
    for path in annotation_files:
        record_id = _annotation_record_id(path)
        if record_id:
            annotation_lookup[record_id] = path

    pairs: List[Tuple[Path, Path]] = []
    for csv_path in csv_files:
        record_id = csv_path.stem
        annotation_path = annotation_lookup.get(record_id)
        if annotation_path is None:
            continue
        pairs.append((csv_path, annotation_path))

    return pairs


def process_directory(
    data_dir: Path,
    window_radius: int,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str]]:
    """Process all record pairs in a directory and return stacked arrays."""
    record_pairs = discover_record_pairs(data_dir)
    if not record_pairs:
        raise FileNotFoundError(
            f"No matching CSV/annotation pairs found in {data_dir.resolve()}."
        )

    all_segments: List[np.ndarray] = []
    all_labels: List[str] = []
    all_leads: Optional[Sequence[str]] = None

    for signal_path, annotation_path in record_pairs:
        segments, labels, leads = extract_record_segments(signal_path, annotation_path, window_radius)
        if not segments:
            continue
        if all_leads is None:
            all_leads = leads
        all_segments.extend(segments)
        all_labels.extend(labels)

    if not all_segments:
        raise RuntimeError(
            "Found record pairs but no valid segments. Check window size or data integrity."
        )

    X = np.stack(all_segments, axis=0)
    y = np.array(all_labels, dtype=object)

    return X, y, all_leads or []


def save_outputs(X: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    """Persist datasets to disk as NumPy arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/omarelsisi/Downloads/archive/mitbih_database"),
        help=(
            "Directory containing paired MIT-BIH CSV and annotation TXT files. "
            "Defaults to the provided mitbih_database archive."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store X.npy and y.npy (defaults to the current working directory).",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=150,
        help="Half window size in samples on each side of the annotation index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.expanduser().resolve()
    output_dir: Path = (
        args.output_dir.expanduser().resolve() if args.output_dir else Path.cwd()
    )

    X, y, leads = process_directory(data_dir, args.window_radius)
    save_outputs(X, y, output_dir)

    window_len = 2 * args.window_radius + 1
    print(f"Saved {X.shape[0]} segments with window length {window_len} and leads: {list(leads)}.")
    print(f"Feature array: {output_dir / 'X.npy'}")
    print(f"Label array:   {output_dir / 'y.npy'}")


if __name__ == "__main__":
    main()
