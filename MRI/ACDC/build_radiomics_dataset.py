#!/usr/bin/env python3
"""
Build radiomics-style features for the ACDC MRI dataset.

Outputs (under processed/):
    - train_features.csv
    - test_features.csv
    - features_npy/<patient_id>.npy   (per-patient vector for quick loading)

The script extracts ED/ES frames, computes volumetric, intensity, texture, and
wall-thickness features, attaches metadata from Info.cfg, and preserves the
original train/test directory split.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

try:
    from skimage.feature import greycomatrix, greycoprops
    HAS_GLCM = True
except ImportError:
    try:
        from skimage.feature.texture import greycomatrix, greycoprops
        HAS_GLCM = True
    except ImportError:
        HAS_GLCM = False
        print("Warning: GLCM features unavailable. Using simplified texture features.")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path("/Users/omarelsisi/Downloads/medical_ai/MRI/ACDC")
DATABASE_DIR = BASE_DIR / "database"
TRAIN_DIR = DATABASE_DIR / "training"
TEST_DIR = DATABASE_DIR / "testing"

EXTERNAL_PROCESSED = Path("/Volumes/Crucial X6/medical_ai_extra/processed")
if EXTERNAL_PROCESSED.exists():
    PROCESSED_DIR = EXTERNAL_PROCESSED
else:
    PROCESSED_DIR = DATABASE_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)
FEATURES_NPY_DIR = PROCESSED_DIR / "features_npy"
FEATURES_NPY_DIR.mkdir(exist_ok=True)

TRAIN_CSV = PROCESSED_DIR / "train_features.csv"
TEST_CSV = PROCESSED_DIR / "test_features.csv"

LABEL_MAP = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}

# Ground-truth label indices in ACDC masks:
# 1 -> Right ventricle (RV) cavity
# 2 -> Myocardium (LV)
# 3 -> Left ventricle (LV) cavity
RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3

GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 32
GLRLM_LEVELS = 16


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def list_patient_dirs(split_dir: Path) -> List[Path]:
    return sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("patient")])


def read_info_cfg(path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            info[key.strip().lower()] = value.strip()
    return info


def load_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    voxel_size = zooms[:3]
    return data, voxel_size


def load_mask(patient_dir: Path, frame_id: int) -> np.ndarray | None:
    mask_path = patient_dir / f"{patient_dir.name}_frame{frame_id:02d}_gt.nii.gz"
    if not mask_path.exists():
        return None
    return nib.load(str(mask_path)).get_fdata()


def extract_frame(volume: np.ndarray, frame_idx: int) -> np.ndarray:
    # Frames are either the first or last axis depending on file orientation.
    if volume.shape[0] <= 50:  # typical number of frames
        return volume[frame_idx]
    return volume[..., frame_idx]


def voxel_volume(voxel_size: Tuple[float, float, float]) -> float:
    return float(np.prod(voxel_size))


def aggregate_stats(values: Iterable[float], prefix: str) -> Dict[str, float]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
    }


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    """Quantize image to specified levels using simple min-max normalization."""
    clipped = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
    # Simple min-max normalization
    img_min, img_max = clipped.min(), clipped.max()
    if img_max > img_min:
        normalized = (clipped - img_min) / (img_max - img_min) * (levels - 1)
    else:
        normalized = np.zeros_like(clipped)
    return normalized.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Feature computations
# --------------------------------------------------------------------------- #

def compute_volumes(mask: np.ndarray, voxel_vol: float) -> Dict[str, float]:
    lv = float(np.sum(mask == LV_LABEL) * voxel_vol)
    rv = float(np.sum(mask == RV_LABEL) * voxel_vol)
    myo = float(np.sum(mask == MYO_LABEL) * voxel_vol)
    return {"lv_volume": lv, "rv_volume": rv, "myocardium_volume": myo}


def compute_intensity_features(image: np.ndarray) -> Dict[str, float]:
    flattened = image.flatten()
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size == 0:
        return {key: np.nan for key in ["mean", "median", "std", "min", "max", "skew", "kurtosis", "entropy"]}

    hist, _ = np.histogram(flattened, bins=64, density=True)
    entropy = -np.sum(hist * np.log(hist + 1e-12))

    return {
        "mean": float(np.mean(flattened)),
        "median": float(np.median(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "skew": float(stats.skew(flattened)),
        "kurtosis": float(stats.kurtosis(flattened)),
        "entropy": float(entropy),
    }


def compute_glcm_features(image: np.ndarray) -> Dict[str, float]:
    """Compute GLCM texture features with fallback if skimage unavailable."""
    if not HAS_GLCM:
        # Simplified texture features using gradient-based measures
        grad_x = np.gradient(image.astype(float), axis=1)
        grad_y = np.gradient(image.astype(float), axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return {
            "contrast": float(np.std(gradient_magnitude)),
            "correlation": float(np.corrcoef(grad_x.flatten(), grad_y.flatten())[0, 1]) if len(grad_x.flatten()) > 1 else 0.0,
            "homogeneity": float(1.0 / (1.0 + np.mean(gradient_magnitude))),
            "energy": float(np.mean(image**2)),
            "ASM": float(np.mean(image**2)),
        }
    
    quantized = quantize_image(image, GLCM_LEVELS)
    matrices = greycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        symmetric=True,
        normed=True,
    )
    features = {}
    for prop in ["contrast", "correlation", "homogeneity", "energy", "ASM"]:
        try:
            values = greycoprops(matrices, prop)
            features[prop] = float(np.mean(values))
        except ValueError:
            features[prop] = np.nan
    return features


def compute_glrlm_features(image: np.ndarray) -> Dict[str, float]:
    quantized = quantize_image(image, GLRLM_LEVELS)
    runs: Dict[int, Counter] = defaultdict(Counter)  # level -> Counter(length -> count)

    for row in quantized:
        current_val = row[0]
        run_len = 1
        for val in row[1:]:
            if val == current_val:
                run_len += 1
            else:
                runs[current_val][run_len] += 1
                current_val = val
                run_len = 1
        runs[current_val][run_len] += 1

    total_runs = sum(sum(lengths.values()) for lengths in runs.values())
    if total_runs == 0:
        return {name: np.nan for name in ["sre", "lre", "gln", "rln"]}

    def compute_metric(func) -> float:
        accum = 0.0
        for level, lengths in runs.items():
            for run_len, count in lengths.items():
                accum += func(level, run_len) * count
        return accum / total_runs

    sre = compute_metric(lambda _, r: 1.0 / (r ** 2))
    lre = compute_metric(lambda _, r: r ** 2)
    gln = compute_metric(lambda level, _: 1.0 * level ** 2)
    rln = compute_metric(lambda _, r: r ** 2)

    return {"sre": float(sre), "lre": float(lre), "gln": float(gln), "rln": float(rln)}


def compute_wall_thickness(mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> Dict[str, float]:
    slice_thickness = voxel_size[2]
    thickness_values: List[float] = []

    lv_cavity = mask == LV_LABEL
    myocardium = mask == MYO_LABEL

    for slice_idx in range(mask.shape[2]):
        lv_slice = lv_cavity[:, :, slice_idx]
        myo_slice = myocardium[:, :, slice_idx]
        if not (lv_slice.any() and myo_slice.any()):
            continue
        distance_map = ndimage.distance_transform_edt(~lv_slice) * voxel_size[0]
        thickness_slice = distance_map[myo_slice]
        if thickness_slice.size > 0:
            thickness_values.append(float(np.mean(thickness_slice)))

    if not thickness_values:
        return {"wall_thickness_mean": np.nan, "wall_thickness_min": np.nan, "wall_thickness_max": np.nan}

    return {
        "wall_thickness_mean": float(np.mean(thickness_values)),
        "wall_thickness_min": float(np.min(thickness_values)),
        "wall_thickness_max": float(np.max(thickness_values)),
    }


def aggregate_slice_features(volume: np.ndarray) -> Dict[str, float]:
    glcm_metrics = defaultdict(list)
    glrlm_metrics = defaultdict(list)

    for slice_idx in range(volume.shape[2]):
        slice_img = volume[:, :, slice_idx]
        if not np.any(np.isfinite(slice_img)):
            continue
        glcm = compute_glcm_features(slice_img)
        glrlm = compute_glrlm_features(slice_img)
        for key, value in glcm.items():
            glcm_metrics[key].append(value)
        for key, value in glrlm.items():
            glrlm_metrics[key].append(value)

    aggregated: Dict[str, float] = {}
    for key, values in glcm_metrics.items():
        aggregated.update(aggregate_stats(values, f"glcm_{key}"))
    for key, values in glrlm_metrics.items():
        aggregated.update(aggregate_stats(values, f"glrlm_{key}"))
    return aggregated


# --------------------------------------------------------------------------- #
# Patient processing
# --------------------------------------------------------------------------- #

def compute_patient_features(patient_dir: Path) -> Dict[str, float]:
    info = read_info_cfg(patient_dir / "Info.cfg")
    volume_path = patient_dir / f"{patient_dir.name}_4d.nii.gz"
    if not volume_path.exists():
        raise FileNotFoundError(f"Missing 4D volume for {patient_dir.name}")

    volume, voxel_size = load_volume(volume_path)
    voxel_vol = voxel_volume(voxel_size)

    height = float(info.get("height", "nan"))
    weight = float(info.get("weight", "nan"))
    ed_frame = int(info.get("ed", 1))
    es_frame = int(info.get("es", 1))
    group = info.get("group")
    if group not in LABEL_MAP:
        raise ValueError(f"Unknown group {group} for {patient_dir.name}")

    frames = {
        "ed": {
            "frame_idx": ed_frame - 1,
            "mask": load_mask(patient_dir, ed_frame),
        },
        "es": {
            "frame_idx": es_frame - 1,
            "mask": load_mask(patient_dir, es_frame),
        },
    }

    patient_features = {
        "patient_id": patient_dir.name,
        "height": height,
        "weight": weight,
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "label": LABEL_MAP[group],
    }

    for phase_name, phase in frames.items():
        frame_idx = phase["frame_idx"]
        image = extract_frame(volume, frame_idx)

        # Intensity stats
        intensity = compute_intensity_features(image)
        patient_features.update({f"{phase_name}_{k}": v for k, v in intensity.items()})

        # Texture stats (aggregated over slices)
        texture = aggregate_slice_features(image)
        patient_features.update({f"{phase_name}_{k}": v for k, v in texture.items()})

        mask = phase["mask"]
        if mask is None:
            continue

        volumes = compute_volumes(mask, voxel_vol)
        patient_features.update({f"{phase_name}_{k}": v for k, v in volumes.items()})

        thickness = compute_wall_thickness(mask, voxel_size)
        patient_features.update({f"{phase_name}_{k}": v for k, v in thickness.items()})

    if all(frames[name]["mask"] is not None for name in ["ed", "es"]):
        lv_ed = patient_features.get("ed_lv_volume", np.nan)
        lv_es = patient_features.get("es_lv_volume", np.nan)
        if np.isfinite(lv_ed) and np.isfinite(lv_es) and lv_ed > 0:
            stroke_volume = lv_ed - lv_es
            ef = stroke_volume / lv_ed
        else:
            stroke_volume = np.nan
            ef = np.nan
        patient_features["stroke_volume_lv"] = stroke_volume
        patient_features["ef_lv"] = ef

    return patient_features


def process_split(split_dir: Path) -> List[Dict[str, float]]:
    features: List[Dict[str, float]] = []
    for patient_dir in list_patient_dirs(split_dir):
        try:
            print(f"Processing {patient_dir.name} ...")
            feat = compute_patient_features(patient_dir)
            features.append(feat)
            np.save(FEATURES_NPY_DIR / f"{patient_dir.name}.npy", feat, allow_pickle=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Skipping {patient_dir.name}: {exc}")
    return features


# --------------------------------------------------------------------------- #
# Plotting helpers (optional)
# --------------------------------------------------------------------------- #

def plot_slice_with_mask(image: np.ndarray, mask: np.ndarray | None, slice_idx: int, title: str) -> None:
    import matplotlib.pyplot as plt

    slice_img = image[:, :, slice_idx]
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_img, cmap="gray")
    if mask is not None:
        slice_mask = mask[:, :, slice_idx]
        plt.contour(slice_mask == LV_LABEL, colors="r", linewidths=0.5)
        plt.contour(slice_mask == RV_LABEL, colors="c", linewidths=0.5)
        plt.contour(slice_mask == MYO_LABEL, colors="y", linewidths=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def save_features(features: List[Dict[str, float]], path: Path) -> None:
    if not features:
        print(f"No features to save for {path}")
        return
    df = pd.DataFrame(features)
    df.sort_values("patient_id", inplace=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows with {len(df.columns) - 2} features to {path}")


def main() -> None:
    print("Starting radiomics feature extraction...")
    train_features = process_split(TRAIN_DIR)
    test_features = process_split(TEST_DIR)

    save_features(train_features, TRAIN_CSV)
    save_features(test_features, TEST_CSV)

    print("Final counts:")
    print(f"  Train patients: {len(train_features)}")
    if train_features:
        print(f"  Train label distribution: {Counter(f['label'] for f in train_features)}")
    print(f"  Test patients : {len(test_features)}")
    if test_features:
        print(f"  Test label distribution : {Counter(f['label'] for f in test_features)}")


if __name__ == "__main__":
    main()

