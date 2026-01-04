#!/usr/bin/env python3
"""
Build advanced radiomics dataset with augmentation and feature expansion.

This script:
1. Loads base radiomics features from train_features.csv / test_features.csv
2. Adds shape descriptors, dynamic features, ratio features, metadata integration
3. Computes wavelet and filtered texture features from volumes
4. Optionally generates augmented samples
5. Saves enhanced datasets to Crucial X6

Outputs:
    - train_features_enhanced.csv
    - test_features_enhanced.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates, rotate, zoom

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("Warning: PyWavelets unavailable. Skipping wavelet features.")

try:
    from skimage import exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image unavailable. Some augmentation features disabled.")

# Configuration
BASE_DIR = Path("/Users/omarelsisi/Downloads/medical_ai/MRI/ACDC")
DATABASE_DIR = BASE_DIR / "database"
TRAINING_DIR = DATABASE_DIR / "training"
TESTING_DIR = DATABASE_DIR / "testing"

EXTERNAL_PROCESSED = Path("/Volumes/Crucial X6/medical_ai_extra/processed")
if EXTERNAL_PROCESSED.exists():
    PROCESSED_DIR = EXTERNAL_PROCESSED
else:
    PROCESSED_DIR = DATABASE_DIR / "processed"

X_IMAGES_DIR = PROCESSED_DIR / "X_images"
BASE_TRAIN_CSV = PROCESSED_DIR / "train_features.csv"
BASE_TEST_CSV = PROCESSED_DIR / "test_features.csv"
META_CSV = PROCESSED_DIR / "meta.csv"

ENHANCED_TRAIN_CSV = PROCESSED_DIR / "train_features_enhanced.csv"
ENHANCED_TEST_CSV = PROCESSED_DIR / "test_features_enhanced.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Augmentation settings - DISABLED by default (very slow on 4D volumes)
ENABLE_AUGMENTATION = False  # Set to True only if you need augmented samples (takes hours)
ENABLE_VOLUME_TEXTURE_FEATURES = False  # Set to True to compute wavelet/texture from volumes (slower)
AUGMENTATION_FRACTION = 0.3  # Fraction of patients to augment (if enabled)
AUGMENTATIONS_PER_PATIENT = 2  # Number of augmented samples per patient


def load_volume(patient_id: str) -> np.ndarray:
    """Load volume from .npy or fallback to .nii.gz"""
    npy_path = X_IMAGES_DIR / f"{patient_id}.npy"
    if npy_path.exists():
        return np.load(npy_path)
    # Fallback to original NIfTI
    for base_dir in [TRAINING_DIR, TESTING_DIR]:
        nii_path = base_dir / patient_id / f"{patient_id}_4d.nii.gz"
        if nii_path.exists():
            return nib.load(str(nii_path)).get_fdata()
    raise FileNotFoundError(f"Could not locate volume for {patient_id}")


def elastic_deform(volume: np.ndarray, alpha: float = 5, sigma: float = 8) -> np.ndarray:
    """Apply elastic deformation to volume"""
    random_state = np.random.RandomState(RANDOM_SEED)
    shape = volume.shape
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

    z, y, x, t = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        np.arange(shape[3]),
        indexing="ij",
    )
    indices = (z + dz, y + dy, x + dx, t)
    distorted = map_coordinates(volume, indices, order=1, mode="reflect")
    return distorted


def augment_volume(volume: np.ndarray) -> List[np.ndarray]:
    """Generate augmented versions of volume"""
    augmented = []

    # Rotations
    angles = [5, -5, 10, -10]
    for angle in angles:
        rotated_vol = rotate(volume, angle=angle, axes=(1, 2), reshape=False, order=1)
        augmented.append(rotated_vol)

    # Zoom/scale
    zoom_factors = [0.95, 1.05]
    for factor in zoom_factors:
        scaled = zoom(volume, (1, factor, factor, 1), order=1)
        center_c = scaled.shape[1] // 2
        center_r = scaled.shape[2] // 2
        target_shape = volume.shape
        start_c = center_c - target_shape[1] // 2
        start_r = center_r - target_shape[2] // 2
        cropped = scaled[:, start_c : start_c + target_shape[1], start_r : start_r + target_shape[2], :]
        augmented.append(cropped)

    # Elastic deformation
    augmented.append(elastic_deform(volume))

    # Noise
    noise_sigma = 0.01
    noise = np.random.normal(0, noise_sigma, size=volume.shape)
    augmented.append(volume + noise)

    # Intensity shift
    shifted = volume + np.random.uniform(-0.05, 0.05)
    augmented.append(shifted)

    # Histogram equalization
    if HAS_SKIMAGE:
        equalized = exposure.equalize_hist(volume, nbins=128)
        augmented.append(equalized)

    # Validate and clip
    validated = []
    for aug in augmented:
        aug = np.clip(aug, np.percentile(volume, 0.5), np.percentile(volume, 99.5))
        validated.append(aug.astype(np.float32))
    return validated


def body_surface_area(height_cm: float, weight_kg: float) -> float:
    """Calculate BSA using Mosteller formula"""
    if np.isnan(height_cm) or np.isnan(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return np.nan
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)


def compute_shape_descriptors(row: pd.Series) -> Dict[str, float]:
    """Compute shape-based features"""
    lv_ed = row.get("ed_lv_volume", np.nan)
    lv_es = row.get("es_lv_volume", np.nan)
    rv_ed = row.get("ed_rv_volume", np.nan)
    rv_es = row.get("es_rv_volume", np.nan)

    def sphericity(volume: float) -> float:
        if not np.isfinite(volume) or volume <= 0:
            return np.nan
        # Simplified sphericity approximation
        return float((36 * np.pi * (volume ** 2)) ** (1 / 3) / ((6 * volume) ** (2 / 3)))

    features = {
        "lv_sphericity_ed": sphericity(lv_ed),
        "lv_sphericity_es": sphericity(lv_es),
        "rv_sphericity_ed": sphericity(rv_ed),
        "rv_sphericity_es": sphericity(rv_es),
    }

    # Chamber elongation
    ed_thick_max = row.get("ed_wall_thickness_max", np.nan)
    ed_thick_min = row.get("ed_wall_thickness_min", np.nan)
    es_thick_max = row.get("es_wall_thickness_max", np.nan)
    es_thick_min = row.get("es_wall_thickness_min", np.nan)

    features["lv_chamber_elongation"] = ed_thick_max / (ed_thick_min + 1e-6) if np.isfinite(ed_thick_max) and np.isfinite(ed_thick_min) else np.nan
    features["rv_chamber_elongation"] = es_thick_max / (es_thick_min + 1e-6) if np.isfinite(es_thick_max) and np.isfinite(es_thick_min) else np.nan

    # Eccentricity
    features["lv_eccentricity"] = (lv_ed - lv_es) / (lv_ed + 1e-6) if np.isfinite(lv_ed) else np.nan
    features["rv_eccentricity"] = (rv_ed - rv_es) / (rv_ed + 1e-6) if np.isfinite(rv_ed) else np.nan

    return features


def compute_dynamic_features(row: pd.Series) -> Dict[str, float]:
    """Compute temporal/dynamic features"""
    lv_ed = row.get("ed_lv_volume", np.nan)
    lv_es = row.get("es_lv_volume", np.nan)
    rv_ed = row.get("ed_rv_volume", np.nan)
    rv_es = row.get("es_rv_volume", np.nan)
    ed_frame = row.get("ed_frame", 1)
    es_frame = row.get("es_frame", 1)

    features = {}
    if np.isfinite(lv_ed) and np.isfinite(lv_es):
        features["lv_stroke_volume"] = lv_ed - lv_es
        features["lv_ef"] = (lv_ed - lv_es) / (lv_ed + 1e-6)
        frame_diff = abs(ed_frame - es_frame) + 1e-6
        features["lv_contraction_rate"] = features["lv_stroke_volume"] / frame_diff

    if np.isfinite(rv_ed) and np.isfinite(rv_es):
        features["rv_stroke_volume"] = rv_ed - rv_es
        features["rv_ef"] = (rv_ed - rv_es) / (rv_ed + 1e-6)

    # Temporal variance of wall thickness
    thickness_keys = [k for k in row.index if "wall_thickness" in k.lower()]
    vals = [row[k] for k in thickness_keys if np.isfinite(row.get(k, np.nan))]
    if vals:
        features["wall_thickness_temporal_std"] = float(np.std(vals))
    else:
        features["wall_thickness_temporal_std"] = np.nan

    return features


def compute_ratio_features(row: pd.Series, meta_row: pd.Series) -> Dict[str, float]:
    """Compute ratio-based physiological features"""
    lv = row.get("ed_lv_volume", np.nan)
    rv = row.get("ed_rv_volume", np.nan)
    thickness_mean = row.get("ed_wall_thickness_mean", np.nan)
    height = meta_row.get("height", np.nan)
    weight = meta_row.get("weight", np.nan)
    bsa = body_surface_area(height, weight)
    stroke = row.get("lv_stroke_volume", np.nan)

    # Chamber radius approximation
    chamber_radius = ((lv / (4 / 3 * np.pi)) ** (1 / 3)) if np.isfinite(lv) and lv > 0 else np.nan

    return {
        "rv_lv_ratio": rv / (lv + 1e-6) if np.isfinite(rv) and np.isfinite(lv) else np.nan,
        "thickness_radius_ratio": thickness_mean / (chamber_radius + 1e-6)
        if np.isfinite(thickness_mean) and np.isfinite(chamber_radius)
        else np.nan,
        "sv_per_bsa": stroke / (bsa + 1e-6) if np.isfinite(stroke) and np.isfinite(bsa) else np.nan,
    }


def wavelet_texture_features(volume: np.ndarray) -> Dict[str, float]:
    """Compute wavelet-based texture features"""
    if not HAS_PYWT:
        return {}
    try:
        frame = volume[..., volume.shape[-1] // 2]
        coeffs = pywt.wavedecn(frame, wavelet="db2", level=1)  # Reduced level to avoid warnings
        energies = []
        for level in coeffs[1:]:
            if isinstance(level, dict):
                for arr in level.values():
                    energies.append(float(np.sqrt(np.sum(arr ** 2))))
            else:
                energies.append(float(np.sqrt(np.sum(level ** 2))))
        return {
            "wavelet_energy_mean": float(np.mean(energies)) if energies else np.nan,
            "wavelet_energy_std": float(np.std(energies)) if energies else np.nan,
            "wavelet_energy_max": float(np.max(energies)) if energies else np.nan,
        }
    except Exception:
        return {"wavelet_energy_mean": np.nan, "wavelet_energy_std": np.nan, "wavelet_energy_max": np.nan}


def filtered_texture_features(volume: np.ndarray) -> Dict[str, float]:
    """Compute filtered texture features"""
    frame = volume[..., volume.shape[-1] // 2]
    gauss = ndimage.gaussian_filter(frame, sigma=1)
    laplace = ndimage.laplace(frame)
    highpass = frame - gauss
    return {
        "gaussian_mean": float(np.mean(gauss)),
        "laplace_std": float(np.std(laplace)),
        "highpass_energy": float(np.sum(highpass ** 2)),
    }


def aggregate_dicts(dicts: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    """Aggregate features from multiple augmented volumes"""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    aggregated = {}
    for key in keys:
        values = [d.get(key, np.nan) for d in dicts if np.isfinite(d.get(key, np.nan))]
        if values:
            aggregated[f"{prefix}_{key}_mean"] = float(np.mean(values))
            aggregated[f"{prefix}_{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{prefix}_{key}_mean"] = np.nan
            aggregated[f"{prefix}_{key}_std"] = np.nan
    return aggregated


def compute_volume_enhancements(patient_id: str, max_aug: int = 3) -> Dict[str, float]:
    """Compute texture features from original and augmented volumes"""
    if not ENABLE_VOLUME_TEXTURE_FEATURES:
        return {}  # Skip expensive volume loading/processing
    
    try:
        volume = load_volume(patient_id)
    except FileNotFoundError:
        return {}

    original_features = {**wavelet_texture_features(volume), **filtered_texture_features(volume)}

    if ENABLE_AUGMENTATION and max_aug > 0:
        # Only do augmentation if explicitly enabled (very slow!)
        augmented_volumes = augment_volume(volume)[:max_aug]
        aug_features = []
        for aug in augmented_volumes:
            feats = {**wavelet_texture_features(aug), **filtered_texture_features(aug)}
            aug_features.append(feats)
        augmented_summary = aggregate_dicts(aug_features, prefix="aug")
        return {**{f"orig_{k}": v for k, v in original_features.items()}, **augmented_summary}
    else:
        return {f"orig_{k}": v for k, v in original_features.items()}


def build_enhanced_row(row: pd.Series, meta_lookup: pd.DataFrame) -> Dict[str, float]:
    """Build enhanced feature row for a patient"""
    patient_id = row.get("patient_id", row.get("PatientID", ""))
    if not patient_id:
        raise ValueError("Patient ID missing from row")

    # Get metadata
    key = patient_id.lower()
    if key not in meta_lookup.index:
        meta_row = pd.Series({}, dtype=float)
    else:
        meta_row = meta_lookup.loc[key]

    # Start with base features
    features = row.to_dict()

    # Add shape descriptors
    features.update(compute_shape_descriptors(row))

    # Add dynamic features
    dyn = compute_dynamic_features(row)
    features.update(dyn)

    # Add ratio features
    features.update(compute_ratio_features({**row, **dyn}, meta_row))

    # Add metadata
    for key in ["height", "weight", "ed", "es"]:
        features[f"meta_{key}"] = meta_row.get(key, np.nan)
    features["meta_bsa"] = body_surface_area(meta_row.get("height", np.nan), meta_row.get("weight", np.nan))

    # Add volume-based texture features
    volume_features = compute_volume_enhancements(patient_id, max_aug=3)
    features.update(volume_features)

    features["is_augmented"] = 0
    return features


def generate_augmented_samples(df: pd.DataFrame, meta_lookup: pd.DataFrame) -> pd.DataFrame:
    """Generate augmented samples"""
    if not ENABLE_AUGMENTATION:
        print("  Skipping augmented samples (ENABLE_AUGMENTATION=False)")
        return pd.DataFrame()

    sample_ids = df["patient_id"].sample(frac=AUGMENTATION_FRACTION, random_state=RANDOM_SEED)
    augmented_records = []

    for patient_id in sample_ids:
        base_row = df[df["patient_id"] == patient_id].iloc[0]
        try:
            volume = load_volume(patient_id)
        except FileNotFoundError:
            continue

        augmented_volumes = augment_volume(volume)[:AUGMENTATIONS_PER_PATIENT]
        for idx, aug in enumerate(augmented_volumes):
            feats = {**wavelet_texture_features(aug), **filtered_texture_features(aug)}
            record = base_row.copy()
            for key, value in feats.items():
                record[f"augmented_{key}"] = value
            record["is_augmented"] = 1
            record["augmentation_id"] = idx
            augmented_records.append(record)

    if augmented_records:
        return pd.DataFrame(augmented_records)
    return pd.DataFrame(columns=df.columns)


def process_split(df: pd.DataFrame, meta_lookup: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Process a split (train or test) to create enhanced features"""
    print(f"\nProcessing {split_name} split ({len(df)} patients)...")
    print(f"  Augmentation: {'ENABLED' if ENABLE_AUGMENTATION else 'DISABLED'}")
    print(f"  Volume texture features: {'ENABLED' if ENABLE_VOLUME_TEXTURE_FEATURES else 'DISABLED'}")

    enhanced_rows = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        patient_id = row.get("patient_id", row.get("PatientID", "unknown"))
        try:
            enhanced = build_enhanced_row(row, meta_lookup)
            enhanced_rows.append(enhanced)
            print(f"  [{idx}/{len(df)}] ✅ {patient_id}")
        except Exception as e:
            print(f"  [{idx}/{len(df)}] ❌ {patient_id}: {e}")
            continue

    enhanced_df = pd.DataFrame(enhanced_rows)

    # Add augmented samples for training split only
    if split_name == "train" and ENABLE_AUGMENTATION:
        print(f"  Generating augmented samples...")
        augmented = generate_augmented_samples(enhanced_df, meta_lookup)
        if not augmented.empty:
            enhanced_df = pd.concat([enhanced_df, augmented], ignore_index=True)
            print(f"  Added {len(augmented)} augmented samples")

    print(f"  Final {split_name} shape: {enhanced_df.shape}")
    return enhanced_df


def main():
    """Main entry point"""
    print("=" * 60)
    print("Building Advanced Radiomics Dataset")
    print("=" * 60)
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Augmentation: {'ENABLED (slow!)' if ENABLE_AUGMENTATION else 'DISABLED (fast)'}")
    print(f"Volume texture features: {'ENABLED (slower)' if ENABLE_VOLUME_TEXTURE_FEATURES else 'DISABLED (fast)'}")
    if not ENABLE_VOLUME_TEXTURE_FEATURES:
        print("  ⚠️  Note: Shape, dynamic, and ratio features will still be computed from CSV data")

    # Load base datasets
    print("\nLoading base datasets...")
    train_df = pd.read_csv(BASE_TRAIN_CSV)
    test_df = pd.read_csv(BASE_TEST_CSV)
    meta_df = pd.read_csv(META_CSV)

    # Normalize patient_id column
    if "PatientID" in train_df.columns:
        train_df["patient_id"] = train_df["PatientID"]
    if "PatientID" in test_df.columns:
        test_df["patient_id"] = test_df["PatientID"]

    # Setup metadata lookup
    meta_lookup = meta_df.copy()
    meta_lookup.columns = [c.lower() for c in meta_lookup.columns]
    meta_lookup = meta_lookup.set_index("patient_id")

    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Meta: {meta_df.shape}")

    # Process splits
    enhanced_train = process_split(train_df, meta_lookup, "train")
    enhanced_test = process_split(test_df, meta_lookup, "test")

    # Save enhanced datasets
    print("\nSaving enhanced datasets...")
    enhanced_train.to_csv(ENHANCED_TRAIN_CSV, index=False)
    enhanced_test.to_csv(ENHANCED_TEST_CSV, index=False)

    print(f"\n✅ Saved enhanced datasets:")
    print(f"  Train: {ENHANCED_TRAIN_CSV}")
    print(f"  Test:  {ENHANCED_TEST_CSV}")
    print(f"\nTrain shape: {enhanced_train.shape}")
    print(f"Test shape:  {enhanced_test.shape}")


if __name__ == "__main__":
    main()

