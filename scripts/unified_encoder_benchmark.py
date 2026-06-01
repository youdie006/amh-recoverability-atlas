#!/usr/bin/env python3
"""Unified cached-array frozen-encoder recoverability benchmark.

This script evaluates only existing cached embeddings. Wang ``.npy`` packs use
the index_wang.csv row order; Wang ``.npz`` packs are aligned by their ``ids``
field to index_wang.csv ``Image`` before probing.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - depends on installed sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "results/diagnostics/cache"
WANG_DIR = REPO / "data/external/mendeley_wang_2026"
WANG_INDEX_PATH = WANG_DIR / "embeddings/index_wang.csv"
WANG_AMH_PATH = WANG_DIR / "amh_recovered_wang.csv"
FUID_DIR = REPO / "data/external/ovarian_us/fuid/extracted"
OUT_PATH = REPO / "results/diagnostics/unified_encoder_benchmark.json"

SEED = 20260601
N_SPLITS = 5
PCA_COMPONENTS = 32
N_BOOTSTRAP = 1000
EPS = 1e-12


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    target_key: str
    target: str
    path: Path
    array_key: str | None = None
    ids_key: str | None = None
    align_to_label_ids: bool = False
    path_note: str | None = None


def embryo_encoder_specs() -> list[EncoderSpec]:
    emb = WANG_DIR / "embeddings"
    return [
        EncoderSpec("DINOv2-base", "embryo_amh", "AMH tertile", emb / "dinov2_z_wang.npy"),
        EncoderSpec("CLIP-L", "embryo_amh", "AMH tertile", emb / "clipl_z_wang.npy"),
        EncoderSpec("FEMI", "embryo_amh", "AMH tertile", emb / "femi_z_wang.npy"),
        EncoderSpec(
            "DINOv2-large",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2g_dinov2_large_encoder/dinov2_large_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "DINOv2-giant",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2_dinov2_giant_encoder/dinov2_giant_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "DINOv3-ViTB",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2A5_dinov3_vitb/dinov3_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "DINOv3-strengthened",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2n_dinov3_strengthened/dinov3_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "V-JEPA2",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2c_vjepa2_encoder/vjepa2_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "SigLIP2",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2x_siglip2_encoder/siglip2_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
        EncoderSpec(
            "RAD-DINO",
            "embryo_amh",
            "AMH tertile",
            REPO / "results/synth/track2y_raddino_encoder/raddino_features.npz",
            array_key="wang",
            ids_key="wang_index",
            align_to_label_ids=True,
        ),
    ]


def array_shape_for_discovery(path: Path) -> tuple[tuple[int, ...] | None, str | None]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        return None, "npz discovery candidate is not a plain array"
    return tuple(int(v) for v in loaded.shape), None


def resolve_fuid_dinov2_path(expected_n: int | None, cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    direct = cache_dir / "fuid_dinov2_cls.npy"
    audit: dict[str, Any] = {
        "requested_path": str(direct),
        "requested_np_load_verified": False,
        "requested_shape": None,
        "searched_cache_glob": str(cache_dir / "*fuid*.npy"),
        "candidates": [],
        "selected_path": None,
        "reason": None,
    }

    shape, err = array_shape_for_discovery(direct) if direct.exists() else (None, "missing")
    audit["requested_np_load_verified"] = err is None
    audit["requested_shape"] = list(shape) if shape is not None else None
    if shape is not None and (expected_n is None or shape == (int(expected_n), 768)):
        audit["selected_path"] = str(direct)
        audit["reason"] = "requested fuid_dinov2_cls.npy exists with expected shape"
        return direct, audit

    if expected_n is None:
        audit["reason"] = "requested fuid_dinov2_cls.npy absent and expected_n was not provided for fallback search"
        return direct, audit

    candidates: list[Path] = []
    for path in sorted(cache_dir.glob("*fuid*.npy")):
        if path.name == "fuid_usfmae_cls.npy":
            continue
        shape, err = array_shape_for_discovery(path)
        accepted = shape == (int(expected_n), 768)
        audit["candidates"].append(
            {
                "path": str(path),
                "np_load_verified": err is None,
                "shape": list(shape) if shape is not None else None,
                "error": err,
                "accepted_as_dinov2_candidate": bool(accepted),
            }
        )
        if accepted:
            candidates.append(path)

    exact_fallback = cache_dir / "fuid_Z.npy"
    if exact_fallback in candidates:
        audit["selected_path"] = str(exact_fallback)
        audit["reason"] = "selected existing 301x768 fuid_Z.npy fallback after requested DINOv2 path was absent"
        return exact_fallback, audit
    if len(candidates) == 1:
        audit["selected_path"] = str(candidates[0])
        audit["reason"] = "selected the only non-USF-MAE 301x768 fuid array found in cache"
        return candidates[0], audit

    audit["reason"] = (
        "no unambiguous non-USF-MAE 301x768 fuid array found"
        if not candidates
        else "multiple non-USF-MAE 301x768 fuid arrays found; DINOv2 fallback not selected"
    )
    return direct, audit


def fuid_encoder_specs(
    expected_n: int | None = None,
    cache_dir: Path = CACHE_DIR,
) -> tuple[list[EncoderSpec], dict[str, Any]]:
    dino_path, dino_audit = resolve_fuid_dinov2_path(expected_n, cache_dir)
    specs = [
        EncoderSpec("DINOv2", "ovary_fuid_phenotype", "FUID phenotype", dino_path),
        EncoderSpec("USF-MAE", "ovary_fuid_phenotype", "FUID phenotype", cache_dir / "fuid_usfmae_cls.npy"),
    ]
    return specs, {"dinov2": dino_audit}


def finite_float(x: Any) -> float | None:
    if x is None:
        return None
    val = float(x)
    if not np.isfinite(val):
        return None
    return val


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return finite_float(obj)
    return obj


def entropy_bits(y: np.ndarray) -> float:
    vals = np.asarray(y)
    _, counts = np.unique(vals, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def quantile_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    arr = np.asarray(values, dtype=float)
    edges = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(arr, edges, right=True).astype(int)
    return labels, [float(edges[0]), float(edges[1])]


def class_counts(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def align_proba_to_labels(proba: np.ndarray, model_classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    out = np.full((p.shape[0], len(labels)), EPS, dtype=float)
    positions = {label: i for i, label in enumerate(np.asarray(labels).tolist())}
    for col, cls in enumerate(np.asarray(model_classes).tolist()):
        if cls in positions:
            out[:, positions[cls]] = np.maximum(p[:, col], EPS)
    out /= out.sum(axis=1, keepdims=True)
    return out


def prior_proba(y_train: np.ndarray, n_rows: int, labels: np.ndarray) -> np.ndarray:
    counts = np.array([np.sum(y_train == label) for label in labels], dtype=float)
    prior = (counts + 0.5) / (counts.sum() + 0.5 * len(labels))
    return np.repeat(prior[None, :], int(n_rows), axis=0)


def vinfo_metrics(
    y: np.ndarray,
    baseline_proba: np.ndarray,
    model_proba: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | None]:
    base_loss = float(log_loss(y, baseline_proba, labels=labels))
    model_loss = float(log_loss(y, model_proba, labels=labels))
    r_bits = (base_loss - model_loss) / math.log(2.0)
    h_bits = entropy_bits(y)
    return {
        "target_entropy_bits": h_bits,
        "baseline_log_loss_nats": base_loss,
        "model_log_loss_nats": model_loss,
        "R_bits": r_bits,
        "R_over_H": r_bits / h_bits if h_bits > 0 else None,
        "mdl_codelength_ratio": model_loss / base_loss if base_loss > 0 else None,
    }


def choose_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    y = np.asarray(y)
    groups = np.asarray(groups).astype(str)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(int(n_splits), n_groups, len(y)))
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())

    if min_count >= n_splits:
        if StratifiedGroupKFold is not None and n_groups >= n_splits:
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
                return "StratifiedGroupKFold", list(splitter.split(np.zeros(len(y)), y, groups=groups))
            except ValueError:
                pass
        if n_groups == len(y):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            return "StratifiedKFold", list(splitter.split(np.zeros(len(y)), y))

    splitter = GroupKFold(n_splits=n_splits)
    return "GroupKFold", list(splitter.split(np.zeros(len(y)), y, groups=groups))


def transform_fold(
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"embedding array must be 2D, got shape={arr.shape}")
    x_train = arr[train_idx].copy()
    x_test = arr[test_idx].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    n_comp = min(int(pca_components), x_train.shape[1], x_train.shape[0] - 1)
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(seed))
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        return x_train, x_test, int(n_comp)
    return x_train, x_test, None


def fit_oof_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    pca_components: int = PCA_COMPONENTS,
    seed: int = SEED,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    splitter_name: str | None = None,
) -> dict[str, Any]:
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups).astype(str)
    labels = np.unique(y_arr)
    if len(labels) < 2:
        raise ValueError("classification probe needs at least two target classes")
    if len(X) != len(y_arr) or len(groups_arr) != len(y_arr):
        raise ValueError("X, y, and groups must have the same row count")

    if splits is None:
        splitter_name, splits = choose_splits(y_arr, groups_arr, n_splits=n_splits, seed=seed)
    elif splitter_name is None:
        splitter_name = "provided"

    model_proba = np.full((len(y_arr), len(labels)), np.nan, dtype=float)
    baseline_proba = np.full_like(model_proba, np.nan)
    fold_details: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        baseline_proba[test_idx] = prior_proba(y_arr[train_idx], len(test_idx), labels)
        train_labels = np.unique(y_arr[train_idx])
        if len(train_labels) < 2:
            model_proba[test_idx] = baseline_proba[test_idx]
            fold_details.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "fallback": "single_train_class",
                }
            )
            continue

        x_train, x_test, used_pca = transform_fold(
            X, train_idx, test_idx, pca_components=pca_components, seed=seed + 101 * fold
        )
        model_kwargs: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 3000,
            "solver": "lbfgs",
            "random_state": seed + 1000 + fold,
        }
        if "multi_class" in inspect.signature(LogisticRegression).parameters:
            model_kwargs["multi_class"] = "multinomial"
        model = LogisticRegression(**model_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train, y_arr[train_idx])
        model_proba[test_idx] = align_proba_to_labels(model.predict_proba(x_test), model.classes_, labels)
        fold_details.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_classes": int(len(train_labels)),
                "pca_components": used_pca,
            }
        )

    if np.any(~np.isfinite(model_proba)) or np.any(~np.isfinite(baseline_proba)):
        raise RuntimeError("out-of-fold predictions contain missing values")

    return {
        "labels": labels,
        "baseline_proba": baseline_proba,
        "model_proba": model_proba,
        "metric": vinfo_metrics(y_arr, baseline_proba, model_proba, labels),
        "splitter": splitter_name,
        "folds": fold_details,
        "splits": splits,
    }


def cluster_bootstrap_ci_r_over_h(
    y: np.ndarray,
    groups: np.ndarray,
    baseline_proba: np.ndarray,
    model_proba: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[float | None]:
    if n_boot <= 0:
        return [None, None]
    rng = np.random.default_rng(int(seed))
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups).astype(str)
    unique_groups = np.array(sorted(np.unique(groups_arr)))
    group_indices = {group: np.flatnonzero(groups_arr == group) for group in unique_groups}
    values: list[float] = []
    for _ in range(int(n_boot)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[group] for group in sampled_groups])
        h_bits = entropy_bits(y_arr[idx])
        if h_bits <= 0:
            continue
        base_loss = float(log_loss(y_arr[idx], baseline_proba[idx], labels=labels))
        model_loss = float(log_loss(y_arr[idx], model_proba[idx], labels=labels))
        values.append(((base_loss - model_loss) / math.log(2.0)) / h_bits)
    if not values:
        return [None, None]
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return [float(lo), float(hi)]


def align_array_to_label_ids(
    arr: np.ndarray,
    ids: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
    ids_arr = np.asarray(ids).astype(str)
    label_arr = np.asarray(label_ids).astype(str)
    alignment: dict[str, Any] = {
        "method": "ids_to_label_index",
        "ids_key": "ids",
        "n_ids": int(len(ids_arr)),
        "n_label_ids": int(len(label_arr)),
        "ids_match_set": False,
        "reordered": None,
        "missing_label_ids": [],
        "extra_pack_ids": [],
    }
    if len(ids_arr) != arr.shape[0]:
        return None, alignment, f"ids length {len(ids_arr)} != feature rows {arr.shape[0]}"
    if len(np.unique(ids_arr)) != len(ids_arr):
        return None, alignment, "pack ids are not unique"
    if len(np.unique(label_arr)) != len(label_arr):
        return None, alignment, "label ids are not unique"

    source_positions = {image_id: i for i, image_id in enumerate(ids_arr.tolist())}
    label_set = set(label_arr.tolist())
    missing = [image_id for image_id in label_arr.tolist() if image_id not in source_positions]
    extra = [image_id for image_id in ids_arr.tolist() if image_id not in label_set]
    alignment["missing_label_ids"] = missing[:10]
    alignment["extra_pack_ids"] = extra[:10]
    alignment["n_missing_label_ids"] = int(len(missing))
    alignment["n_extra_pack_ids"] = int(len(extra))
    if missing or extra:
        return None, alignment, "pack ids do not exactly match label Image ids"

    order = np.asarray([source_positions[image_id] for image_id in label_arr.tolist()], dtype=int)
    alignment["ids_match_set"] = True
    alignment["reordered"] = bool(not np.array_equal(order, np.arange(len(order))))
    return np.asarray(arr)[order], alignment, None


def verify_load_array(
    spec: EncoderSpec,
    expected_n: int,
    label_ids: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    row: dict[str, Any] = {
        "path": str(spec.path),
        "path_note": spec.path_note,
        "np_load_verified": False,
        "status": "excluded",
        "reason": None,
        "n": None,
        "dim": None,
        "format": None,
        "array_key": spec.array_key,
        "ids_key": spec.ids_key,
        "alignment": None,
    }
    try:
        loaded = np.load(spec.path, allow_pickle=True)
    except Exception as exc:
        row["reason"] = f"np.load failed: {type(exc).__name__}: {exc}"
        return None, row
    row["np_load_verified"] = True

    if isinstance(loaded, np.lib.npyio.NpzFile):
        row["format"] = "npz"
        row["npz_keys"] = list(loaded.files)
        if spec.array_key is None:
            loaded.close()
            row["reason"] = "npz pack requires array_key but none was configured"
            return None, row
        if spec.array_key not in loaded.files:
            loaded.close()
            row["reason"] = f"npz pack missing feature key {spec.array_key!r}"
            return None, row
        arr = loaded[spec.array_key]
        if spec.align_to_label_ids:
            if spec.ids_key is None or spec.ids_key not in loaded.files:
                loaded.close()
                row["reason"] = f"npz pack missing ids key {spec.ids_key!r}"
                return None, row
            if label_ids is None:
                loaded.close()
                row["reason"] = "label_ids are required for id-aligned npz pack"
                return None, row
            arr, alignment, err = align_array_to_label_ids(arr, loaded[spec.ids_key], np.asarray(label_ids))
            alignment["ids_key"] = spec.ids_key
            row["alignment"] = alignment
            if err is not None:
                loaded.close()
                row["reason"] = err
                return None, row
        else:
            row["alignment"] = {"method": "npz_row_order", "reordered": False}
        loaded.close()
    else:
        row["format"] = "npy"
        arr = loaded
        row["alignment"] = {
            "method": "assumed_label_index_order",
            "reordered": False,
            "ids_checked": False,
        }

    if arr.ndim != 2:
        row["reason"] = f"array is not 2D: shape={arr.shape}"
        return None, row
    row["n"] = int(arr.shape[0])
    row["dim"] = int(arr.shape[1])
    if arr.shape[0] != expected_n:
        row["reason"] = f"row count {arr.shape[0]} != expected labels {expected_n}"
        return None, row
    row["status"] = "used"
    row["reason"] = None
    return np.asarray(arr, dtype=np.float32), row


def load_wang_target() -> dict[str, Any]:
    idx = pd.read_csv(WANG_INDEX_PATH)
    amh = pd.read_csv(WANG_AMH_PATH)
    required_index = {"Image", "patient_id", "amh"}
    required_amh = {"Image", "AMH_raw"}
    missing_index = sorted(required_index - set(idx.columns))
    missing_amh = sorted(required_amh - set(amh.columns))
    if missing_index or missing_amh:
        raise RuntimeError(f"missing Wang columns: index={missing_index}, amh={missing_amh}")
    same_n = len(idx) == len(amh)
    same_image_order = same_n and idx["Image"].astype(str).equals(amh["Image"].astype(str))
    same_amh = same_n and np.allclose(idx["amh"].astype(float).to_numpy(), amh["AMH_raw"].astype(float).to_numpy())
    if not same_image_order or not same_amh:
        raise RuntimeError("Wang index_wang.csv and amh_recovered_wang.csv are not row-aligned")
    y, edges = quantile_tertiles(idx["amh"].astype(float).to_numpy())
    return {
        "y": y,
        "groups": idx["patient_id"].astype(str).to_numpy(),
        "label_ids": idx["Image"].astype(str).to_numpy(),
        "n": int(len(idx)),
        "target": "AMH tertile",
        "label_source": str(WANG_INDEX_PATH),
        "label_audit_source": str(WANG_AMH_PATH),
        "class_names": {"0": "low_AMH", "1": "mid_AMH", "2": "high_AMH"},
        "class_counts": class_counts(y),
        "tertile_edges_amh": edges,
        "n_groups": int(idx["patient_id"].nunique()),
        "alignment_checks": {
            "index_rows": int(len(idx)),
            "amh_rows": int(len(amh)),
            "same_image_order": bool(same_image_order),
            "same_amh_values": bool(same_amh),
        },
    }


def load_fuid_target() -> dict[str, Any]:
    paths: list[str] = []
    labels: list[int] = []
    class_order = [("Normal", 0), ("Dominant", 1), ("PCO", 2)]
    folder_for_class = {"Normal": "Normal", "Dominant": "Dominant_Follicle", "PCO": "PCO"}
    for class_name, label in class_order:
        folder = folder_for_class[class_name]
        for path in sorted(glob.glob(str(FUID_DIR / "**" / folder / "*"), recursive=True)):
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(path)
                labels.append(label)
    y = np.asarray(labels, dtype=int)
    return {
        "y": y,
        "groups": np.array([f"fuid_image_{i}" for i in range(len(y))], dtype=object),
        "n": int(len(y)),
        "target": "FUID phenotype",
        "label_source": str(FUID_DIR),
        "class_names": {"0": "Normal", "1": "Dominant", "2": "PCO"},
        "class_counts": class_counts(y),
        "n_groups": int(len(y)),
        "alignment_checks": {
            "label_order": "sorted recursive image paths by Normal, Dominant_Follicle, PCO",
            "n_image_paths": int(len(paths)),
        },
    }


def empty_encoder_result(spec: EncoderSpec, load_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": spec.target,
        "path": str(spec.path),
        "path_note": spec.path_note,
        "status": load_row["status"],
        "reason": load_row["reason"],
        "n": load_row["n"],
        "dim": load_row["dim"],
        "format": load_row.get("format"),
        "array_key": load_row.get("array_key"),
        "ids_key": load_row.get("ids_key"),
        "alignment": load_row.get("alignment"),
        "R/H": None,
        "bootstrap_CI": [None, None],
        "control_task_selectivity": None,
        "control_RH": None,
        "R_bits": None,
        "target_entropy_bits": None,
        "baseline_log_loss_nats": None,
        "model_log_loss_nats": None,
        "mdl_codelength_ratio": None,
        "probe": {
            "splitter": None,
            "n_splits": int(N_SPLITS),
            "model": f"StandardScaler + PCA(min({PCA_COMPONENTS}, n_features, n_train-1)) + multinomial LogisticRegression(lbfgs)",
            "baseline": "train-fold class prior with Jeffreys smoothing",
        },
        "folds": [],
        "np_load_verified": bool(load_row["np_load_verified"]),
    }


def evaluate_encoder(
    spec: EncoderSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    splitter_name: str,
    control_y: np.ndarray,
) -> dict[str, Any]:
    real = fit_oof_probe(X, y, groups, seed=seed, splits=splits, splitter_name=splitter_name)
    control = fit_oof_probe(
        X,
        control_y,
        groups,
        seed=seed + 100_000,
        splits=splits,
        splitter_name=f"{splitter_name} (same folds as real target)",
    )
    ci = cluster_bootstrap_ci_r_over_h(
        y,
        groups,
        real["baseline_proba"],
        real["model_proba"],
        real["labels"],
        n_boot=n_bootstrap,
        seed=seed + 200_000,
    )
    rh = finite_float(real["metric"]["R_over_H"])
    control_rh = finite_float(control["metric"]["R_over_H"])
    selectivity = finite_float(rh - control_rh) if rh is not None and control_rh is not None else None
    return {
        "target": spec.target,
        "path": str(spec.path),
        "path_note": spec.path_note,
        "status": "used",
        "reason": None,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "format": None,
        "array_key": spec.array_key,
        "ids_key": spec.ids_key,
        "alignment": None,
        "R/H": rh,
        "bootstrap_CI": ci,
        "control_task_selectivity": selectivity,
        "control_RH": control_rh,
        "R_bits": finite_float(real["metric"]["R_bits"]),
        "target_entropy_bits": finite_float(real["metric"]["target_entropy_bits"]),
        "baseline_log_loss_nats": finite_float(real["metric"]["baseline_log_loss_nats"]),
        "model_log_loss_nats": finite_float(real["metric"]["model_log_loss_nats"]),
        "mdl_codelength_ratio": finite_float(real["metric"]["mdl_codelength_ratio"]),
        "control_task": {
            "label_permutation": "single fixed row-level random permutation preserving class counts",
            "R/H": control_rh,
            "R_bits": finite_float(control["metric"]["R_bits"]),
            "target_entropy_bits": finite_float(control["metric"]["target_entropy_bits"]),
            "baseline_log_loss_nats": finite_float(control["metric"]["baseline_log_loss_nats"]),
            "model_log_loss_nats": finite_float(control["metric"]["model_log_loss_nats"]),
            "mdl_codelength_ratio": finite_float(control["metric"]["mdl_codelength_ratio"]),
        },
        "probe": {
            "splitter": splitter_name,
            "n_splits": int(N_SPLITS),
            "model": f"StandardScaler + PCA(min({PCA_COMPONENTS}, n_features, n_train-1)) + multinomial LogisticRegression(lbfgs)",
            "baseline": "train-fold class prior with Jeffreys smoothing",
        },
        "folds": real["folds"],
        "np_load_verified": True,
    }


def evaluate_target(
    specs: list[EncoderSpec],
    target_info: dict[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = np.asarray(target_info["y"])
    groups = np.asarray(target_info["groups"]).astype(str)
    label_ids = target_info.get("label_ids")
    splitter_name, splits = choose_splits(y, groups, n_splits=N_SPLITS, seed=seed)
    rng = np.random.default_rng(seed + 91_337)
    control_y = rng.permutation(y)
    results: dict[str, Any] = {}
    loaded_arrays: dict[str, np.ndarray] = {}
    for spec in specs:
        print(f"[{spec.target_key}] verifying {spec.name}: {spec.path}", flush=True)
        arr, load_row = verify_load_array(spec, expected_n=len(y), label_ids=label_ids)
        if arr is None:
            results[spec.name] = empty_encoder_result(spec, load_row)
            print(f"  excluded {spec.name}: {load_row['reason']}", flush=True)
            continue
        print(f"  used {spec.name}: shape={arr.shape}", flush=True)
        row = evaluate_encoder(
            spec,
            arr,
            y,
            groups,
            n_bootstrap=n_bootstrap,
            seed=seed,
            splits=splits,
            splitter_name=splitter_name,
            control_y=control_y,
        )
        row["format"] = load_row.get("format")
        row["alignment"] = load_row.get("alignment")
        row["npz_keys"] = load_row.get("npz_keys")
        results[spec.name] = row
        loaded_arrays[spec.name] = arr
    return results, loaded_arrays


def finite_feature_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"CKA expects 2D arrays, got shape={arr.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(arr), arr, med)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    x = finite_feature_matrix(X)
    y = finite_feature_matrix(Y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA arrays must have the same row count")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = float(np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro"))
    if denom <= 0:
        return float("nan")
    return float((np.linalg.norm(xy, ord="fro") ** 2) / denom)


def average_ranks(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks.tolist()


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    if not all(np.isfinite(v) for v in x + y):
        return None
    rx = np.asarray(average_ranks(x), dtype=float)
    ry = np.asarray(average_ranks(y), dtype=float)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cka_summary(wang_arrays: dict[str, np.ndarray], embryo_results: dict[str, Any]) -> dict[str, Any]:
    names = [spec.name for spec in embryo_encoder_specs() if spec.name in wang_arrays]
    if len(names) < 2:
        return {
            "encoder_names": names,
            "matrix": None,
            "recoverability_tracking": None,
            "summary": "CKA not computed because fewer than two Wang encoders loaded.",
        }

    matrix = np.eye(len(names), dtype=float)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            val = linear_cka(wang_arrays[left], wang_arrays[right])
            matrix[i, j] = val
            matrix[j, i] = val

    mean_cka: list[float] = []
    rh: list[float] = []
    for i, name in enumerate(names):
        mean_cka.append(float(np.mean(np.delete(matrix[i], i))))
        val = embryo_results[name]["R/H"]
        rh.append(float(val) if val is not None and np.isfinite(val) else float("nan"))

    pairwise_cka: list[float] = []
    pairwise_abs_rh_gap: list[float] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            if not (np.isfinite(rh[i]) and np.isfinite(rh[j])):
                continue
            pairwise_cka.append(float(matrix[i, j]))
            pairwise_abs_rh_gap.append(float(abs(rh[i] - rh[j])))

    tracking = {
        "mean_pairwise_CKA": {name: float(mean_cka[i]) for i, name in enumerate(names)},
        "R/H": {name: finite_float(rh[i]) for i, name in enumerate(names)},
        "recoverability_rank_descending_RH": {
            name: finite_float(rank) for name, rank in zip(names, average_ranks([-v for v in rh]), strict=True)
        }
        if all(np.isfinite(v) for v in rh)
        else None,
        "spearman_mean_CKA_vs_RH": spearman_rho(mean_cka, rh) if all(np.isfinite(v) for v in rh) else None,
        "spearman_pairwise_CKA_vs_abs_RH_gap": spearman_rho(pairwise_cka, pairwise_abs_rh_gap),
        "note": "Descriptive only: n=10 encoders and the AMH recoverability range is near null, so CKA/rank correlations are not inferential.",
    }

    return {
        "encoder_names": names,
        "matrix": matrix.tolist(),
        "recoverability_tracking": tracking,
        "summary": f"Pairwise linear CKA computed on centered cached embeddings for {len(names)} Wang encoders; recoverability tracking is descriptive only.",
    }


def ci_overlap(a: list[float | None], b: list[float | None]) -> bool | None:
    if len(a) != 2 or len(b) != 2 or any(v is None for v in a + b):
        return None
    return not (float(a[1]) < float(b[0]) or float(b[1]) < float(a[0]))


def make_verdict(results: dict[str, Any]) -> str:
    embryo = results["embryo_amh"]["encoders"]
    ovary = results["ovary_fuid_phenotype"]["encoders"]
    embryo_used = {name: row for name, row in embryo.items() if row["status"] == "used"}
    ovary_used = {name: row for name, row in ovary.items() if row["status"] == "used"}
    if len(embryo_used) < 10:
        return (
            f"Verified-path run is incomplete: {len(embryo_used)}/10 Wang encoders and "
            f"{len(ovary_used)}/2 FUID encoders loaded, so no embryo-AMH cross-encoder verdict is claimed."
        )
    embryo_vals = [float(row["R/H"]) for row in embryo_used.values() if row["R/H"] is not None]
    ovary_vals = [float(row["R/H"]) for row in ovary_used.values() if row["R/H"] is not None]
    null_like = 0
    for row in embryo_used.values():
        rh = row["R/H"]
        ci = row["bootstrap_CI"]
        crosses_zero = len(ci) == 2 and ci[0] is not None and ci[1] is not None and float(ci[0]) <= 0.0 <= float(ci[1])
        if rh is not None and (float(rh) <= 0.0 or crosses_zero):
            null_like += 1
    embryo_part = (
        f"embryo-AMH R/H range {min(embryo_vals):.4f} to {max(embryo_vals):.4f}; "
        f"{null_like}/{len(embryo_used)} encoders have R/H <= 0 or a 95% cluster-bootstrap CI crossing 0"
    )
    ovary_part = (
        f"FUID phenotype positive-control R/H range {min(ovary_vals):.4f} to {max(ovary_vals):.4f}"
        if ovary_vals
        else "FUID phenotype positive control was not available"
    )
    return f"{embryo_part}; {ovary_part}; CKA/recoverability correlations are descriptive only."


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    wang = load_wang_target()
    fuid = load_fuid_target()
    fuid_specs, fuid_discovery = fuid_encoder_specs(expected_n=fuid["n"])

    embryo_results, wang_arrays = evaluate_target(
        embryo_encoder_specs(),
        wang,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    fuid_results, _ = evaluate_target(
        fuid_specs,
        fuid,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 50_000,
    )

    result: dict[str, Any] = {
        "benchmark": "unified_frozen_encoder_recoverability",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "seed": int(args.seed),
            "n_splits": int(N_SPLITS),
            "n_bootstrap": int(args.n_bootstrap),
            "pca_components": int(PCA_COMPONENTS),
            "verified_existing_paths": True,
            "feature_extraction": "none; np.load cached arrays only",
            "fold_seed_policy": "one seed per target; same folds and same fixed permuted control labels across encoders within target",
        },
        "embryo_amh": {
            "target": "AMH tertile",
            "n": int(wang["n"]),
            "n_groups": int(wang["n_groups"]),
            "class_names": wang["class_names"],
            "class_counts": wang["class_counts"],
            "target_entropy_bits": finite_float(entropy_bits(wang["y"])),
            "tertile_edges_amh": wang["tertile_edges_amh"],
            "label_source": wang["label_source"],
            "label_audit_source": wang["label_audit_source"],
            "alignment_checks": wang["alignment_checks"],
            "encoders": embryo_results,
        },
        "ovary_fuid_phenotype": {
            "target": "FUID phenotype",
            "n": int(fuid["n"]),
            "n_groups": int(fuid["n_groups"]),
            "class_names": fuid["class_names"],
            "class_counts": fuid["class_counts"],
            "target_entropy_bits": finite_float(entropy_bits(fuid["y"])),
            "label_source": fuid["label_source"],
            "alignment_checks": fuid["alignment_checks"],
            "encoder_discovery": fuid_discovery,
            "encoders": fuid_results,
        },
        "cross_encoder_geometry": cka_summary(wang_arrays, embryo_results),
    }
    result["verdict"] = make_verdict(result)
    return result


def format_float(x: float | None) -> str:
    return "null" if x is None else f"{x:.6f}"


def print_summary(result: dict[str, Any]) -> None:
    print("\nUnified frozen-encoder benchmark")
    for section_key in ["embryo_amh", "ovary_fuid_phenotype"]:
        section = result[section_key]
        print(f"\n{section_key}: {section['target']} n={section['n']}")
        for name, row in section["encoders"].items():
            if row["status"] != "used":
                print(f"  {name}: excluded ({row['reason']})")
                continue
            ci = row["bootstrap_CI"]
            print(
                f"  {name}: R/H={format_float(row['R/H'])}, "
                f"CI=[{format_float(ci[0])}, {format_float(ci[1])}], "
                f"control_RH={format_float(row['control_RH'])}, "
                f"selectivity={format_float(row['control_task_selectivity'])}"
            )
    cka = result["cross_encoder_geometry"]
    print(f"\nCKA: {cka['summary']}")
    print(f"Verdict: {result['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    result = run_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
