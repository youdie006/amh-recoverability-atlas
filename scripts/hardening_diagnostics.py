#!/usr/bin/env python3
"""Hardening diagnostics from cached Wang embryo features.

This script performs two fixed, data-in-hand analyses:

1. Target-substitution inflation: compare registered AMH-tertile probing
   against a predeclared menu of nuisance/random targets.
2. Non-linear null-hardening: compare the existing linear AMH probe with a
   fixed HistGradientBoostingClassifier probe on PCA(32) and PCA(64).

No image feature extraction is performed; all encoders are loaded from cached
``.npy`` arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import unified_encoder_benchmark as bench  # noqa: E402


REPO = bench.REPO
OUT_DIR = REPO / "results/diagnostics"
TARGET_SUB_OUT = OUT_DIR / "target_substitution_inflation.json"
NONLINEAR_OUT = OUT_DIR / "embryo_amh_nonlinear_probe.json"
LINEAR_BENCHMARK_PATH = OUT_DIR / "unified_encoder_benchmark.json"

SEED = bench.SEED
N_SPLITS = bench.N_SPLITS
N_BOOTSTRAP = bench.N_BOOTSTRAP
PRIMARY_ENCODER_NAMES = ("DINOv2-base", "CLIP-L", "FEMI")
PCA_NONLINEAR = (32, 64)

AGE_ALIASES = {
    "age",
    "maternalage",
    "maternalageyears",
    "maternalageyear",
    "femaleage",
    "womanage",
    "patientage",
}
SITE_BATCH_TOKENS = ("clinic", "site", "batch", "center", "centre", "lab", "laboratory")


@dataclass(frozen=True)
class TargetSpec:
    key: str
    display_name: str
    y: np.ndarray
    groups: np.ndarray
    row_indices: np.ndarray
    kind: str
    label_source_column: str
    classes: np.ndarray
    class_counts: dict[str, int]
    n: int
    n_groups: int
    target_entropy_bits: float | None
    tertile_edges: list[float] | None = None
    category_mapping: dict[str, int] | None = None
    generation: str | None = None


EstimatorFactory = Callable[[int], Any]


def normalize_column(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_age_column(columns: list[str] | pd.Index) -> str | None:
    for col in columns:
        if normalize_column(str(col)) in AGE_ALIASES:
            return str(col)
    return None


def find_site_batch_column(columns: list[str] | pd.Index) -> str | None:
    for col in columns:
        norm = normalize_column(str(col))
        if any(token in norm for token in SITE_BATCH_TOKENS):
            return str(col)
    return None


def stable_hash_3class(values: np.ndarray | pd.Series) -> np.ndarray:
    labels: list[int] = []
    for value in np.asarray(values).astype(str):
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        labels.append(int.from_bytes(digest[:8], "big") % 3)
    return np.asarray(labels, dtype=int)


def make_target_spec(
    *,
    key: str,
    display_name: str,
    y: np.ndarray,
    groups: np.ndarray,
    row_indices: np.ndarray,
    kind: str,
    label_source_column: str,
    tertile_edges: list[float] | None = None,
    category_mapping: dict[str, int] | None = None,
    generation: str | None = None,
) -> TargetSpec:
    y_arr = np.asarray(y, dtype=int)
    group_arr = np.asarray(groups).astype(str)
    classes = np.unique(y_arr)
    return TargetSpec(
        key=key,
        display_name=display_name,
        y=y_arr,
        groups=group_arr,
        row_indices=np.asarray(row_indices, dtype=int),
        kind=kind,
        label_source_column=str(label_source_column),
        classes=classes,
        class_counts=bench.class_counts(y_arr),
        n=int(len(y_arr)),
        n_groups=int(len(np.unique(group_arr))),
        target_entropy_bits=bench.finite_float(bench.entropy_bits(y_arr)),
        tertile_edges=tertile_edges,
        category_mapping=category_mapping,
        generation=generation,
    )


def tertile_target_from_column(
    frame: pd.DataFrame,
    *,
    key: str,
    display_name: str,
    column: str,
) -> TargetSpec | None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & frame["patient_id"].notna().to_numpy()
    if int(mask.sum()) < 3:
        return None
    y, edges = bench.quantile_tertiles(values[mask])
    if len(np.unique(y)) < 2:
        return None
    return make_target_spec(
        key=key,
        display_name=display_name,
        y=y,
        groups=frame.loc[mask, "patient_id"].astype(str).to_numpy(),
        row_indices=np.flatnonzero(mask),
        kind="continuous_tertile",
        label_source_column=column,
        tertile_edges=edges,
    )


def categorical_partition_from_column(
    frame: pd.DataFrame,
    *,
    key: str,
    display_name: str,
    column: str,
) -> TargetSpec | None:
    mask = frame[column].notna().to_numpy() & frame["patient_id"].notna().to_numpy()
    if int(mask.sum()) < 3:
        return None
    vals = frame.loc[mask, column].astype(str)
    counts = vals.value_counts(sort=True)
    if len(counts) < 3:
        return None
    if len(counts) == 3:
        categories = sorted(counts.index.tolist())
        mapping = {cat: i for i, cat in enumerate(categories)}
        y = vals.map(mapping).to_numpy(dtype=int)
    else:
        top = counts.index[:2].tolist()
        mapping = {str(top[0]): 0, str(top[1]): 1, "__OTHER__": 2}
        y = vals.map(lambda x: mapping.get(str(x), 2)).to_numpy(dtype=int)
    if len(np.unique(y)) < 3:
        return None
    return make_target_spec(
        key=key,
        display_name=display_name,
        y=y,
        groups=frame.loc[mask, "patient_id"].astype(str).to_numpy(),
        row_indices=np.flatnonzero(mask),
        kind="categorical_partition",
        label_source_column=column,
        category_mapping=mapping,
    )


def build_target_menu(frame: pd.DataFrame, seed: int) -> tuple[dict[str, TargetSpec], dict[str, Any]]:
    required = {"patient_id", "amh"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing required target menu columns: {missing}")

    targets: dict[str, TargetSpec] = {}
    omissions: dict[str, Any] = {}

    amh = tertile_target_from_column(
        frame,
        key="registered_amh_tertile",
        display_name="REGISTERED serum AMH tertile",
        column="amh",
    )
    if amh is None:
        raise RuntimeError("registered AMH tertile target could not be constructed")
    targets[amh.key] = amh

    age_col = find_age_column(frame.columns)
    if age_col is None:
        omissions["maternal_age_tertile"] = {
            "status": "omitted",
            "reason": "no age/maternal-age column found",
            "searched_aliases": sorted(AGE_ALIASES),
        }
    else:
        age = tertile_target_from_column(
            frame,
            key="maternal_age_tertile",
            display_name="maternal age tertile",
            column=age_col,
        )
        if age is None:
            omissions["maternal_age_tertile"] = {
                "status": "omitted",
                "reason": f"age column {age_col!r} did not yield a usable classification target",
            }
        else:
            targets[age.key] = age

    site_batch_col = find_site_batch_column(frame.columns)
    if site_batch_col is None:
        omissions["clinic_site_batch"] = {
            "status": "omitted",
            "reason": "no clinic/site/batch column found",
            "searched_tokens": list(SITE_BATCH_TOKENS),
        }
    else:
        series = frame[site_batch_col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == series.notna().sum() and series.nunique(dropna=True) > 5:
            site = tertile_target_from_column(
                frame,
                key="clinic_site_batch_partition",
                display_name=f"{site_batch_col} tertile partition",
                column=site_batch_col,
            )
        else:
            site = categorical_partition_from_column(
                frame,
                key="clinic_site_batch_partition",
                display_name=f"{site_batch_col} 3-class partition",
                column=site_batch_col,
            )
        if site is None:
            omissions["clinic_site_batch"] = {
                "status": "omitted",
                "reason": f"column {site_batch_col!r} did not yield a usable 3-class target",
            }
        else:
            targets[site.key] = site

    full_mask = frame["patient_id"].notna().to_numpy()
    row_indices = np.flatnonzero(full_mask)
    hash_y = stable_hash_3class(frame.loc[full_mask, "patient_id"])
    targets["patient_hash_3class"] = make_target_spec(
        key="patient_hash_3class",
        display_name="within-cohort patient_id SHA256 modulo-3 partition",
        y=hash_y,
        groups=frame.loc[full_mask, "patient_id"].astype(str).to_numpy(),
        row_indices=row_indices,
        kind="nuisance_hash_partition",
        label_source_column="patient_id",
        generation="SHA256(patient_id) modulo 3; fixed positive-control-for-inflation nuisance target",
    )

    rng = np.random.default_rng(int(seed) + 17_017)
    random_y = rng.integers(0, 3, size=len(row_indices), endpoint=False)
    targets["uniform_random_3class"] = make_target_spec(
        key="uniform_random_3class",
        display_name="uniform random 3-class label",
        y=random_y,
        groups=frame.loc[full_mask, "patient_id"].astype(str).to_numpy(),
        row_indices=row_indices,
        kind="random_floor",
        label_source_column="generated",
        generation=f"np.random.default_rng({int(seed) + 17_017}).integers(0, 3, n_rows)",
    )

    return targets, omissions


def target_to_json(target: TargetSpec) -> dict[str, Any]:
    return {
        "display_name": target.display_name,
        "kind": target.kind,
        "label_source_column": target.label_source_column,
        "n": target.n,
        "n_groups": target.n_groups,
        "classes": target.classes.tolist(),
        "class_counts": target.class_counts,
        "target_entropy_bits": target.target_entropy_bits,
        "tertile_edges": target.tertile_edges,
        "category_mapping": target.category_mapping,
        "generation": target.generation,
    }


def primary_embryo_encoder_specs() -> list[bench.EncoderSpec]:
    by_name = {spec.name: spec for spec in bench.embryo_encoder_specs()}
    return [by_name[name] for name in PRIMARY_ENCODER_NAMES]


def load_primary_encoder_arrays(expected_n: int, label_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for spec in primary_embryo_encoder_specs():
        arr, audit = bench.verify_load_array(spec, expected_n=expected_n, label_ids=label_ids)
        audits[spec.name] = audit
        if arr is not None:
            arrays[spec.name] = arr
    return arrays, audits


def fit_oof_probe_with_factory(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    pca_components: int,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]] | None,
    splitter_name: str | None,
    estimator_factory: EstimatorFactory,
    model_label: str,
) -> dict[str, Any]:
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups).astype(str)
    labels = np.unique(y_arr)
    if len(labels) < 2:
        raise ValueError("classification probe needs at least two target classes")
    if len(X) != len(y_arr) or len(groups_arr) != len(y_arr):
        raise ValueError("X, y, and groups must have the same row count")

    if splits is None:
        splitter_name, splits = bench.choose_splits(y_arr, groups_arr, n_splits=N_SPLITS, seed=seed)
    elif splitter_name is None:
        splitter_name = "provided"

    model_proba = np.full((len(y_arr), len(labels)), np.nan, dtype=float)
    baseline_proba = np.full_like(model_proba, np.nan)
    fold_details: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        baseline_proba[test_idx] = bench.prior_proba(y_arr[train_idx], len(test_idx), labels)
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

        x_train, x_test, used_pca = bench.transform_fold(
            X,
            train_idx,
            test_idx,
            pca_components=pca_components,
            seed=seed + 101 * fold,
        )
        model = estimator_factory(seed + 1000 + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train, y_arr[train_idx])
        model_proba[test_idx] = bench.align_proba_to_labels(model.predict_proba(x_test), model.classes_, labels)
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
        "metric": bench.vinfo_metrics(y_arr, baseline_proba, model_proba, labels),
        "splitter": splitter_name,
        "folds": fold_details,
        "splits": splits,
        "model_label": model_label,
    }


def logistic_probe(
    X: np.ndarray,
    target: TargetSpec,
    *,
    seed: int,
    n_bootstrap: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    splitter_name: str,
) -> dict[str, Any]:
    probe = bench.fit_oof_probe(
        X,
        target.y,
        target.groups,
        seed=seed,
        pca_components=bench.PCA_COMPONENTS,
        splits=splits,
        splitter_name=splitter_name,
    )
    return probe_result_json(
        probe,
        target.y,
        target.groups,
        seed=seed + 200_000,
        n_bootstrap=n_bootstrap,
        model_label=f"StandardScaler + PCA({bench.PCA_COMPONENTS}) + multinomial LogisticRegression(lbfgs)",
        pca_components=bench.PCA_COMPONENTS,
    )


def hist_gradient_boosting_factory(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=int(seed))


def nonlinear_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    pca_components: int,
    seed: int,
    n_bootstrap: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    splitter_name: str,
) -> dict[str, Any]:
    probe = fit_oof_probe_with_factory(
        X,
        y,
        groups,
        pca_components=pca_components,
        seed=seed,
        splits=splits,
        splitter_name=splitter_name,
        estimator_factory=hist_gradient_boosting_factory,
        model_label=f"StandardScaler + PCA({pca_components}) + HistGradientBoostingClassifier(defaults, fixed random_state)",
    )
    return probe_result_json(
        probe,
        y,
        groups,
        seed=seed + 200_000,
        n_bootstrap=n_bootstrap,
        model_label=probe["model_label"],
        pca_components=pca_components,
    )


def probe_result_json(
    probe: dict[str, Any],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    model_label: str,
    pca_components: int,
) -> dict[str, Any]:
    metric = probe["metric"]
    ci = bench.cluster_bootstrap_ci_r_over_h(
        y,
        groups,
        probe["baseline_proba"],
        probe["model_proba"],
        probe["labels"],
        n_boot=int(n_bootstrap),
        seed=int(seed),
    )
    return {
        "R/H": bench.finite_float(metric["R_over_H"]),
        "bootstrap_CI": ci,
        "R_bits": bench.finite_float(metric["R_bits"]),
        "target_entropy_bits": bench.finite_float(metric["target_entropy_bits"]),
        "baseline_log_loss_nats": bench.finite_float(metric["baseline_log_loss_nats"]),
        "model_log_loss_nats": bench.finite_float(metric["model_log_loss_nats"]),
        "mdl_codelength_ratio": bench.finite_float(metric["mdl_codelength_ratio"]),
        "probe": {
            "splitter": probe["splitter"],
            "n_splits": int(len(probe["splits"])),
            "model": model_label,
            "pca_components_requested": int(pca_components),
            "baseline": "train-fold class prior with Jeffreys smoothing",
        },
        "folds": probe["folds"],
    }


def null_like(metric_row: dict[str, Any]) -> bool | None:
    rh = metric_row.get("R/H")
    ci = metric_row.get("bootstrap_CI")
    if rh is None:
        return None
    if float(rh) <= 0.0:
        return True
    if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        return bool(float(ci[0]) <= 0.0 <= float(ci[1]))
    return None


def inflation_summary(per_target: dict[str, dict[str, Any]], registered_key: str) -> dict[str, Any]:
    values = {
        key: float(row["R/H"])
        for key, row in per_target.items()
        if row.get("R/H") is not None and np.isfinite(float(row["R/H"]))
    }
    registered = values.get(registered_key)
    if registered is None or not values:
        return {
            "registered_target": registered_key,
            "registered_RH": bench.finite_float(registered),
            "target_agnostic_max_target": None,
            "target_agnostic_max_RH": None,
            "inflation_statistic": None,
            "max_alternative_target": None,
            "max_alternative_RH": None,
            "max_alternative_minus_registered": None,
        }

    max_target = max(values, key=values.get)
    alternatives = {k: v for k, v in values.items() if k != registered_key}
    if alternatives:
        max_alt_target = max(alternatives, key=alternatives.get)
        max_alt_rh: float | None = alternatives[max_alt_target]
        max_alt_gap: float | None = max_alt_rh - registered
    else:
        max_alt_target = None
        max_alt_rh = None
        max_alt_gap = None
    return {
        "registered_target": registered_key,
        "registered_RH": bench.finite_float(registered),
        "target_agnostic_max_target": max_target,
        "target_agnostic_max_RH": bench.finite_float(values[max_target]),
        "inflation_statistic": bench.finite_float(values[max_target] - registered),
        "max_alternative_target": max_alt_target,
        "max_alternative_RH": bench.finite_float(max_alt_rh),
        "max_alternative_minus_registered": bench.finite_float(max_alt_gap),
    }


def load_wang_frames() -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = pd.read_csv(bench.WANG_INDEX_PATH)
    amh = pd.read_csv(bench.WANG_AMH_PATH)
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

    frame = idx.copy()
    index_age_col = find_age_column(idx.columns)
    recovered_age_col = find_age_column(amh.columns)
    if index_age_col is None and recovered_age_col is not None:
        frame[recovered_age_col] = amh[recovered_age_col]

    audit = {
        "index_path": str(bench.WANG_INDEX_PATH),
        "amh_recovered_path": str(bench.WANG_AMH_PATH),
        "index_shape": [int(idx.shape[0]), int(idx.shape[1])],
        "amh_recovered_shape": [int(amh.shape[0]), int(amh.shape[1])],
        "index_columns": [str(col) for col in idx.columns],
        "amh_recovered_columns": [str(col) for col in amh.columns],
        "same_image_order": bool(same_image_order),
        "same_amh_values": bool(same_amh),
        "age_column_in_index": index_age_col,
        "age_column_in_amh_recovered": recovered_age_col,
        "age_column_used": index_age_col or recovered_age_col,
    }
    return frame, audit


def target_substitution_verdict(result: dict[str, Any]) -> dict[str, Any]:
    gaps: list[float] = []
    max_null_like = 0
    used = 0
    for encoder_row in result["encoders"].values():
        stat = encoder_row["inflation"]
        gap = stat["inflation_statistic"]
        if gap is not None:
            gaps.append(float(gap))
        max_target = stat["target_agnostic_max_target"]
        if max_target is not None:
            used += 1
            row = encoder_row["targets"][max_target]
            if null_like(row) is True:
                max_null_like += 1
    positive = sum(gap > 0.0 for gap in gaps)
    if not gaps:
        statement = "Inflation statistic could not be computed for any encoder."
        strength = "uncomputed"
    elif positive == 0:
        statement = (
            "The target-agnostic maximum did not exceed the registered AMH R/H for any used encoder; "
            "cross-target inflation is not demonstrated in this run."
        )
        strength = "none"
    else:
        statement = (
            f"The target-agnostic maximum exceeded registered AMH for {positive}/{len(gaps)} encoders; "
            f"inflation statistic range {min(gaps):.6f} to {max(gaps):.6f} R/H."
        )
        strength = "strong" if positive == len(gaps) and min(gaps) > 0.01 else "weak"
    if used and max_null_like == used:
        statement += " The menu maximum is null-like for every used encoder by R/H<=0 or CI crossing 0."
    elif used:
        statement += f" The menu maximum is null-like for {max_null_like}/{used} used encoders."
    return {
        "strength": strength,
        "n_encoders_with_positive_inflation": int(positive),
        "n_encoders_evaluated": int(len(gaps)),
        "n_target_agnostic_max_null_like": int(max_null_like),
        "statement": statement,
    }


def run_target_substitution(args: argparse.Namespace) -> dict[str, Any]:
    frame, column_audit = load_wang_frames()
    target_menu, omissions = build_target_menu(frame, seed=args.seed)
    arrays, load_audits = load_primary_encoder_arrays(
        expected_n=len(frame),
        label_ids=frame["Image"].astype(str).to_numpy(),
    )

    result: dict[str, Any] = {
        "analysis": "target_substitution_cross_target_inflation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "seed": int(args.seed),
            "n_splits": int(N_SPLITS),
            "n_bootstrap": int(args.n_bootstrap),
            "registered_target": "registered_amh_tertile",
            "probe_conventions": "StratifiedGroupKFold by patient_id; StandardScaler + PCA(32); train-fold class-prior baseline; R/H in fraction-of-entropy; patient-cluster bootstrap CI",
            "feature_extraction": "none; cached np.load arrays only",
            "menu_policy": "predeclared targets only; omitted unavailable age/site/batch arms are reported rather than substituted",
        },
        "column_audit": column_audit,
        "targets": {key: target_to_json(target) for key, target in target_menu.items()},
        "omitted_targets": omissions,
        "encoders": {},
    }

    for spec in primary_embryo_encoder_specs():
        encoder_row: dict[str, Any] = {
            "path": str(spec.path),
            "load_audit": load_audits.get(spec.name),
            "targets": {},
            "inflation": None,
        }
        arr = arrays.get(spec.name)
        if arr is None:
            encoder_row["status"] = "excluded"
            encoder_row["reason"] = load_audits.get(spec.name, {}).get("reason")
            result["encoders"][spec.name] = encoder_row
            continue
        encoder_row["status"] = "used"
        encoder_row["reason"] = None
        for target_i, (target_key, target) in enumerate(target_menu.items()):
            splitter_name, splits = bench.choose_splits(
                target.y,
                target.groups,
                n_splits=N_SPLITS,
                seed=int(args.seed) + 10_000 * target_i,
            )
            x = arr[target.row_indices]
            metric = logistic_probe(
                x,
                target,
                seed=int(args.seed) + 10_000 * target_i,
                n_bootstrap=int(args.n_bootstrap),
                splits=splits,
                splitter_name=splitter_name,
            )
            metric["target"] = target_to_json(target)
            metric["null_like"] = null_like(metric)
            encoder_row["targets"][target_key] = metric
        encoder_row["inflation"] = inflation_summary(encoder_row["targets"], "registered_amh_tertile")
        result["encoders"][spec.name] = encoder_row

    result["verdict"] = target_substitution_verdict(result)
    return result


def metric_subset(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "R/H": None,
            "bootstrap_CI": [None, None],
            "R_bits": None,
            "target_entropy_bits": None,
            "baseline_log_loss_nats": None,
            "model_log_loss_nats": None,
            "mdl_codelength_ratio": None,
        }
    return {
        "R/H": row.get("R/H"),
        "bootstrap_CI": row.get("bootstrap_CI", [None, None]),
        "R_bits": row.get("R_bits"),
        "target_entropy_bits": row.get("target_entropy_bits"),
        "baseline_log_loss_nats": row.get("baseline_log_loss_nats"),
        "model_log_loss_nats": row.get("model_log_loss_nats"),
        "mdl_codelength_ratio": row.get("mdl_codelength_ratio"),
        "null_like": null_like(row),
    }


def load_linear_benchmark_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    if not LINEAR_BENCHMARK_PATH.exists():
        return {}, {"status": "missing", "path": str(LINEAR_BENCHMARK_PATH)}
    payload = json.loads(LINEAR_BENCHMARK_PATH.read_text(encoding="utf-8"))
    rows = payload.get("embryo_amh", {}).get("encoders", {})
    return rows, {
        "status": "loaded",
        "path": str(LINEAR_BENCHMARK_PATH),
        "benchmark": payload.get("benchmark"),
        "created_utc": payload.get("created_utc"),
    }


def nonlinear_verdict(result: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for enc in result["encoders"].values():
        if enc.get("status") != "used":
            continue
        for pca_key, row in enc["nonlinear_hist_gradient_boosting"].items():
            rows.append(row)
    nulls = [row.get("null_like") is True for row in rows]
    positive_nonnull = [
        row
        for row in rows
        if row.get("R/H") is not None
        and float(row["R/H"]) > 0.0
        and row.get("bootstrap_CI", [None, None])[0] is not None
        and float(row["bootstrap_CI"][0]) > 0.0
    ]
    if not rows:
        statement = "No non-linear probe rows were computed."
        survives = None
    elif all(nulls):
        statement = (
            f"All {len(rows)} non-linear AMH probe arms are null-like by R/H<=0 or CI crossing 0; "
            "the embryo->AMH null survives this fixed non-linear probe check."
        )
        survives = True
    elif positive_nonnull:
        statement = (
            f"{len(positive_nonnull)}/{len(rows)} non-linear arms have R/H>0 with a CI lower bound above 0; "
            "the null does not uniformly survive this check."
        )
        survives = False
    else:
        statement = (
            f"{sum(nulls)}/{len(rows)} non-linear arms are null-like; "
            "the non-linear check is mixed and should not be summarized as a clean null."
        )
        survives = False
    return {
        "null_survives_nonlinear_probe": survives,
        "n_nonlinear_arms": int(len(rows)),
        "n_positive_nonnull_arms": int(len(positive_nonnull)),
        "statement": statement,
    }


def run_nonlinear_probe(args: argparse.Namespace) -> dict[str, Any]:
    frame, column_audit = load_wang_frames()
    target_menu, omissions = build_target_menu(frame, seed=args.seed)
    target = target_menu["registered_amh_tertile"]
    arrays, load_audits = load_primary_encoder_arrays(
        expected_n=len(frame),
        label_ids=frame["Image"].astype(str).to_numpy(),
    )
    splitter_name, splits = bench.choose_splits(target.y, target.groups, n_splits=N_SPLITS, seed=int(args.seed))
    rng = np.random.default_rng(int(args.seed) + 91_337)
    control_y = rng.permutation(target.y)
    linear_rows, linear_audit = load_linear_benchmark_rows()

    hgb_sig = inspect.signature(HistGradientBoostingClassifier)
    result: dict[str, Any] = {
        "analysis": "embryo_amh_nonlinear_probe",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "seed": int(args.seed),
            "n_splits": int(N_SPLITS),
            "n_bootstrap": int(args.n_bootstrap),
            "target": "registered_amh_tertile",
            "pca_components": list(PCA_NONLINEAR),
            "probe_conventions": "same patient groups, train-fold class-prior baseline, R/H in fraction-of-entropy, patient-cluster bootstrap CI",
            "nonlinear_model": "HistGradientBoostingClassifier with sklearn defaults except fixed random_state",
            "nonlinear_model_signature": str(hgb_sig),
            "feature_extraction": "none; cached np.load arrays only",
            "random_control": "single fixed row-level random permutation preserving AMH tertile class counts; same folds as real target",
        },
        "column_audit": column_audit,
        "target": target_to_json(target),
        "omitted_targets_seen_but_not_used": omissions,
        "linear_source": linear_audit,
        "encoders": {},
    }

    for spec in primary_embryo_encoder_specs():
        encoder_row: dict[str, Any] = {
            "path": str(spec.path),
            "load_audit": load_audits.get(spec.name),
            "linear_logistic_pca32_from_unified_encoder_benchmark": metric_subset(linear_rows.get(spec.name)),
            "nonlinear_hist_gradient_boosting": {},
        }
        arr = arrays.get(spec.name)
        if arr is None:
            encoder_row["status"] = "excluded"
            encoder_row["reason"] = load_audits.get(spec.name, {}).get("reason")
            result["encoders"][spec.name] = encoder_row
            continue
        encoder_row["status"] = "used"
        encoder_row["reason"] = None
        x = arr[target.row_indices]
        for pca_components in PCA_NONLINEAR:
            probe_seed = int(args.seed) + 50_000 * (1 + PCA_NONLINEAR.index(pca_components))
            real = nonlinear_probe(
                x,
                target.y,
                target.groups,
                pca_components=int(pca_components),
                seed=probe_seed,
                n_bootstrap=int(args.n_bootstrap),
                splits=splits,
                splitter_name=splitter_name,
            )
            control = nonlinear_probe(
                x,
                control_y,
                target.groups,
                pca_components=int(pca_components),
                seed=probe_seed + 100_000,
                n_bootstrap=int(args.n_bootstrap),
                splits=splits,
                splitter_name=f"{splitter_name} (same folds as real AMH target)",
            )
            real["control_task"] = {
                "label_permutation": "single fixed row-level random permutation preserving AMH tertile class counts",
                **metric_subset(control),
            }
            real["control_RH"] = control["R/H"]
            real["control_task_selectivity"] = (
                bench.finite_float(float(real["R/H"]) - float(control["R/H"]))
                if real["R/H"] is not None and control["R/H"] is not None
                else None
            )
            real["null_like"] = null_like(real)
            encoder_row["nonlinear_hist_gradient_boosting"][f"pca{pca_components}"] = real
        result["encoders"][spec.name] = encoder_row

    result["verdict"] = nonlinear_verdict(result)
    return result


def format_float(x: Any) -> str:
    return "null" if x is None else f"{float(x):.6f}"


def print_target_substitution_summary(result: dict[str, Any]) -> None:
    print("\nTarget-substitution inflation")
    for name, row in result["encoders"].items():
        if row["status"] != "used":
            print(f"  {name}: excluded ({row['reason']})")
            continue
        stat = row["inflation"]
        print(
            f"  {name}: registered={format_float(stat['registered_RH'])}, "
            f"max={format_float(stat['target_agnostic_max_RH'])} ({stat['target_agnostic_max_target']}), "
            f"inflation={format_float(stat['inflation_statistic'])}"
        )
        for target_key, metric in row["targets"].items():
            ci = metric["bootstrap_CI"]
            print(
                f"    {target_key}: R/H={format_float(metric['R/H'])}, "
                f"CI=[{format_float(ci[0])}, {format_float(ci[1])}]"
            )
    print(f"  verdict: {result['verdict']['statement']}")


def print_nonlinear_summary(result: dict[str, Any]) -> None:
    print("\nEmbryo AMH non-linear probe")
    for name, row in result["encoders"].items():
        if row["status"] != "used":
            print(f"  {name}: excluded ({row['reason']})")
            continue
        linear = row["linear_logistic_pca32_from_unified_encoder_benchmark"]
        print(f"  {name}: linear_pca32={format_float(linear['R/H'])}")
        for pca_key, metric in row["nonlinear_hist_gradient_boosting"].items():
            ci = metric["bootstrap_CI"]
            print(
                f"    HGB {pca_key}: R/H={format_float(metric['R/H'])}, "
                f"CI=[{format_float(ci[0])}, {format_float(ci[1])}], "
                f"control_RH={format_float(metric['control_RH'])}, "
                f"selectivity={format_float(metric['control_task_selectivity'])}"
            )
    print(f"  verdict: {result['verdict']['statement']}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bench.json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=["both", "target-substitution", "nonlinear"], default="both")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--target-substitution-out", type=Path, default=TARGET_SUB_OUT)
    parser.add_argument("--nonlinear-out", type=Path, default=NONLINEAR_OUT)
    args = parser.parse_args()

    if args.analysis in {"both", "target-substitution"}:
        target_result = run_target_substitution(args)
        write_json(args.target_substitution_out, target_result)
        print_target_substitution_summary(target_result)
        print(f"saved {args.target_substitution_out}")

    if args.analysis in {"both", "nonlinear"}:
        nonlinear_result = run_nonlinear_probe(args)
        write_json(args.nonlinear_out, nonlinear_result)
        print_nonlinear_summary(nonlinear_result)
        print(f"saved {args.nonlinear_out}")


if __name__ == "__main__":
    main()
