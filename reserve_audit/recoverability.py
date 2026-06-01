"""Target-separated recoverability audit.

The classification path preserves the recoverability-atlas v2 probe mechanics:
fixed observed-label splits, fold-local impute/scale/PCA, logistic probes,
inner-fold prior blending, and cluster-aware target permutations.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - depends on installed sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


Task = Literal["classification", "regression"]
EPS = 1e-12
PROBA_SMOOTH = 1e-5


@dataclass(frozen=True)
class _FoldPlan:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    used_pca_components: int | None
    inner_sub_idx: np.ndarray | None
    inner_cal_idx: np.ndarray | None
    X_inner_sub: np.ndarray | None
    X_inner_cal: np.ndarray | None


@dataclass(frozen=True)
class _ConditionalFoldPlan:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    baseline_train: np.ndarray
    baseline_test: np.ndarray
    augmented_train: np.ndarray
    augmented_test: np.ndarray
    used_modality_pca_components: int | None
    inner_sub_idx: np.ndarray | None
    inner_cal_idx: np.ndarray | None
    baseline_inner_sub: np.ndarray | None
    baseline_inner_cal: np.ndarray | None
    augmented_inner_sub: np.ndarray | None
    augmented_inner_cal: np.ndarray | None


@dataclass(frozen=True)
class _RegressionFoldPlan:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    baseline_train: np.ndarray | None
    baseline_test: np.ndarray | None
    augmented_train: np.ndarray
    augmented_test: np.ndarray


def _round_float(x: Any, ndigits: int = 6) -> float | None:
    if x is None:
        return None
    val = float(x)
    if not np.isfinite(val):
        return None
    if abs(val) < 0.5 * 10 ** (-ndigits):
        val = 0.0
    return round(val, ndigits)


def _finite_feature_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("features must be a 1D or 2D array")
    return arr


def _check_rows(X: np.ndarray, y: np.ndarray, name: str = "X") -> None:
    if len(X) != len(y):
        raise ValueError(f"{name} rows ({len(X)}) must match y length ({len(y)})")


def entropy_bits(y: np.ndarray) -> float:
    vals = np.asarray(y)
    _, counts = np.unique(vals, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _class_counts(y: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(y == label)) for label in labels}


def _align_proba_to_labels(
    proba: np.ndarray,
    model_classes: np.ndarray,
    labels: np.ndarray,
    smooth: float = PROBA_SMOOTH,
) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    out = np.full((p.shape[0], len(labels)), smooth / max(len(labels), 1), dtype=float)
    label_pos = {label: i for i, label in enumerate(labels.tolist())}
    for j, cls in enumerate(np.asarray(model_classes).tolist()):
        if cls in label_pos:
            out[:, label_pos[cls]] += (1.0 - smooth) * p[:, j]
    out = np.clip(out, EPS, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _baseline_prior_proba(y_train: np.ndarray, n_rows: int, labels: np.ndarray) -> np.ndarray:
    counts = np.array([np.sum(y_train == label) for label in labels], dtype=float)
    prior = (counts + 0.5) / (counts.sum() + 0.5 * len(labels))
    return np.repeat(prior[None, :], int(n_rows), axis=0)


def _blend_proba(baseline_proba: np.ndarray, model_proba: np.ndarray, alpha: float) -> np.ndarray:
    p = (1.0 - float(alpha)) * np.asarray(baseline_proba, dtype=float) + float(alpha) * np.asarray(
        model_proba, dtype=float
    )
    p = np.clip(p, EPS, None)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _select_blend_alpha(y: np.ndarray, baseline_proba: np.ndarray, model_proba: np.ndarray, labels: np.ndarray) -> float:
    best_alpha = 0.0
    best_loss = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        p = _blend_proba(baseline_proba, model_proba, float(alpha))
        loss = float(log_loss(y, p, labels=labels))
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_alpha = float(alpha)
    return best_alpha


def _classification_splits(
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
        if StratifiedGroupKFold is not None and n_groups > n_splits:
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


def _regression_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    groups = np.asarray(groups).astype(str)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(int(n_splits), n_groups, len(y)))
    if n_groups == len(y):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        return "KFold", list(splitter.split(np.zeros(len(y))))
    splitter = GroupKFold(n_splits=n_splits)
    return "GroupKFold", list(splitter.split(np.zeros(len(y)), y, groups=groups))


def _impute_scale_pca(
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_components: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    X = _finite_feature_matrix(X)
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, med)
    X_test = np.where(np.isfinite(X_test), X_test, med)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    used_components = None
    if pca_components is not None and X_train.shape[1] > int(pca_components):
        n_comp = min(int(pca_components), X_train.shape[0] - 1, X_train.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(seed))
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            used_components = int(n_comp)
    return X_train, X_test, used_components


def _impute_scale(
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train, test, _ = _impute_scale_pca(X, train_idx, test_idx, pca_components=None, seed=0)
    return train, test


def predictive_vinfo_from_proba(
    y: np.ndarray,
    baseline_proba: np.ndarray,
    model_proba: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    base_loss = float(log_loss(y, baseline_proba, labels=labels))
    model_loss = float(log_loss(y, model_proba, labels=labels))
    bits = (base_loss - model_loss) / math.log(2.0)
    h = entropy_bits(y)
    return {
        "bits": bits,
        "fraction_of_entropy": bits / h if h > 0 else np.nan,
        "baseline_log_loss_nats": base_loss,
        "model_log_loss_nats": model_loss,
    }


def conditional_vinfo_from_proba(
    y: np.ndarray,
    baseline_proba: np.ndarray,
    augmented_proba: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | str]:
    base_loss = float(log_loss(y, baseline_proba, labels=labels))
    model_loss = float(log_loss(y, augmented_proba, labels=labels))
    bits = (base_loss - model_loss) / math.log(2.0)
    h = entropy_bits(y)
    return {
        "bits": bits,
        "fraction_of_entropy": bits / h if h > 0 else np.nan,
        "baseline_log_loss_nats": base_loss,
        "model_log_loss_nats": model_loss,
        "baseline_name": "baseline",
        "augmented_name": "baseline_plus_features",
    }


class _LinearProbePlan:
    def __init__(
        self,
        X: np.ndarray,
        y_observed: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        n_splits: int,
        seed: int,
        pca_components: int | None,
    ) -> None:
        self.X = _finite_feature_matrix(X)
        self.y_observed = np.asarray(y_observed)
        self.groups = np.asarray(groups).astype(str)
        self.labels = np.asarray(labels)
        self.seed = int(seed)
        self.pca_components = pca_components
        self.splitter_name, splits = _classification_splits(
            self.y_observed, self.groups, n_splits=n_splits, seed=self.seed
        )
        self.folds = [self._make_fold_plan(i, train, test) for i, (train, test) in enumerate(splits, start=1)]

    def _make_fold_plan(self, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> _FoldPlan:
        X_train, X_test, used_pca = _impute_scale_pca(
            self.X,
            train_idx,
            test_idx,
            pca_components=self.pca_components,
            seed=self.seed + 101 * fold,
        )
        inner_seed = self.seed + 701 * fold
        inner_sub_idx = None
        inner_cal_idx = None
        X_inner_sub = None
        X_inner_cal = None
        y_train = self.y_observed[train_idx]
        groups_train = self.groups[train_idx]
        if len(np.unique(y_train)) >= 2 and len(np.unique(groups_train)) >= 3:
            try:
                _, inner_splits = _classification_splits(
                    y_train,
                    groups_train,
                    n_splits=min(3, len(np.unique(groups_train))),
                    seed=inner_seed,
                )
            except ValueError:
                inner_splits = []
            for inner_sub, inner_cal in inner_splits:
                sub_abs = train_idx[inner_sub]
                cal_abs = train_idx[inner_cal]
                if len(np.unique(self.y_observed[sub_abs])) < 2:
                    continue
                X_inner_sub, X_inner_cal, _ = _impute_scale_pca(
                    self.X,
                    sub_abs,
                    cal_abs,
                    pca_components=self.pca_components,
                    seed=inner_seed + 17,
                )
                inner_sub_idx = sub_abs
                inner_cal_idx = cal_abs
                break
        return _FoldPlan(
            fold=int(fold),
            train_idx=train_idx,
            test_idx=test_idx,
            X_train=X_train,
            X_test=X_test,
            used_pca_components=used_pca,
            inner_sub_idx=inner_sub_idx,
            inner_cal_idx=inner_cal_idx,
            X_inner_sub=X_inner_sub,
            X_inner_cal=X_inner_cal,
        )

    @staticmethod
    def _model(seed: int) -> LogisticRegression:
        return LogisticRegression(C=0.5, max_iter=1000, solver="newton-cholesky", random_state=int(seed))

    def fit(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        y = np.asarray(y)
        model_proba = np.full((len(y), len(self.labels)), np.nan, dtype=float)
        baseline_proba = np.full_like(model_proba, np.nan)
        fold_details: list[dict[str, Any]] = []
        for plan in self.folds:
            train_idx = plan.train_idx
            test_idx = plan.test_idx
            baseline_proba[test_idx] = _baseline_prior_proba(y[train_idx], len(test_idx), self.labels)
            train_labels = np.unique(y[train_idx])
            if len(train_labels) < 2:
                model_proba[test_idx] = baseline_proba[test_idx]
                fold_details.append({"fold": plan.fold, "fallback": "single_train_class"})
                continue

            alpha = 0.0
            if (
                plan.inner_sub_idx is not None
                and plan.inner_cal_idx is not None
                and plan.X_inner_sub is not None
                and plan.X_inner_cal is not None
                and len(np.unique(y[plan.inner_sub_idx])) >= 2
            ):
                inner_seed = self.seed + 701 * plan.fold
                inner_model = self._model(inner_seed + 19)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    inner_model.fit(plan.X_inner_sub, y[plan.inner_sub_idx])
                inner_raw = _align_proba_to_labels(
                    inner_model.predict_proba(plan.X_inner_cal), inner_model.classes_, self.labels
                )
                inner_base = _baseline_prior_proba(y[plan.inner_sub_idx], len(plan.inner_cal_idx), self.labels)
                alpha = _select_blend_alpha(y[plan.inner_cal_idx], inner_base, inner_raw, self.labels)

            model = self._model(self.seed + 1000 + plan.fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(plan.X_train, y[train_idx])
            raw = _align_proba_to_labels(model.predict_proba(plan.X_test), model.classes_, self.labels)
            model_proba[test_idx] = _blend_proba(baseline_proba[test_idx], raw, alpha)
            fold_details.append(
                {
                    "fold": int(plan.fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_classes": int(len(train_labels)),
                    "pca_components": plan.used_pca_components,
                    "prior_blend_alpha": _round_float(alpha),
                }
            )

        if np.any(~np.isfinite(model_proba)) or np.any(~np.isfinite(baseline_proba)):
            raise RuntimeError("out-of-fold predictions contain missing values")
        return model_proba, baseline_proba, {"splitter": self.splitter_name, "folds": fold_details}


class _ConditionalProbePlan:
    def __init__(
        self,
        X: np.ndarray,
        baseline_features: np.ndarray,
        y_observed: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        n_splits: int,
        seed: int,
        pca_components: int | None,
    ) -> None:
        self.X = _finite_feature_matrix(X)
        self.baseline_features = _finite_feature_matrix(baseline_features)
        if self.X.shape[0] != self.baseline_features.shape[0]:
            raise ValueError("features and baseline rows must match")
        self.y_observed = np.asarray(y_observed)
        self.groups = np.asarray(groups).astype(str)
        self.labels = np.asarray(labels)
        self.seed = int(seed)
        self.pca_components = pca_components
        self.splitter_name, splits = _classification_splits(
            self.y_observed, self.groups, n_splits=n_splits, seed=self.seed
        )
        self.folds = [self._make_fold_plan(i, train, test) for i, (train, test) in enumerate(splits, start=1)]

    @staticmethod
    def _model(seed: int) -> LogisticRegression:
        return LogisticRegression(C=0.5, max_iter=300, solver="lbfgs", random_state=int(seed))

    @staticmethod
    def _stack(baseline_block: np.ndarray, feature_block: np.ndarray) -> np.ndarray:
        return np.column_stack([baseline_block, feature_block])

    def _transform_blocks(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
        b_train, b_test, _ = _impute_scale_pca(self.baseline_features, train_idx, test_idx, None, seed)
        x_train, x_test, used_pca = _impute_scale_pca(self.X, train_idx, test_idx, self.pca_components, seed)
        return b_train, b_test, self._stack(b_train, x_train), self._stack(b_test, x_test), used_pca

    def _make_fold_plan(self, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> _ConditionalFoldPlan:
        b_train, b_test, aug_train, aug_test, used_pca = self._transform_blocks(
            train_idx, test_idx, seed=self.seed + 101 * fold
        )
        inner_seed = self.seed + 701 * fold
        inner_sub_idx = None
        inner_cal_idx = None
        b_inner_sub = None
        b_inner_cal = None
        aug_inner_sub = None
        aug_inner_cal = None
        y_train = self.y_observed[train_idx]
        groups_train = self.groups[train_idx]
        if len(np.unique(y_train)) >= 2 and len(np.unique(groups_train)) >= 3:
            try:
                _, inner_splits = _classification_splits(
                    y_train,
                    groups_train,
                    n_splits=min(3, len(np.unique(groups_train))),
                    seed=inner_seed,
                )
            except ValueError:
                inner_splits = []
            for inner_sub, inner_cal in inner_splits:
                sub_abs = train_idx[inner_sub]
                cal_abs = train_idx[inner_cal]
                if len(np.unique(self.y_observed[sub_abs])) < 2:
                    continue
                b_inner_sub, b_inner_cal, aug_inner_sub, aug_inner_cal, _ = self._transform_blocks(
                    sub_abs, cal_abs, seed=inner_seed + 17
                )
                inner_sub_idx = sub_abs
                inner_cal_idx = cal_abs
                break
        return _ConditionalFoldPlan(
            fold=int(fold),
            train_idx=train_idx,
            test_idx=test_idx,
            baseline_train=b_train,
            baseline_test=b_test,
            augmented_train=aug_train,
            augmented_test=aug_test,
            used_modality_pca_components=used_pca,
            inner_sub_idx=inner_sub_idx,
            inner_cal_idx=inner_cal_idx,
            baseline_inner_sub=b_inner_sub,
            baseline_inner_cal=b_inner_cal,
            augmented_inner_sub=aug_inner_sub,
            augmented_inner_cal=aug_inner_cal,
        )

    def _fit_family(self, y: np.ndarray, family: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        y = np.asarray(y)
        model_proba = np.full((len(y), len(self.labels)), np.nan, dtype=float)
        fold_details: list[dict[str, Any]] = []
        for plan in self.folds:
            train_idx = plan.train_idx
            test_idx = plan.test_idx
            prior = _baseline_prior_proba(y[train_idx], len(test_idx), self.labels)
            train_labels = np.unique(y[train_idx])
            if len(train_labels) < 2:
                model_proba[test_idx] = prior
                fold_details.append({"fold": plan.fold, "fallback": "single_train_class"})
                continue
            if family == "baseline":
                train_features = plan.baseline_train
                test_features = plan.baseline_test
                inner_sub_features = plan.baseline_inner_sub
                inner_cal_features = plan.baseline_inner_cal
            elif family == "augmented":
                train_features = plan.augmented_train
                test_features = plan.augmented_test
                inner_sub_features = plan.augmented_inner_sub
                inner_cal_features = plan.augmented_inner_cal
            else:  # pragma: no cover - internal guard
                raise ValueError(f"unknown family: {family}")

            alpha = 0.0
            if (
                plan.inner_sub_idx is not None
                and plan.inner_cal_idx is not None
                and inner_sub_features is not None
                and inner_cal_features is not None
                and len(np.unique(y[plan.inner_sub_idx])) >= 2
            ):
                inner_seed = self.seed + 701 * plan.fold
                inner_model = self._model(inner_seed + 19)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    inner_model.fit(inner_sub_features, y[plan.inner_sub_idx])
                inner_raw = _align_proba_to_labels(
                    inner_model.predict_proba(inner_cal_features), inner_model.classes_, self.labels
                )
                inner_base = _baseline_prior_proba(y[plan.inner_sub_idx], len(plan.inner_cal_idx), self.labels)
                alpha = _select_blend_alpha(y[plan.inner_cal_idx], inner_base, inner_raw, self.labels)

            model = self._model(self.seed + 1000 + plan.fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(train_features, y[train_idx])
            raw = _align_proba_to_labels(model.predict_proba(test_features), model.classes_, self.labels)
            model_proba[test_idx] = _blend_proba(prior, raw, alpha)
            fold_details.append(
                {
                    "fold": int(plan.fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_classes": int(len(train_labels)),
                    "modality_pca_components": plan.used_modality_pca_components,
                    "prior_blend_alpha": _round_float(alpha),
                }
            )
        if np.any(~np.isfinite(model_proba)):
            raise RuntimeError(f"{family} out-of-fold predictions contain missing values")
        return model_proba, fold_details

    def fit(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        baseline_proba, baseline_folds = self._fit_family(y, "baseline")
        augmented_proba, augmented_folds = self._fit_family(y, "augmented")
        meta = {
            "splitter": self.splitter_name,
            "split_policy": "fixed observed-label split plan reused for all permutation draws",
            "baseline_folds": baseline_folds,
            "augmented_folds": augmented_folds,
            "feature_transform": (
                "baseline covariates are scaled separately and always retained; feature block is "
                "scaled/PCA-compressed before concatenation"
            ),
        }
        return augmented_proba, baseline_proba, meta


class _RegressionProbePlan:
    def __init__(
        self,
        X: np.ndarray,
        y_observed: np.ndarray,
        groups: np.ndarray,
        baseline_features: np.ndarray | None,
        n_splits: int,
        seed: int,
    ) -> None:
        self.X = _finite_feature_matrix(X)
        self.baseline_features = None if baseline_features is None else _finite_feature_matrix(baseline_features)
        if self.baseline_features is not None and self.baseline_features.shape[0] != self.X.shape[0]:
            raise ValueError("features and baseline rows must match")
        self.y_observed = np.asarray(y_observed, dtype=float)
        self.groups = np.asarray(groups).astype(str)
        self.seed = int(seed)
        self.splitter_name, splits = _regression_splits(self.y_observed, self.groups, n_splits, self.seed)
        self.folds = [self._make_fold_plan(i, train, test) for i, (train, test) in enumerate(splits, start=1)]

    def _make_fold_plan(self, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> _RegressionFoldPlan:
        x_train, x_test = _impute_scale(self.X, train_idx, test_idx)
        if self.baseline_features is None:
            return _RegressionFoldPlan(fold, train_idx, test_idx, None, None, x_train, x_test)
        b_train, b_test = _impute_scale(self.baseline_features, train_idx, test_idx)
        aug_train = np.column_stack([b_train, x_train])
        aug_test = np.column_stack([b_test, x_test])
        return _RegressionFoldPlan(fold, train_idx, test_idx, b_train, b_test, aug_train, aug_test)

    def fit(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        y = np.asarray(y, dtype=float)
        baseline_pred = np.full(len(y), np.nan, dtype=float)
        model_pred = np.full(len(y), np.nan, dtype=float)
        fold_details: list[dict[str, Any]] = []
        for plan in self.folds:
            train_idx = plan.train_idx
            test_idx = plan.test_idx
            if plan.baseline_train is None or plan.baseline_test is None:
                baseline_pred[test_idx] = float(np.mean(y[train_idx]))
            else:
                base_model = LinearRegression()
                base_model.fit(plan.baseline_train, y[train_idx])
                baseline_pred[test_idx] = base_model.predict(plan.baseline_test)
            model = LinearRegression()
            model.fit(plan.augmented_train, y[train_idx])
            model_pred[test_idx] = model.predict(plan.augmented_test)
            fold_details.append({"fold": int(plan.fold), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        if np.any(~np.isfinite(model_pred)) or np.any(~np.isfinite(baseline_pred)):
            raise RuntimeError("regression out-of-fold predictions contain missing values")
        return model_pred, baseline_pred, {"splitter": self.splitter_name, "folds": fold_details}


def _incremental_r2(y: np.ndarray, baseline_pred: np.ndarray, model_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    model_pred = np.asarray(model_pred, dtype=float)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    baseline_sse = float(np.sum((y - baseline_pred) ** 2))
    model_sse = float(np.sum((y - model_pred) ** 2))
    baseline_r2 = 1.0 - baseline_sse / sst if sst > 0 else np.nan
    model_r2 = 1.0 - model_sse / sst if sst > 0 else np.nan
    inc = (baseline_sse - model_sse) / sst if sst > 0 else np.nan
    return {
        "incremental_r2": inc,
        "baseline_r2_heldout": baseline_r2,
        "model_r2_heldout": model_r2,
        "baseline_sse": baseline_sse,
        "model_sse": model_sse,
    }


def permute_target_cluster_aware(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y).copy()
    groups = np.asarray(groups).astype(str)
    uniq = np.array(sorted(np.unique(groups)))
    out = y.copy()
    by_size: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for group in uniq:
        idx = np.flatnonzero(groups == group)
        by_size.setdefault(len(idx), []).append((idx, y[idx].copy()))
    for blocks in by_size.values():
        source_order = rng.permutation(len(blocks))
        for target_block, source_i in zip(blocks, source_order, strict=False):
            target_idx, _ = target_block
            out[target_idx] = blocks[int(source_i)][1]
    return out


def _permutation_summary(observed: float, null_values: np.ndarray, scale: float | None = None) -> dict[str, Any]:
    arr = np.asarray(null_values, dtype=float)
    if len(arr) == 0:
        return {
            "n_permutations": 0,
            "alternative": "greater",
            "empirical_p_value": None,
            "statistic_mean": None,
            "statistic_sd": None,
            "statistic_p95_raw": None,
            "fraction_mean": None,
            "fraction_sd": None,
            "fraction_p95_floor": None,
        }
    p95 = float(np.percentile(arr, 95))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "n_permutations": int(len(arr)),
        "alternative": "greater",
        "empirical_p_value": float((1 + np.sum(arr >= float(observed))) / (len(arr) + 1)),
        "statistic_mean": float(np.mean(arr)),
        "statistic_sd": sd,
        "statistic_p95_raw": p95,
        "statistic_p95_floor": float(max(0.0, p95)),
        "fraction_mean": float(np.mean(arr) / scale) if scale and scale > 0 else None,
        "fraction_sd": float(sd / scale) if scale and scale > 0 else None,
        "fraction_p95_floor": float(max(0.0, p95) / scale) if scale and scale > 0 else None,
    }


def _json_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: (value if isinstance(value, str) else _round_float(value)) for key, value in metric.items()}


def recoverability(
    X: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray | None = None,
    task: Task = "classification",
    n_splits: int = 5,
    seed: int = 20260530,
    groups: np.ndarray | None = None,
    n_permutations: int = 200,
    pca_components: int | None = 32,
) -> dict[str, Any]:
    """Estimate held-out recoverability with an optional conditional baseline.

    Classification returns predictive V-information as log-loss reduction in
    bits and as a fraction of target entropy. Regression returns incremental
    held-out R2. When ``baseline`` is supplied, the statistic is computed beyond
    that baseline feature block.
    """

    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    X_arr = _finite_feature_matrix(X)
    y_arr = np.asarray(y, dtype=float if task == "regression" else None)
    _check_rows(X_arr, y_arr)
    if groups is None:
        groups_arr = np.array([f"row_{i}" for i in range(len(y_arr))], dtype=str)
    else:
        groups_arr = np.asarray(groups).astype(str)
        _check_rows(groups_arr, y_arr, "groups")
    baseline_arr = None if baseline is None else _finite_feature_matrix(baseline)
    if baseline_arr is not None:
        _check_rows(baseline_arr, y_arr, "baseline")

    if task == "classification":
        labels = np.unique(y_arr)
        if len(labels) < 2:
            raise ValueError("classification recoverability requires at least two target classes")
        if baseline_arr is None:
            plan: Any = _LinearProbePlan(
                X_arr, y_arr, groups_arr, labels, n_splits=n_splits, seed=seed, pca_components=pca_components
            )
            model_proba, baseline_proba, meta = plan.fit(y_arr)
            metric = predictive_vinfo_from_proba(y_arr, baseline_proba, model_proba, labels)
            mode = "marginal"
        else:
            plan = _ConditionalProbePlan(
                X_arr,
                baseline_arr,
                y_arr,
                groups_arr,
                labels,
                n_splits=n_splits,
                seed=seed,
                pca_components=pca_components,
            )
            model_proba, baseline_proba, meta = plan.fit(y_arr)
            metric = conditional_vinfo_from_proba(y_arr, baseline_proba, model_proba, labels)
            mode = "conditional"
        observed = float(metric["bits"])
        null_values: list[float] = []
        rng = np.random.default_rng(int(seed) + 300)
        for _ in range(int(n_permutations)):
            yp = permute_target_cluster_aware(y_arr, groups_arr, rng)
            perm_model, perm_base, _ = plan.fit(yp)
            if baseline_arr is None:
                perm_metric = predictive_vinfo_from_proba(yp, perm_base, perm_model, labels)
            else:
                perm_metric = conditional_vinfo_from_proba(yp, perm_base, perm_model, labels)
            null_values.append(float(perm_metric["bits"]))
        h_bits = entropy_bits(y_arr)
        return {
            "task": "classification",
            "mode": mode,
            "n": int(len(y_arr)),
            "n_groups": int(len(np.unique(groups_arr))),
            "n_classes": int(len(labels)),
            "class_counts": _class_counts(y_arr, labels),
            "target_entropy_bits": _round_float(h_bits),
            "R": _json_metric(metric),
            "permutation_null": _json_metric(_permutation_summary(observed, np.asarray(null_values), h_bits)),
            "probe": {
                "model": (
                    "StandardScaler + PCA if needed + LogisticRegression; conditional baseline is fit "
                    "as baseline-only versus baseline+features"
                ),
                "splitter": meta["splitter"],
                "split_policy": "fixed observed-label split plan reused for all permutation draws",
            },
            "folds": meta.get("folds") or meta,
        }

    plan = _RegressionProbePlan(
        X_arr,
        y_arr,
        groups_arr,
        baseline_features=baseline_arr,
        n_splits=n_splits,
        seed=seed,
    )
    model_pred, baseline_pred, meta = plan.fit(y_arr)
    metric = _incremental_r2(y_arr, baseline_pred, model_pred)
    observed = float(metric["incremental_r2"])
    null_values = []
    rng = np.random.default_rng(int(seed) + 300)
    for _ in range(int(n_permutations)):
        yp = permute_target_cluster_aware(y_arr, groups_arr, rng).astype(float)
        perm_model, perm_base, _ = plan.fit(yp)
        perm_metric = _incremental_r2(yp, perm_base, perm_model)
        null_values.append(float(perm_metric["incremental_r2"]))
    return {
        "task": "regression",
        "mode": "conditional" if baseline_arr is not None else "marginal",
        "n": int(len(y_arr)),
        "n_groups": int(len(np.unique(groups_arr))),
        "R": _json_metric(metric),
        "permutation_null": _json_metric(_permutation_summary(observed, np.asarray(null_values), None)),
        "probe": {
            "model": "StandardScaler + LinearRegression; statistic is incremental held-out R2",
            "splitter": meta["splitter"],
            "split_policy": "fixed observed-target split plan reused for all permutation draws",
        },
        "folds": meta["folds"],
    }
