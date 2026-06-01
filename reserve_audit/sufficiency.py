"""Model-class sufficiency audit with TOST equivalence."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


Task = Literal["classification", "regression"]
EPS = 1e-15


def percentile_ci(values: np.ndarray, level: float = 0.95) -> list[float | None]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [None, None]
    lo = (1.0 - float(level)) / 2.0
    hi = 1.0 - lo
    return [float(np.quantile(vals, lo)), float(np.quantile(vals, hi))]


def tost_equivalence(differences: np.ndarray, margin: float) -> dict[str, Any]:
    """Bootstrap TOST-style equivalence check.

    ``differences`` should be paired held-out score differences, complex minus
    simple. Negative values favor the complex model.
    """

    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    ci90 = percentile_ci(diffs, 0.90)
    ci95 = percentile_ci(diffs, 0.95)
    lower_ok = ci90[0] is not None and ci90[0] > -float(margin)
    upper_ok = ci90[1] is not None and ci90[1] < float(margin)
    equivalent = bool(lower_ok and upper_ok)
    return {
        "margin": float(margin),
        "difference_definition": "complex_score_minus_simple_score; negative favors complex",
        "ci90": ci90,
        "ci95": ci95,
        "equivalent_by_90ci": equivalent,
        "p_greater_than_negative_margin_bootstrap": (
            float((np.sum(diffs <= -float(margin)) + 1) / (len(diffs) + 1)) if len(diffs) else None
        ),
        "p_less_than_positive_margin_bootstrap": (
            float((np.sum(diffs >= float(margin)) + 1) / (len(diffs) + 1)) if len(diffs) else None
        ),
        "statement": (
            f"Equivalent within +/-{float(margin):.6g} by bootstrap 90% CI."
            if equivalent
            else f"Not equivalent within +/-{float(margin):.6g} by bootstrap 90% CI."
        ),
    }


def _finite_feature_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D numeric array")
    return arr


def _classification_loss_vector(y: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    pos = {label: i for i, label in enumerate(labels.tolist())}
    idx = np.array([pos[value] for value in y], dtype=int)
    return -np.log(proba[np.arange(len(y)), idx])


def _bootstrap_means(values: np.ndarray, n_bootstrap: int, seed: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, len(vals), size=len(vals))
        boot[i] = float(np.mean(vals[idx]))
    return boot


def _align_proba(proba: np.ndarray, model_classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.full((proba.shape[0], len(labels)), EPS, dtype=float)
    pos = {label: i for i, label in enumerate(labels.tolist())}
    for j, cls in enumerate(np.asarray(model_classes).tolist()):
        if cls in pos:
            out[:, pos[cls]] = proba[:, j]
    out = np.clip(out, EPS, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _classification_oof(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    labels = np.unique(y)
    simple = np.full((len(y), len(labels)), np.nan, dtype=float)
    complex_ = np.full_like(simple, np.nan)
    folds: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        simple_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=int(seed + fold)),
        )
        complex_model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.01,
            early_stopping=False,
            random_state=int(seed + 1000 + fold),
        )
        simple_model.fit(X[train_idx], y[train_idx])
        complex_model.fit(X[train_idx], y[train_idx])
        simple[test_idx] = _align_proba(simple_model.predict_proba(X[test_idx]), simple_model[-1].classes_, labels)
        complex_[test_idx] = _align_proba(complex_model.predict_proba(X[test_idx]), complex_model.classes_, labels)
        folds.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "simple_log_loss": float(log_loss(y[test_idx], simple[test_idx], labels=labels)),
                "complex_log_loss": float(log_loss(y[test_idx], complex_[test_idx], labels=labels)),
            }
        )
    return simple, complex_, folds


def _regression_oof(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    simple = np.full(len(y), np.nan, dtype=float)
    complex_ = np.full(len(y), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        simple_model = make_pipeline(StandardScaler(), LinearRegression())
        complex_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.01,
            random_state=int(seed + 1000 + fold),
        )
        simple_model.fit(X[train_idx], y[train_idx])
        complex_model.fit(X[train_idx], y[train_idx])
        simple[test_idx] = simple_model.predict(X[test_idx])
        complex_[test_idx] = complex_model.predict(X[test_idx])
        folds.append({"fold": int(fold), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
    return simple, complex_, folds


def model_class_sufficiency(
    X: np.ndarray,
    y: np.ndarray,
    task: Task = "classification",
    n_splits: int = 5,
    seed: int = 20260530,
    margin: float = 0.02,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Compare simple and complex model classes by paired held-out scores."""

    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    X_arr = _finite_feature_matrix(X)
    if len(X_arr) != len(y):
        raise ValueError("X rows must match y length")

    if task == "classification":
        y_arr = np.asarray(y)
        labels = np.unique(y_arr)
        simple_proba, complex_proba, folds = _classification_oof(X_arr, y_arr, n_splits, seed)
        simple_losses = _classification_loss_vector(y_arr, simple_proba, labels)
        complex_losses = _classification_loss_vector(y_arr, complex_proba, labels)
        score_name = "log_loss_nats"
    else:
        y_arr = np.asarray(y, dtype=float)
        simple_pred, complex_pred, folds = _regression_oof(X_arr, y_arr, n_splits, seed)
        simple_losses = (y_arr - simple_pred) ** 2
        complex_losses = (y_arr - complex_pred) ** 2
        score_name = "squared_error"

    diffs = complex_losses - simple_losses
    boot = _bootstrap_means(diffs, n_bootstrap=n_bootstrap, seed=seed + 401)
    return {
        "task": task,
        "n": int(len(y_arr)),
        "score": score_name,
        "simple_model": "standardized LogisticRegression" if task == "classification" else "standardized LinearRegression",
        "complex_model": "HistGradientBoostingClassifier" if task == "classification" else "HistGradientBoostingRegressor",
        "simple_score_mean": float(np.mean(simple_losses)),
        "complex_score_mean": float(np.mean(complex_losses)),
        "difference_complex_minus_simple": {
            "point": float(np.mean(diffs)),
            "ci95": percentile_ci(boot, 0.95),
            "bootstrap_n": int(len(boot)),
            "bootstrap_method": "paired row bootstrap of held-out score differences; models are not refit inside bootstrap",
        },
        "tost_equivalence": tost_equivalence(boot, margin=margin),
        "folds": folds,
    }
