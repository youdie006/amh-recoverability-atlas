"""Age-adjusted quantity-vs-quality contrast utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


EPS = 1e-12


@dataclass(frozen=True)
class CorrelationEstimate:
    r: float
    z: float
    se_z: float
    ci_low: float
    ci_high: float
    n: int
    n_controls: int
    p: float | None


def rank_for_spearman(x: np.ndarray | pd.Series) -> np.ndarray:
    return stats.rankdata(np.asarray(x, dtype=float), method="average")


def correlation_ci_from_fisher(r: float, n: int, n_controls: int = 1) -> CorrelationEstimate:
    if n <= n_controls + 3:
        raise ValueError(f"n={n} is too small for Fisher CI with {n_controls} controls")
    r = float(np.clip(r, -0.999999, 0.999999))
    z = float(np.arctanh(r))
    se_z = float(1.0 / math.sqrt(n - n_controls - 3))
    ci_low = float(np.tanh(z - 1.96 * se_z))
    ci_high = float(np.tanh(z + 1.96 * se_z))
    df = n - n_controls - 2
    if abs(r) >= 0.999999:
        p = 0.0
    else:
        t_stat = r * math.sqrt(df / max(1.0 - r * r, EPS))
        p = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return CorrelationEstimate(r=r, z=z, se_z=se_z, ci_low=ci_low, ci_high=ci_high, n=int(n), n_controls=n_controls, p=p)


def partial_spearman(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    age: np.ndarray | pd.Series,
) -> CorrelationEstimate:
    """Age-adjusted partial Spearman via Pearson correlation of rank residuals."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    age_arr = np.asarray(age, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(age_arr)
    n = int(mask.sum())
    n_controls = 1
    if n <= n_controls + 3:
        raise ValueError("too few complete cases for partial Spearman")
    rx = rank_for_spearman(x_arr[mask])
    ry = rank_for_spearman(y_arr[mask])
    rage = rank_for_spearman(age_arr[mask])
    design = np.column_stack([np.ones(n), rage])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) < EPS or np.std(ey) < EPS:
        raise ValueError("zero residual variance after age adjustment")
    r = float(np.corrcoef(ex, ey)[0, 1])
    return correlation_ci_from_fisher(r, n=n, n_controls=n_controls)


def reml_tau2(y: np.ndarray, se: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    var = np.maximum(np.asarray(se, dtype=float) ** 2, EPS)
    if len(y) <= 1:
        return 0.0

    def objective(tau2: float) -> float:
        v = var + max(float(tau2), 0.0)
        w = 1.0 / v
        mu = float(np.sum(w * y) / np.sum(w))
        resid = y - mu
        return 0.5 * (np.sum(np.log(v)) + math.log(np.sum(w)) + np.sum(w * resid * resid))

    upper = max(1.0, float(np.var(y, ddof=1) * 20.0 + np.max(var) * 20.0))
    opt = optimize.minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    tau2 = float(opt.x) if opt.success else 0.0
    if objective(0.0) <= objective(tau2) + 1e-10:
        return 0.0
    return max(0.0, tau2)


def reml_hksj_pool(effects: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool Fisher-z effects with REML tau2 and Hartung-Knapp-Sidik-Jonkman SE."""

    usable = [
        {
            "label": str(e.get("label", f"effect_{i}")),
            "z": float(e["z"]),
            "se_z": max(float(e["se_z"]), 1e-6),
            **{k: v for k, v in e.items() if k not in {"label", "z", "se_z"}},
        }
        for i, e in enumerate(effects)
        if np.isfinite(float(e.get("z", np.nan))) and np.isfinite(float(e.get("se_z", np.nan)))
    ]
    if len(usable) < 3:
        return {"status": "not_pooled", "reason": "k<3", "k": int(len(usable)), "effects": usable}
    y = np.array([e["z"] for e in usable], dtype=float)
    se = np.array([e["se_z"] for e in usable], dtype=float)
    var = se * se
    tau2 = reml_tau2(y, se)
    w = 1.0 / (var + tau2)
    pooled_z = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - pooled_z) ** 2))
    df = len(usable) - 1
    hksj_scale = float(max(q / df, EPS))
    se_hksj = float(math.sqrt(hksj_scale / np.sum(w)))
    crit = float(stats.t.ppf(0.975, df=df))
    ci_z = [pooled_z - crit * se_hksj, pooled_z + crit * se_hksj]
    t_stat = pooled_z / se_hksj if se_hksj > 0 else math.nan
    p = float(2.0 * stats.t.sf(abs(t_stat), df=df)) if math.isfinite(t_stat) else None
    fixed_w = 1.0 / var
    fixed_z = float(np.sum(fixed_w * y) / np.sum(fixed_w))
    q_fixed = float(np.sum(fixed_w * (y - fixed_z) ** 2))
    i2 = float(max(0.0, (q_fixed - df) / q_fixed) * 100.0) if q_fixed > 0 else 0.0
    return {
        "status": "ok",
        "method": "REML_HKSJ",
        "k": int(len(usable)),
        "z": pooled_z,
        "se_z_hksj": se_hksj,
        "ci95_z": ci_z,
        "r": float(np.tanh(pooled_z)),
        "r_ci95": [float(np.tanh(ci_z[0])), float(np.tanh(ci_z[1]))],
        "p_hksj": p,
        "tau2_reml": tau2,
        "hksj_scale": hksj_scale,
        "q_fixed": q_fixed,
        "i2_fixed_q": i2,
        "effects": usable,
    }


def _endpoint_zs(
    frame: pd.DataFrame,
    marker: str,
    endpoints: list[str],
    age: str,
) -> tuple[list[float], list[CorrelationEstimate]]:
    zs: list[float] = []
    estimates: list[CorrelationEstimate] = []
    for endpoint in endpoints:
        est = partial_spearman(frame[marker], frame[endpoint], frame[age])
        zs.append(est.z)
        estimates.append(est)
    return zs, estimates


def quantity_quality_contrast(
    frame: pd.DataFrame,
    marker: str,
    age: str,
    quantity_endpoints: list[str],
    quality_endpoints: list[str],
    n_bootstrap: int = 1000,
    seed: int = 20260530,
) -> dict[str, Any]:
    """Contrast age-adjusted marker associations with quantity and quality endpoints."""

    q_zs, q_est = _endpoint_zs(frame, marker, quantity_endpoints, age)
    qual_zs, qual_est = _endpoint_zs(frame, marker, quality_endpoints, age)
    quantity_z = float(np.mean(q_zs))
    quality_z = float(np.mean(qual_zs))
    gap_z = quantity_z - quality_z
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    failed = 0
    for _ in range(int(n_bootstrap)):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))].reset_index(drop=True)
        try:
            q_boot, _ = _endpoint_zs(sample, marker, quantity_endpoints, age)
            qual_boot, _ = _endpoint_zs(sample, marker, quality_endpoints, age)
        except ValueError:
            failed += 1
            continue
        if all(np.isfinite(q_boot)) and all(np.isfinite(qual_boot)):
            boot.append(float(np.mean(q_boot) - np.mean(qual_boot)))
        else:
            failed += 1
    boot_arr = np.asarray(boot, dtype=float)
    if len(boot_arr) >= max(30, int(n_bootstrap) // 4) and np.std(boot_arr, ddof=1) > EPS:
        se_z = float(np.std(boot_arr, ddof=1))
        ci_z = [float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))]
        ci_method = "row_bootstrap_percentile"
    else:
        endpoint_ses = np.array([e.se_z for e in q_est + qual_est], dtype=float)
        se_z = float(math.sqrt(np.sum(endpoint_ses**2)) / len(endpoint_ses))
        ci_z = [gap_z - 1.96 * se_z, gap_z + 1.96 * se_z]
        ci_method = "fisher_endpoint_average_fallback"
    return {
        "marker": marker,
        "age": age,
        "contrast": "quantity_minus_quality",
        "quantity_z": quantity_z,
        "quality_z": quality_z,
        "gap_z": gap_z,
        "gap_r_approx": float(np.tanh(quantity_z) - np.tanh(quality_z)),
        "se_z": max(se_z, 1e-6),
        "ci95_z": ci_z,
        "ci_method": ci_method,
        "quantity_endpoint_names": quantity_endpoints,
        "quality_endpoint_names": quality_endpoints,
        "bootstrap_replicates": int(len(boot_arr)),
        "bootstrap_failed_replicates": int(failed),
    }
