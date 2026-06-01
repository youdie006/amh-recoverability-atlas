"""Command line interface for the reserve audit harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reserve_audit.recoverability import recoverability
from reserve_audit.sufficiency import model_class_sufficiency


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _parse_columns(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV file containing features, target, and optional baseline/group columns")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--features", default=None, help="Comma-separated feature columns; default uses all non-target/baseline/group columns")
    parser.add_argument("--baseline", default=None, help="Comma-separated baseline feature columns for conditional recoverability")
    parser.add_argument("--groups", default=None, help="Optional cluster/group column for split and permutation exchangeability")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--sufficiency", action="store_true", help="Also run simple-vs-complex model-class sufficiency")
    parser.add_argument("--sufficiency-margin", type=float, default=0.02)
    parser.add_argument("--sufficiency-bootstrap", type=int, default=2000)
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.data)
    baseline_cols = _parse_columns(args.baseline)
    feature_cols = _parse_columns(args.features)
    if not feature_cols:
        excluded = {args.target, *baseline_cols}
        if args.groups:
            excluded.add(args.groups)
        feature_cols = [col for col in df.columns if col not in excluded]
    missing = [col for col in [args.target, *feature_cols, *baseline_cols] if col not in df.columns]
    if args.groups and args.groups not in df.columns:
        missing.append(args.groups)
    if missing:
        raise ValueError(f"missing columns in {args.data}: {missing}")

    cols = [args.target, *feature_cols, *baseline_cols]
    if args.groups:
        cols.append(args.groups)
    work = df.loc[:, cols].copy()
    for col in [*feature_cols, *baseline_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if args.task == "regression":
        work[args.target] = pd.to_numeric(work[args.target], errors="coerce")
    work = work.dropna(subset=[args.target, *feature_cols, *baseline_cols]).reset_index(drop=True)

    X = work[feature_cols].to_numpy(dtype=float)
    y = work[args.target].to_numpy(dtype=float) if args.task == "regression" else work[args.target].to_numpy()
    baseline = work[baseline_cols].to_numpy(dtype=float) if baseline_cols else None
    groups = work[args.groups].astype(str).to_numpy() if args.groups else None

    payload = recoverability(
        X,
        y,
        baseline=baseline,
        task=args.task,
        n_splits=args.n_splits,
        seed=args.seed,
        groups=groups,
        n_permutations=args.n_permutations,
        pca_components=args.pca_components,
    )
    payload["input"] = {
        "data": str(args.data),
        "target": args.target,
        "features": feature_cols,
        "baseline": baseline_cols,
        "groups": args.groups,
        "dropped_rows": int(len(df) - len(work)),
    }
    if args.sufficiency:
        suff_X = X if baseline is None else np.column_stack([baseline, X])
        payload["sufficiency"] = model_class_sufficiency(
            suff_X,
            y,
            task=args.task,
            n_splits=args.n_splits,
            seed=args.seed,
            margin=args.sufficiency_margin,
            n_bootstrap=args.sufficiency_bootstrap,
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_cli(args)
    print(json.dumps(_json_safe(payload), sort_keys=True))


if __name__ == "__main__":
    main()
