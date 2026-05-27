#!/usr/bin/env python3
"""
Run the paired AMH counterfactual contrast for joint FM and diffusion NLLs.

For each non-blank-AMH test row, compare the true joint NLL against the mean
joint NLL after replacing only the AMH scalar with AMH values sampled from
other test rows. Positive delta means the true AMH has lower NLL than
counterfactual AMH values.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

import compute_diffusion_pf_ode_nll as diffusion_nll
import compute_flow_matching_nll as flow_nll
from train_diffusion import PARAMETERIZATION


SEED = 10_123
EXPECTED_FEATURE_DIM = 768
EXPECTED_JOINT_DIM = 769
EXPECTED_EMBEDDING_ROWS = 754
EXPECTED_TEST_ROWS = 113
EXPECTED_AMH_NORMALIZATION_SHA256 = (
    "7acddc74a794ed3beed7f00c69d122d736c222bd74ed2fd4a3b9501a46cda5da"
)
EXPECTED_INPUT_SHA256 = {
    "embeddings": "66462f061a10a3e12b483f8b087a48b7113bd55bc8fa0646b6e88e982ee36bc3",
    "embeddings_index": "af85b66b68765423498aaddcd9cd5263835d2f63678ca23e09282d74df9d12fe",
    "test_csv": "b700fa7514724007a0a7aa7b730a16d14ef3aab5793223f5bf540f8ffcd2deb9",
    "amh_recovered_csv": "1cc757447d53c37e659b5c9d5ba8f9a77e23bf145e5af8386d22eaa4d85d67ed",
    "amh_normalization_json": EXPECTED_AMH_NORMALIZATION_SHA256,
}
EXPECTED_SHA256 = {
    "flow_matching": {
        "model_joint": "3d25ce4506239303e0da3166f0c8ad5af0e3c37cbc1c37f9b96cb7ddf72b8588",
        "training_log_joint": "fc515e4ce90270fddbb6ba2af1953f5c383a6eff23a034d899a7abf81fd4c417",
    },
    "diffusion": {
        "model_joint": "dbf16182258db9bc66d1e52445e343a1479b871b231e23f60efe44303bdb4b24",
        "training_log_joint": "f89964a7fe6c777ec0afffa2ecf65e4ceb223e4321b24d41ca908aab39b6cf52",
    },
}
BUCKETS = ("clean", "xx_mmm", "mmm_xx")


@dataclass(frozen=True)
class AmhRows:
    rows: pd.DataFrame
    z: np.ndarray
    amh_raw: np.ndarray
    amh_normalized: np.ndarray
    amh_mean: float
    amh_std: float


@dataclass(frozen=True)
class BackendSpec:
    key: str
    label: str
    model_path: Path


@dataclass
class BackendRun:
    summary: dict[str, Any]
    finite_permutation_evals: int
    finite_true_evals: int
    deltas: np.ndarray


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def library_versions() -> dict[str, str]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torchdiffeq": package_version("torchdiffeq"),
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "transformers": package_version("transformers"),
        "diffusers": package_version("diffusers"),
    }


def git_short_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def git_dirty(repo_root: Path) -> bool | str:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], cwd=repo_root, check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False
        ).returncode
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return bool(unstaged or staged or untracked)
    except Exception:
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--device",
        default="cuda",
        help="Must be the literal value 'cuda'. CPU fallback is forbidden.",
    )
    parser.add_argument(
        "--embeddings",
        default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/dinov2_z.npy",
        help="Path to DINOv2 embedding matrix.",
    )
    parser.add_argument(
        "--embeddings-index",
        default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/index.csv",
        help="Path to embedding index CSV.",
    )
    parser.add_argument(
        "--splits-dir",
        default="code/repos/kromp-blastocyst-dataset-audit/data/splits",
        help="Directory containing test.csv.",
    )
    parser.add_argument(
        "--amh-recovered",
        default="code/repos/kromp-blastocyst-dataset-audit/data/amh_recovered.csv",
        help="Recovered AMH CSV used to validate and source AMH_hypA.",
    )
    parser.add_argument(
        "--amh-normalization",
        default="results/baseline/amh_normalization.json",
        help="AMH normalization JSON from Day 2.",
    )
    parser.add_argument(
        "--fm-model",
        default="results/flow_matching/model_joint.pt",
        help="Flow-matching joint checkpoint.",
    )
    parser.add_argument(
        "--diff-model",
        default="results/diffusion/model_joint.pt",
        help="Diffusion joint checkpoint.",
    )
    parser.add_argument(
        "--backend",
        default="both",
        choices=("both", "flow_matching", "fm", "diffusion", "diff"),
        help="Backend to run.",
    )
    parser.add_argument(
        "--k-permutations",
        type=int,
        default=20,
        help="AMH permutations sampled with replacement per row.",
    )
    parser.add_argument(
        "--hutchinson-samples",
        type=int,
        default=5,
        help="Hutchinson probes per NLL evaluation.",
    )
    parser.add_argument(
        "--ode-method",
        default="dopri5",
        choices=("dopri5",),
        help="ODE method. Diffusion PF-ODE uses dopri5.",
    )
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--epsilon-clip",
        type=float,
        default=1e-3,
        help="Diffusion lower time bound.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Paired bootstrap iterations.",
    )
    parser.add_argument(
        "--out-json",
        default="results/diagnostics/delta_amh.json",
        help="Output diagnostic JSON.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--euler-steps",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    assert args.device == "cuda", "--device must be the literal value 'cuda'"
    assert args.seed == SEED, "seed must be 10123"
    assert args.k_permutations > 0, "--k-permutations must be positive"
    assert args.hutchinson_samples > 0, "--hutchinson-samples must be positive"
    assert args.ode_atol > 0, "--ode-atol must be positive"
    assert args.ode_rtol > 0, "--ode-rtol must be positive"
    assert math.isclose(args.epsilon_clip, 1e-3), "--epsilon-clip must be 1e-3"
    assert args.bootstrap_iters > 0, "--bootstrap-iters must be positive"
    return args


def resolve_cuda_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise RuntimeError("This diagnostic requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Stopping with no CPU fallback."
        )
    return torch.device("cuda")


def backend_specs(args: argparse.Namespace) -> list[BackendSpec]:
    fm = BackendSpec(
        key="flow_matching",
        label="flow_matching",
        model_path=Path(args.fm_model).resolve(),
    )
    diff = BackendSpec(
        key="diffusion",
        label="diffusion",
        model_path=Path(args.diff_model).resolve(),
    )
    if args.backend in ("flow_matching", "fm"):
        return [fm]
    if args.backend in ("diffusion", "diff"):
        return [diff]
    return [fm, diff]


def assert_no_surprise_object_columns(df: pd.DataFrame, path: Path) -> None:
    allowed = {"Image", "prefix", "AMH", "AMH_raw", "AMH_bucket"}
    surprise = sorted(set(df.select_dtypes(include=["object"]).columns) - allowed)
    assert not surprise, f"unexpected object columns in {path}: {surprise}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict), f"{path} did not contain a JSON object"
    return payload


def assert_expected_sha256(label: str, path: Path) -> str:
    actual = sha256_of(path)
    expected = EXPECTED_INPUT_SHA256[label]
    assert actual == expected, f"{label} SHA mismatch: {actual} != {expected}"
    return actual


def load_amh_normalization(path: Path) -> dict[str, Any]:
    actual_sha = assert_expected_sha256("amh_normalization_json", path)
    assert actual_sha == EXPECTED_AMH_NORMALIZATION_SHA256, (
        f"AMH normalization SHA mismatch: {actual_sha}"
    )
    payload = load_json(path)
    assert payload.get("scope") == "computed on train.csv rows with AMH_bucket != 'blank'"
    assert int(payload.get("n_used", 0)) > 0, "AMH normalization n_used invalid"
    assert np.isfinite(float(payload["mean"])), "AMH normalization mean is non-finite"
    assert float(payload["std"]) > 0, "AMH normalization std must be positive"
    assert int(payload.get("std_ddof", 0)) == 0, "AMH normalization must use population std"
    assert int(payload.get("seed", SEED)) == SEED, "AMH normalization seed mismatch"
    return payload


def load_embeddings(
    embeddings_path: Path,
    embeddings_index_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    assert_expected_sha256("embeddings", embeddings_path)
    assert_expected_sha256("embeddings_index", embeddings_index_path)
    features = np.load(embeddings_path)
    assert features.shape == (EXPECTED_EMBEDDING_ROWS, EXPECTED_FEATURE_DIM), (
        f"embedding shape: {features.shape}"
    )
    assert features.dtype == np.float32, f"embedding dtype: {features.dtype}"
    assert np.isfinite(features).all(), "embeddings contain non-finite values"
    index = pd.read_csv(
        embeddings_index_path,
        dtype={"Image": str, "prefix": str},
        keep_default_na=True,
    )
    assert len(index) == EXPECTED_EMBEDDING_ROWS, f"index rows: {len(index)}"
    assert index["Image"].is_unique, "embedding index Image values are not unique"
    assert not index[["Image", "prefix"]].isna().any().any(), "index has missing values"
    assert_no_surprise_object_columns(index, embeddings_index_path)
    index = index.copy()
    index["_feature_row"] = np.arange(len(index), dtype=np.int64)
    return features, index


def load_test_split(path: Path) -> pd.DataFrame:
    assert_expected_sha256("test_csv", path)
    dtype = {
        "Image": str,
        "AMH": str,
        "AMH_bucket": str,
        "prefix": str,
        "HA": "Int64",
        "LB": "Int64",
    }
    df = pd.read_csv(path, dtype=dtype, keep_default_na=True)
    assert len(df) == EXPECTED_TEST_ROWS, f"test rows: {len(df)}"
    assert df["Image"].is_unique, "test Image values are not unique"
    assert not df["Image"].isna().any(), "test split has missing Image"
    for col in ("AMH_bucket", "AMH_hypA", "AMH_hypB", "prefix", "HA", "LB"):
        assert col in df.columns, f"{col} missing in {path}"
    for col in ("AMH_hypA", "AMH_hypB"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[
        ["Image", "AMH", "AMH_bucket", "AMH_hypA", "AMH_hypB", "prefix", "HA", "LB"]
    ].copy()
    assert_no_surprise_object_columns(df, path)
    return df


def load_amh_recovered(path: Path) -> pd.DataFrame:
    assert_expected_sha256("amh_recovered_csv", path)
    dtype = {
        "Image": str,
        "AMH_raw": str,
        "AMH_bucket": str,
        "is_palindrome": "boolean",
    }
    df = pd.read_csv(path, dtype=dtype, keep_default_na=True)
    assert len(df) == EXPECTED_EMBEDDING_ROWS, f"AMH recovered rows: {len(df)}"
    assert df["Image"].is_unique, "AMH recovered Image values are not unique"
    for col in ("AMH_hypA", "AMH_hypB"):
        assert col in df.columns, f"{col} missing in {path}"
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[["Image", "AMH_raw", "AMH_bucket", "AMH_hypA", "AMH_hypB"]].copy()
    assert_no_surprise_object_columns(df, path)
    return df


def load_amh_rows(
    embeddings_path: Path,
    embeddings_index_path: Path,
    test_split_path: Path,
    amh_recovered_path: Path,
    amh_normalization_path: Path,
) -> AmhRows:
    embeddings, embeddings_index = load_embeddings(embeddings_path, embeddings_index_path)
    split = load_test_split(test_split_path)
    recovered = load_amh_recovered(amh_recovered_path)
    norm = load_amh_normalization(amh_normalization_path)

    split = split.copy()
    split["_split_order"] = np.arange(len(split), dtype=np.int64)
    merged = split.merge(
        recovered,
        on="Image",
        how="left",
        validate="one_to_one",
        suffixes=("_split", "_recovered"),
        sort=False,
    )
    assert len(merged) == len(split), "AMH recovered merge changed test row count"
    assert not merged["AMH_bucket_recovered"].isna().any(), (
        "test row missing in amh_recovered.csv"
    )
    assert (
        merged["AMH_bucket_split"].astype(str)
        == merged["AMH_bucket_recovered"].astype(str)
    ).all(), "AMH_bucket mismatch between test split and amh_recovered.csv"
    for col in ("AMH_hypA", "AMH_hypB"):
        split_col = f"{col}_split"
        recovered_col = f"{col}_recovered"
        split_vals = merged[split_col].to_numpy(dtype=np.float64)
        recovered_vals = merged[recovered_col].to_numpy(dtype=np.float64)
        equal_or_both_nan = (split_vals == recovered_vals) | (
            np.isnan(split_vals) & np.isnan(recovered_vals)
        )
        assert equal_or_both_nan.all(), (
            f"{col} mismatch between test split and amh_recovered.csv"
        )

    merged = merged.merge(
        embeddings_index[["Image", "prefix", "_feature_row"]],
        on="Image",
        how="left",
        validate="one_to_one",
        suffixes=("", "_embedding"),
        sort=False,
    )
    assert not merged["_feature_row"].isna().any(), "test Images missing embeddings"
    prefix_mismatch = (
        merged["prefix"].astype(str).to_numpy()
        != merged["prefix_embedding"].astype(str).to_numpy()
    )
    assert not prefix_mismatch.any(), (
        f"prefix mismatch for {merged.loc[prefix_mismatch, 'Image'].iloc[0]}"
    )
    merged = merged.sort_values("_split_order").reset_index(drop=True)
    feature_rows = merged["_feature_row"].astype(np.int64).to_numpy()
    z_all = embeddings[feature_rows].astype(np.float32, copy=True)
    assert z_all.shape == (EXPECTED_TEST_ROWS, EXPECTED_FEATURE_DIM), z_all.shape
    assert np.isfinite(z_all).all(), "aligned test embeddings contain non-finite values"

    bucket = merged["AMH_bucket_split"].astype(str)
    nonblank = bucket.ne("blank").to_numpy()
    amh_raw_all = merged["AMH_hypA_recovered"].to_numpy(dtype=np.float64)
    assert np.isfinite(amh_raw_all[nonblank]).all(), "nonblank raw AMH is non-finite"
    z = z_all[nonblank].astype(np.float32, copy=True)
    amh_raw = amh_raw_all[nonblank].astype(np.float64, copy=True)
    amh_mean = float(norm["mean"])
    amh_std = float(norm["std"])
    amh_normalized = ((amh_raw - amh_mean) / amh_std).astype(np.float32)
    assert np.isfinite(amh_normalized).all(), "standardized AMH is non-finite"

    rows = pd.DataFrame(
        {
            "Image": merged.loc[nonblank, "Image"].astype(str).to_numpy(),
            "split": "test",
            "AMH_bucket": bucket.loc[nonblank].astype(str).to_numpy(),
            "AMH_hypA": amh_raw,
            "AMH_normalized": amh_normalized.astype(np.float32),
            "HA": merged.loc[nonblank, "HA"].astype("Int64").to_numpy(),
            "LB": merged.loc[nonblank, "LB"].astype("Int64").to_numpy(),
            "prefix": merged.loc[nonblank, "prefix"].astype(str).to_numpy(),
        }
    )
    assert len(rows) >= 80, f"nonblank-AMH test rows: {len(rows)}"
    assert len(rows) == len(z) == len(amh_raw) == len(amh_normalized)
    assert set(rows["AMH_bucket"].unique()).issubset(set(BUCKETS)), (
        f"unexpected AMH buckets: {sorted(rows['AMH_bucket'].unique())}"
    )
    return AmhRows(
        rows=rows.reset_index(drop=True),
        z=z,
        amh_raw=amh_raw,
        amh_normalized=amh_normalized,
        amh_mean=amh_mean,
        amh_std=amh_std,
    )


def make_joint_state(z: np.ndarray, amh_normalized: float) -> np.ndarray:
    assert z.shape == (EXPECTED_FEATURE_DIM,), z.shape
    out = np.empty((EXPECTED_JOINT_DIM,), dtype=np.float32)
    out[:EXPECTED_FEATURE_DIM] = z.astype(np.float32, copy=False)
    out[EXPECTED_FEATURE_DIM] = np.float32(amh_normalized)
    assert np.isfinite(out).all(), "joint state has non-finite values"
    return out


def verify_flow_artifacts(model_path: Path) -> dict[str, Any]:
    assert model_path.name == "model_joint.pt", f"expected model_joint.pt, got {model_path}"
    log_path = model_path.parent / "training_log_joint.json"
    assert model_path.exists(), f"missing flow-matching checkpoint: {model_path}"
    assert log_path.exists(), f"missing flow-matching training log: {log_path}"
    model_sha = sha256_of(model_path)
    log_sha = sha256_of(log_path)
    assert model_sha == EXPECTED_SHA256["flow_matching"]["model_joint"], (
        f"flow_matching model_joint SHA mismatch: {model_sha}"
    )
    assert log_sha == EXPECTED_SHA256["flow_matching"]["training_log_joint"], (
        f"flow_matching training_log_joint SHA mismatch: {log_sha}"
    )
    log_payload = load_json(log_path)
    config = log_payload.get("config", {})
    assert log_payload.get("device") == "cuda", "flow_matching provenance device != cuda"
    assert log_payload.get("cuda_available") is True, (
        "flow_matching provenance cuda_available is not true"
    )
    assert log_payload.get("variant") == "joint", "flow_matching log variant mismatch"
    assert int(config.get("input_dim")) == EXPECTED_JOINT_DIM, (
        "flow_matching joint input_dim mismatch"
    )
    assert config.get("joint_amh_conditioning_source") == (
        "current_interpolated_state_last_dim"
    ), "flow_matching joint conditioning source mismatch"
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt_config = checkpoint["config"]
    assert ckpt_config.get("variant") == "joint", "flow_matching checkpoint variant mismatch"
    assert int(ckpt_config.get("input_dim")) == EXPECTED_JOINT_DIM
    assert int(ckpt_config.get("seed")) == SEED
    return {
        "model_sha256": model_sha,
        "training_log_sha256": log_sha,
        "device": log_payload.get("device"),
        "cuda_available": bool(log_payload.get("cuda_available")),
        "input_dim": EXPECTED_JOINT_DIM,
        "lock_source": "10_Wiki/Artifacts/Flow Matching Joint Day 4.md",
    }


def verify_diffusion_artifacts(model_path: Path) -> dict[str, Any]:
    assert model_path.name == "model_joint.pt", f"expected model_joint.pt, got {model_path}"
    log_path = model_path.parent / "training_log_joint.json"
    assert model_path.exists(), f"missing diffusion checkpoint: {model_path}"
    assert log_path.exists(), f"missing diffusion training log: {log_path}"
    model_sha = sha256_of(model_path)
    log_sha = sha256_of(log_path)
    assert model_sha == EXPECTED_SHA256["diffusion"]["model_joint"], (
        f"diffusion model_joint SHA mismatch: {model_sha}"
    )
    assert log_sha == EXPECTED_SHA256["diffusion"]["training_log_joint"], (
        f"diffusion training_log_joint SHA mismatch: {log_sha}"
    )
    log_payload = load_json(log_path)
    config = log_payload.get("config", {})
    assert log_payload.get("device") == "cuda", "diffusion provenance device != cuda"
    assert log_payload.get("cuda_available") is True, (
        "diffusion provenance cuda_available is not true"
    )
    assert log_payload.get("parameterization") == PARAMETERIZATION, (
        "diffusion log parameterization mismatch"
    )
    assert config.get("parameterization") == PARAMETERIZATION, (
        "diffusion config parameterization mismatch"
    )
    assert config.get("loss_target") == "epsilon_noise"
    assert config.get("score_recovery") == "-epsilon_hat / sigma(t)"
    assert config.get("noise_schedule") == "vp"
    assert int(config.get("input_dim")) == EXPECTED_JOINT_DIM
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt_config = checkpoint["config"]
    assert checkpoint.get("parameterization") == PARAMETERIZATION, (
        "diffusion checkpoint parameterization mismatch"
    )
    assert ckpt_config.get("parameterization") == PARAMETERIZATION
    assert ckpt_config.get("noise_schedule") == "vp"
    assert ckpt_config.get("score_recovery") == "-epsilon_hat / sigma(t)"
    assert ckpt_config.get("variant") == "joint"
    assert int(ckpt_config.get("input_dim")) == EXPECTED_JOINT_DIM
    assert int(ckpt_config.get("seed")) == SEED
    return {
        "model_sha256": model_sha,
        "training_log_sha256": log_sha,
        "device": log_payload.get("device"),
        "cuda_available": bool(log_payload.get("cuda_available")),
        "parameterization": log_payload.get("parameterization"),
        "input_dim": EXPECTED_JOINT_DIM,
        "lock_source": "10_Wiki/Artifacts/Diffusion PF-ODE NLL Day 6.md",
    }


def load_backend_model(spec: BackendSpec, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if spec.key == "flow_matching":
        provenance = verify_flow_artifacts(spec.model_path)
        model = flow_nll.load_flow_model(spec.model_path.parent, "joint", device)
        return model, provenance
    if spec.key == "diffusion":
        provenance = verify_diffusion_artifacts(spec.model_path)
        model = diffusion_nll.load_diffusion_model(spec.model_path.parent, "joint", device)
        return model, provenance
    raise AssertionError(f"unknown backend: {spec.key}")


def compute_backend_nll(
    spec: BackendSpec,
    model: torch.nn.Module,
    x_np: np.ndarray,
    generator: torch.Generator,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    if spec.key == "flow_matching":
        result = flow_nll.compute_variant_nll(
            model=model,
            variant="joint",
            x_np=x_np,
            generator=generator,
            device=device,
            method=args.ode_method,
            atol=args.ode_atol,
            rtol=args.ode_rtol,
            euler_steps=args.euler_steps,
            hutchinson_samples=args.hutchinson_samples,
        )
    elif spec.key == "diffusion":
        result = diffusion_nll.compute_variant_nll(
            model=model,
            variant="joint",
            x_np=x_np,
            generator=generator,
            device=device,
            atol=args.ode_atol,
            rtol=args.ode_rtol,
            epsilon_clip=args.epsilon_clip,
            hutchinson_samples=args.hutchinson_samples,
        )
    else:
        raise AssertionError(f"unknown backend: {spec.key}")
    value = float(result.nll_mean)
    assert math.isfinite(value), f"{spec.key} NLL is non-finite"
    return value


def make_permutation_indices(n_rows: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, k), dtype=np.int64)
    all_indices = np.arange(n_rows, dtype=np.int64)
    for row_idx in range(n_rows):
        candidates = all_indices[all_indices != row_idx]
        assert len(candidates) == n_rows - 1
        out[row_idx] = rng.choice(candidates, size=k, replace=True)
    assert out.shape == (n_rows, k)
    assert not (out == np.arange(n_rows, dtype=np.int64)[:, None]).any()
    return out


def bootstrap_mean(
    values: np.ndarray,
    bootstrap_iters: int,
    seed: int,
) -> tuple[list[float], float]:
    arr = np.asarray(values, dtype=np.float64)
    assert arr.ndim == 1 and len(arr) > 0
    assert np.isfinite(arr).all(), "bootstrap values contain non-finite entries"
    rng = np.random.default_rng(seed)
    means = np.empty((bootstrap_iters,), dtype=np.float64)
    n = len(arr)
    for idx in range(bootstrap_iters):
        sample_indices = rng.integers(0, n, size=n)
        means[idx] = arr[sample_indices].mean()
    ci = np.percentile(means, [2.5, 97.5])
    p_value = float(np.mean(means <= 0.0))
    return [float(ci[0]), float(ci[1])], p_value


def bucket_summary(
    deltas: np.ndarray,
    buckets: np.ndarray,
    bucket: str,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)[buckets == bucket]
    assert len(values) > 0, f"no rows for AMH_bucket={bucket}"
    ci, _ = bootstrap_mean(values, bootstrap_iters, seed)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": ci,
    }


def aggregate_backend_result(
    rows: AmhRows,
    nll_true: np.ndarray,
    nll_perm: np.ndarray,
    bootstrap_iters: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    assert nll_true.shape == (len(rows.rows),)
    assert nll_perm.ndim == 2 and nll_perm.shape[0] == len(rows.rows)
    assert np.isfinite(nll_true).all(), "true NLL contains non-finite values"
    assert np.isfinite(nll_perm).all(), "permutation NLL contains non-finite values"
    perm_mean = nll_perm.mean(axis=1)
    perm_std = (
        nll_perm.std(axis=1, ddof=1)
        if nll_perm.shape[1] > 1
        else np.zeros((len(rows.rows),), dtype=np.float64)
    )
    perm_se = (
        perm_std / math.sqrt(nll_perm.shape[1])
        if nll_perm.shape[1] > 1
        else np.zeros((len(rows.rows),), dtype=np.float64)
    )
    deltas = perm_mean - nll_true
    assert deltas.shape == (len(rows.rows),)
    assert np.isfinite(deltas).all(), "delta contains non-finite values"
    assert np.isfinite(perm_se).all(), "per-row permutation SE contains non-finite values"

    q25, q75 = np.percentile(deltas, [25, 75])
    ci, p_value = bootstrap_mean(deltas, bootstrap_iters, seed)
    bucket_arr = rows.rows["AMH_bucket"].astype(str).to_numpy()
    by_bucket = {
        bucket: bucket_summary(deltas, bucket_arr, bucket, bootstrap_iters, seed)
        for bucket in BUCKETS
    }
    per_row = []
    for idx, row in rows.rows.iterrows():
        per_row.append(
            {
                "Image": str(row["Image"]),
                "AMH_bucket": str(row["AMH_bucket"]),
                "AMH_hypA": float(rows.amh_raw[idx]),
                "AMH_normalized": float(rows.amh_normalized[idx]),
                "NLL_true": float(nll_true[idx]),
                "NLL_perm_mean": float(perm_mean[idx]),
                "NLL_perm_std": float(perm_std[idx]),
                "delta_i": float(deltas[idx]),
                "delta_i_std": float(perm_se[idx]),
            }
        )
    summary = {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "iqr_delta": [float(q25), float(q75)],
        "ci95_mean_delta_paired_bootstrap": ci,
        "one_sided_p_value": p_value,
        "by_bucket": by_bucket,
        "delta_i_std_definition": "std_k(NLL_perm_k_i) / sqrt(k_permutations)",
        "per_row": per_row,
    }
    return summary, deltas


def run_sign_sanity(
    spec: BackendSpec,
    model: torch.nn.Module,
    rows: AmhRows,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    z0 = rows.z[0]
    true_state = make_joint_state(z0, float(rows.amh_normalized[0]))
    perturbed_raw = float(rows.amh_raw[0] + 10.0 * rows.amh_std)
    perturbed_norm = float((perturbed_raw - rows.amh_mean) / rows.amh_std)
    perturbed_state = make_joint_state(z0, perturbed_norm)
    true_generator = torch.Generator(device=device).manual_seed(args.seed)
    perturbed_generator = torch.Generator(device=device).manual_seed(args.seed)
    true_nll = compute_backend_nll(
        spec=spec,
        model=model,
        x_np=true_state,
        generator=true_generator,
        device=device,
        args=args,
    )
    perturbed_nll = compute_backend_nll(
        spec=spec,
        model=model,
        x_np=perturbed_state,
        generator=perturbed_generator,
        device=device,
        args=args,
    )
    passed = bool(perturbed_nll > true_nll)
    assert passed, (
        f"{spec.key} sign-sanity failed: perturbed AMH NLL {perturbed_nll} "
        f"<= true AMH NLL {true_nll}"
    )
    return {
        "NLL_true_first_row": true_nll,
        "NLL_perturbed_first_row": perturbed_nll,
        "perturbed_raw_amh": perturbed_raw,
        "perturbed_normalized_amh": perturbed_norm,
        "passed": passed,
    }


def run_backend(
    spec: BackendSpec,
    model: torch.nn.Module,
    rows: AmhRows,
    permutation_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> BackendRun:
    n_rows, k = permutation_indices.shape
    assert n_rows == len(rows.rows)
    nll_true = np.empty((n_rows,), dtype=np.float64)
    nll_perm = np.empty((n_rows, k), dtype=np.float64)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    backend_start = time.perf_counter()

    for row_idx in range(n_rows):
        z_i = rows.z[row_idx]
        true_state = make_joint_state(z_i, float(rows.amh_normalized[row_idx]))
        nll_true[row_idx] = compute_backend_nll(
            spec=spec,
            model=model,
            x_np=true_state,
            generator=generator,
            device=device,
            args=args,
        )
        for perm_idx, source_idx in enumerate(permutation_indices[row_idx]):
            perm_state = make_joint_state(z_i, float(rows.amh_normalized[source_idx]))
            nll_perm[row_idx, perm_idx] = compute_backend_nll(
                spec=spec,
                model=model,
                x_np=perm_state,
                generator=generator,
                device=device,
                args=args,
            )
        assert np.isfinite(nll_true[row_idx]), f"{spec.key} true NLL non-finite"
        assert np.isfinite(nll_perm[row_idx]).all(), (
            f"{spec.key} permutation NLL non-finite at row {row_idx}"
        )
        if (row_idx + 1) % 5 == 0 or row_idx == n_rows - 1:
            print(
                "event=delta_progress "
                f"backend={spec.key} rows_done={row_idx + 1}/{n_rows} "
                f"nll_evals_done={(row_idx + 1) * (k + 1)} "
                f"elapsed_seconds={time.perf_counter() - backend_start:.3f}",
                flush=True,
            )
        if (row_idx + 1) % 25 == 0:
            torch.cuda.empty_cache()

    summary, deltas = aggregate_backend_result(
        rows=rows,
        nll_true=nll_true,
        nll_perm=nll_perm,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    return BackendRun(
        summary=summary,
        finite_permutation_evals=int(np.isfinite(nll_perm).sum()),
        finite_true_evals=int(np.isfinite(nll_true).sum()),
        deltas=deltas,
    )


def assert_acceptance(
    payload: dict[str, Any],
    backend_runs: dict[str, BackendRun],
    selected_backends: list[BackendSpec],
) -> None:
    n_rows = int(payload["n_rows"])
    k = int(payload["k_permutations"])
    assert n_rows >= 80, f"n_rows {n_rows} < 80"
    expected_perm_evals = n_rows * k * len(selected_backends)
    observed_perm_evals = sum(run.finite_permutation_evals for run in backend_runs.values())
    assert observed_perm_evals == expected_perm_evals, (
        f"finite permutation NLL evaluations {observed_perm_evals} != "
        f"{expected_perm_evals}"
    )
    expected_true_evals = n_rows * len(selected_backends)
    observed_true_evals = sum(run.finite_true_evals for run in backend_runs.values())
    assert observed_true_evals == expected_true_evals, (
        f"finite true NLL evaluations {observed_true_evals} != {expected_true_evals}"
    )
    for spec in selected_backends:
        run = backend_runs[spec.key]
        assert run.deltas.shape == (n_rows,), f"{spec.key} delta length mismatch"
        assert np.isfinite(run.deltas).all(), f"{spec.key} delta has non-finite entries"
        backend_result = payload["results"][spec.key]
        ci = backend_result["ci95_mean_delta_paired_bootstrap"]
        assert len(ci) == 2 and np.isfinite(np.asarray(ci, dtype=np.float64)).all()
        for bucket in BUCKETS:
            assert bucket in backend_result["by_bucket"], f"{bucket} missing for {spec.key}"
            bucket_entry = backend_result["by_bucket"][bucket]
            assert int(bucket_entry["n"]) > 0, f"{spec.key} bucket {bucket} is empty"
            assert len(bucket_entry["ci95"]) == 2
        assert len(backend_result["per_row"]) == n_rows


def write_json_roundtrip(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["n_rows"] == payload["n_rows"], "JSON round-trip n_rows mismatch"
    assert loaded["results"].keys() == payload["results"].keys(), (
        "JSON round-trip results mismatch"
    )


def print_final_report(
    payload: dict[str, Any],
    out_json: Path,
    out_json_sha: str,
    script_path: Path,
    script_sha: str,
    total_elapsed: float,
    peak_vram_mib: int,
) -> None:
    print("=== Δ_AMH Diagnostic Complete ===", flush=True)
    print("", flush=True)
    for backend in ("flow_matching", "diffusion"):
        if backend not in payload["results"]:
            continue
        result = payload["results"][backend]
        ci = result["ci95_mean_delta_paired_bootstrap"]
        print(f"Backend: {backend}", flush=True)
        print(f"  n = {payload['n_rows']}", flush=True)
        print(
            f"  mean Δ = {result['mean_delta']:.6f} "
            f"[{ci[0]:.6f}, {ci[1]:.6f}]  (95% paired bootstrap CI)",
            flush=True,
        )
        print(f"  median Δ = {result['median_delta']:.6f}", flush=True)
        print(
            "  one-sided p-value (H0: mean Δ <= 0) = "
            f"{result['one_sided_p_value']:.6f}",
            flush=True,
        )
        print("  By bucket:", flush=True)
        for bucket in BUCKETS:
            entry = result["by_bucket"][bucket]
            ci_bucket = entry["ci95"]
            print(
                f"    {bucket:<7} (n={entry['n']}): mean Δ = "
                f"{entry['mean']:.6f} [{ci_bucket[0]:.6f}, {ci_bucket[1]:.6f}]",
                flush=True,
            )
        print("", flush=True)
    print("Interpretation:", flush=True)
    print("  - If both backends mean Δ > 0 with CI excluding 0:", flush=True)
    print("    Models DO use AMH; dimension-dominance is NOT the full story.", flush=True)
    print("  - If both backends mean Δ ≈ 0 with CI crossing 0:", flush=True)
    print("    Dimension-dominance hypothesis empirically confirmed.", flush=True)
    print("    Recommendation: invest in direct conditional CNF (a as FiLM context).", flush=True)
    print("  - If backends disagree: investigate convention mismatch first.", flush=True)
    print("", flush=True)
    print("SHA256:", flush=True)
    print(f"  delta_amh.json: {out_json_sha}", flush=True)
    print(f"  delta_amh_counterfactual.py: {script_sha}", flush=True)
    print("", flush=True)
    print(f"Total runtime: {total_elapsed:.3f}s", flush=True)
    print(f"Peak VRAM: {peak_vram_mib} MiB", flush=True)
    print(f"Output: {out_json}", flush=True)


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_cuda_device(args.device)
    script_path = Path(__file__).resolve()
    workspace_root = Path.cwd().resolve()
    embeddings_path = Path(args.embeddings).resolve()
    embeddings_index_path = Path(args.embeddings_index).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    test_split_path = splits_dir / "test.csv"
    amh_recovered_path = Path(args.amh_recovered).resolve()
    amh_normalization_path = Path(args.amh_normalization).resolve()
    out_json = Path(args.out_json).resolve()
    selected_backends = backend_specs(args)
    command = " ".join(shlex.quote(part) for part in ["python3", *sys.argv])

    print(
        "event=environment "
        f"python={'.'.join(map(str, sys.version_info[:3]))} "
        f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device={device} backend={args.backend} seed={args.seed}",
        flush=True,
    )

    rows = load_amh_rows(
        embeddings_path=embeddings_path,
        embeddings_index_path=embeddings_index_path,
        test_split_path=test_split_path,
        amh_recovered_path=amh_recovered_path,
        amh_normalization_path=amh_normalization_path,
    )
    permutation_indices = make_permutation_indices(
        n_rows=len(rows.rows),
        k=args.k_permutations,
        seed=args.seed,
    )
    print(
        "event=rows_loaded "
        f"test_rows={EXPECTED_TEST_ROWS} nonblank_amh_rows={len(rows.rows)} "
        f"bucket_counts={json.dumps(rows.rows['AMH_bucket'].value_counts().to_dict(), sort_keys=True)}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: dict[str, Any] = {}
    backend_runs: dict[str, BackendRun] = {}
    provenance: dict[str, Any] = {}
    sign_sanity: dict[str, Any] = {}

    for spec in selected_backends:
        print(f"event=backend_start backend={spec.key}", flush=True)
        model, backend_provenance = load_backend_model(spec, device)
        provenance[spec.key] = backend_provenance
        sign_sanity[spec.key] = run_sign_sanity(
            spec=spec,
            model=model,
            rows=rows,
            device=device,
            args=args,
        )
        print(
            "event=sign_sanity "
            f"backend={spec.key} "
            f"true_nll={sign_sanity[spec.key]['NLL_true_first_row']:.6f} "
            f"perturbed_nll={sign_sanity[spec.key]['NLL_perturbed_first_row']:.6f} "
            f"passed={sign_sanity[spec.key]['passed']}",
            flush=True,
        )
        backend_run = run_backend(
            spec=spec,
            model=model,
            rows=rows,
            permutation_indices=permutation_indices,
            device=device,
            args=args,
        )
        backend_runs[spec.key] = backend_run
        results[spec.key] = backend_run.summary
        print(f"event=backend_complete backend={spec.key}", flush=True)
        del model
        torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mib = int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
    else:
        peak_vram_mib = 0

    input_sha256 = {
        "embeddings": assert_expected_sha256("embeddings", embeddings_path),
        "embeddings_index": assert_expected_sha256(
            "embeddings_index", embeddings_index_path
        ),
        "test_csv": assert_expected_sha256("test_csv", test_split_path),
        "amh_recovered_csv": assert_expected_sha256(
            "amh_recovered_csv", amh_recovered_path
        ),
        "amh_normalization_json": assert_expected_sha256(
            "amh_normalization_json", amh_normalization_path
        ),
        "delta_amh_counterfactual_py": sha256_of(script_path),
    }
    for spec in selected_backends:
        input_sha256[f"{spec.key}_model_joint_pt"] = provenance[spec.key][
            "model_sha256"
        ]
        input_sha256[f"{spec.key}_training_log_joint_json"] = provenance[spec.key][
            "training_log_sha256"
        ]

    payload = {
        "experiment": "AMH counterfactual paired contrast",
        "reference": "GPT-Pro Q3 review 2026-05-19; data_audit.md §8",
        "n_rows": int(len(rows.rows)),
        "k_permutations": int(args.k_permutations),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_protocol": {
            "type": "paired row-level resampling",
            "n_resamples": int(args.bootstrap_iters),
            "seed": int(args.seed),
            "rng": "np.random.default_rng",
            "statistic": "mean(delta)",
            "ci": "percentile 2.5/97.5",
            "p_value": "fraction of bootstrap resamples where mean(delta) <= 0",
        },
        "seed": int(args.seed),
        "device": str(device),
        "results": results,
        "interpretation_hint": (
            "mean_delta > 0 with CI excluding 0 → model uses AMH; "
            "mean_delta ≈ 0 with CI crossing 0 → dimension-dominance "
            "hypothesis empirically confirmed"
        ),
        "lock": {
            "input_sha256": input_sha256,
            "model_provenance": provenance,
            "sign_sanity": sign_sanity,
            "permutation_sampling": {
                "mode": "with_replacement",
                "excludes_same_row": True,
                "rng": "np.random.default_rng(10123)",
            },
            "bootstrap": {
                "type": "paired row-level resampling",
                "n_resamples": int(args.bootstrap_iters),
                "seed": int(args.seed),
                "rng": "np.random.default_rng(10123)",
            },
            "library_versions": library_versions(),
            "git_short_sha": git_short_sha(workspace_root),
            "git_dirty": git_dirty(workspace_root),
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": torch.__version__,
            "command": command,
        },
        "timestamp_utc": utc_now(),
    }

    assert_acceptance(payload, backend_runs, selected_backends)
    write_json_roundtrip(out_json, payload)
    out_json_sha = sha256_of(out_json)
    script_sha = sha256_of(script_path)
    total_elapsed = time.perf_counter() - start
    print_final_report(
        payload=payload,
        out_json=out_json,
        out_json_sha=out_json_sha,
        script_path=script_path,
        script_sha=script_sha,
        total_elapsed=total_elapsed,
        peak_vram_mib=peak_vram_mib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
