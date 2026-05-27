import os
#!/usr/bin/env python3
"""
Compute conditional NLLs for p(z | AMH) under Card 1 models.

Both backends integrate only the 768-dimensional z state. AMH is passed as a
detached FiLM context, and every Hutchinson trace is with respect to z only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

from compute_flow_matching_nll import (
    EXPECTED_FEATURE_DIM,
    EXPECTED_SPLIT_ROWS,
    load_amh_normalization,
    load_embeddings,
    load_split_csv,
    merge_split_with_embeddings,
    nll_from_prior_state as fm_nll_from_prior_state,
    rademacher,
    standardize_amh,
)
from train_conditional_diffusion import (
    ConditionalDiffusionNoisePredictor,
    analytic_trace_integral,
    beta_t,
)
from train_conditional_flow_matching import (
    CONDITIONAL_AMH_HANDLING,
    EXPECTED_AMH_DIM,
    ConditionalFlowVectorField,
)
from train_diffusion import (
    PARAMETERIZATION as DIFFUSION_PARAMETERIZATION,
    VP_EPSILON_CLIP,
    vp_alpha_sigma,
)


SEED = 10_123
EXPECTED_ROWS = int(__import__("os").environ.get("EXPECTED_ROWS", "227"))
BACKENDS = ("fm", "diffusion", "both")


@dataclass(frozen=True)
class LoadedRows:
    rows: pd.DataFrame
    z: np.ndarray
    amh: np.ndarray
    amh_available: np.ndarray


@dataclass(frozen=True)
class VariantResult:
    nll_mean: float
    hutchinson_std: float
    fallback_count: int = 0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
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
        "scipy": package_version("scipy"),
        "torchdiffeq": package_version("torchdiffeq"),
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
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=repo_root, check=False).returncode
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False).returncode
        return bool(unstaged or staged)
    except Exception:
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--embeddings", default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/dinov2_z.npy")
    parser.add_argument("--embeddings-index", default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/index.csv")
    parser.add_argument("--splits-dir", default="code/repos/kromp-blastocyst-dataset-audit/data/splits")
    parser.add_argument("--amh-recovered", default="code/repos/kromp-blastocyst-dataset-audit/data/amh_recovered.csv")
    parser.add_argument("--amh-normalization", default="results/baseline/amh_normalization.json")
    parser.add_argument("--fm-model", default="results/conditional/flow_matching/model_conditional.pt")
    parser.add_argument("--diff-model", default="results/conditional/diffusion/model_conditional.pt")
    parser.add_argument("--out-csv", default="results/conditional/nll.csv")
    parser.add_argument("--backend", choices=BACKENDS, default="both")
    parser.add_argument("--hutchinson-samples", type=int, default=5)
    parser.add_argument("--ode-method", default="dopri5", choices=("dopri5", "euler"))
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--euler-steps", type=int, default=100)
    parser.add_argument("--epsilon-clip", type=float, default=VP_EPSILON_CLIP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.hutchinson_samples == 5, "Card 1 NLL requires exactly 5 Hutchinson samples"
    assert args.ode_atol > 0, "--ode-atol must be positive"
    assert args.ode_rtol > 0, "--ode-rtol must be positive"
    assert args.euler_steps > 0, "--euler-steps must be positive"
    assert math.isclose(args.epsilon_clip, VP_EPSILON_CLIP), "epsilon_clip must remain 1e-3"
    assert args.seed == SEED, "Card 1 seed must remain 10123"
    return args


def resolve_cuda_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise RuntimeError("Card 1 NLL requires --device cuda; CPU/auto fallback is forbidden.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Card 1 NLL requires CUDA, but torch.cuda.is_available() is False. "
            "Stopping before model loading or NLL computation."
        )
    return torch.device("cuda")


def load_conditional_rows(
    embeddings_path: Path,
    embeddings_index_path: Path,
    splits_dir: Path,
    amh_normalization_path: Path,
) -> LoadedRows:
    embeddings, embeddings_index = load_embeddings(embeddings_path, embeddings_index_path)
    norm = load_amh_normalization(amh_normalization_path)
    pieces: list[pd.DataFrame] = []
    z_arrays: list[np.ndarray] = []
    amh_arrays: list[np.ndarray] = []
    available_arrays: list[np.ndarray] = []
    for split_name in ("val", "test"):
        split_raw = load_split_csv(splits_dir / f"{split_name}.csv", split_name)
        split_rows, split_z = merge_split_with_embeddings(
            split_raw, embeddings_index, embeddings, split_name
        )
        split_rows = split_rows.copy()
        split_rows["split"] = split_name
        standardized = standardize_amh(split_rows, norm)
        available = split_rows["AMH_bucket"].astype(str).ne("blank").to_numpy()
        amh = np.full((len(split_rows), EXPECTED_AMH_DIM), np.nan, dtype=np.float32)
        amh[available, 0] = standardized[available]
        assert np.isfinite(split_z).all(), f"{split_name} z states are non-finite"
        assert np.isfinite(amh[available]).all(), f"{split_name} nonblank AMH context is non-finite"
        pieces.append(split_rows)
        z_arrays.append(split_z.astype(np.float32, copy=True))
        amh_arrays.append(amh)
        available_arrays.append(available)
    rows = pd.concat(pieces, axis=0, ignore_index=True)
    z = np.concatenate(z_arrays, axis=0).astype(np.float32, copy=False)
    amh = np.concatenate(amh_arrays, axis=0).astype(np.float32, copy=False)
    available = np.concatenate(available_arrays, axis=0)
    assert len(rows) == EXPECTED_ROWS, f"combined rows: {len(rows)} != 227"
    assert z.shape == (EXPECTED_ROWS, EXPECTED_FEATURE_DIM), z.shape
    assert amh.shape == (EXPECTED_ROWS, EXPECTED_AMH_DIM), amh.shape
    return LoadedRows(rows=rows, z=z, amh=amh, amh_available=available)


def load_flow_model(path: Path, device: torch.device) -> torch.nn.Module:
    assert path.exists(), f"missing conditional FM checkpoint: {path}"
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    assert config.get("variant") == "conditional", "FM checkpoint variant mismatch"
    assert int(config.get("input_dim")) == EXPECTED_FEATURE_DIM, "FM checkpoint input_dim mismatch"
    assert int(config.get("amh_dim")) == EXPECTED_AMH_DIM, "FM checkpoint amh_dim mismatch"
    assert int(config.get("seed")) == SEED, "FM checkpoint seed mismatch"
    assert config.get("parameterization") == "flow_matching", "FM checkpoint parameterization mismatch"
    assert config.get("conditional_amh_handling") == CONDITIONAL_AMH_HANDLING
    model = ConditionalFlowVectorField(
        input_dim=EXPECTED_FEATURE_DIM,
        time_embed_dim=int(config.get("time_embed_dim", 128)),
        amh_embed_dim=int(config.get("amh_embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        n_blocks=int(config.get("n_blocks", 4)),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not incompatible.missing_keys, "FM checkpoint missing keys"
    assert not incompatible.unexpected_keys, "FM checkpoint unexpected keys"
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if device.type == "cuda":
        try:
            model = torch.compile(model, mode="default", dynamic=False)
            print("event=torch_compile_applied backend=fm mode=default", flush=True)
        except Exception as exc:
            print(
                f"event=torch_compile_skipped backend=fm reason={type(exc).__name__} message={exc}",
                flush=True,
            )
    return model


def load_diffusion_model(path: Path, device: torch.device) -> torch.nn.Module:
    assert path.exists(), f"missing conditional diffusion checkpoint: {path}"
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    assert checkpoint.get("parameterization") == DIFFUSION_PARAMETERIZATION
    assert config.get("variant") == "conditional", "diffusion checkpoint variant mismatch"
    assert int(config.get("input_dim")) == EXPECTED_FEATURE_DIM, "diffusion checkpoint input_dim mismatch"
    assert int(config.get("amh_dim")) == EXPECTED_AMH_DIM, "diffusion checkpoint amh_dim mismatch"
    assert int(config.get("seed")) == SEED, "diffusion checkpoint seed mismatch"
    assert config.get("parameterization") == DIFFUSION_PARAMETERIZATION
    assert config.get("loss_target") == "epsilon_noise"
    assert config.get("score_recovery") == "-epsilon_hat / sigma(t)"
    assert config.get("conditional_amh_handling") == CONDITIONAL_AMH_HANDLING
    model = ConditionalDiffusionNoisePredictor(
        input_dim=EXPECTED_FEATURE_DIM,
        time_embed_dim=int(config.get("time_embed_dim", 128)),
        amh_embed_dim=int(config.get("amh_embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        n_blocks=int(config.get("n_blocks", 4)),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not incompatible.missing_keys, "diffusion checkpoint missing keys"
    assert not incompatible.unexpected_keys, "diffusion checkpoint unexpected keys"
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if device.type == "cuda":
        try:
            model = torch.compile(model, mode="default", dynamic=False)
            print("event=torch_compile_applied backend=diffusion mode=default", flush=True)
        except Exception as exc:
            print(
                f"event=torch_compile_skipped backend=diffusion reason={type(exc).__name__} message={exc}",
                flush=True,
            )
    return model


def fm_model_time(s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(1.0, device=z.device, dtype=z.dtype) - s.to(device=z.device, dtype=z.dtype)
    return t.reshape(1).expand(z.shape[0])


def conditional_fm_vector_field_and_trace(
    model: torch.nn.Module,
    z: torch.Tensor,
    amh: torch.Tensor,
    s: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_value = z.detach().to(dtype=torch.float32)
    amh_value = amh.detach().to(device=z_value.device, dtype=torch.float32)
    t_value = fm_model_time(s, z_value)
    with torch.no_grad():
        v_value = model(z_value, t_value, amh_value).detach()
    with torch.enable_grad():
        z_req = z.detach().to(dtype=torch.float32).requires_grad_(True)
        amh_req = amh.detach().to(device=z_req.device, dtype=torch.float32)
        t_req = fm_model_time(s, z_req)
        out = model(z_req, t_req, amh_req)
        eps_req = eps.to(device=z_req.device, dtype=out.dtype)
        jvp = torch.autograd.grad(
            out,
            z_req,
            grad_outputs=eps_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        trace = (jvp * eps_req).sum(dim=1).detach().to(dtype=torch.float64)
        del z_req, amh_req, out, eps_req, jvp
    return v_value, trace


def conditional_fm_augmented_dynamics(
    model: torch.nn.Module,
    z: torch.Tensor,
    amh: torch.Tensor,
    log_det: torch.Tensor,
    s: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del log_det
    v_value, trace = conditional_fm_vector_field_and_trace(model, z, amh, s, eps)
    return -v_value, trace


def solve_conditional_fm_probe_dopri5(
    model: torch.nn.Module,
    x: torch.Tensor,
    amh: torch.Tensor,
    eps: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_det0 = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float64)
    s_span = torch.tensor([0.0, 1.0], device=x.device, dtype=torch.float32)

    def ode_func(
        s: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_state, log_det_state = state
        return conditional_fm_augmented_dynamics(model, z_state, amh, log_det_state, s, eps)

    z_traj, log_det_traj = odeint(
        ode_func,
        (x, log_det0),
        s_span,
        method="dopri5",
        atol=atol,
        rtol=rtol,
    )
    return z_traj[-1].detach(), log_det_traj[-1].detach()


def solve_conditional_fm_probe_euler(
    model: torch.nn.Module,
    x: torch.Tensor,
    amh: torch.Tensor,
    eps: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = x.detach()
    log_det = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float64)
    dt = torch.as_tensor(1.0 / steps, device=x.device, dtype=torch.float64)
    for step in range(steps):
        s = torch.as_tensor(step / steps, device=x.device, dtype=torch.float32)
        dz, dlog_det = conditional_fm_augmented_dynamics(model, z, amh, log_det, s, eps)
        z = (z + dt.to(dtype=dz.dtype) * dz).detach()
        log_det = (log_det + dt * dlog_det).detach()
    return z, log_det


def compute_conditional_fm_nll(
    model: torch.nn.Module,
    z_np: np.ndarray,
    amh_value: float,
    generator: torch.Generator,
    device: torch.device,
    method: str,
    atol: float,
    rtol: float,
    euler_steps: int,
    hutchinson_samples: int,
) -> VariantResult:
    assert z_np.shape == (EXPECTED_FEATURE_DIM,), f"FM row shape: {z_np.shape}"
    assert np.isfinite(z_np).all(), "FM z row is non-finite"
    assert math.isfinite(float(amh_value)), "FM AMH context is non-finite"
    x_single = torch.from_numpy(z_np.astype(np.float32, copy=False)).reshape(1, EXPECTED_FEATURE_DIM)
    x = x_single.repeat(hutchinson_samples, 1).to(device)
    amh = torch.full((hutchinson_samples, EXPECTED_AMH_DIM), float(amh_value), device=device, dtype=torch.float32)
    eps = rademacher((hutchinson_samples, EXPECTED_FEATURE_DIM), generator, device, torch.float32)
    fallback_count = 0
    try:
        if method == "euler":
            z0, log_det = solve_conditional_fm_probe_euler(model, x, amh, eps, euler_steps)
        else:
            z0, log_det = solve_conditional_fm_probe_dopri5(model, x, amh, eps, atol, rtol)
        assert torch.isfinite(z0).all() and torch.isfinite(log_det).all()
    except Exception:
        if method == "euler":
            raise
        fallback_count = hutchinson_samples
        z0, log_det = solve_conditional_fm_probe_euler(model, x, amh, eps, euler_steps)
        assert torch.isfinite(z0).all() and torch.isfinite(log_det).all()
    nll_array = fm_nll_from_prior_state(z0, log_det, EXPECTED_FEATURE_DIM)
    assert nll_array.shape == (hutchinson_samples,)
    hutchinson_std = float(nll_array.std(ddof=1)) if len(nll_array) > 1 else 0.0
    nll_mean = float(np.mean(nll_array))
    assert math.isfinite(nll_mean), "FM mean NLL is non-finite"
    del x_single, x, amh, eps, z0, log_det, nll_array
    return VariantResult(
        nll_mean=float(np.float32(nll_mean)),
        hutchinson_std=hutchinson_std,
        fallback_count=fallback_count,
    )


def diffusion_t_batch(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return t.to(device=z.device, dtype=z.dtype).reshape(1).expand(z.shape[0])


def conditional_diffusion_drift_and_trace(
    model: torch.nn.Module,
    z: torch.Tensor,
    amh: torch.Tensor,
    t: torch.Tensor,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_value = z.detach().to(dtype=torch.float32)
    amh_value = amh.detach().to(device=z_value.device, dtype=torch.float32)
    t_value = diffusion_t_batch(t, z_value)
    beta_value = beta_t(t.to(device=z_value.device, dtype=z_value.dtype))
    _, sigma_value = vp_alpha_sigma(t_value[:1])
    with torch.no_grad():
        eps_hat_value = model(z_value, t_value, amh_value).detach()
        drift = -0.5 * beta_value * z_value + 0.5 * beta_value * eps_hat_value / sigma_value.reshape(1, 1)
    with torch.enable_grad():
        z_req = z.detach().to(dtype=torch.float32).requires_grad_(True)
        amh_req = amh.detach().to(device=z_req.device, dtype=torch.float32)
        t_req = diffusion_t_batch(t, z_req)
        beta_req = beta_t(t.to(device=z_req.device, dtype=z_req.dtype))
        _, sigma_req = vp_alpha_sigma(t_req[:1])
        eps_hat_trace = model(z_req, t_req, amh_req)
        probe_req = probe.to(device=z_req.device, dtype=eps_hat_trace.dtype)
        jvp = torch.autograd.grad(
            eps_hat_trace,
            z_req,
            grad_outputs=probe_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        trace_eps = (jvp * probe_req).sum(dim=1).detach().to(dtype=torch.float64)
        stochastic_trace = (
            0.5
            * beta_req.detach().to(dtype=torch.float64)
            / sigma_req.detach().to(dtype=torch.float64).reshape(())
            * trace_eps
        )
        del z_req, amh_req, eps_hat_trace, probe_req, jvp, trace_eps
    return drift, stochastic_trace


def conditional_diffusion_augmented_dynamics(
    model: torch.nn.Module,
    z: torch.Tensor,
    amh: torch.Tensor,
    log_det_partial: torch.Tensor,
    t: torch.Tensor,
    probe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del log_det_partial
    return conditional_diffusion_drift_and_trace(model, z, amh, t, probe)


def solve_conditional_diffusion_probe_dopri5(
    model: torch.nn.Module,
    x: torch.Tensor,
    amh: torch.Tensor,
    probe: torch.Tensor,
    atol: float,
    rtol: float,
    epsilon_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_det0 = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float64)
    t_span = torch.tensor([epsilon_clip, 1.0], device=x.device, dtype=torch.float32)

    def ode_func(
        t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_state, log_det_state = state
        return conditional_diffusion_augmented_dynamics(
            model, z_state, amh, log_det_state, t, probe
        )

    z_traj, log_det_partial_traj = odeint(
        ode_func,
        (x, log_det0),
        t_span,
        method="dopri5",
        atol=atol,
        rtol=rtol,
    )
    return z_traj[-1].detach(), log_det_partial_traj[-1].detach()


def diffusion_nll_from_prior_state(
    z1: torch.Tensor,
    log_det_partial: torch.Tensor,
    epsilon_clip: float,
) -> np.ndarray:
    z1_double = z1.detach().to(dtype=torch.float64)
    quadratic = torch.sum(z1_double * z1_double, dim=1)
    normalizer = EXPECTED_FEATURE_DIM * math.log(2.0 * math.pi)
    log_p1 = -0.5 * (quadratic + normalizer)
    log_det_total = log_det_partial.to(dtype=torch.float64) + analytic_trace_integral(
        EXPECTED_FEATURE_DIM, epsilon_clip
    )
    nll = -(log_p1 + log_det_total)
    values = nll.detach().cpu().numpy().astype(np.float64, copy=False)
    assert np.isfinite(values).all(), "diffusion NLL contains non-finite values"
    return values


def compute_conditional_diffusion_nll(
    model: torch.nn.Module,
    z_np: np.ndarray,
    amh_value: float,
    generator: torch.Generator,
    device: torch.device,
    atol: float,
    rtol: float,
    epsilon_clip: float,
    hutchinson_samples: int,
) -> VariantResult:
    assert z_np.shape == (EXPECTED_FEATURE_DIM,), f"diffusion row shape: {z_np.shape}"
    assert np.isfinite(z_np).all(), "diffusion z row is non-finite"
    assert math.isfinite(float(amh_value)), "diffusion AMH context is non-finite"
    x_single = torch.from_numpy(z_np.astype(np.float32, copy=False)).reshape(1, EXPECTED_FEATURE_DIM)
    x = x_single.repeat(hutchinson_samples, 1).to(device)
    amh = torch.full((hutchinson_samples, EXPECTED_AMH_DIM), float(amh_value), device=device, dtype=torch.float32)
    probe = rademacher((hutchinson_samples, EXPECTED_FEATURE_DIM), generator, device, torch.float32)
    z1, log_det_partial = solve_conditional_diffusion_probe_dopri5(
        model=model,
        x=x,
        amh=amh,
        probe=probe,
        atol=atol,
        rtol=rtol,
        epsilon_clip=epsilon_clip,
    )
    assert torch.isfinite(z1).all(), "diffusion prior endpoint is non-finite"
    assert torch.isfinite(log_det_partial).all(), "diffusion log-det partial is non-finite"
    nll_array = diffusion_nll_from_prior_state(z1, log_det_partial, epsilon_clip)
    assert nll_array.shape == (hutchinson_samples,)
    hutchinson_std = float(nll_array.std(ddof=1)) if len(nll_array) > 1 else 0.0
    nll_mean = float(np.mean(nll_array))
    assert math.isfinite(nll_mean), "diffusion mean NLL is non-finite"
    del x_single, x, amh, probe, z1, log_det_partial, nll_array
    return VariantResult(nll_mean=float(np.float32(nll_mean)), hutchinson_std=hutchinson_std)


def run_sign_sanity(
    loaded: LoadedRows,
    models: dict[str, torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    val_nonblank = np.flatnonzero(
        loaded.rows["split"].astype(str).eq("val").to_numpy() & loaded.amh_available
    )
    assert val_nonblank.size > 0, "no nonblank validation rows for sign sanity"
    row_idx = int(val_nonblank[0])
    z_np = loaded.z[row_idx]
    a_true = float(loaded.amh[row_idx, 0])
    a_perturbed = a_true + 10.0
    result: dict[str, Any] = {
        "row_index": row_idx,
        "Image": str(loaded.rows.loc[row_idx, "Image"]),
        "a_true_standardized": a_true,
        "a_perturbed_standardized": a_perturbed,
    }
    if "fm" in models:
        gen_true = torch.Generator(device=device).manual_seed(args.seed + 120_001)
        gen_pert = torch.Generator(device=device).manual_seed(args.seed + 120_001)
        true_nll = compute_conditional_fm_nll(
            models["fm"], z_np, a_true, gen_true, device, args.ode_method,
            args.ode_atol, args.ode_rtol, args.euler_steps, args.hutchinson_samples
        ).nll_mean
        pert_nll = compute_conditional_fm_nll(
            models["fm"], z_np, a_perturbed, gen_pert, device, args.ode_method,
            args.ode_atol, args.ode_rtol, args.euler_steps, args.hutchinson_samples
        ).nll_mean
        passed = bool(pert_nll > true_nll)
        assert passed, f"FM sign sanity failed: perturbed {pert_nll} <= true {true_nll}"
        result["fm"] = {"nll_true": true_nll, "nll_perturbed_by_10sigma": pert_nll, "passed": passed}
    if "diffusion" in models:
        gen_true = torch.Generator(device=device).manual_seed(args.seed + 130_001)
        gen_pert = torch.Generator(device=device).manual_seed(args.seed + 130_001)
        true_nll = compute_conditional_diffusion_nll(
            models["diffusion"], z_np, a_true, gen_true, device, args.ode_atol,
            args.ode_rtol, args.epsilon_clip, args.hutchinson_samples
        ).nll_mean
        pert_nll = compute_conditional_diffusion_nll(
            models["diffusion"], z_np, a_perturbed, gen_pert, device, args.ode_atol,
            args.ode_rtol, args.epsilon_clip, args.hutchinson_samples
        ).nll_mean
        passed = bool(pert_nll > true_nll)
        assert passed, f"diffusion sign sanity failed: perturbed {pert_nll} <= true {true_nll}"
        result["diffusion"] = {"nll_true": true_nll, "nll_perturbed_by_10sigma": pert_nll, "passed": passed}
    return result


def distribution_summary(values: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n_finite": 0, "n_nan": int(np.isnan(arr).sum()), "std": float("nan"), "std_lt_200": False}
    return {
        "n_finite": int(finite.size),
        "n_nan": int(np.isnan(arr).sum()),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=0)),
        "std_lt_200": bool(np.std(finite, ddof=0) < 200.0),
    }


def finite_median_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def assert_acceptance(out: pd.DataFrame, backend: str, sign_sanity: dict[str, Any]) -> dict[str, Any]:
    expected_columns = ["Image", "split", "NLL_conditional_fm", "NLL_conditional_diff", "AMH_bucket"]
    assert list(out.columns) == expected_columns, f"conditional nll columns: {list(out.columns)}"
    assert len(out) == EXPECTED_ROWS, f"conditional nll rows: {len(out)} != 227"
    split_counts = out["split"].value_counts().to_dict()
    # Dataset-agnostic check (Wang multi-cohort patch)
    assert split_counts.get("val", 0) > 0, f"val empty: {split_counts}"
    assert split_counts.get("test", 0) > 0, f"test empty: {split_counts}"
    blank = out["AMH_bucket"].astype(str).eq("blank").to_numpy()
    selected = ["NLL_conditional_fm", "NLL_conditional_diff"] if backend == "both" else (
        ["NLL_conditional_fm"] if backend == "fm" else ["NLL_conditional_diff"]
    )
    for col in selected:
        values = out[col].to_numpy(dtype=np.float64)
        assert np.isnan(values[blank]).all(), f"{col} must be NaN on blank-AMH rows"
        assert np.isfinite(values[~blank]).all(), f"{col} must be finite on nonblank AMH rows"
    if backend == "both":
        assert "fm" in sign_sanity and "diffusion" in sign_sanity
    elif backend == "fm":
        assert "fm" in sign_sanity
    else:
        assert "diffusion" in sign_sanity
    return {
        "227_rows": True,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "nan_only_blank_for_selected_backends": True,
        "sign_sanity": sign_sanity,
        "distribution_summary": {
            col: distribution_summary(out[col].to_numpy(dtype=np.float64))
            for col in ["NLL_conditional_fm", "NLL_conditional_diff"]
        },
        "std_lt_200_note": "cosmetic only, not a hard gate",
    }


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_cuda_device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    repo_root = Path.cwd().resolve()
    script_path = Path(__file__).resolve()
    embeddings_path = Path(args.embeddings).resolve()
    embeddings_index_path = Path(args.embeddings_index).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    amh_normalization_path = Path(args.amh_normalization).resolve()
    out_csv = Path(args.out_csv).resolve()

    print(
        "event=environment conditional_nll=true "
        f"python={'.'.join(map(str, sys.version_info[:3]))} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} device={device} backend={args.backend}",
        flush=True,
    )
    loaded = load_conditional_rows(
        embeddings_path=embeddings_path,
        embeddings_index_path=embeddings_index_path,
        splits_dir=splits_dir,
        amh_normalization_path=amh_normalization_path,
    )
    print(
        "event=rows_loaded "
        f"rows={len(loaded.rows)} val={EXPECTED_SPLIT_ROWS['val']} "
        f"test={EXPECTED_SPLIT_ROWS['test']} blank_amh={int((~loaded.amh_available).sum())}",
        flush=True,
    )
    models: dict[str, torch.nn.Module] = {}
    if args.backend in ("fm", "both"):
        models["fm"] = load_flow_model(Path(args.fm_model).resolve(), device)
    if args.backend in ("diffusion", "both"):
        models["diffusion"] = load_diffusion_model(Path(args.diff_model).resolve(), device)

    sign_sanity = run_sign_sanity(loaded, models, args, device)
    print(f"event=sign_sanity payload={json.dumps(sign_sanity, sort_keys=True)}", flush=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    fm_nll: list[float] = []
    diff_nll: list[float] = []
    fm_std: list[float] = []
    diff_std: list[float] = []
    fallback_counts = {"fm": 0, "diffusion": 0}

    for row_idx, row in loaded.rows.iterrows():
        if loaded.amh_available[row_idx] and "fm" in models:
            result = compute_conditional_fm_nll(
                model=models["fm"],
                z_np=loaded.z[row_idx],
                amh_value=float(loaded.amh[row_idx, 0]),
                generator=generator,
                device=device,
                method=args.ode_method,
                atol=args.ode_atol,
                rtol=args.ode_rtol,
                euler_steps=args.euler_steps,
                hutchinson_samples=args.hutchinson_samples,
            )
            fm_nll.append(result.nll_mean)
            fm_std.append(result.hutchinson_std)
            fallback_counts["fm"] += result.fallback_count
        else:
            fm_nll.append(float("nan"))
            fm_std.append(float("nan"))
        if loaded.amh_available[row_idx] and "diffusion" in models:
            result = compute_conditional_diffusion_nll(
                model=models["diffusion"],
                z_np=loaded.z[row_idx],
                amh_value=float(loaded.amh[row_idx, 0]),
                generator=generator,
                device=device,
                atol=args.ode_atol,
                rtol=args.ode_rtol,
                epsilon_clip=args.epsilon_clip,
                hutchinson_samples=args.hutchinson_samples,
            )
            diff_nll.append(result.nll_mean)
            diff_std.append(result.hutchinson_std)
            fallback_counts["diffusion"] += result.fallback_count
        else:
            diff_nll.append(float("nan"))
            diff_std.append(float("nan"))
        if (row_idx + 1) % 10 == 0 or row_idx == len(loaded.rows) - 1:
            print(
                "event=nll_progress "
                f"rows_done={row_idx + 1}/{len(loaded.rows)} elapsed_seconds={time.perf_counter() - start:.3f}",
                flush=True,
            )
        if device.type == "cuda" and (row_idx + 1) % 25 == 0:
            torch.cuda.empty_cache()

    out = pd.DataFrame(
        {
            "Image": loaded.rows["Image"].astype(str).to_numpy(),
            "split": loaded.rows["split"].astype(str).to_numpy(),
            "NLL_conditional_fm": np.asarray(fm_nll, dtype=np.float32),
            "NLL_conditional_diff": np.asarray(diff_nll, dtype=np.float32),
            "AMH_bucket": loaded.rows["AMH_bucket"].astype(str).to_numpy(),
        }
    )
    # Save-before-assert (PoC Day 6 lesson — don't lose NLL data to acceptance fail)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    acceptance = assert_acceptance(out, args.backend, sign_sanity)
    reloaded = pd.read_csv(
        out_csv,
        dtype={
            "Image": str,
            "split": str,
            "NLL_conditional_fm": "float32",
            "NLL_conditional_diff": "float32",
            "AMH_bucket": str,
        },
        keep_default_na=True,
    )
    reloaded_acceptance = assert_acceptance(reloaded, args.backend, sign_sanity)
    assert reloaded_acceptance["227_rows"] == acceptance["227_rows"]

    out_sha = sha256_of(out_csv)
    script_sha = sha256_of(script_path)
    input_sha256 = {
        "embeddings_npy": sha256_of(embeddings_path),
        "embeddings_index_csv": sha256_of(embeddings_index_path),
        "val_csv": sha256_of(splits_dir / "val.csv"),
        "test_csv": sha256_of(splits_dir / "test.csv"),
        "amh_normalization_json": sha256_of(amh_normalization_path),
        "fm_model": sha256_of(Path(args.fm_model).resolve()) if Path(args.fm_model).resolve().exists() else None,
        "diff_model": sha256_of(Path(args.diff_model).resolve()) if Path(args.diff_model).resolve().exists() else None,
    }
    print("event=summary_start", flush=True)
    print(f"script={script_path}", flush=True)
    print(f"script_sha256={script_sha}", flush=True)
    print(f"output_csv={out_csv}", flush=True)
    print(f"output_csv_sha256={out_sha}", flush=True)
    print(f"backend={args.backend}", flush=True)
    print(f"conditional_amh_handling={CONDITIONAL_AMH_HANDLING}", flush=True)
    print("input_dim=768", flush=True)
    print("amh_dim=1", flush=True)
    print("divergence_wrt=z_only", flush=True)
    print(f"acceptance={json.dumps(acceptance, sort_keys=True)}", flush=True)
    print(f"fallback_counts={json.dumps(fallback_counts, sort_keys=True)}", flush=True)
    print(
        "hutchinson_std_median="
        f"{json.dumps({'fm': finite_median_or_none(fm_std), 'diffusion': finite_median_or_none(diff_std)}, sort_keys=True)}",
        flush=True,
    )
    print(f"input_sha256={json.dumps(input_sha256, sort_keys=True)}", flush=True)
    print(f"code_revision={git_short_sha(repo_root)}", flush=True)
    print(f"git_dirty={git_dirty(repo_root)}", flush=True)
    print(f"library_versions={json.dumps(library_versions(), sort_keys=True)}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"device={device}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"timestamp_utc={utc_now()}", flush=True)
    print(f"elapsed_seconds={time.perf_counter() - start:.3f}", flush=True)
    print(
        f"peak_vram_mib={int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))}",
        flush=True,
    )
    print("event=summary_end", flush=True)
    print(
        "event=conditional_nll_done "
        f"out_csv={out_csv} out_csv_sha256={out_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
