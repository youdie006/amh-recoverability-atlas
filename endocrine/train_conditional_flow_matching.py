import os
#!/usr/bin/env python3
"""
Train a conditional flow-matching CNF for p(z | AMH).

AMH is used only as FiLM context. It is never concatenated to the noised state,
and the vector field is defined on the 768-dimensional image embedding z.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_flow_matching import (
    EXPECTED_EMBEDDING_ROWS,
    EXPECTED_FEATURE_DIM,
    EXPECTED_SPLIT_ROWS,
    ResidualFiLMBlock,
    SinusoidalTimeEmbedding,
    device_memory_summary,
    git_dirty,
    git_short_sha,
    library_versions,
    load_amh_normalization,
    load_embeddings,
    load_split_csv,
    merge_split_with_embeddings,
    seed_everything,
    sha256_of,
    standardize_amh,
    utc_now,
)


SEED = 10_123
CONDITIONAL_AMH_HANDLING = "context_film_not_noised_state"
PARAMETERIZATION = "flow_matching"
EXPECTED_AMH_DIM = 1


@dataclass
class ConditionalData:
    train_z: np.ndarray
    train_a: np.ndarray
    val_z: np.ndarray
    val_a: np.ndarray
    row_counts: dict[str, int]
    amh_mean: float
    amh_std: float
    amh_normalization_sha256: str


@dataclass
class TrainResult:
    checkpoint_path: Path
    training_log_path: Path
    train_loss_curve: list[float]
    val_loss_curve: list[float]
    best_epoch: int
    best_val_loss: float
    epochs_trained: int
    acceptance: dict[str, Any]
    model_sha256: str
    training_log_sha256: str


class ConditionalFlowVectorField(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = EXPECTED_FEATURE_DIM,
        time_embed_dim: int = 128,
        amh_embed_dim: int = 128,
        hidden_dim: int = 512,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        assert input_dim == EXPECTED_FEATURE_DIM, f"conditional FM input_dim must be 768, got {input_dim}"
        assert time_embed_dim > 0, "time_embed_dim must be positive"
        assert amh_embed_dim > 0, "amh_embed_dim must be positive"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert n_blocks > 0, "n_blocks must be positive"
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.amh_embed_dim = amh_embed_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.amh_mlp = torch.nn.Sequential(
            torch.nn.Linear(EXPECTED_AMH_DIM, amh_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(amh_embed_dim, amh_embed_dim),
        )
        combined_embed_dim = time_embed_dim + amh_embed_dim
        self.blocks = torch.nn.ModuleList(
            [
                ResidualFiLMBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embed_dim=combined_embed_dim,
                )
                for _ in range(n_blocks)
            ]
        )
        self.out_norm = torch.nn.LayerNorm(input_dim)
        self.out = torch.nn.Linear(input_dim, input_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, amh: torch.Tensor) -> torch.Tensor:
        assert z.ndim == 2 and z.shape[1] == self.input_dim, (
            f"z shape {tuple(z.shape)} incompatible with input_dim={self.input_dim}"
        )
        assert amh.ndim == 2 and amh.shape == (z.shape[0], EXPECTED_AMH_DIM), (
            f"amh must have shape (batch, 1), got {tuple(amh.shape)}"
        )
        time_emb = self.time_mlp(self.time_embedding(t.to(dtype=z.dtype)))
        amh_emb = self.amh_mlp(amh.detach().to(device=z.device, dtype=z.dtype))
        emb = torch.cat([time_emb, amh_emb], dim=1)
        h = z
        for block in self.blocks:
            h = block(h, emb)
        return self.out(self.out_norm(h))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_canonical_torch_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "canonical.pt"
        torch.save(payload, tmp_path)
        path.write_bytes(tmp_path.read_bytes())


def assert_cuda_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise RuntimeError("Card 1 requires the literal flag --device cuda; CPU/auto fallback is forbidden.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Card 1 requires CUDA, but torch.cuda.is_available() is False. "
            "Stopping before training."
        )
    return torch.device("cuda")


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--embeddings", default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/dinov2_z.npy")
    parser.add_argument("--embeddings-index", default="code/repos/kromp-blastocyst-dataset-audit/data/embeddings/index.csv")
    parser.add_argument("--splits-dir", default="code/repos/kromp-blastocyst-dataset-audit/data/splits")
    parser.add_argument("--amh-recovered", default="code/repos/kromp-blastocyst-dataset-audit/data/amh_recovered.csv")
    parser.add_argument("--amh-normalization", default="results/baseline/amh_normalization.json")
    parser.add_argument("--out-dir", default="results/conditional/flow_matching")
    parser.add_argument("--epochs", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--amh-embed-dim", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--sign-sanity-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.epochs > 0, "--epochs must be positive"
    assert args.batch_size > 0, "--batch-size must be positive"
    assert args.lr > 0, "--lr must be positive"
    assert args.weight_decay >= 0, "--weight-decay must be non-negative"
    assert args.early_stop_patience > 0, "--early-stop-patience must be positive"
    assert args.min_epochs > 0, "--min-epochs must be positive"
    assert args.val_every > 0, "--val-every must be positive"
    assert args.hidden_dim > 0, "--hidden-dim must be positive"
    assert args.time_embed_dim > 0, "--time-embed-dim must be positive"
    assert args.amh_embed_dim > 0, "--amh-embed-dim must be positive"
    assert args.n_blocks > 0, "--n-blocks must be positive"
    assert args.sign_sanity_steps > 0, "--sign-sanity-steps must be positive"
    assert args.seed == SEED, "Card 1 seed must remain 10123"
    return args


def prepare_conditional_arrays(
    embeddings_path: Path,
    embeddings_index_path: Path,
    splits_dir: Path,
    amh_normalization_path: Path,
) -> ConditionalData:
    embeddings, embeddings_index = load_embeddings(embeddings_path, embeddings_index_path, strict_expected_rows=False)
    norm = load_amh_normalization(amh_normalization_path)
    train_raw = load_split_csv(splits_dir / "train.csv", "train", strict_expected_rows=False)
    val_raw = load_split_csv(splits_dir / "val.csv", "val", strict_expected_rows=False)
    train_rows, train_z = merge_split_with_embeddings(train_raw, embeddings_index, embeddings, "train")
    val_rows, val_z = merge_split_with_embeddings(val_raw, embeddings_index, embeddings, "val")
    train_mask = train_rows["AMH_bucket"].astype(str).ne("blank").to_numpy()
    val_mask = val_rows["AMH_bucket"].astype(str).ne("blank").to_numpy()
    assert train_mask.any(), "conditional train split has no nonblank AMH rows"
    assert val_mask.any(), "conditional val split has no nonblank AMH rows"
    train_xx_before = int(train_rows["AMH_bucket"].astype(str).eq("xx_mmm").sum())
    val_xx_before = int(val_rows["AMH_bucket"].astype(str).eq("xx_mmm").sum())
    train_xx_after = int(train_rows.loc[train_mask, "AMH_bucket"].astype(str).eq("xx_mmm").sum())
    val_xx_after = int(val_rows.loc[val_mask, "AMH_bucket"].astype(str).eq("xx_mmm").sum())
    assert train_xx_after == train_xx_before, "conditional filtering removed xx_mmm train rows"
    assert val_xx_after == val_xx_before, "conditional filtering removed xx_mmm val rows"
    train_a = standardize_amh(train_rows.loc[train_mask].copy(), norm)[:, None]
    val_a = standardize_amh(val_rows.loc[val_mask].copy(), norm)[:, None]
    train_z = train_z[train_mask].astype(np.float32, copy=True)
    val_z = val_z[val_mask].astype(np.float32, copy=True)
    train_a = train_a.astype(np.float32, copy=True)
    val_a = val_a.astype(np.float32, copy=True)
    assert train_z.shape[1] == EXPECTED_FEATURE_DIM and val_z.shape[1] == EXPECTED_FEATURE_DIM
    assert train_a.shape[1] == EXPECTED_AMH_DIM and val_a.shape[1] == EXPECTED_AMH_DIM
    assert np.isfinite(train_z).all() and np.isfinite(val_z).all(), "z arrays contain non-finite values"
    assert np.isfinite(train_a).all() and np.isfinite(val_a).all(), "AMH context arrays contain non-finite values"
    row_counts = {
        "train_before_filter": int(len(train_rows)),
        "val_before_filter": int(len(val_rows)),
        "train_blank_dropped": int((~train_mask).sum()),
        "val_blank_dropped": int((~val_mask).sum()),
        "train_xx_mmm_retained": train_xx_after,
        "val_xx_mmm_retained": val_xx_after,
        "train_after_filter": int(train_z.shape[0]),
        "val_after_filter": int(val_z.shape[0]),
    }
    return ConditionalData(
        train_z=train_z,
        train_a=train_a,
        val_z=val_z,
        val_a=val_a,
        row_counts=row_counts,
        amh_mean=float(norm["mean"]),
        amh_std=float(norm["std"]),
        amh_normalization_sha256=sha256_of(amh_normalization_path),
    )


def conditional_flow_matching_loss(
    model: ConditionalFlowVectorField,
    z1: torch.Tensor,
    amh: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    z1 = z1.to(device)
    amh = amh.to(device)
    assert z1.shape[0] == amh.shape[0], "z/AMH batch size mismatch"
    z0 = torch.randn(z1.shape, device=device, dtype=z1.dtype, generator=generator)
    t = torch.rand((z1.shape[0],), device=device, dtype=z1.dtype, generator=generator)
    zt = (1.0 - t[:, None]) * z0 + t[:, None] * z1
    target = z1 - z0
    pred = model(zt, t, amh)
    assert pred.shape == target.shape, "conditional vector field output shape mismatch"
    loss = F.mse_loss(pred, target)
    assert torch.isfinite(loss), "conditional flow matching loss is not finite"
    return loss


@torch.no_grad()
def evaluate_flow_loss(
    model: ConditionalFlowVectorField,
    z: torch.Tensor,
    amh: torch.Tensor,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    loss = conditional_flow_matching_loss(model, z.to(device), amh.to(device), generator, device)
    value = float(loss.detach().cpu().item())
    assert math.isfinite(value), "validation loss is not finite"
    return value


def learning_rate_for_epoch(epoch: int, base_lr: float, epochs: int) -> float:
    warmup_epochs = max(1, int(math.ceil(epochs * 0.05)))
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    return base_lr


def rademacher_like(z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    eps = torch.randint(0, 2, z.shape, generator=generator, device=z.device, dtype=torch.int64)
    return eps.to(dtype=z.dtype).mul_(2.0).sub_(1.0)


def standard_normal_log_prob(z0: torch.Tensor) -> torch.Tensor:
    z0_double = z0.detach().to(dtype=torch.float64)
    quadratic = torch.sum(z0_double * z0_double, dim=1)
    normalizer = EXPECTED_FEATURE_DIM * math.log(2.0 * math.pi)
    return -0.5 * (quadratic + normalizer)


def euler_conditional_fm_nll(
    model: ConditionalFlowVectorField,
    z_np: np.ndarray,
    amh_value: float,
    device: torch.device,
    seed: int,
    steps: int,
    hutchinson_samples: int = 5,
) -> float:
    assert z_np.shape == (EXPECTED_FEATURE_DIM,)
    assert math.isfinite(float(amh_value))
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.from_numpy(z_np.astype(np.float32, copy=False)).reshape(1, -1).repeat(hutchinson_samples, 1).to(device)
    amh = torch.full((hutchinson_samples, 1), float(amh_value), device=device, dtype=torch.float32)
    eps = rademacher_like(z, generator)
    log_det = torch.zeros((hutchinson_samples,), device=device, dtype=torch.float64)
    dt = torch.as_tensor(1.0 / steps, device=device, dtype=torch.float64)
    model.eval()
    for step in range(steps):
        s = torch.as_tensor(step / steps, device=device, dtype=torch.float32)
        t = (torch.as_tensor(1.0, device=device, dtype=torch.float32) - s).reshape(1).expand(z.shape[0])
        z_value = z.detach().to(dtype=torch.float32)
        with torch.no_grad():
            v_value = model(z_value, t, amh.detach()).detach()
        with torch.enable_grad():
            z_req = z.detach().to(dtype=torch.float32).requires_grad_(True)
            t_req = (torch.as_tensor(1.0, device=device, dtype=torch.float32) - s).reshape(1).expand(z_req.shape[0])
            out = model(z_req, t_req, amh.detach())
            jvp = torch.autograd.grad(
                out,
                z_req,
                grad_outputs=eps.to(dtype=out.dtype),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            trace = (jvp * eps.to(dtype=jvp.dtype)).sum(dim=1).detach().to(dtype=torch.float64)
            del z_req, out, jvp
        z = (z - dt.to(dtype=v_value.dtype) * v_value).detach()
        log_det = (log_det + dt * trace).detach()
    nll = -standard_normal_log_prob(z) + log_det
    values = nll.detach().cpu().numpy().astype(np.float64)
    assert np.isfinite(values).all(), "sign-sanity NLL contains non-finite values"
    return float(np.mean(values))


def run_sign_sanity(
    model: ConditionalFlowVectorField,
    data: ConditionalData,
    device: torch.device,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    true_a = float(data.val_a[0, 0])
    perturbed_a = true_a + 10.0
    true_nll = euler_conditional_fm_nll(model, data.val_z[0], true_a, device, seed + 70_001, steps)
    perturbed_nll = euler_conditional_fm_nll(model, data.val_z[0], perturbed_a, device, seed + 70_001, steps)
    passed = bool(perturbed_nll > true_nll)
    assert passed, (
        "conditional FM sign sanity failed: NLL(z, a+10sigma) "
        f"{perturbed_nll} <= NLL(z, a_true) {true_nll}"
    )
    return {
        "backend": "flow_matching",
        "row": "first_nonblank_val",
        "a_true_standardized": true_a,
        "a_perturbed_standardized": perturbed_a,
        "nll_true": true_nll,
        "nll_perturbed_by_10sigma": perturbed_nll,
        "passed": passed,
        "method": f"euler_{steps}_steps_5_hutchinson",
    }


def build_acceptance_summary(
    train_loss_curve: list[float],
    val_loss_curve: list[float],
    model_path: Path,
    sign_sanity: dict[str, Any],
) -> dict[str, Any]:
    assert len(train_loss_curve) >= 1, "need at least one training loss"
    assert len(val_loss_curve) >= 2, "need at least two validation losses"
    assert all(math.isfinite(v) for v in train_loss_curve + val_loss_curve), "loss curve has non-finite values"
    initial = float(val_loss_curve[0])
    best = float(min(val_loss_curve))
    final = float(val_loss_curve[-1])
    rel_decrease = (initial - best) / max(abs(initial), 1e-12)
    summary = {
        "initial_val_loss": initial,
        "final_val_loss": final,
        "best_val_loss": best,
        "best_val_loss_relative_decrease": float(rel_decrease),
        "validation_loss_decreased_at_least_10pct": bool(rel_decrease >= 0.10),
        "final_lt_initial": bool(final < initial),
        "finite_loss_curves": True,
        "model_file_bytes": int(model_path.stat().st_size),
        "model_file_lt_30mb": bool(model_path.stat().st_size < 30_000_000),
        "sign_sanity": sign_sanity,
    }
    summary["acceptance_passed"] = bool(
        summary["validation_loss_decreased_at_least_10pct"]
        and summary["final_lt_initial"]
        and summary["model_file_lt_30mb"]
        and bool(sign_sanity["passed"])
    )
    assert summary["validation_loss_decreased_at_least_10pct"], "validation loss did not decrease by at least 10%"
    assert summary["final_lt_initial"], "final validation loss did not improve over initial validation loss"
    assert summary["model_file_lt_30mb"], "model checkpoint is >= 30 MB"
    assert summary["acceptance_passed"], "conditional FM training acceptance failed"
    return summary


def build_checkpoint_payload(
    model_state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    data: ConditionalData,
    train_loss_curve: list[float],
    val_loss_curve: list[float],
) -> dict[str, Any]:
    param_groups = optimizer_state_dict.get("param_groups", [])
    optim_state = {
        "omitted_full_state": True,
        "reason": "Inference and NLL require state_dict/config; AdamW moments are omitted to keep checkpoint compact.",
        "optimizer_class": "AdamW",
        "param_group_count": int(len(param_groups)),
        "param_groups": [
            {
                key: value
                for key, value in group.items()
                if key != "params" and isinstance(value, (int, float, str, bool, type(None)))
            }
            for group in param_groups
        ],
    }
    return {
        "state_dict": model_state_dict,
        "optim_state": optim_state,
        "config": config,
        "epoch": int(epoch),
        "amh_mean": data.amh_mean,
        "amh_std": data.amh_std,
        "train_loss_curve": [float(v) for v in train_loss_curve],
        "val_loss_curve": [float(v) for v in val_loss_curve],
    }


def train_model(
    data: ConditionalData,
    args: argparse.Namespace,
    device: torch.device,
    repo_root: Path,
    start_time: float,
) -> TrainResult:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "model_conditional.pt"
    training_log_path = out_dir / "training_log_conditional.json"
    model = ConditionalFlowVectorField(
        input_dim=EXPECTED_FEATURE_DIM,
        time_embed_dim=args.time_embed_dim,
        amh_embed_dim=args.amh_embed_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_z = torch.from_numpy(data.train_z).to(device)
    train_a = torch.from_numpy(data.train_a).to(device)
    val_z = torch.from_numpy(data.val_z).to(device)
    val_a = torch.from_numpy(data.val_a).to(device)
    batch_generator = torch.Generator(device=device).manual_seed(args.seed)
    flow_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    train_loss_curve: list[float] = []
    val_loss_curve: list[float] = []
    epoch_records: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_optim_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    n_train = train_z.shape[0]
    assert n_train > 0, "empty conditional training tensor"

    for epoch in range(1, args.epochs + 1):
        model.train()
        lr = learning_rate_for_epoch(epoch, args.lr, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        perm = torch.randperm(n_train, generator=batch_generator, device=device)
        batch_losses: list[float] = []
        for batch_start in range(0, n_train, args.batch_size):
            idx = perm[batch_start : batch_start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = conditional_flow_matching_loss(
                model=model,
                z1=train_z[idx],
                amh=train_a[idx],
                generator=flow_generator,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            assert math.isfinite(loss_value), f"non-finite train loss at epoch {epoch}"
            batch_losses.append(loss_value)
        train_loss = float(np.mean(batch_losses))
        train_loss_curve.append(train_loss)
        record: dict[str, Any] = {"epoch": int(epoch), "train_loss": train_loss, "lr": float(lr)}
        should_validate = epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs
        if should_validate:
            val_loss = evaluate_flow_loss(model, val_z, val_a, device, args.seed + 10_000 + epoch)
            val_loss_curve.append(val_loss)
            record["val_loss"] = val_loss
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                best_optim_state = copy.deepcopy(optimizer.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.val_every if epoch != 1 else 1
            print(
                "event=epoch backend=flow_matching conditional=true "
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
        epoch_records.append(record)
        if (
            epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stop_patience
            and len(val_loss_curve) >= 2
        ):
            print(
                "event=early_stop backend=flow_matching conditional=true "
                f"epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            break

    assert best_state_dict is not None, "no best model state was captured"
    assert best_optim_state is not None, "no best optimizer state was captured"
    checkpoint_config = {
        "variant": "conditional",
        "input_dim": EXPECTED_FEATURE_DIM,
        "amh_dim": EXPECTED_AMH_DIM,
        "time_embed_dim": int(args.time_embed_dim),
        "amh_embed_dim": int(args.amh_embed_dim),
        "combined_film_embed_dim": int(args.time_embed_dim + args.amh_embed_dim),
        "n_blocks": int(args.n_blocks),
        "hidden_dim": int(args.hidden_dim),
        "seed": int(args.seed),
        "epochs_trained": int(len(train_loss_curve)),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "weight_decay": float(args.weight_decay),
        "parameterization": PARAMETERIZATION,
        "conditional_amh_handling": CONDITIONAL_AMH_HANDLING,
        "state_space": "z_only",
        "divergence_wrt": "z_only",
    }
    checkpoint = build_checkpoint_payload(
        model_state_dict=best_state_dict,
        optimizer_state_dict=best_optim_state,
        config=checkpoint_config,
        epoch=best_epoch,
        data=data,
        train_loss_curve=train_loss_curve,
        val_loss_curve=val_loss_curve,
    )
    save_canonical_torch_checkpoint(checkpoint, checkpoint_path)
    model_sha = sha256_of(checkpoint_path)
    model.load_state_dict(best_state_dict, strict=True)
    sign_sanity = run_sign_sanity(model, data, device, args.seed, args.sign_sanity_steps)
    acceptance = build_acceptance_summary(train_loss_curve, val_loss_curve, checkpoint_path, sign_sanity)
    input_sha256 = {
        "embeddings_npy": sha256_of(Path(args.embeddings).resolve()),
        "embeddings_index_csv": sha256_of(Path(args.embeddings_index).resolve()),
        "train_csv": sha256_of(Path(args.splits_dir).resolve() / "train.csv"),
        "val_csv": sha256_of(Path(args.splits_dir).resolve() / "val.csv"),
        "amh_normalization_json": data.amh_normalization_sha256,
    }
    amh_recovered_path = Path(args.amh_recovered).resolve()
    if amh_recovered_path.exists():
        input_sha256["amh_recovered_csv"] = sha256_of(amh_recovered_path)
    log_payload = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "parameterization": PARAMETERIZATION,
        "conditional_amh_handling": CONDITIONAL_AMH_HANDLING,
        "input_dim": EXPECTED_FEATURE_DIM,
        "amh_dim": EXPECTED_AMH_DIM,
        "row_counts": data.row_counts,
        "config": checkpoint_config,
        "epochs": epoch_records,
        "train_loss_curve": checkpoint["train_loss_curve"],
        "val_loss_curve": checkpoint["val_loss_curve"],
        "acceptance": acceptance,
        "sign_sanity": sign_sanity,
        "input_sha256": input_sha256,
        "output_sha256": {"model_conditional_pt": model_sha},
        "amh_normalization": {
            "path": str(Path(args.amh_normalization).resolve()),
            "sha256": data.amh_normalization_sha256,
            "mean": data.amh_mean,
            "std": data.amh_std,
        },
        "code_revision": git_short_sha(repo_root),
        "git_dirty": git_dirty(repo_root),
        "library_versions": library_versions(),
        "seed": int(args.seed),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_memory": device_memory_summary(device),
        "torch_version": torch.__version__,
        "timestamp_utc": utc_now(),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    assert log_payload["device"] == "cuda"
    assert log_payload["cuda_available"] is True
    write_json(training_log_path, log_payload)
    training_log_sha = sha256_of(training_log_path)
    log_payload["output_sha256"]["training_log_conditional_json_initial"] = training_log_sha
    write_json(training_log_path, log_payload)
    training_log_sha = sha256_of(training_log_path)
    print(
        "event=train_done backend=flow_matching conditional=true "
        f"epochs={len(train_loss_curve)} best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.6f} model_sha256={model_sha} "
        f"training_log_sha256={training_log_sha}",
        flush=True,
    )
    return TrainResult(
        checkpoint_path=checkpoint_path,
        training_log_path=training_log_path,
        train_loss_curve=train_loss_curve,
        val_loss_curve=val_loss_curve,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        epochs_trained=len(train_loss_curve),
        acceptance=acceptance,
        model_sha256=model_sha,
        training_log_sha256=training_log_sha,
    )


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = assert_cuda_device(args.device)
    reset_peak_vram(device)
    repo_root = Path.cwd().resolve()
    print(
        "event=environment backend=flow_matching conditional=true "
        f"python={'.'.join(map(str, sys.version_info[:3]))} "
        f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}",
        flush=True,
    )
    data = prepare_conditional_arrays(
        embeddings_path=Path(args.embeddings).resolve(),
        embeddings_index_path=Path(args.embeddings_index).resolve(),
        splits_dir=Path(args.splits_dir).resolve(),
        amh_normalization_path=Path(args.amh_normalization).resolve(),
    )
    result = train_model(data, args, device, repo_root, start)
    assert result.acceptance["acceptance_passed"], "acceptance summary did not pass"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
