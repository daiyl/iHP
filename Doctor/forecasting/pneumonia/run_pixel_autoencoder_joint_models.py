"""Joint pixel-loss forecasting for multiple temporal models.

This is the multi-model version of the pixel-level image forecasting pipeline:

1. Pretrain/load a 9-channel image autoencoder.
2. For each temporal model, copy the pretrained encoder/decoder.
3. Jointly train Encoder + temporal model + Decoder using:

       L = pixel_loss + lambda_latent * latent_loss + lambda_recon * recon_loss

The temporal model sees past latent/action tokens and future action tokens, then
predicts future image latents. The decoder maps those predicted latents back to
future images, and pixel MSE/MAE are computed on the decoded images.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the sibling runner directly.  Some extracted copies of this project
# contain only the experiment runners in ``phenotype_decoders``; importing the
# package would execute its optional decoder imports before we reach this file.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pixel_autoencoder_dlinear as base

DEFAULT_TSL_ROOT = Path(__file__).resolve().parents[1] / "core"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "generated_observations"
    / "pneumonia_pixel_joint_models"
)
DEFAULT_MODELS = [
    "LSTM",
    "xLSTM",
    "PatchTST",
    "iTransformer",
    "Mamba",
    "SCINet",
    "TimesNet",
    "DLinear",
    "TiDE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint pixel-loss pneumonia image forecasts.")
    parser.add_argument("--source-dir", type=Path, default=base.DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tsl-root", type=Path, default=DEFAULT_TSL_ROOT)
    parser.add_argument("--models", nargs="+", default=["DLinear"])
    parser.add_argument("--image-size", type=int, choices=[64], default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--forecast-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--forecast-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--moving-avg", type=int, default=3)
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.001)
    parser.add_argument("--recon-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-recon-frames-per-batch", type=int, default=0)
    parser.add_argument("--freeze-autoencoder-during-forecast", action="store_true")
    parser.add_argument("--max-ae-train-frames", type=int, default=0)
    parser.add_argument("--max-ae-val-frames", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--retrain-ae", action="store_true")
    parser.add_argument("--retrain-forecast", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-latents",
        dest="cache_latents",
        action="store_true",
        help="Cache all frozen-encoder latents once and share them across forecast models (default).",
    )
    cache_group.add_argument(
        "--no-cache-latents",
        dest="cache_latents",
        action="store_false",
        help="Use the original per-window PNG loading and encoder execution path.",
    )
    parser.set_defaults(cache_latents=True)
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Rebuild the shared latent cache even when a compatible cache exists.",
    )
    parser.add_argument(
        "--latent-cache-batch-size",
        type=int,
        default=256,
        help="Number of image stacks encoded at once while building the latent cache.",
    )

    # Temporal model hyperparameters shared with Time-Series-Library models.
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-kernels", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--xlstm-backend", choices=["vanilla", "cuda"], default="vanilla")
    parser.add_argument("--individual", action="store_true")
    return parser.parse_args()


def expand_models(models: list[str]) -> list[str]:
    if len(models) == 1 and models[0].lower() == "all":
        return list(DEFAULT_MODELS)
    unknown = [model for model in models if model not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Choose from {DEFAULT_MODELS} or all.")
    return models


def autoencoder_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Return a stable hash so latents are never reused with different AE weights."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def latent_cache_metadata(
    args: argparse.Namespace,
    pretrained_autoencoder_state: dict[str, torch.Tensor],
    sample_dirs: list[Path],
) -> dict[str, object]:
    return {
        "version": 1,
        "source_dir": str(args.source_dir),
        "sample_names": [
            f"{sample_dir.parent.name}/{sample_dir.name}" for sample_dir in sample_dirs
        ],
        "num_sequences": len(sample_dirs),
        "time_steps": 200,
        "image_size": args.image_size,
        "latent_dim": args.latent_dim,
        "modalities": list(base.MODALITIES),
        "dtype": "float32",
        "autoencoder_sha256": autoencoder_state_sha256(pretrained_autoencoder_state),
    }


def load_or_create_latent_cache(
    args: argparse.Namespace,
    autoencoder: base.ConvImageAutoencoder,
    pretrained_autoencoder_state: dict[str, torch.Tensor],
    sample_dirs: list[Path],
    device: torch.device,
) -> np.ndarray:
    """Encode every time step once and return a read-only memory-mapped array."""
    if args.latent_cache_batch_size <= 0:
        raise ValueError("--latent-cache-batch-size must be positive")

    cache_dir = args.output_dir / "_autoencoder" / "latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_path = cache_dir / "latents.npy"
    metadata_path = cache_dir / "metadata.json"
    expected = latent_cache_metadata(args, pretrained_autoencoder_state, sample_dirs)
    expected_shape = (len(sample_dirs), 200, args.latent_dim)

    if latent_path.exists() and metadata_path.exists() and not args.reencode:
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_latents = np.load(latent_path, mmap_mode="r")
            if (
                cached_metadata == expected
                and cached_latents.shape == expected_shape
                and cached_latents.dtype == np.float32
            ):
                print(
                    f"[latent-cache] loaded {latent_path} "
                    f"shape={cached_latents.shape}"
                )
                return cached_latents
            print("[latent-cache] existing cache is incompatible; rebuilding")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[latent-cache] could not load existing cache ({exc}); rebuilding")

    building_latent_path = cache_dir / "latents.building.npy"
    building_metadata_path = cache_dir / "metadata.building.json"
    latent_array = np.lib.format.open_memmap(
        building_latent_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    print(
        f"[latent-cache] encoding {len(sample_dirs) * 200} frames once "
        f"to shape={expected_shape}"
    )
    with torch.no_grad():
        for sample_idx, sample_dir in enumerate(sample_dirs):
            for start in range(0, 200, args.latent_cache_batch_size):
                end = min(200, start + args.latent_cache_batch_size)
                images = np.stack(
                    [
                        base.load_frame_stack(sample_dir, time_idx, args.image_size)
                        for time_idx in range(start, end)
                    ],
                    axis=0,
                )
                image_tensor = torch.from_numpy(images).to(device)
                latent_array[sample_idx, start:end] = (
                    autoencoder.encode(image_tensor).detach().cpu().numpy().astype(np.float32)
                )
            if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(sample_dirs):
                print(
                    f"[latent-cache] encoded {sample_idx + 1}/{len(sample_dirs)} sequences"
                )

    latent_array.flush()
    del latent_array
    building_metadata_path.write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )
    os.replace(building_latent_path, latent_path)
    os.replace(building_metadata_path, metadata_path)

    cached_latents = np.load(latent_path, mmap_mode="r")
    print(f"[latent-cache] saved {latent_path} shape={cached_latents.shape}")
    return cached_latents


def make_tsl_config(args: argparse.Namespace, model_name: str) -> Namespace:
    input_dim = args.latent_dim + 3
    input_len = args.seq_len + args.pred_len
    return Namespace(
        task_name="long_term_forecast",
        seq_len=input_len,
        label_len=args.pred_len,
        pred_len=args.pred_len,
        enc_in=input_dim,
        dec_in=input_dim,
        c_out=1 if model_name == "TiDE" else input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        moving_avg=args.moving_avg,
        factor=args.factor,
        distil=True,
        dropout=args.dropout,
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        individual=args.individual,
        top_k=args.top_k,
        num_kernels=args.num_kernels,
        expand=args.expand,
        d_conv=args.d_conv,
        xlstm_backend=args.xlstm_backend,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        num_class=3,
    )


class TSLForecastAdapter(nn.Module):
    def __init__(self, model_name: str, args: argparse.Namespace):
        super().__init__()
        self.model_name = model_name
        self.pred_len = args.pred_len
        self.input_dim = args.latent_dim + 3
        self.latent_dim = args.latent_dim

        tsl_root = args.tsl_root.resolve()
        if not tsl_root.exists():
            raise FileNotFoundError(tsl_root)
        if str(tsl_root) not in sys.path:
            sys.path.insert(0, str(tsl_root))

        module_name = 'model_xlstm' if model_name == 'xLSTM' else model_name
        model_module = importlib.import_module('models.' + module_name)
        self.core = model_module.Model(make_tsl_config(args, model_name))
        self.output_projection = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        device = tokens.device
        x_mark = torch.zeros(batch_size, tokens.shape[1], 4, dtype=tokens.dtype, device=device)
        dec_len = self.pred_len + self.pred_len
        x_dec = torch.zeros(batch_size, dec_len, self.input_dim, dtype=tokens.dtype, device=device)
        y_mark = torch.zeros(batch_size, dec_len, 4, dtype=tokens.dtype, device=device)
        output = self.core(tokens, x_mark, x_dec, y_mark)
        if output.dim() == 4:
            output = output.squeeze(-2) if output.shape[-2] == 1 else output.flatten(2)
        output = output[:, -self.pred_len:, :]
        if output.shape[-1] < self.input_dim:
            pad = torch.zeros(
                output.shape[0],
                output.shape[1],
                self.input_dim - output.shape[-1],
                dtype=output.dtype,
                device=output.device,
            )
            output = torch.cat([output, pad], dim=-1)
        elif output.shape[-1] > self.input_dim:
            output = output[:, :, : self.input_dim]
        return self.output_projection(output)


def window_tag(args: argparse.Namespace) -> str:
    if args.max_train_windows == 0 and args.max_val_windows == 0 and args.max_test_windows == 0:
        return "full"
    return f"mw{args.max_train_windows}_{args.max_val_windows}_{args.max_test_windows}"


def make_model_args(args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    model_args = copy.copy(args)
    model_args.output_dir = args.output_dir / model_name / f"{args.seq_len}_to_{args.pred_len}"
    model_args.output_dir.mkdir(parents=True, exist_ok=True)
    return model_args


def evaluate_validation(
    args: argparse.Namespace,
    autoencoder: base.ConvImageAutoencoder,
    forecaster: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    return base.evaluate_joint_losses(args, autoencoder, forecaster, loader, device)


def cached_forward_losses(
    args: argparse.Namespace,
    autoencoder: base.ConvImageAutoencoder,
    forecaster: nn.Module,
    tokens: torch.Tensor,
    true_future_latent: torch.Tensor,
    future_pixels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Frozen-AE loss path using precomputed encoder outputs."""
    batch_size = tokens.shape[0]
    pred_len = true_future_latent.shape[1]
    pred_latent = forecaster(tokens)
    pred_pixels = autoencoder.decode(pred_latent.reshape(-1, args.latent_dim)).view(
        batch_size, pred_len, 9, args.image_size, args.image_size
    )
    pixel_loss = F.mse_loss(pred_pixels, future_pixels)
    latent_loss = F.mse_loss(pred_latent, true_future_latent)
    recon_loss = pixel_loss.new_tensor(0.0)
    total_loss = (
        args.pixel_loss_weight * pixel_loss
        + args.latent_loss_weight * latent_loss
        + args.recon_loss_weight * recon_loss
    )
    losses = {
        "total": total_loss.detach(),
        "pixel": pixel_loss.detach(),
        "latent": latent_loss.detach(),
        "recon": recon_loss.detach(),
    }
    return total_loss, losses, pred_latent, pred_pixels


def evaluate_cached_losses(
    args: argparse.Namespace,
    autoencoder: base.ConvImageAutoencoder,
    forecaster: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    autoencoder.eval()
    forecaster.eval()
    totals = {"total": 0.0, "pixel": 0.0, "latent": 0.0, "recon": 0.0}
    count = 0
    with torch.no_grad():
        for tokens, true_future_latent, future_pixels, _, _ in loader:
            tokens = tokens.to(device)
            true_future_latent = true_future_latent.to(device)
            future_pixels = future_pixels.to(device)
            _, losses, _, _ = cached_forward_losses(
                args,
                autoencoder,
                forecaster,
                tokens,
                true_future_latent,
                future_pixels,
            )
            batch_size = tokens.shape[0]
            for key in totals:
                totals[key] += losses[key].item() * batch_size
            count += batch_size
    return {key: value / count for key, value in totals.items()}


def train_one_model(
    args: argparse.Namespace,
    model_name: str,
    pretrained_autoencoder_state: dict[str, torch.Tensor],
    sample_dirs: list[Path],
    actions: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    latent_cache: np.ndarray | None,
) -> tuple[base.ConvImageAutoencoder, nn.Module]:
    model_args = make_model_args(args, model_name)
    checkpoint_path = (
        model_args.output_dir
        / f"{model_name}_joint_{args.seq_len}_to_{args.pred_len}_{window_tag(args)}.pt"
    )

    autoencoder = base.ConvImageAutoencoder(args.latent_dim).to(device)
    autoencoder.load_state_dict(pretrained_autoencoder_state)
    forecaster = TSLForecastAdapter(model_name, args).to(device)

    if checkpoint_path.exists() and not args.retrain_forecast:
        payload = torch.load(checkpoint_path, map_location=device)
        forecaster.load_state_dict(payload["forecaster"])
        autoencoder.load_state_dict(payload["autoencoder"])
        if latent_cache is not None:
            checkpoint_hash = autoencoder_state_sha256(payload["autoencoder"])
            pretrained_hash = autoencoder_state_sha256(pretrained_autoencoder_state)
            if checkpoint_hash != pretrained_hash:
                raise ValueError(
                    f"{checkpoint_path} contains a different autoencoder, so its "
                    "latents do not match the shared cache. Use --retrain-forecast "
                    "or --no-cache-latents."
                )
        print(f"[{model_name}] loaded {checkpoint_path}")
        return autoencoder, forecaster

    train_indices, val_indices, _ = split
    use_cached_latents = latent_cache is not None
    if use_cached_latents:
        train_dataset = base.LatentActionWindowDataset(
            latent_cache,
            actions,
            sample_dirs,
            train_indices,
            args.seq_len,
            args.pred_len,
            args.image_size,
            include_pixels=True,
            max_windows=args.max_train_windows,
            seed=args.seed,
        )
        val_dataset = base.LatentActionWindowDataset(
            latent_cache,
            actions,
            sample_dirs,
            val_indices,
            args.seq_len,
            args.pred_len,
            args.image_size,
            include_pixels=True,
            max_windows=args.max_val_windows,
            seed=args.seed + 1,
        )
    else:
        train_dataset = base.PixelActionWindowDataset(
            sample_dirs,
            actions,
            train_indices,
            args.seq_len,
            args.pred_len,
            args.image_size,
            args.max_train_windows,
            args.seed,
        )
        val_dataset = base.PixelActionWindowDataset(
            sample_dirs,
            actions,
            val_indices,
            args.seq_len,
            args.pred_len,
            args.image_size,
            args.max_val_windows,
            args.seed + 1,
        )
    train_loader = DataLoader(train_dataset, batch_size=args.forecast_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.forecast_batch_size, shuffle=False, num_workers=0)

    for parameter in autoencoder.parameters():
        parameter.requires_grad_(not args.freeze_autoencoder_during_forecast)
    parameters = list(forecaster.parameters())
    if not args.freeze_autoencoder_during_forecast:
        parameters += list(autoencoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_forecaster = None
    best_autoencoder = None
    stale = 0
    for epoch in range(1, args.forecast_epochs + 1):
        forecaster.train()
        if args.freeze_autoencoder_during_forecast:
            autoencoder.eval()
        else:
            autoencoder.train()

        running = {"total": 0.0, "pixel": 0.0, "latent": 0.0, "recon": 0.0}
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_cached_latents:
                tokens, true_future_latent, future_pixels, _, _ = batch
                tokens = tokens.to(device)
                true_future_latent = true_future_latent.to(device)
                future_pixels = future_pixels.to(device)
                total_loss, losses, _, _ = cached_forward_losses(
                    args,
                    autoencoder,
                    forecaster,
                    tokens,
                    true_future_latent,
                    future_pixels,
                )
                batch_size = tokens.shape[0]
            else:
                past_pixels, future_pixels, past_actions, future_actions, _, _ = batch
                past_pixels = past_pixels.to(device)
                future_pixels = future_pixels.to(device)
                past_actions = past_actions.to(device)
                future_actions = future_actions.to(device)
                total_loss, losses, _, _ = base.joint_forward_losses(
                    args,
                    autoencoder,
                    forecaster,
                    past_pixels,
                    future_pixels,
                    past_actions,
                    future_actions,
                    include_recon=not args.freeze_autoencoder_during_forecast,
                )
                batch_size = past_pixels.shape[0]
            total_loss.backward()
            optimizer.step()

            for key in running:
                running[key] += losses[key].item() * batch_size
            count += batch_size

        train_metrics = {key: value / count for key, value in running.items()}
        if use_cached_latents:
            val_metrics = evaluate_cached_losses(
                args, autoencoder, forecaster, val_loader, device
            )
        else:
            val_metrics = evaluate_validation(
                args, autoencoder, forecaster, val_loader, device
            )
        print(
            "[{model}] epoch={epoch} train_total={train_total:.6f} train_pixel={train_pixel:.6f} "
            "val_total={val_total:.6f} val_pixel={val_pixel:.6f} val_latent={val_latent:.6f} "
            "val_recon={val_recon:.6f}".format(
                model=model_name,
                epoch=epoch,
                train_total=train_metrics["total"],
                train_pixel=train_metrics["pixel"],
                val_total=val_metrics["total"],
                val_pixel=val_metrics["pixel"],
                val_latent=val_metrics["latent"],
                val_recon=val_metrics["recon"],
            )
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_forecaster = {key: value.detach().cpu() for key, value in forecaster.state_dict().items()}
            best_autoencoder = {key: value.detach().cpu() for key, value in autoencoder.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}")
                break

    if best_forecaster is not None:
        forecaster.load_state_dict(best_forecaster)
    if best_autoencoder is not None:
        autoencoder.load_state_dict(best_autoencoder)

    torch.save(
        {
            "model_name": model_name,
            "forecaster": forecaster.state_dict(),
            "autoencoder": autoencoder.state_dict(),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "latent_dim": args.latent_dim,
            "loss_weights": {
                "pixel": args.pixel_loss_weight,
                "latent": args.latent_loss_weight,
                "recon": args.recon_loss_weight,
            },
            "latent_cache": use_cached_latents,
        },
        checkpoint_path,
    )
    print(f"[{model_name}] saved {checkpoint_path}")
    return autoencoder, forecaster


def evaluate_cached_pixel_forecast(
    args: argparse.Namespace,
    autoencoder: base.ConvImageAutoencoder,
    forecaster: nn.Module,
    latent_cache: np.ndarray,
    sample_dirs: list[Path],
    actions: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> base.PixelMetrics:
    dataset = base.LatentActionWindowDataset(
        latent_cache,
        actions,
        sample_dirs,
        test_indices,
        args.seq_len,
        args.pred_len,
        args.image_size,
        include_pixels=True,
        max_windows=args.max_test_windows,
        seed=args.seed + 2,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.forecast_batch_size,
        shuffle=False,
        num_workers=0,
    )
    autoencoder.eval()
    forecaster.eval()

    latent_sse = latent_sae = 0.0
    latent_count = 0
    pixel_sse = pixel_sae = 0.0
    pixel_count = 0
    preview_saved = 0
    pred_latents = []
    true_latents = []
    preview_dir = args.output_dir / f"previews_joint_{args.seq_len}_to_{args.pred_len}"

    with torch.no_grad():
        for batch_idx, (tokens, true_latent, true_pixels, _, _) in enumerate(loader):
            tokens = tokens.to(device)
            true_latent = true_latent.to(device)
            true_pixels = true_pixels.to(device)
            pred_latent = forecaster(tokens)
            pred_images = autoencoder.decode(
                pred_latent.reshape(-1, args.latent_dim)
            ).view(
                pred_latent.shape[0],
                args.pred_len,
                9,
                args.image_size,
                args.image_size,
            )

            latent_diff = pred_latent - true_latent
            latent_sse += torch.square(latent_diff).sum().item()
            latent_sae += torch.abs(latent_diff).sum().item()
            latent_count += true_latent.numel()

            pixel_diff = pred_images - true_pixels
            pixel_sse += torch.square(pixel_diff).sum().item()
            pixel_sae += torch.abs(pixel_diff).sum().item()
            pixel_count += true_pixels.numel()

            pred_latents.append(pred_latent.detach().cpu().numpy())
            true_latents.append(true_latent.detach().cpu().numpy())

            previews_in_batch = min(
                args.preview_count - preview_saved, pred_latent.shape[0]
            )
            for item_idx in range(max(0, previews_in_batch)):
                item_dir = preview_dir / f"case_{batch_idx:04d}_{item_idx:02d}"
                for horizon_idx in range(args.pred_len):
                    base.save_frame_stack(
                        pred_images[item_idx, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"pred_h{horizon_idx:02d}",
                    )
                    base.save_frame_stack(
                        true_pixels[item_idx, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"true_h{horizon_idx:02d}",
                    )
                preview_saved += 1

    np.save(
        args.output_dir / f"pred_latents_joint_{args.seq_len}_to_{args.pred_len}.npy",
        np.concatenate(pred_latents, axis=0),
    )
    np.save(
        args.output_dir / f"true_latents_joint_{args.seq_len}_to_{args.pred_len}.npy",
        np.concatenate(true_latents, axis=0),
    )
    return base.PixelMetrics(
        latent_mse=latent_sse / latent_count,
        latent_mae=latent_sae / latent_count,
        pixel_mse=pixel_sse / pixel_count,
        pixel_mae=pixel_sae / pixel_count,
    )


def write_summary_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = [
        "model",
        "seq_len",
        "pred_len",
        "latent_mse",
        "latent_mae",
        "pixel_mse",
        "pixel_rmse",
        "pixel_mae",
        "output_dir",
    ]
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.tsl_root = args.tsl_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.seed_everything(args.seed)
    device = base.resolve_device(args.device)
    models = expand_models(args.models)
    print(f"[device] {device}")
    print(f"[models] {models}")
    print(f"[tsl_root] {args.tsl_root}")

    sample_dirs = base.discover_sample_dirs(args.source_dir)
    split = base.split_indices(len(sample_dirs), args.seed, args.train_ratio, args.val_ratio)
    train_indices, val_indices, test_indices = split
    actions = np.stack([base.read_actions(sample_dir) for sample_dir in sample_dirs], axis=0)

    autoencoder_args = copy.copy(args)
    autoencoder_args.output_dir = args.output_dir / "_autoencoder"
    autoencoder_args.output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_autoencoder = base.train_autoencoder(
        autoencoder_args, sample_dirs, train_indices, val_indices, device
    )
    pretrained_state = {
        key: value.detach().cpu().clone()
        for key, value in pretrained_autoencoder.state_dict().items()
    }

    latent_cache = None
    if args.cache_latents and args.freeze_autoencoder_during_forecast:
        latent_cache = load_or_create_latent_cache(
            args,
            pretrained_autoencoder,
            pretrained_state,
            sample_dirs,
            device,
        )
    elif args.cache_latents:
        print(
            "[latent-cache] disabled because the autoencoder is being fine-tuned; "
            "use --freeze-autoencoder-during-forecast to enable safe caching"
        )
    else:
        print("[latent-cache] disabled by --no-cache-latents")

    metadata = {
        "source_dir": str(args.source_dir),
        "tsl_root": str(args.tsl_root),
        "models": models,
        "num_sequences": len(sample_dirs),
        "time_steps": 200,
        "image_shape": [9, args.image_size, args.image_size],
        "latent_dim": args.latent_dim,
        "action_dim": 3,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "freeze_autoencoder_during_forecast": args.freeze_autoencoder_during_forecast,
        "latent_cache_enabled": latent_cache is not None,
        "loss": "pixel + latent_weight * latent + recon_weight * recon",
        "loss_weights": {
            "pixel": args.pixel_loss_weight,
            "latent": args.latent_loss_weight,
            "recon": args.recon_loss_weight,
        },
        "split_counts": {
            "train": int(train_indices.size),
            "val": int(val_indices.size),
            "test": int(test_indices.size),
        },
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    summary_path = args.output_dir / f"summary_{args.seq_len}_to_{args.pred_len}.csv"
    for model_name in models:
        print(f"[start] {model_name}")
        autoencoder, forecaster = train_one_model(
            args,
            model_name,
            pretrained_state,
            sample_dirs,
            actions,
            split,
            device,
            latent_cache,
        )
        model_args = make_model_args(args, model_name)
        if latent_cache is not None:
            metrics = evaluate_cached_pixel_forecast(
                model_args,
                autoencoder,
                forecaster,
                latent_cache,
                sample_dirs,
                actions,
                test_indices,
                device,
            )
        else:
            metrics = base.evaluate_joint_pixel_forecast(
                model_args,
                autoencoder,
                forecaster,
                sample_dirs,
                actions,
                test_indices,
                device,
            )
        metrics_payload = {
            "model": model_name,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "latent_mse": metrics.latent_mse,
            "latent_mae": metrics.latent_mae,
            "pixel_mse": metrics.pixel_mse,
            "pixel_rmse": float(np.sqrt(metrics.pixel_mse)),
            "pixel_mae": metrics.pixel_mae,
            "latent_cache": latent_cache is not None,
            "loss_weights": {
                "pixel": args.pixel_loss_weight,
                "latent": args.latent_loss_weight,
                "recon": args.recon_loss_weight,
            },
        }
        metrics_path = model_args.output_dir / f"pixel_metrics_joint_{args.seq_len}_to_{args.pred_len}.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        write_summary_row(
            summary_path,
            {
                "model": model_name,
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "latent_mse": metrics.latent_mse,
                "latent_mae": metrics.latent_mae,
                "pixel_mse": metrics.pixel_mse,
                "pixel_rmse": float(np.sqrt(metrics.pixel_mse)),
                "pixel_mae": metrics.pixel_mae,
                "output_dir": str(model_args.output_dir),
            },
        )
        print(f"[done] {model_name} pixel_mse={metrics.pixel_mse:.6f} pixel_mae={metrics.pixel_mae:.6f}")

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
