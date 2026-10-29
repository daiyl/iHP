"""Pixel-level image forecasting with an image autoencoder and DLinear.

Pipeline:
1. Train/load a convolutional autoencoder for the three image modalities.
2. Encode every 200-step image sequence into latent vectors.
3. Train a DLinear-style forecaster on [latent, action] tokens.
4. Decode predicted future latents back to images.
5. Report pixel-level MSE/MAE against the true future images.

Each frame is represented as a 9-channel image:
pneumonia RGB + fundus RGB + oct RGB.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_SOURCE_DIR = (
    Path(__file__).resolve().parent
    / "generated_observations"
    / "pneumonia_200_sequences"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "generated_observations"
    / "pneumonia_pixel_dlinear"
)
MODALITIES = ("pneumonia", "fundus", "oct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pixel-level pneumonia image forecasting with DLinear.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, choices=[64], default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--ae-epochs", type=int, default=8)
    parser.add_argument("--forecast-epochs", type=int, default=10)
    parser.add_argument("--forecast-loss", choices=("pixel", "latent", "combined"), default="latent")
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--forecast-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--moving-avg", type=int, default=3)
    parser.add_argument("--loss-mode", choices=("latent", "joint"), default="joint")
    parser.add_argument("--recon-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--freeze-autoencoder-during-forecast",
        action="store_true",
        help="Keep encoder/decoder fixed during joint forecasting training.",
    )
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument(
        "--max-recon-frames-per-batch",
        type=int,
        default=64,
        help="Maximum image frames used for the AE reconstruction term in each joint batch.",
    )
    parser.add_argument("--max-ae-train-frames", type=int, default=0)
    parser.add_argument("--max-ae-val-frames", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--retrain-ae", action="store_true")
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--retrain-forecast", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(choice)


def discover_sample_dirs(source_dir: Path) -> list[Path]:
    sample_dirs = sorted(
        path
        for path in source_dir.glob("*/*")
        if path.is_dir() and path.name.startswith("sample_")
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* directories found under {source_dir}")
    return sample_dirs


def split_indices(num_samples: int, seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_end = int(round(num_samples * train_ratio))
    val_end = train_end + int(round(num_samples * val_ratio))
    return (
        np.sort(indices[:train_end]).astype(np.int64),
        np.sort(indices[train_end:val_end]).astype(np.int64),
        np.sort(indices[val_end:]).astype(np.int64),
    )


def read_actions(sample_dir: Path) -> np.ndarray:
    csv_path = sample_dir / "observation_sequence.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append([float(row["a1"]), float(row["a2"]), float(row["a3"])])
    if len(rows) != 200:
        raise ValueError(f"{csv_path} should contain 200 rows, got {len(rows)}")
    return np.asarray(rows, dtype=np.float32)


def load_frame_stack(sample_dir: Path, time_idx: int, image_size: int) -> np.ndarray:
    channels = []
    resample = getattr(Image, "Resampling", Image).BILINEAR
    filename = f"frame_{time_idx:04d}.png"
    for modality in MODALITIES:
        path = sample_dir / modality / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size), resample)
            array = np.asarray(image, dtype=np.float32) / 255.0
        channels.append(np.transpose(array, (2, 0, 1)))
    return np.concatenate(channels, axis=0).astype(np.float32)


def tensor_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0.0, 1.0)
    array = np.transpose(array, (1, 2, 0))
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def save_frame_stack(stack: np.ndarray, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, modality in enumerate(MODALITIES):
        image = tensor_to_image(stack[idx * 3 : (idx + 1) * 3])
        image.save(output_dir / f"{prefix}_{modality}.png")


class FrameDataset(Dataset):
    def __init__(
        self,
        sample_dirs: list[Path],
        indices: Iterable[int],
        image_size: int,
        max_frames: int,
        seed: int,
    ) -> None:
        self.sample_dirs = sample_dirs
        self.image_size = image_size
        self.items = [(int(sample_idx), time_idx) for sample_idx in indices for time_idx in range(200)]
        if max_frames > 0 and len(self.items) > max_frames:
            rng = random.Random(seed)
            self.items = rng.sample(self.items, max_frames)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample_idx, time_idx = self.items[index]
        return torch.from_numpy(load_frame_stack(self.sample_dirs[sample_idx], time_idx, self.image_size))


class ConvImageAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(128 * 4 * 4, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder_conv(image)
        return self.encoder_fc(hidden.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_fc(latent).view(latent.shape[0], 128, 4, 4)
        return self.decoder_conv(hidden)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image))


def evaluate_autoencoder(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    count = 0
    with torch.no_grad():
        for image in loader:
            image = image.to(device)
            recon = model(image)
            diff = recon - image
            sse += torch.square(diff).sum().item()
            sae += torch.abs(diff).sum().item()
            count += image.numel()
    return sse / count, sae / count


def train_autoencoder(
    args: argparse.Namespace,
    sample_dirs: list[Path],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
) -> ConvImageAutoencoder:
    checkpoint_path = args.output_dir / "autoencoder.pt"
    model = ConvImageAutoencoder(args.latent_dim).to(device)
    if checkpoint_path.exists() and not args.retrain_ae:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"])
        print(f"[autoencoder] loaded {checkpoint_path}")
        return model

    train_dataset = FrameDataset(
        sample_dirs, train_indices, args.image_size, args.max_ae_train_frames, args.seed
    )
    val_dataset = FrameDataset(
        sample_dirs, val_indices, args.image_size, args.max_ae_val_frames, args.seed + 1
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.ae_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for image in train_loader:
            image = image.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(image)
            loss = F.mse_loss(recon, image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.shape[0]
            total_count += image.shape[0]
        train_loss = total_loss / total_count
        val_mse, val_mae = evaluate_autoencoder(model, val_loader, device)
        print(f"[autoencoder] epoch={epoch} train_mse={train_loss:.6f} val_mse={val_mse:.6f} val_mae={val_mae:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "modalities": MODALITIES,
        },
        checkpoint_path,
    )
    print(f"[autoencoder] saved {checkpoint_path}")
    return model


def encode_sequences(
    args: argparse.Namespace,
    model: ConvImageAutoencoder,
    sample_dirs: list[Path],
    actions: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    latent_path = args.output_dir / "latent_dataset.npz"
    if latent_path.exists() and not args.reencode:
        loaded = np.load(latent_path, allow_pickle=True)
        if int(loaded["latent_dim"]) == args.latent_dim:
            print(f"[latent] loaded {latent_path}")
            return {key: loaded[key] for key in loaded.files}

    model.eval()
    sequences = []
    with torch.no_grad():
        for sample_idx, sample_dir in enumerate(sample_dirs):
            latents = []
            for start in range(0, 200, args.batch_size):
                end = min(200, start + args.batch_size)
                images = np.stack(
                    [load_frame_stack(sample_dir, time_idx, args.image_size) for time_idx in range(start, end)],
                    axis=0,
                )
                image_tensor = torch.from_numpy(images).to(device)
                latent = model.encode(image_tensor).detach().cpu().numpy()
                latents.append(latent)
            sequences.append(np.concatenate(latents, axis=0))
            if (sample_idx + 1) % 10 == 0:
                print(f"[latent] encoded {sample_idx + 1}/{len(sample_dirs)} sequences")

    train_indices, val_indices, test_indices = split
    latent_array = np.stack(sequences, axis=0).astype(np.float32)
    run_names = np.array(
        [f"{sample_dir.parent.name}/{sample_dir.name}" for sample_dir in sample_dirs],
        dtype=object,
    )
    np.savez_compressed(
        latent_path,
        sequences=latent_array,
        actions=actions.astype(np.float32),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        run_names=run_names,
        latent_dim=np.array(args.latent_dim, dtype=np.int64),
        image_size=np.array(args.image_size, dtype=np.int64),
    )
    print(f"[latent] saved {latent_path}")
    return dict(
        sequences=latent_array,
        actions=actions.astype(np.float32),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        run_names=run_names,
        latent_dim=np.array(args.latent_dim, dtype=np.int64),
        image_size=np.array(args.image_size, dtype=np.int64),
    )


class LatentActionWindowDataset(Dataset):
    def __init__(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        sample_dirs: list[Path],
        indices: Iterable[int],
        seq_len: int,
        pred_len: int,
        image_size: int,
        include_pixels: bool = False,
        max_windows: int = 0,
        seed: int = 0,
    ) -> None:
        self.latents = latents
        self.actions = actions
        self.sample_dirs = sample_dirs
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.image_size = image_size
        self.include_pixels = include_pixels
        self.samples = []
        for sample_idx in indices:
            for start in range(200 - seq_len - pred_len + 1):
                self.samples.append((int(sample_idx), start))
        if max_windows > 0 and len(self.samples) > max_windows:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_windows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_idx, start = self.samples[index]
        past_end = start + self.seq_len
        pred_end = past_end + self.pred_len

        past_latent = self.latents[sample_idx, start:past_end]
        past_action = self.actions[sample_idx, start:past_end]
        future_action = self.actions[sample_idx, past_end:pred_end]
        future_latent_zeros = np.zeros((self.pred_len, self.latents.shape[-1]), dtype=np.float32)

        past_tokens = np.concatenate([past_latent, past_action], axis=-1)
        future_tokens = np.concatenate([future_latent_zeros, future_action], axis=-1)
        x = np.concatenate([past_tokens, future_tokens], axis=0).astype(np.float32)
        y = self.latents[sample_idx, past_end:pred_end].astype(np.float32)

        if not self.include_pixels:
            return torch.from_numpy(x), torch.from_numpy(y)

        pixels = np.stack(
            [load_frame_stack(self.sample_dirs[sample_idx], time_idx, self.image_size) for time_idx in range(past_end, pred_end)],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(pixels), sample_idx, start


class PixelActionWindowDataset(Dataset):
    def __init__(
        self,
        sample_dirs: list[Path],
        actions: np.ndarray,
        indices: Iterable[int],
        seq_len: int,
        pred_len: int,
        image_size: int,
        max_windows: int,
        seed: int,
    ) -> None:
        self.sample_dirs = sample_dirs
        self.actions = actions
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.image_size = image_size
        self.samples = []
        for sample_idx in indices:
            for start in range(200 - seq_len - pred_len + 1):
                self.samples.append((int(sample_idx), start))
        if max_windows > 0 and len(self.samples) > max_windows:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_windows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_idx, start = self.samples[index]
        past_end = start + self.seq_len
        pred_end = past_end + self.pred_len
        past_pixels = np.stack(
            [load_frame_stack(self.sample_dirs[sample_idx], time_idx, self.image_size) for time_idx in range(start, past_end)],
            axis=0,
        ).astype(np.float32)
        future_pixels = np.stack(
            [load_frame_stack(self.sample_dirs[sample_idx], time_idx, self.image_size) for time_idx in range(past_end, pred_end)],
            axis=0,
        ).astype(np.float32)
        past_actions = self.actions[sample_idx, start:past_end].astype(np.float32)
        future_actions = self.actions[sample_idx, past_end:pred_end].astype(np.float32)
        return (
            torch.from_numpy(past_pixels),
            torch.from_numpy(future_pixels),
            torch.from_numpy(past_actions),
            torch.from_numpy(future_actions),
            sample_idx,
            start,
        )


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=self.kernel_size, stride=1)
        return x.permute(0, 2, 1)


class ConditionalDLinear(nn.Module):
    """DLinear temporal core with a channel mixer for action-conditioned output."""

    def __init__(self, input_len: int, pred_len: int, input_dim: int, target_dim: int, moving_avg: int):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.moving_average = MovingAverage(moving_avg)
        self.linear_seasonal = nn.Linear(input_len, pred_len)
        self.linear_trend = nn.Linear(input_len, pred_len)
        self.channel_projection = nn.Linear(input_dim, target_dim)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / input_len)
        nn.init.constant_(self.linear_trend.weight, 1.0 / input_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        seasonal = x - trend
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend = self.linear_trend(trend.permute(0, 2, 1))
        out = (seasonal + trend).permute(0, 2, 1)
        return self.channel_projection(out)


def evaluate_forecast_latent(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            diff = pred - y
            sse += torch.square(diff).sum().item()
            sae += torch.abs(diff).sum().item()
            count += y.numel()
    return sse / count, sae / count


def compute_forecast_loss(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    pred_latent: torch.Tensor,
    true_latent: torch.Tensor,
    true_pixels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    latent_loss = F.mse_loss(pred_latent, true_latent)
    if args.forecast_loss == "latent":
        return latent_loss, latent_loss.detach(), None

    if true_pixels is None:
        raise ValueError("Pixel or combined forecast loss requires true future pixels")

    pred_pixels = autoencoder.decode(pred_latent.reshape(-1, args.latent_dim))
    pred_pixels = pred_pixels.view(
        pred_latent.shape[0], args.pred_len, 9, args.image_size, args.image_size
    )
    pixel_loss = F.mse_loss(pred_pixels, true_pixels)
    if args.forecast_loss == "pixel":
        return args.pixel_loss_weight * pixel_loss, latent_loss.detach(), pixel_loss.detach()

    total_loss = args.pixel_loss_weight * pixel_loss + args.latent_loss_weight * latent_loss
    return total_loss, latent_loss.detach(), pixel_loss.detach()


def evaluate_forecast_training_loss(
    args: argparse.Namespace,
    model: nn.Module,
    autoencoder: ConvImageAutoencoder,
    loader: DataLoader,
    device: torch.device,
    include_pixels: bool,
) -> tuple[float, float, float | None]:
    model.eval()
    autoencoder.eval()
    total_loss = 0.0
    total_latent_loss = 0.0
    total_pixel_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            if include_pixels:
                x, y, true_pixels, _, _ = batch
                true_pixels = true_pixels.to(device)
            else:
                x, y = batch
                true_pixels = None
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss, latent_loss, pixel_loss = compute_forecast_loss(
                args, autoencoder, pred, y, true_pixels
            )
            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_latent_loss += latent_loss.item() * batch_size
            if pixel_loss is not None:
                total_pixel_loss += pixel_loss.item() * batch_size
            total_count += batch_size

    avg_pixel = None if args.forecast_loss == "latent" else total_pixel_loss / total_count
    return total_loss / total_count, total_latent_loss / total_count, avg_pixel


def train_forecaster(
    args: argparse.Namespace,
    latent_data: dict[str, np.ndarray],
    sample_dirs: list[Path],
    autoencoder: ConvImageAutoencoder,
    device: torch.device,
) -> ConditionalDLinear:
    checkpoint_path = args.output_dir / f"dlinear_{args.forecast_loss}_{args.seq_len}_to_{args.pred_len}.pt"
    input_dim = args.latent_dim + 3
    input_len = args.seq_len + args.pred_len
    model = ConditionalDLinear(input_len, args.pred_len, input_dim, args.latent_dim, args.moving_avg).to(device)
    if checkpoint_path.exists() and not args.retrain_forecast:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"])
        print(f"[forecast] loaded {checkpoint_path}")
        return model

    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    include_pixels = args.forecast_loss in ("pixel", "combined")
    train_dataset = LatentActionWindowDataset(
        latent_data["sequences"],
        latent_data["actions"],
        sample_dirs,
        latent_data["train_indices"],
        args.seq_len,
        args.pred_len,
        args.image_size,
        include_pixels=include_pixels,
    )
    val_dataset = LatentActionWindowDataset(
        latent_data["sequences"],
        latent_data["actions"],
        sample_dirs,
        latent_data["val_indices"],
        args.seq_len,
        args.pred_len,
        args.image_size,
        include_pixels=include_pixels,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.forecast_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.forecast_batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.forecast_epochs + 1):
        model.train()
        total_loss = 0.0
        total_latent_loss = 0.0
        total_pixel_loss = 0.0
        total_count = 0
        for batch in train_loader:
            if include_pixels:
                x, y, true_pixels, _, _ = batch
                true_pixels = true_pixels.to(device)
            else:
                x, y = batch
                true_pixels = None
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, latent_loss, pixel_loss = compute_forecast_loss(
                args, autoencoder, pred, y, true_pixels
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.shape[0]
            total_latent_loss += latent_loss.item() * y.shape[0]
            if pixel_loss is not None:
                total_pixel_loss += pixel_loss.item() * y.shape[0]
            total_count += y.shape[0]
        train_loss = total_loss / total_count
        train_latent = total_latent_loss / total_count
        train_pixel = None if not include_pixels else total_pixel_loss / total_count
        val_loss, val_latent, val_pixel = evaluate_forecast_training_loss(
            args, model, autoencoder, val_loader, device, include_pixels
        )
        message = (
            f"[forecast] epoch={epoch} train_loss={train_loss:.6f} "
            f"train_latent_mse={train_latent:.6f} val_loss={val_loss:.6f} "
            f"val_latent_mse={val_latent:.6f}"
        )
        if train_pixel is not None and val_pixel is not None:
            message += f" train_pixel_mse={train_pixel:.6f} val_pixel_mse={val_pixel:.6f}"
        print(message)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "latent_dim": args.latent_dim,
            "input_dim": input_dim,
            "forecast_loss": args.forecast_loss,
            "pixel_loss_weight": args.pixel_loss_weight,
            "latent_loss_weight": args.latent_loss_weight,
        },
        checkpoint_path,
    )
    print(f"[forecast] saved {checkpoint_path}")
    return model


def build_forecast_tokens(
    past_latent: torch.Tensor,
    past_actions: torch.Tensor,
    future_actions: torch.Tensor,
) -> torch.Tensor:
    future_latent = torch.zeros(
        future_actions.shape[0],
        future_actions.shape[1],
        past_latent.shape[-1],
        dtype=past_latent.dtype,
        device=past_latent.device,
    )
    past_tokens = torch.cat([past_latent, past_actions], dim=-1)
    future_tokens = torch.cat([future_latent, future_actions], dim=-1)
    return torch.cat([past_tokens, future_tokens], dim=1)


def joint_forward_losses(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    forecaster: ConditionalDLinear,
    past_pixels: torch.Tensor,
    future_pixels: torch.Tensor,
    past_actions: torch.Tensor,
    future_actions: torch.Tensor,
    include_recon: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch_size = past_pixels.shape[0]
    pred_len = future_pixels.shape[1]
    past_flat = past_pixels.reshape(-1, 9, args.image_size, args.image_size)
    future_flat = future_pixels.reshape(-1, 9, args.image_size, args.image_size)

    past_latent = autoencoder.encode(past_flat).view(batch_size, args.seq_len, args.latent_dim)
    with torch.no_grad():
        true_future_latent = autoencoder.encode(future_flat).view(batch_size, pred_len, args.latent_dim)

    tokens = build_forecast_tokens(past_latent, past_actions, future_actions)
    pred_latent = forecaster(tokens)
    pred_pixels = autoencoder.decode(pred_latent.reshape(-1, args.latent_dim)).view(
        batch_size, pred_len, 9, args.image_size, args.image_size
    )

    pixel_loss = F.mse_loss(pred_pixels, future_pixels)
    latent_loss = F.mse_loss(pred_latent, true_future_latent)
    if include_recon and args.recon_loss_weight > 0:
        recon_frames = torch.cat([past_flat, future_flat], dim=0)
        if args.max_recon_frames_per_batch > 0 and recon_frames.shape[0] > args.max_recon_frames_per_batch:
            frame_idx = torch.randperm(recon_frames.shape[0], device=recon_frames.device)[
                : args.max_recon_frames_per_batch
            ]
            recon_frames = recon_frames[frame_idx]
        recon_output = autoencoder(recon_frames)
        recon_loss = F.mse_loss(recon_output, recon_frames)
    else:
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


def evaluate_joint_losses(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    forecaster: ConditionalDLinear,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    autoencoder.eval()
    forecaster.eval()
    totals = {"total": 0.0, "pixel": 0.0, "latent": 0.0, "recon": 0.0}
    count = 0
    with torch.no_grad():
        for past_pixels, future_pixels, past_actions, future_actions, _, _ in loader:
            past_pixels = past_pixels.to(device)
            future_pixels = future_pixels.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)
            _, losses, _, _ = joint_forward_losses(
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
            for key in totals:
                totals[key] += losses[key].item() * batch_size
            count += batch_size
    return {key: value / count for key, value in totals.items()}


def train_joint_forecaster(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    sample_dirs: list[Path],
    actions: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> ConditionalDLinear:
    window_tag = (
        "full"
        if args.max_train_windows == 0 and args.max_val_windows == 0 and args.max_test_windows == 0
        else f"mw{args.max_train_windows}_{args.max_val_windows}_{args.max_test_windows}"
    )
    checkpoint_path = args.output_dir / f"dlinear_joint_{args.seq_len}_to_{args.pred_len}_{window_tag}.pt"
    input_dim = args.latent_dim + 3
    input_len = args.seq_len + args.pred_len
    forecaster = ConditionalDLinear(input_len, args.pred_len, input_dim, args.latent_dim, args.moving_avg).to(device)
    if checkpoint_path.exists() and not args.retrain_forecast:
        payload = torch.load(checkpoint_path, map_location=device)
        forecaster.load_state_dict(payload["forecaster"])
        if "autoencoder" in payload:
            autoencoder.load_state_dict(payload["autoencoder"])
        print(f"[joint] loaded {checkpoint_path}")
        return forecaster

    train_indices, val_indices, _ = split
    train_dataset = PixelActionWindowDataset(
        sample_dirs,
        actions,
        train_indices,
        args.seq_len,
        args.pred_len,
        args.image_size,
        args.max_train_windows,
        args.seed,
    )
    val_dataset = PixelActionWindowDataset(
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
        if args.freeze_autoencoder_during_forecast:
            autoencoder.eval()
        else:
            autoencoder.train()
        forecaster.train()
        running = {"total": 0.0, "pixel": 0.0, "latent": 0.0, "recon": 0.0}
        count = 0
        for past_pixels, future_pixels, past_actions, future_actions, _, _ in train_loader:
            past_pixels = past_pixels.to(device)
            future_pixels = future_pixels.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            total_loss, losses, _, _ = joint_forward_losses(
                args,
                autoencoder,
                forecaster,
                past_pixels,
                future_pixels,
                past_actions,
                future_actions,
                include_recon=not args.freeze_autoencoder_during_forecast,
            )
            total_loss.backward()
            optimizer.step()
            batch_size = past_pixels.shape[0]
            for key in running:
                running[key] += losses[key].item() * batch_size
            count += batch_size

        train_metrics = {key: value / count for key, value in running.items()}
        val_metrics = evaluate_joint_losses(args, autoencoder, forecaster, val_loader, device)
        print(
            "[joint] epoch={epoch} "
            "train_total={train_total:.6f} train_pixel={train_pixel:.6f} "
            "val_total={val_total:.6f} val_pixel={val_pixel:.6f} "
            "val_latent={val_latent:.6f} val_recon={val_recon:.6f}".format(
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
                break

    if best_forecaster is not None:
        forecaster.load_state_dict(best_forecaster)
    if best_autoencoder is not None:
        autoencoder.load_state_dict(best_autoencoder)
    torch.save(
        {
            "forecaster": forecaster.state_dict(),
            "autoencoder": autoencoder.state_dict(),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "latent_dim": args.latent_dim,
            "loss_mode": "joint",
            "loss_weights": {
                "pixel": args.pixel_loss_weight,
                "latent": args.latent_loss_weight,
                "recon": args.recon_loss_weight,
            },
            "freeze_autoencoder": args.freeze_autoencoder_during_forecast,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "model": autoencoder.state_dict(),
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "modalities": MODALITIES,
            "joint_finetuned": not args.freeze_autoencoder_during_forecast,
        },
        args.output_dir / "autoencoder_joint.pt",
    )
    print(f"[joint] saved {checkpoint_path}")
    return forecaster


@dataclass
class PixelMetrics:
    latent_mse: float
    latent_mae: float
    pixel_mse: float
    pixel_mae: float


def evaluate_pixel_forecast(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    forecaster: ConditionalDLinear,
    latent_data: dict[str, np.ndarray],
    sample_dirs: list[Path],
    device: torch.device,
) -> PixelMetrics:
    dataset = LatentActionWindowDataset(
        latent_data["sequences"],
        latent_data["actions"],
        sample_dirs,
        latent_data["test_indices"],
        args.seq_len,
        args.pred_len,
        args.image_size,
        include_pixels=True,
    )
    loader = DataLoader(dataset, batch_size=args.forecast_batch_size, shuffle=False, num_workers=0)
    autoencoder.eval()
    forecaster.eval()

    latent_sse = latent_sae = 0.0
    latent_count = 0
    pixel_sse = pixel_sae = 0.0
    pixel_count = 0
    preview_saved = 0
    pred_latents = []
    true_latents = []
    preview_dir = args.output_dir / f"previews_{args.seq_len}_to_{args.pred_len}"

    with torch.no_grad():
        for batch_idx, (x, y, true_pixels, sample_idx, start) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            true_pixels = true_pixels.to(device)
            pred = forecaster(x)
            latent_diff = pred - y
            latent_sse += torch.square(latent_diff).sum().item()
            latent_sae += torch.abs(latent_diff).sum().item()
            latent_count += y.numel()

            pred_images = autoencoder.decode(pred.reshape(-1, args.latent_dim))
            pred_images = pred_images.view(pred.shape[0], args.pred_len, 9, args.image_size, args.image_size)
            pixel_diff = pred_images - true_pixels
            pixel_sse += torch.square(pixel_diff).sum().item()
            pixel_sae += torch.abs(pixel_diff).sum().item()
            pixel_count += true_pixels.numel()

            pred_latents.append(pred.detach().cpu().numpy())
            true_latents.append(y.detach().cpu().numpy())

            while preview_saved < args.preview_count and preview_saved < pred.shape[0]:
                item_dir = preview_dir / f"case_{batch_idx:04d}_{preview_saved:02d}"
                for horizon_idx in range(args.pred_len):
                    save_frame_stack(
                        pred_images[preview_saved, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"pred_h{horizon_idx:02d}",
                    )
                    save_frame_stack(
                        true_pixels[preview_saved, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"true_h{horizon_idx:02d}",
                    )
                preview_saved += 1

    np.save(args.output_dir / f"pred_latents_{args.seq_len}_to_{args.pred_len}.npy", np.concatenate(pred_latents, axis=0))
    np.save(args.output_dir / f"true_latents_{args.seq_len}_to_{args.pred_len}.npy", np.concatenate(true_latents, axis=0))
    return PixelMetrics(
        latent_mse=latent_sse / latent_count,
        latent_mae=latent_sae / latent_count,
        pixel_mse=pixel_sse / pixel_count,
        pixel_mae=pixel_sae / pixel_count,
    )


def evaluate_joint_pixel_forecast(
    args: argparse.Namespace,
    autoencoder: ConvImageAutoencoder,
    forecaster: ConditionalDLinear,
    sample_dirs: list[Path],
    actions: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> PixelMetrics:
    dataset = PixelActionWindowDataset(
        sample_dirs,
        actions,
        test_indices,
        args.seq_len,
        args.pred_len,
        args.image_size,
        args.max_test_windows,
        args.seed + 2,
    )
    loader = DataLoader(dataset, batch_size=args.forecast_batch_size, shuffle=False, num_workers=0)
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
        for batch_idx, (past_pixels, future_pixels, past_actions, future_actions, _, _) in enumerate(loader):
            past_pixels = past_pixels.to(device)
            future_pixels = future_pixels.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)

            batch_size = past_pixels.shape[0]
            pred_len = future_pixels.shape[1]
            past_latent = autoencoder.encode(
                past_pixels.reshape(-1, 9, args.image_size, args.image_size)
            ).view(batch_size, args.seq_len, args.latent_dim)
            true_future_latent = autoencoder.encode(
                future_pixels.reshape(-1, 9, args.image_size, args.image_size)
            ).view(batch_size, pred_len, args.latent_dim)
            tokens = build_forecast_tokens(past_latent, past_actions, future_actions)
            pred = forecaster(tokens)
            pred_images = autoencoder.decode(pred.reshape(-1, args.latent_dim)).view(
                batch_size, pred_len, 9, args.image_size, args.image_size
            )

            latent_diff = pred - true_future_latent
            latent_sse += torch.square(latent_diff).sum().item()
            latent_sae += torch.abs(latent_diff).sum().item()
            latent_count += true_future_latent.numel()

            pixel_diff = pred_images - future_pixels
            pixel_sse += torch.square(pixel_diff).sum().item()
            pixel_sae += torch.abs(pixel_diff).sum().item()
            pixel_count += future_pixels.numel()

            pred_latents.append(pred.detach().cpu().numpy())
            true_latents.append(true_future_latent.detach().cpu().numpy())

            while preview_saved < args.preview_count and preview_saved < pred.shape[0]:
                item_dir = preview_dir / f"case_{batch_idx:04d}_{preview_saved:02d}"
                for horizon_idx in range(args.pred_len):
                    save_frame_stack(
                        pred_images[preview_saved, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"pred_h{horizon_idx:02d}",
                    )
                    save_frame_stack(
                        future_pixels[preview_saved, horizon_idx].detach().cpu().numpy(),
                        item_dir,
                        f"true_h{horizon_idx:02d}",
                    )
                preview_saved += 1

    np.save(args.output_dir / f"pred_latents_joint_{args.seq_len}_to_{args.pred_len}.npy", np.concatenate(pred_latents, axis=0))
    np.save(args.output_dir / f"true_latents_joint_{args.seq_len}_to_{args.pred_len}.npy", np.concatenate(true_latents, axis=0))
    return PixelMetrics(
        latent_mse=latent_sse / latent_count,
        latent_mae=latent_sae / latent_count,
        pixel_mse=pixel_sse / pixel_count,
        pixel_mae=pixel_sae / pixel_count,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {device}")

    sample_dirs = discover_sample_dirs(args.source_dir)
    split = split_indices(len(sample_dirs), args.seed, args.train_ratio, args.val_ratio)
    train_indices, val_indices, test_indices = split
    actions = np.stack([read_actions(sample_dir) for sample_dir in sample_dirs], axis=0)
    write_json(
        args.output_dir / "dataset_metadata.json",
        {
            "source_dir": str(args.source_dir),
            "num_sequences": len(sample_dirs),
            "time_steps": 200,
            "image_shape": [9, args.image_size, args.image_size],
            "latent_dim": args.latent_dim,
            "action_dim": 3,
            "model": "ConditionalDLinear",
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "loss_mode": args.loss_mode,
            "forecast_loss": args.forecast_loss,
            "loss_weights": {
                "pixel": args.pixel_loss_weight,
                "latent": args.latent_loss_weight,
                "recon": args.recon_loss_weight,
            },
            "freeze_autoencoder_during_forecast": args.freeze_autoencoder_during_forecast,
            "split_counts": {
                "train": int(train_indices.size),
                "val": int(val_indices.size),
                "test": int(test_indices.size),
            },
            "input_definition": "past latent+past action tokens plus future action tokens with zero future latent placeholders",
            "target_definition": "future image latent decoded to future images for pixel metrics",
        },
    )

    autoencoder = train_autoencoder(args, sample_dirs, train_indices, val_indices, device)
    if args.loss_mode == "latent":
        latent_data = encode_sequences(args, autoencoder, sample_dirs, actions, split, device)
        forecaster = train_forecaster(args, latent_data, sample_dirs, autoencoder, device)
        metrics = evaluate_pixel_forecast(args, autoencoder, forecaster, latent_data, sample_dirs, device)
    else:
        forecaster = train_joint_forecaster(args, autoencoder, sample_dirs, actions, split, device)
        metrics = evaluate_joint_pixel_forecast(args, autoencoder, forecaster, sample_dirs, actions, test_indices, device)

    metrics_payload = {
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "loss_mode": args.loss_mode,
        "forecast_loss": args.forecast_loss,
        "loss_weights": {
            "pixel": args.pixel_loss_weight,
            "latent": args.latent_loss_weight,
            "recon": args.recon_loss_weight,
        },
        "latent_mse": metrics.latent_mse,
        "latent_mae": metrics.latent_mae,
        "pixel_mse": metrics.pixel_mse,
        "pixel_rmse": float(np.sqrt(metrics.pixel_mse)),
        "pixel_mae": metrics.pixel_mae,
    }
    write_json(args.output_dir / f"pixel_metrics_{args.loss_mode}_{args.seq_len}_to_{args.pred_len}.json", metrics_payload)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
