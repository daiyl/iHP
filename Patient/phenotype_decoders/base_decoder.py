from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


State = Tuple[float, float, float]
StateImageFile = Tuple[State, str]

THREE_STATE_IMAGE_FILES: Tuple[StateImageFile, ...] = (
    ((1.0, 0.0, 0.0), "(1,0,0).png"),
    ((0.0, 1.0, 0.0), "(0,1,0).png"),
    ((0.0, 0.0, 1.0), "(0,0,1).png"),
)

DEFAULT_STATE_IMAGE_FILES = THREE_STATE_IMAGE_FILES

# Backward-compatible aliases for code that imports the old names directly.
STATE_IMAGE_FILES = DEFAULT_STATE_IMAGE_FILES

TRAINING_STATES = np.array([state for state, _ in STATE_IMAGE_FILES], dtype=np.float32)


def normalize_state(state: Iterable[float]) -> np.ndarray:
    values = np.array(list(state), dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("State must contain exactly three values")
    if np.any(values < 0):
        raise ValueError("State probabilities cannot be negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("State probabilities must sum to a positive value")
    return values / total


def parse_state(value: str) -> np.ndarray:
    cleaned = value.strip().replace("(", "").replace(")", "")
    parts = [float(part.strip()) for part in cleaned.split(",") if part.strip()]
    return normalize_state(parts)


def state_label(state: Iterable[float]) -> str:
    return "_".join(f"{x:g}" for x in normalize_state(state))


def training_states_from_image_files(state_image_files: Tuple[StateImageFile, ...]) -> np.ndarray:
    return np.array([state for state, _ in state_image_files], dtype=np.float32)


def nearest_training_state(
    state: Iterable[float],
    training_states: Optional[np.ndarray] = None,
) -> np.ndarray:
    normalized = normalize_state(state)
    states = TRAINING_STATES if training_states is None else np.asarray(training_states, dtype=np.float32)
    distances = np.linalg.norm(states - normalized, axis=1)
    return states[int(np.argmin(distances))].copy()


class StateImageDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        state_image_files: Tuple[StateImageFile, ...] = DEFAULT_STATE_IMAGE_FILES,
        image_size: int = 64,
    ):
        self.samples = []
        self.image_size = image_size
        self.training_states = training_states_from_image_files(state_image_files)
        for state, filename in state_image_files:
            image_path = image_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Missing training image: {image_path}")
            self.samples.append((np.array(state, dtype=np.float32), image_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        state, image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        return torch.tensor(state, dtype=torch.float32), torch.tensor(image_array, dtype=torch.float32)


class StateImageDecoder(nn.Module):
    """Neural decoder from a 3D latent health-state probability to an image."""

    def __init__(self, state_dim: int = 3, latent_dim: int = 128, base_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 4 * 4 * base_channels * 4),
            nn.ReLU(),
            nn.Unflatten(1, (base_channels * 4, 4, 4)),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, state_probs: torch.Tensor) -> torch.Tensor:
        return self.network(state_probs)


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    image_dir: Path
    checkpoint_name: str
    decoder_class: Type[StateImageDecoder] = StateImageDecoder
    state_image_files: Tuple[StateImageFile, ...] = DEFAULT_STATE_IMAGE_FILES
    force_discrete_decode: bool = False


def prepare_decoder_state(
    state: Iterable[float],
    round_to_training_state: bool = True,
    training_states: Optional[np.ndarray] = None,
) -> np.ndarray:
    normalized = normalize_state(state)
    if round_to_training_state:
        return nearest_training_state(normalized, training_states=training_states)
    return normalized


def image_from_tensor(output: torch.Tensor) -> Image.Image:
    array = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_decoder(spec, checkpoint_dir: Path, device: torch.device) -> StateImageDecoder:
    checkpoint_name = spec.checkpoint_name if isinstance(spec, DecoderSpec) else spec
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder_class = spec.decoder_class if isinstance(spec, DecoderSpec) else StateImageDecoder
    decoder = decoder_class().to(device)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    if isinstance(spec, DecoderSpec):
        decoder.training_states = training_states_from_image_files(spec.state_image_files)
        decoder.force_discrete_decode = spec.force_discrete_decode
    else:
        decoder.training_states = np.asarray(checkpoint.get("training_states", TRAINING_STATES), dtype=np.float32)
        decoder.force_discrete_decode = False
    decoder.eval()
    return decoder


def decode_state(
    decoder: StateImageDecoder,
    state: Iterable[float],
    device: torch.device,
    round_to_training_state: bool = True,
) -> Image.Image:
    should_round = bool(getattr(decoder, "force_discrete_decode", False)) or round_to_training_state
    prepared_state = prepare_decoder_state(
        state,
        round_to_training_state=should_round,
        training_states=getattr(decoder, "training_states", None),
    )
    state_tensor = torch.tensor(prepared_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = decoder(state_tensor)
    return image_from_tensor(output)


def decode_states(
    decoder: StateImageDecoder,
    states: List[Iterable[float]],
    device: torch.device,
    round_to_training_state: bool = True,
) -> List[Image.Image]:
    should_round = bool(getattr(decoder, "force_discrete_decode", False)) or round_to_training_state
    training_states = getattr(decoder, "training_states", None)
    prepared_states = [
        prepare_decoder_state(
            state,
            round_to_training_state=should_round,
            training_states=training_states,
        )
        for state in states
    ]
    keys = [tuple(np.round(state, 6)) for state in prepared_states]
    unique_states = {}
    for key, state in zip(keys, prepared_states):
        unique_states.setdefault(key, state)

    decoded_by_key = {}
    with torch.no_grad():
        for key, state in unique_states.items():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            decoded_by_key[key] = image_from_tensor(decoder(state_tensor))
    return [decoded_by_key[key].copy() for key in keys]


def train_decoder(
    spec: DecoderSpec,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    device: torch.device,
) -> Path:
    dataset = StateImageDataset(spec.image_dir, state_image_files=spec.state_image_files, image_size=image_size)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    decoder = spec.decoder_class().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    decoder.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for state_probs, target_images in loader:
            state_probs = state_probs.to(device)
            target_images = target_images.to(device)
            output = decoder(state_probs)
            loss = criterion(output, target_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_probs.size(0)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[{spec.name}] epoch {epoch:04d}/{epochs}, loss={total_loss / len(dataset):.6f}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / spec.checkpoint_name
    torch.save(
        {
            "modality": spec.name,
            "image_size": image_size,
            "model_state_dict": decoder.state_dict(),
            "architecture": "StateImageDecoder(state_dim=3, latent_dim=128, base_channels=64)",
            "training_states": dataset.training_states,
        },
        checkpoint_path,
    )
    print(f"[{spec.name}] saved {checkpoint_path}")
    return checkpoint_path
