import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch

from Patient.decoders import DECODER_SPECS
from Patient.decoders.base_decoder import train_decoder


def selected_decoders(names: Iterable[str]) -> Dict[str, object]:
    selected = {}
    for name in names:
        if name not in DECODER_SPECS:
            raise ValueError(f"Unknown decoder {name}; choose from {sorted(DECODER_SPECS)}")
        selected[name] = DECODER_SPECS[name]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pneumonia, fundus, and OCT neural decoders.")
    parser.add_argument("--modalities", nargs="+", default=list(DECODER_SPECS.keys()))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("Patient/decoders/checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for spec in selected_decoders(args.modalities).values():
        train_decoder(
            spec=spec,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            device=device,
        )


if __name__ == "__main__":
    main()

