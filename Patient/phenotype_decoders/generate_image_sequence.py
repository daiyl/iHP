import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from Patient.decoders import DECODER_SPECS
from Patient.decoders.base_decoder import decode_states, load_decoder


def selected_decoders(names: Iterable[str]) -> Dict[str, object]:
    if "all" in names:
        return DECODER_SPECS
    selected = {}
    for name in names:
        if name not in DECODER_SPECS:
            raise ValueError(f"Unknown decoder {name}; choose from {sorted(DECODER_SPECS)} or all")
        selected[name] = DECODER_SPECS[name]
    return selected


def read_sequence_csv(path: Path) -> List[np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        np.array([float(row["s1"]), float(row["s2"]), float(row["s3"])], dtype=np.float32)
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate observation image sequences from a latent-state CSV.")
    parser.add_argument("--sequence", required=True, type=Path, help="CSV sequence with s1,s2,s3 columns")
    parser.add_argument("--modalities", nargs="+", default=["all"], help="all, pneumonia, fundus, oct")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("Patient/decoders/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("Patient/observation_data/generated_observations"))
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Decode continuous states directly instead of first mapping them to nearest decoder training states.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = read_sequence_csv(args.sequence)
    base_output_dir = args.output_dir / "sequences" / args.sequence.stem
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in selected_decoders(args.modalities).items():
        decoder = load_decoder(spec, args.checkpoint_dir, device)
        modality_dir = base_output_dir / name
        modality_dir.mkdir(parents=True, exist_ok=True)
        images = decode_states(decoder, states, device, round_to_training_state=not args.continuous)
        for idx, image in enumerate(images):
            image.save(modality_dir / f"frame_{idx:04d}.png")
        print(f"Saved {len(images)} frames to {modality_dir}")


if __name__ == "__main__":
    main()

