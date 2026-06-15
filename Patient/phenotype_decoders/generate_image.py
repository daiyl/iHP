import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch

from Patient.decoders import DECODER_SPECS
from Patient.decoders.base_decoder import decode_state, load_decoder, parse_state, state_label


def selected_decoders(names: Iterable[str]) -> Dict[str, object]:
    if "all" in names:
        return DECODER_SPECS
    selected = {}
    for name in names:
        if name not in DECODER_SPECS:
            raise ValueError(f"Unknown decoder {name}; choose from {sorted(DECODER_SPECS)} or all")
        selected[name] = DECODER_SPECS[name]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate observation images from one 3D latent state.")
    parser.add_argument("--state", required=True, type=str, help="Single 3D state, e.g. '(0,0.1,0.9)'")
    parser.add_argument("--modalities", nargs="+", default=["all"], help="all, pneumonia, fundus, oct")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("Patient/decoders/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("Patient/observation_data/generated_observations"))
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Decode the continuous input directly instead of first mapping it to the nearest decoder training state.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = parse_state(args.state)
    output_dir = args.output_dir / "single_state" / f"state_{state_label(state)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in selected_decoders(args.modalities).items():
        decoder = load_decoder(spec, args.checkpoint_dir, device)
        image = decode_state(decoder, state, device, round_to_training_state=not args.continuous)
        output_path = output_dir / f"{name}.png"
        image.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

