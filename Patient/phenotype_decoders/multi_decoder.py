from pathlib import Path
from typing import Dict, Tuple

import torch

from Patient.phenotype_decoders import DECODER_SPECS
from Patient.phenotype_decoders.base_decoder import decode_state, load_decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
_DECODER_CACHE: Dict[Tuple[str, Path, str], object] = {}


def _selected_specs(modalities=None):
    if isinstance(modalities, str):
        modalities = [modalities]
    if modalities is None or "all" in modalities:
        return DECODER_SPECS
    selected = {}
    for name in modalities:
        if name not in DECODER_SPECS:
            raise ValueError(f"Unknown decoder {name}; choose from {sorted(DECODER_SPECS)} or all")
        selected[name] = DECODER_SPECS[name]
    return selected


def _get_decoder(name, spec, checkpoint_dir):
    key = (name, checkpoint_dir.resolve(), str(device))
    if key not in _DECODER_CACHE:
        _DECODER_CACHE[key] = load_decoder(spec, checkpoint_dir, device)
    return _DECODER_CACHE[key]


def multi_decoder(state, return_state=False, modalities=None, checkpoint_dir=None):
    decoder_state = tuple(float(x) for x in state)
    active_checkpoint_dir = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else Path(checkpoint_dir)
    images = []
    for name, spec in _selected_specs(modalities).items():
        decoder = _get_decoder(name, spec, active_checkpoint_dir)
        image = decode_state(decoder, decoder_state, device, round_to_training_state=True)
        images.append(image)

    if return_state:
        return images, decoder_state
    return images
