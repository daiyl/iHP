from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder


class PneumoniaDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="pneumonia",
    image_dir=Path("Patient/decoders/training_data/pneumonia"),
    checkpoint_name="pneumonia_decoder.pt",
    decoder_class=PneumoniaDecoder,
)


__all__ = ["PneumoniaDecoder", "SPEC"]
