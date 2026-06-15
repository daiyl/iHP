from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder


class OCTDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="oct",
    image_dir=Path("Patient/decoders/training_data/oct"),
    checkpoint_name="oct_decoder.pt",
    decoder_class=OCTDecoder,
)


__all__ = ["OCTDecoder", "SPEC"]
