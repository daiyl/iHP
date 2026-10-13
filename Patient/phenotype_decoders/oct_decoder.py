from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder, THREE_STATE_IMAGE_FILES


class OCTDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="oct",
    image_dir=Path("Patient/models/training_data/oct"),
    checkpoint_name="oct_decoder.pt",
    decoder_class=OCTDecoder,
    state_image_files=THREE_STATE_IMAGE_FILES,
    force_discrete_decode=True,
)


__all__ = ["OCTDecoder", "SPEC"]
