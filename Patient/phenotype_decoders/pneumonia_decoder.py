from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder, THREE_STATE_IMAGE_FILES


class PneumoniaDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="pneumonia",
    image_dir=Path("Patient/models/training_data/pneumonia"),
    checkpoint_name="pneumonia_decoder.pt",
    decoder_class=PneumoniaDecoder,
    state_image_files=THREE_STATE_IMAGE_FILES,
    force_discrete_decode=True,
)


__all__ = ["PneumoniaDecoder", "SPEC"]
