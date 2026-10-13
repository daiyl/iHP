from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder, THREE_STATE_IMAGE_FILES


class FundusDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="fundus",
    image_dir=Path("Patient/models/training_data/fundus"),
    checkpoint_name="fundus_decoder.pt",
    decoder_class=FundusDecoder,
    state_image_files=THREE_STATE_IMAGE_FILES,
    force_discrete_decode=True,
)


__all__ = ["FundusDecoder", "SPEC"]
