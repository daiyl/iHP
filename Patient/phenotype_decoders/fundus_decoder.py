from pathlib import Path

from .base_decoder import DecoderSpec, StateImageDecoder


class FundusDecoder(StateImageDecoder):
    pass


SPEC = DecoderSpec(
    name="fundus",
    image_dir=Path("Patient/decoders/training_data/fundus"),
    checkpoint_name="fundus_decoder.pt",
    decoder_class=FundusDecoder,
)


__all__ = ["FundusDecoder", "SPEC"]
