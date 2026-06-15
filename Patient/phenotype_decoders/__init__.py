from .base_decoder import StateImageDecoder
from .fundus_decoder import FundusDecoder, SPEC as FUNDUS_DECODER
from .oct_decoder import OCTDecoder, SPEC as OCT_DECODER
from .pneumonia_decoder import PneumoniaDecoder, SPEC as PNEUMONIA_DECODER


DECODER_SPECS = {
    "pneumonia": PNEUMONIA_DECODER,
    "fundus": FUNDUS_DECODER,
    "oct": OCT_DECODER,
}


__all__ = [
    "DECODER_SPECS",
    "FUNDUS_DECODER",
    "FundusDecoder",
    "OCT_DECODER",
    "OCTDecoder",
    "PNEUMONIA_DECODER",
    "PneumoniaDecoder",
    "StateImageDecoder",
]
