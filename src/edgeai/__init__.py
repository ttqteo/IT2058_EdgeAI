"""Bounded host-only primitives for the IT2058 EdgeAI prototype.

The frontend constants and arithmetic in this package are the approved
``EXPERIMENTAL_HOST_ONLY`` v0.1 design.  They are useful for host/C parity
work, but do not establish the final frontend, tensor layout, USB protocol,
or board runtime contract.
"""

from .frontend import (
    CONTRACT_DISCOVERY,
    CONTRACT_UNRESOLVED,
    EXPERIMENTAL_HOST_ONLY,
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LOG_FLOOR,
    NUM_FRAMES,
    NUM_FREQ_BINS,
    NUM_MELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    WavLoadReport,
    export_filterbank_header,
    load_wav,
    log_mel,
    mel_filterbank,
)
from .matching import (
    MatchResult,
    cosine_similarity,
    enroll,
    l2_distance,
    match_1_to_1,
    match_1_to_n,
    normalize_embedding,
)

__all__ = [
    "CONTRACT_DISCOVERY",
    "CONTRACT_UNRESOLVED",
    "EXPERIMENTAL_HOST_ONLY",
    "FFT_LENGTH",
    "FRAME_LENGTH",
    "FRAME_STEP",
    "LOG_FLOOR",
    "NUM_FRAMES",
    "NUM_FREQ_BINS",
    "NUM_MELS",
    "NUM_SAMPLES",
    "SAMPLE_RATE",
    "WavLoadReport",
    "export_filterbank_header",
    "load_wav",
    "log_mel",
    "mel_filterbank",
    "MatchResult",
    "cosine_similarity",
    "enroll",
    "l2_distance",
    "match_1_to_1",
    "match_1_to_n",
    "normalize_embedding",
]
