"""A pretrained neural audio codec (EnCodec, 24 kHz) as a perceptual latent space.

    audio (24 kHz mono) -> encoder -> latents (128, frames at 75 Hz) -> decoder -> audio

The decoder was trained adversarially to sound like real audio, so latents near the encoder's range decode to
natural timbre and phase. `snap` projects latents onto the residual-quantiser codebooks, which are the only latents
the decoder saw in training.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from encodec import EncodecModel

from synthesize import decode_mono

CODEC_RATE = 24_000
LATENTS = 128
HOP = 320  # samples per latent frame (75 Hz)
BANDWIDTH = 6.0  # kbps: 8 codebooks
TARGET_RMS = 0.05


@lru_cache(maxsize=1)
def model() -> EncodecModel:
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(BANDWIDTH)
    return codec.eval()


def load(path: Path) -> torch.Tensor:
    """Mono audio at CODEC_RATE, scaled to TARGET_RMS and trimmed to whole latent frames."""
    audio = decode_mono(path, CODEC_RATE)
    audio = audio / audio.pow(2).mean().sqrt() * TARGET_RMS
    return audio[: audio.numel() // HOP * HOP]


@torch.no_grad()
def encode(audio: torch.Tensor) -> torch.Tensor:
    """(n,) audio -> (LATENTS, n / HOP) continuous latents."""
    return model().encoder(audio[None, None])[0]


@torch.no_grad()
def decode(latents: torch.Tensor) -> torch.Tensor:
    """(LATENTS, frames) -> (frames * HOP,) audio."""
    return model().decoder(latents[None])[0, 0]


@torch.no_grad()
def snap(latents: torch.Tensor, bandwidth: float = BANDWIDTH) -> torch.Tensor:
    """Nearest points of the residual-quantiser codebooks (more bandwidth = more codebooks = a finer grid)."""
    codec = model()
    codes = codec.quantizer.encode(latents[None], codec.frame_rate, bandwidth)
    return codec.quantizer.decode(codes)[0]
