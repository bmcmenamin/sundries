"""Amplitude / instantaneous-frequency view of the wavelet coefficients.

Coding the complex baseband coefficients splits one steady partial into several
features with different phase offsets (a partial off its band's centre rotates).
Here each band is described instead by
  channel 0  compressed amplitude  |c|^0.5
  channel 1  instantaneous frequency offset: the phase advance per frame, as a
             fraction of the band's centre frequency (so a transposed note has
             the same value), clipped and gated by amplitude so silence is 0.
A steady partial is constant in both channels, and both are invariant to time shifts.

`coefficients_from` integrates the frequency back into a phase (initial phases are
free); `reconstruct_audio` then makes the waveform consistent by optimising it so
its own analysis reproduces the target features.
"""

from __future__ import annotations

import math

import torch

from texture_stats import SAMPLE_RATE
from wavelet import DECIMATION, Wavelet, centre_frequencies

FRAME_RATE = SAMPLE_RATE / DECIMATION
AMPLITUDE_POWER = 0.5
GATE = 0.03  # magnitudes well below this count as silence for the frequency channel
FREQ_SCALE = 0.15  # relative frequency offset that maps to 1.0
FREQ_CLIP = 2.0


def _radians_per_frame() -> torch.Tensor:
    """(B, 1): phase advance per frame for a relative frequency offset of 1.0."""
    return 2 * math.pi * centre_frequencies()[:, None] / FRAME_RATE


def relative_offset(coefficients: torch.Tensor) -> torch.Tensor:
    """(B, W) complex -> (B, W) frequency offset per frame, relative to the band centre, scaled and clipped."""
    product = coefficients[:, 1:] * coefficients[:, :-1].conj()
    advance = torch.atan2(product.imag, product.real + 1e-12)
    advance = torch.cat([advance, advance[:, -1:]], dim=-1)
    return (advance / _radians_per_frame() / FREQ_SCALE).clamp(-FREQ_CLIP, FREQ_CLIP)


def features(coefficients: torch.Tensor) -> torch.Tensor:
    """(B, W) complex -> (2, B, W) real."""
    magnitude = coefficients.abs()
    offset = relative_offset(coefficients)
    gate = magnitude / (magnitude + GATE)
    return torch.stack([(magnitude + 1e-6).pow(AMPLITUDE_POWER), gate * offset])


def coefficients_from(target: torch.Tensor) -> torch.Tensor:
    """(2, B, W) -> (B, W) complex, with the phase integrated from the frequency channel."""
    magnitude = target[0].clamp(min=0).pow(1 / AMPLITUDE_POWER)
    gate = magnitude / (magnitude + GATE)
    offset = (target[1] / (gate + 1e-3)).clamp(-FREQ_CLIP, FREQ_CLIP) * (gate > 0.2)
    advance = offset * FREQ_SCALE * _radians_per_frame()
    return torch.polar(magnitude, torch.cumsum(advance, dim=-1) - advance)


def reconstruct_audio(target: torch.Tensor, wavelet: Wavelet, iters: int) -> torch.Tensor:
    """Waveform whose analysis has features close to `target`, started from the integrated phase."""
    x = wavelet.synthesize(coefficients_from(target)).clone().requires_grad_()
    optimiser = torch.optim.LBFGS(
        [x], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = (features(wavelet.analyse(x)) - target).pow(2).mean()
        loss.backward()
        return loss

    optimiser.step(closure)
    return x.detach()


def snr_db(target: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(10 * torch.log10(target.pow(2).sum() / ((target - estimate).pow(2).sum() + 1e-12)))
