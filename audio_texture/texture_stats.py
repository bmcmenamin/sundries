"""Auditory texture statistics after McDermott & Simoncelli (2011).

A waveform goes through a cochlear filterbank, the subband envelopes are
compressed, and statistics are measured on the envelopes and on their
modulation-filtered versions. Every step is differentiable, so a waveform can
be optimised until its statistics match those of a target sound.

C2 correlations capture phase relations between octave-spaced modulation
bands (onset sharpness, asymmetry, pulse shape) within each cochlear channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Iterable

import torch
import torch.nn.functional as F

SAMPLE_RATE = 12_000  # "a bit better than a phone speaker": ~5 kHz of bandwidth
ENV_RATE = 400
ENV_DECIMATION = SAMPLE_RATE // ENV_RATE
N_COCHLEAR = 90
COCHLEAR_RANGE_HZ = (80.0, 5_000.0)
COCHLEAR_HALF_WIDTH_ERB = 1.0
N_MODULATION = 16
MODULATION_LOW_HZ = 0.5
MODULATION_WIDTH_OCTAVES = 0.3
ENVELOPE_COMPRESSION = 0.3
C1_OFFSETS = (1, 2, 3)
EPS = 1e-8


def _erb_number(freq_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1 + 0.00437 * freq_hz)


@lru_cache(maxsize=None)
def cochlear_filters(n_samples: int) -> torch.Tensor:
    """(N_COCHLEAR, n_samples // 2 + 1) cosine filters, equally spaced on the ERB scale.

    Width is fixed in ERBs (not tied to band count) so adding bands oversamples
    the same resolution instead of narrowing it below what the ear resolves.
    """
    freqs = torch.fft.rfftfreq(n_samples, 1 / SAMPLE_RATE)
    lo, hi = _erb_number(torch.tensor(COCHLEAR_RANGE_HZ))
    centres = torch.linspace(lo, hi, N_COCHLEAR)
    offset = (_erb_number(freqs)[None, :] - centres[:, None]) / COCHLEAR_HALF_WIDTH_ERB
    return torch.cos(math.pi / 2 * offset) * (offset.abs() < 1)


@lru_cache(maxsize=None)
def modulation_filters(n_env: int) -> torch.Tensor:
    """(N_MODULATION, n_env // 2 + 1) log-Gaussian filters, half-octave spaced from 0.5 Hz."""
    freqs = torch.fft.rfftfreq(n_env, 1 / ENV_RATE).clamp(min=1e-3)
    centres = MODULATION_LOW_HZ * 2 ** (torch.arange(N_MODULATION) / 2)
    octaves = torch.log2(freqs[None, :] / centres[:, None])
    return torch.exp(-0.5 * (octaves / MODULATION_WIDTH_OCTAVES) ** 2)


def _analytic(one_sided_spectrum: torch.Tensor, n: int) -> torch.Tensor:
    """Time-domain analytic signal from a doubled positive-frequency spectrum."""
    padding = torch.zeros_like(one_sided_spectrum[..., : n - one_sided_spectrum.shape[-1]])
    return torch.fft.ifft(torch.cat([one_sided_spectrum, padding], dim=-1), n=n)


def _power(z: torch.Tensor) -> torch.Tensor:
    return z.real**2 + z.imag**2


def compressed_envelopes(x: torch.Tensor) -> torch.Tensor:
    """(N_COCHLEAR, n // ENV_DECIMATION) power-law compressed subband envelopes."""
    analytic = _analytic(2 * torch.fft.rfft(x) * cochlear_filters(x.shape[-1]), x.shape[-1])
    envelope = (_power(analytic) + 1e-12).sqrt() ** ENVELOPE_COMPRESSION
    return F.avg_pool1d(envelope[None], ENV_DECIMATION)[0]


def _standardise(x: torch.Tensor) -> torch.Tensor:
    centred = x - x.mean(-1, keepdim=True)
    return centred / (centred.pow(2).mean(-1, keepdim=True) + EPS).sqrt()


def _envelope_moments(envelope: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = envelope.mean(-1)
    centred = envelope - mean[:, None]
    std = (centred.pow(2).mean(-1) + EPS).sqrt()
    skew = centred.pow(3).mean(-1) / std**3
    kurtosis = centred.pow(4).mean(-1) / std**4
    return mean, std / mean, skew, kurtosis


def _cross_band_correlations(envelope: torch.Tensor) -> torch.Tensor:
    z = _standardise(envelope)
    pairs = torch.triu_indices(N_COCHLEAR, N_COCHLEAR, offset=1)
    return (z @ z.T / z.shape[-1])[pairs[0], pairs[1]]


def _modulation_analytic(envelope: torch.Tensor) -> torch.Tensor:
    """(N_COCHLEAR, N_MODULATION, n_env) complex analytic modulation-filtered envelopes."""
    n_env = envelope.shape[-1]
    return _analytic(2 * torch.fft.rfft(envelope)[:, None, :] * modulation_filters(n_env)[None], n_env)


def _modulation_power(bands: torch.Tensor, envelope: torch.Tensor) -> torch.Tensor:
    return bands.pow(2).mean(-1) / (envelope.var(-1, unbiased=False)[:, None] + EPS)


def _c1_correlations(bands: torch.Tensor) -> torch.Tensor:
    """Correlation of the same modulation band across cochlear channels."""
    z = _standardise(bands)
    return torch.cat([(z[:-o] * z[o:]).mean(-1).flatten() for o in C1_OFFSETS])


def _c2_correlations(analytic: torch.Tensor) -> torch.Tensor:
    """Phase-doubled low band vs. the band an octave above, normalised to [-1, 1] (real, imag)."""
    low, high = analytic[:, :-2], analytic[:, 2:]
    c2 = (low**2 * high.conj()).mean(-1) / ((_power(low) ** 2).mean(-1) * _power(high).mean(-1) + EPS).sqrt()
    return torch.cat([c2.real.flatten(), c2.imag.flatten()])


@dataclass(frozen=True)
class TextureStats:
    env_mean: torch.Tensor
    env_cv: torch.Tensor
    env_skew: torch.Tensor
    env_kurtosis: torch.Tensor
    cross_band_corr: torch.Tensor
    modulation_power: torch.Tensor
    c1_corr: torch.Tensor
    c2_corr: torch.Tensor

    @classmethod
    def from_waveform(cls, x: torch.Tensor) -> TextureStats:
        """`x` is mono at SAMPLE_RATE, with length a multiple of ENV_DECIMATION."""
        return cls.from_envelope(compressed_envelopes(x))

    @classmethod
    def from_envelope(cls, envelope: torch.Tensor) -> TextureStats:
        analytic = _modulation_analytic(envelope)
        bands = analytic.real
        mean, cv, skew, kurtosis = _envelope_moments(envelope)
        return cls(
            env_mean=mean,
            env_cv=cv,
            env_skew=skew,
            env_kurtosis=kurtosis,
            cross_band_corr=_cross_band_correlations(envelope),
            modulation_power=_modulation_power(bands, envelope),
            c1_corr=_c1_correlations(bands),
            c2_corr=_c2_correlations(analytic),
        )

    @classmethod
    def average(cls, stats: Iterable[TextureStats]) -> TextureStats:
        stats = list(stats)
        return cls(**{f.name: torch.stack([getattr(s, f.name) for s in stats]).mean(0) for f in fields(cls)})

    def distance(self, target: TextureStats) -> torch.Tensor:
        """Sum over statistic classes of squared error, relative to the target's scale."""
        return sum(
            (getattr(self, f.name) - getattr(target, f.name)).pow(2).mean()
            / (getattr(target, f.name).pow(2).mean() + 1e-2)
            for f in fields(self)
        )
