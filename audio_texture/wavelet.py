"""Complex log-frequency wavelet transform, demodulated to baseband and decimated.

A real signal x is mapped to coefficients c[b, m] (BANDS x frames), one complex
value per band per 1/ (SAMPLE_RATE / DECIMATION) s. Band b is an analytic filter
Gaussian in log-frequency, centred at LOW_HZ * 2^(b / BANDS_PER_OCTAVE); its
output is shifted down to baseband and decimated, so a steady partial becomes a
nearly constant complex value (phase is kept, but nothing oscillates quickly).

Everything is done on the FFT of the whole signal: the bins around each band are
cut out and re-centred. `synthesize` is the canonical dual of `analyse`
(filters divided by the summed squared response), so the round trip is
near-exact inside the covered band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from texture_stats import SAMPLE_RATE

BANDS_PER_OCTAVE = 12
LOW_HZ = 80.0
HIGH_HZ = 4_000.0
DECIMATION = 8
N_BANDS = int(math.log2(HIGH_HZ / LOW_HZ) * BANDS_PER_OCTAVE) + 1
BAND_SIGMA_OCTAVES = 1 / BANDS_PER_OCTAVE
COVERAGE_FLOOR = 0.05  # fraction of the peak summed response below which a bin is uncovered


def centre_frequencies() -> torch.Tensor:
    return LOW_HZ * 2 ** (torch.arange(N_BANDS) / BANDS_PER_OCTAVE)


def _response(freqs: torch.Tensor) -> torch.Tensor:
    """One-sided Gaussian-in-log-frequency filters; zero for f <= 0.

    `freqs` is (n,) for every band at the same frequencies, or (N_BANDS, n) for
    per-band frequencies; the result is (N_BANDS, n) either way.
    """
    freqs = freqs.expand(N_BANDS, -1) if freqs.ndim == 1 else freqs
    octaves = torch.log2(freqs.clamp(min=1e-3)) - torch.log2(centre_frequencies())[:, None]
    return torch.exp(-0.5 * (octaves / BAND_SIGMA_OCTAVES) ** 2) * (freqs > 0)


@dataclass(frozen=True)
class Wavelet:
    n_samples: int
    index: torch.Tensor  # (N_BANDS, frames) rfft bin of each window position (clamped)
    filters: torch.Tensor  # (N_BANDS, frames) filter value there; zero outside (0, Nyquist]
    inverse_norm: torch.Tensor  # (n // 2 + 1,) 1 / (2 * summed squared response), 0 if uncovered

    @property
    def frames(self) -> int:
        return self.n_samples // DECIMATION

    @classmethod
    @lru_cache(maxsize=None)
    def for_length(cls, n_samples: int) -> Wavelet:
        """`n_samples` must be a multiple of 2 * DECIMATION."""
        if n_samples % (2 * DECIMATION):
            raise ValueError(f"n_samples must be a multiple of {2 * DECIMATION}")
        frames = n_samples // DECIMATION
        bin_hz = SAMPLE_RATE / n_samples
        centre_bins = (centre_frequencies() / bin_hz).round().long()
        raw_index = centre_bins[:, None] + torch.arange(-frames // 2, frames // 2)[None, :]
        valid = (raw_index > 0) & (raw_index <= n_samples // 2)
        index = raw_index.clamp(0, n_samples // 2)
        filters = _response(index * bin_hz) * valid
        full = _response(torch.arange(n_samples // 2 + 1) * bin_hz)
        summed = full.pow(2).sum(0)
        inverse_norm = torch.where(summed > COVERAGE_FLOOR * summed.max(), 1 / (2 * summed), torch.zeros_like(summed))
        return cls(n_samples, index, filters, inverse_norm)

    def analyse(self, x: torch.Tensor) -> torch.Tensor:
        """(n,) real -> (N_BANDS, frames) complex baseband coefficients."""
        window = 2 * torch.fft.rfft(x)[self.index] * self.filters
        return torch.fft.ifft(torch.fft.ifftshift(window, dim=-1), dim=-1) / DECIMATION

    def synthesize(self, coefficients: torch.Tensor) -> torch.Tensor:
        """(N_BANDS, frames) complex -> (n,) real; the dual of `analyse`."""
        window = torch.fft.fftshift(torch.fft.fft(coefficients, dim=-1), dim=-1) * DECIMATION
        spectrum = torch.zeros(self.n_samples // 2 + 1, dtype=window.dtype).index_add(
            0, self.index.flatten(), (window * self.filters).flatten()
        )
        return torch.fft.irfft(spectrum * self.inverse_norm, n=self.n_samples)

    def covered(self, x: torch.Tensor) -> torch.Tensor:
        """`x` restricted to the band the transform represents (the round-trip's ideal output)."""
        return torch.fft.irfft(torch.fft.rfft(x) * (self.inverse_norm > 0), n=self.n_samples)
