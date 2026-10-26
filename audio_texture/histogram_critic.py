"""Histogram features of wavelet coefficients and a regularised linear critic.

A "pseudo-GAN": the discriminator is a ridge-regularised linear classifier (an
LDA direction) over a large bank of histograms, and the generator is trained
against it. For a linear critic the game reduces to histogram matching with
adaptive weights: features that vary a lot between windows of the *real* stem
count for little, features that the current sample gets wrong while being
stable in real audio count for a lot.

The features are deliberately impoverished. They are 1-D and 2-D histograms of
local, phase-sensitive scalars pooled over small band-groups and 1 s windows, so
they say what kinds of local events occur in which register, never when, and
cannot reconstruct an arbitrary waveform. Phase-sensitive scalars (Re/Im of
smoothed unit phasors):
  advance     phase advance over 5 / 16 / 50 ms in a band (frequency stability of a partial)
  neighbour   phase difference between adjacent bands (onset timing / chirp)
  harmonic h  theta(b + delta) - h * theta(b) for the exact-octave and
              near-harmonic band offsets: whether partials are phase-locked
Magnitude scalars: level, rise (16 / 100 ms), level 50 / 200 ms later (persistence
of a steady line), spectral slope, and harmonic level ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from wavelet import DECIMATION, Wavelet

STEP = 4  # frames are subsampled by this before histogramming
WINDOW = 375  # histogram windows, in subsampled steps (1 s)
HOP = 94
MAX_LAG = 300  # frames at 1500 Hz (200 ms)
HARMONICS = ((12, 2), (19, 3), (24, 4), (28, 5))  # (band offset, harmonic number)
ADVANCE_LAGS = (8, 24, 75)  # frames: 5, 16, 50 ms
FLOOR = 1e-3  # magnitude floor inside the log
PHASOR_DAMPING = 0.03  # unit phasors fade to zero below this magnitude
LEVEL_RANGE = (-3.0, 3.0)
PHASOR_RANGE = (-1.0, 1.0)


@dataclass(frozen=True)
class FeatureConfig:
    group: int = 3  # bands per pooling group (12 bands per octave)
    bins: int = 24  # 1-D histogram bins
    joint_bins: int = 12  # per axis of the 2-D histograms


@dataclass(frozen=True)
class Reference:
    """Per-band log-magnitude statistics of the source, used to z-score levels."""

    mean: torch.Tensor  # (B, 1)
    std: torch.Tensor  # (B, 1)

    @classmethod
    def from_coefficients(cls, coefficients: torch.Tensor) -> Reference:
        log_magnitude = torch.log(coefficients.abs() + FLOOR)
        return cls(log_magnitude.mean(-1, keepdim=True), log_magnitude.std(-1, keepdim=True) + 1e-3)


@lru_cache(maxsize=None)
def carrier(n_samples: int) -> torch.Tensor:
    """exp(i * demodulation phase) per band and frame, undoing the baseband shift (exact, integer arithmetic)."""
    wavelet = Wavelet.for_length(n_samples)
    centre_bins = wavelet.index[:, wavelet.frames // 2]
    frames = torch.arange(wavelet.frames)
    cycles = (centre_bins[:, None] * frames[None, :]) % wavelet.frames / wavelet.frames
    return torch.polar(torch.ones_like(cycles, dtype=torch.float32), (2 * torch.pi * cycles).float())


def _phasor(z: torch.Tensor) -> torch.Tensor:
    return z / (z.abs() + PHASOR_DAMPING)


def scalar_maps(coefficients: torch.Tensor, reference: Reference) -> dict[str, torch.Tensor]:
    """Real-valued (bands', steps) maps; band axis starts at band 0 for every map."""
    frames = coefficients.shape[1]
    steps = frames - MAX_LAG
    level = (torch.log(coefficients.abs() + FLOOR) - reference.mean) / reference.std
    analytic = _phasor(coefficients * carrier(frames * DECIMATION))
    baseband = _phasor(coefficients)

    def at(x: torch.Tensor, lag: int = 0) -> torch.Tensor:
        return x[:, lag : lag + steps : STEP]

    maps = {
        "level": at(level),
        "rise16": at(level, 24) - at(level),
        "rise100": at(level, 150) - at(level),
        "later50": at(level, 75),
        "later200": at(level, MAX_LAG),
        "slope": at(level)[1:] - at(level)[:-1],
    }
    phasors = {f"advance{lag}": at(baseband) * at(baseband, lag).conj() for lag in ADVANCE_LAGS}
    phasors["neighbour"] = at(analytic)[1:] * at(analytic)[:-1].conj()
    for offset, harmonic in HARMONICS:
        phasors[f"harmonic{harmonic}"] = at(analytic)[offset:] * at(analytic)[:-offset].conj() ** harmonic
        maps[f"ratio{harmonic}"] = at(level)[offset:] - at(level)[:-offset]
    for name, z in phasors.items():
        maps[f"{name}_re"], maps[f"{name}_im"] = z.real, z.imag
    return maps


def soft_assign(values: torch.Tensor, bins: int, low: float, high: float) -> torch.Tensor:
    """(..., bins) linear-interpolation histogram weights; values outside [low, high] clamp to the edge."""
    position = ((values - low) / (high - low) * bins - 0.5).clamp(0, bins - 1)
    lower = position.floor()
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = (position - lower)[..., None]
    zeros = torch.zeros(*values.shape, bins)
    return zeros.scatter_add(-1, lower.long()[..., None], 1 - fraction).scatter_add(-1, upper.long()[..., None], fraction)


def _group_slices(n_bands: int, group: int) -> list[tuple[int, int]]:
    return [(b, min(b + group, n_bands)) for b in range(0, n_bands, group)]


def _window_pool(x: torch.Tensor) -> torch.Tensor:
    """(steps, channels) -> (windows, channels): mean over sliding windows."""
    return F.avg_pool1d(x.T[None], WINDOW, stride=HOP)[0].T


def _join_groups(blocks: list[torch.Tensor | None]) -> torch.Tensor:
    """Concatenate per-group blocks; groups beyond a map's bands contribute zeros."""
    reference = next(b for b in blocks if b is not None)
    return torch.cat([torch.zeros_like(reference) if b is None else b for b in blocks], dim=-1)


def marginal(x: torch.Tensor, value_range: tuple[float, float], n_bands: int, config: FeatureConfig) -> torch.Tensor:
    """(windows, n_groups * bins): per band-group histogram of x in every window."""
    return _join_groups([
        _window_pool(soft_assign(x[b0:b1], config.bins, *value_range).mean(0)) if b0 < x.shape[0] else None
        for b0, b1 in _group_slices(n_bands, config.group)
    ])


def joint(x: torch.Tensor, y: torch.Tensor, x_range, y_range, n_bands: int, config: FeatureConfig) -> torch.Tensor:
    """(windows, n_groups * joint_bins^2): per band-group 2-D histogram of (x, y) in every window."""
    bands = min(x.shape[0], y.shape[0])
    blocks = []
    for b0, b1 in _group_slices(n_bands, config.group):
        if b0 >= bands:
            blocks.append(None)
            continue
        b1 = min(b1, bands)
        ax = soft_assign(x[b0:b1], config.joint_bins, *x_range)
        ay = soft_assign(y[b0:b1], config.joint_bins, *y_range)
        blocks.append(_window_pool(torch.einsum("btk,btl->tkl", ax, ay).flatten(1) / (b1 - b0)))
    return _join_groups(blocks)


def window_features(coefficients: torch.Tensor, reference: Reference, config: FeatureConfig = FeatureConfig()) -> torch.Tensor:
    """(windows, p): every histogram, per window."""
    n_bands = coefficients.shape[0]
    maps = scalar_maps(coefficients, reference)
    ranges = {name: PHASOR_RANGE if name.endswith(("_re", "_im")) else LEVEL_RANGE for name in maps}
    parts = [marginal(m, ranges[name], n_bands, config) for name, m in maps.items()]
    phasor_names = [*(f"advance{lag}" for lag in ADVANCE_LAGS), "neighbour", *(f"harmonic{h}" for _, h in HARMONICS)]
    pairs = [(f"{p}_re", f"{p}_im") for p in phasor_names]
    pairs += [("level", f"{p}_re") for p in ("advance8", "advance24", "advance75", "harmonic2", "harmonic3")]
    pairs += [("level", f"ratio{h}") for _, h in HARMONICS[:3]]
    pairs += [("rise16", "rise100"), ("level", "rise16"), ("level", "later50"), ("level", "later200"), ("level", "slope")]
    parts += [joint(maps[a], maps[b], ranges[a], ranges[b], n_bands, config) for a, b in pairs]
    return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class Critic:
    """Regularised linear discriminant between real windows and the current samples' windows."""

    real_mean: torch.Tensor  # (p,)
    centred: torch.Tensor  # (p, n): real and generated windows, each centred on its own mean
    inverse: torch.Tensor  # (n, n): (lambda * n * I + Z^T Z)^-1
    ridge: float

    @classmethod
    def fit(cls, real: torch.Tensor, generated: torch.Tensor, ridge_rel: float) -> Critic:
        """`ridge_rel` scales the ridge to the mean within-class feature variance."""
        centred = torch.cat([real - real.mean(0), generated - generated.mean(0)]).T.contiguous()
        n = centred.shape[1]
        ridge = ridge_rel * float(centred.pow(2).sum() / (n * centred.shape[0]))
        inverse = torch.linalg.inv(ridge * n * torch.eye(n) + centred.T @ centred)
        return cls(real.mean(0), centred, inverse, ridge)

    def loss(self, generated: torch.Tensor) -> torch.Tensor:
        """Regularised Mahalanobis distance between mean histograms, per feature.

        delta^T (Z Z^T / n + ridge I)^-1 delta / p, by the Woodbury identity.
        """
        delta = generated.mean(0) - self.real_mean
        projected = self.centred.T @ delta
        return (delta.pow(2).sum() - projected @ self.inverse @ projected) / (self.ridge * delta.numel())
