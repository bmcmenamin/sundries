"""Sliced-Wasserstein loss on spectrotemporal patches (after Heitz et al. 2021).

A patch is a block of a time-frequency map. Two maps are used:
  - the compressed cochleagram envelope (all channels x 25 frames), decimated to
    give 0.25 s, 1 s and 4 s patches: attacks, whole notes, phrase dynamics;
  - a compressed long-window STFT magnitude (64 bins x 16 frames), whose ~12 Hz
    bins resolve individual partials, which the envelope cannot (a steady partial
    has a flat envelope wherever it sits inside a cochlear band).

The loss compares the *distribution* of patches in a synthesis against a bank of
patches from the source: project both onto random directions, sort, and compare
the sorted values (1-D optimal transport). Directions and patch subsets are
drawn once per `PatchRound` so the objective is deterministic within a round,
which L-BFGS needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F

from texture_stats import N_COCHLEAR, compressed_envelopes

SPEC_N_FFT = 1024
SPEC_HOP = 128
SPEC_BINS = slice(7, 427)  # ~80 Hz .. 5 kHz at 11.7 Hz per bin
SPEC_COMPRESSION = 0.3
MAX_BANK = 16_384
N_PROJECTIONS = 256


@lru_cache(maxsize=None)
def _window() -> torch.Tensor:
    return torch.hann_window(SPEC_N_FFT)


def compressed_spectrogram(x: torch.Tensor) -> torch.Tensor:
    """(bins, frames) power-law compressed STFT magnitude."""
    stft = torch.stft(x, SPEC_N_FFT, hop_length=SPEC_HOP, window=_window(), return_complex=True)
    return ((stft.real**2 + stft.imag**2 + 1e-10).sqrt() ** SPEC_COMPRESSION)[SPEC_BINS]


@dataclass(frozen=True)
class Features:
    envelope: torch.Tensor  # (N_COCHLEAR, frames) at ENV_RATE
    spectrogram: torch.Tensor  # (bins, frames) at SAMPLE_RATE / SPEC_HOP

    @classmethod
    def from_waveform(cls, x: torch.Tensor) -> Features:
        return cls(compressed_envelopes(x), compressed_spectrogram(x))


@dataclass(frozen=True)
class PatchSpec:
    source: str  # a `Features` field name
    decimation: int  # frame-axis average pooling
    window: tuple[int, int]  # (rows, frames)
    row_stride: int

    def feature_map(self, features: Features) -> torch.Tensor:
        return F.avg_pool1d(getattr(features, self.source)[None], self.decimation)[0]

    def fits(self, features: Features) -> bool:
        rows, frames = self.feature_map(features).shape
        return rows >= self.window[0] and frames >= self.window[1]

    def view(self, feature_map: torch.Tensor) -> torch.Tensor:
        """(row positions, frame positions, rows, frames) sliding-window view."""
        return feature_map.unfold(0, self.window[0], self.row_stride).unfold(1, self.window[1], 1)


PATCH_SPECS = (
    PatchSpec("envelope", 4, (N_COCHLEAR, 25), N_COCHLEAR),
    PatchSpec("envelope", 16, (N_COCHLEAR, 25), N_COCHLEAR),
    PatchSpec("envelope", 64, (N_COCHLEAR, 25), N_COCHLEAR),
    PatchSpec("spectrogram", 1, (64, 16), 32),
    PatchSpec("spectrogram", 4, (64, 16), 32),
)


@dataclass(frozen=True)
class PatchScale:
    spec: PatchSpec
    mean: torch.Tensor  # (rows, 1)
    std: torch.Tensor  # (rows, 1)
    bank: torch.Tensor  # (n_patches, rows * frames), standardised

    @classmethod
    def from_features(cls, spec: PatchSpec, windows: Sequence[Features], generator: torch.Generator) -> PatchScale:
        maps = [spec.feature_map(f) for f in windows]
        pooled = torch.cat(maps, dim=-1)
        mean, std = pooled.mean(-1, keepdim=True), pooled.std(-1, keepdim=True) + 1e-6
        per_window = -(-MAX_BANK // len(maps))
        bank = torch.cat(
            [_subsample(spec.view((m - mean) / std).flatten(2).flatten(0, 1), per_window, generator) for m in maps]
        )
        return cls(spec, mean, std, bank)

    def _view(self, features: Features) -> torch.Tensor:
        return self.spec.view((self.spec.feature_map(features) - self.mean) / self.std)

    def positions(self, features: Features) -> int:
        row_positions, frame_positions = self._view(features).shape[:2]
        return row_positions * frame_positions

    def gather(self, features: Features, picks: torch.Tensor) -> torch.Tensor:
        view = self._view(features)
        return view[picks // view.shape[1], picks % view.shape[1]].flatten(1)


def _subsample(patches: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    return patches[torch.randperm(patches.shape[0], generator=generator)[:n]]


@dataclass(frozen=True)
class PatchBank:
    scales: tuple[PatchScale, ...]

    @classmethod
    def from_features(
        cls, windows: Sequence[Features], generator: torch.Generator, sources: Sequence[str]
    ) -> PatchBank:
        """Specs whose patches would not fit in the shortest window are dropped."""
        return cls(
            tuple(
                PatchScale.from_features(spec, windows, generator)
                for spec in PATCH_SPECS
                if spec.source in sources and all(spec.fits(f) for f in windows)
            )
        )


@dataclass(frozen=True)
class PatchRound:
    """One fixed draw of projection directions and patch subsets."""

    scales: tuple[PatchScale, ...]
    directions: tuple[torch.Tensor, ...]  # (dim, N_PROJECTIONS) unit columns
    synth_picks: tuple[torch.Tensor, ...]
    sorted_targets: tuple[torch.Tensor, ...]  # (m, N_PROJECTIONS)

    @classmethod
    def draw(cls, bank: PatchBank, features: Features, generator: torch.Generator) -> PatchRound:
        draws = [_draw_scale(scale, features, generator) for scale in bank.scales]
        return cls(bank.scales, *(tuple(d) for d in zip(*draws)))

    def loss(self, features: Features) -> torch.Tensor:
        return sum(
            _sliced_loss(scale.gather(features, picks) @ directions, target)
            for scale, directions, picks, target in zip(
                self.scales, self.directions, self.synth_picks, self.sorted_targets
            )
        )


def _draw_scale(scale: PatchScale, features: Features, generator: torch.Generator):
    n_synth = scale.positions(features)
    n_bank, dim = scale.bank.shape
    m = min(n_synth, n_bank)
    directions = torch.randn(dim, N_PROJECTIONS, generator=generator)
    directions = directions / directions.norm(dim=0)
    synth_picks = torch.randperm(n_synth, generator=generator)[:m]
    bank_picks = torch.randperm(n_bank, generator=generator)[:m]
    return directions, synth_picks, (scale.bank[bank_picks] @ directions).sort(0).values


def _sliced_loss(projected: torch.Tensor, sorted_target: torch.Tensor) -> torch.Tensor:
    return (projected.sort(0).values - sorted_target).pow(2).mean() / (sorted_target.pow(2).mean() + 1e-2)
