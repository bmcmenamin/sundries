"""Per-stem texture synthesis in a pretrained codec's latent space, by matching statistics.

    venv/bin/python codec_texture.py walk_1                       # every audible stem
    venv/bin/python codec_texture.py walk_1 --stems piano guitar

Per stem:
  1. encode the separated stem with EnCodec (codec.py): 128 latent channels at 75 Hz
  2. reduce to the leading principal components (decorrelated, and the latents are far from full rank)
  3. optimise a new coefficient sequence until it matches the stem's statistics (`Statistics`):
       values        each component's histogram
       motion        each component's change over a set of lags (how far and fast it moves, at several timescales)
       speed         the length of the whole vector's change over the same lags
       curvature     second differences (how sharply motion turns: attacks and decays)
       radius        the length of the whole vector, raw and in units of each component's spread: components can
                     each be ordinary while several are extreme at once, which no real frame is
       autocovariance  of every component and of the speed envelope out to 2 s: whether motion repeats (rhythm)
       cross-spectrum  magnitude of the cross-spectral matrix of the components' energy envelopes, in log-spaced
                     modulation bands from 0.5 to 15 Hz: which components pulse together, and at what rate
       activity      the speed averaged over 0.25 s ... 4 s, as a log-scale distribution: how long it stays active or still
                     (phrase and pause lengths; a histogram of single steps cannot say, and a share of the total energy
                     would let the pauses slip)
       joint         the leading components together, each in units of its spread, along 64 random projections: how they
                     depend on one another (bright only when loud), which separate histograms cannot say
       dynamics      position together with velocity (changes over 1 and 3 frames) of the leading components, along random
                     projections: how fast it moves where (a quiet state that stays still, a loud one that moves)
       ica           values and changes (lags 1, 3) along an overcomplete set of non-Gaussian axes of the source (FastICA,
                     3 random starts pooled): the heavy-tailed directions (attacks, fuzz) that PCA axes spread out
     histograms are compared as sorted values (1-D optimal transport), so the objective is deterministic
  4. optionally pull every run of `--window` frames toward the stem's own runs (`--manifold`): the nearest-neighbour
     distance guards against combinations the codec never saw, which can decode to an out-of-place sound; with a
     window longer than 1 it also demands that the trajectory (an attack, a decay) occurs in the source. With
     `--tolerance` the penalty is a hinge: runs within that many times the source's own novelty (how far its runs are
     from the rest of itself) are free, so the sample is held near the source but not pulled onto it. `--floor` adds
     the other side: runs closer than that are pushed away, since the closest ones are literal replays (waveform
     matches of 0.8 and above all had distances below about 0.3 of the source's novelty)
  5. invert the PCA and decode with the codec
Stems are written at their own relative loudness with one shared gain, so a set of stems mixes the way the source did:
  out/codec_texture/01_generate_<tag>/<input>/<stem>/<uuid>.{wav,pt}   (same <uuid> for the stems of one set; .pt = the
                                                                       synthesised PCA coefficients)
  out/codec_texture/02_mix_<tag>/<input>/<uuid>.wav               (their sum)
"""

from __future__ import annotations

import argparse
import math
import random
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy.io import wavfile

import codec
from paths import CODEC_TEXTURE, cache_dir, step_dir, stems_of
from perturb import CACHE_KIND as PERTURB_CACHE, effects_of
from synthesize import OUTPUT_PEAK, decode_mono
from synthesize_stems import stem_levels

LAG_SETS = {
    "sparse": (1, 3, 9, 27, 81),  # 13 ms ... 1.1 s at 75 Hz
    # dense enough to resolve a period: spacing of 2-4 frames (27-53 ms) out to 0.4 s, then widening to 2 s
    "dense": (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 36, 40, 45, 50, 60, 70, 80, 100, 125, 150),
}
LAG_SETS["half"] = LAG_SETS["dense"][::2] + (150,)  # every other lag of "dense": half the sorting for much the same resolution
CURVATURE_LAGS = (1, 3, 9)
ACF_LAGS = 150  # 2 s
FRAME_RATE = codec.CODEC_RATE / codec.HOP  # 75 Hz
MOD_COMPONENTS = 16  # leading components whose energy envelopes enter the cross-spectral matrix
MOD_BANDS = 20
MOD_RANGE = (0.5, 15.0)  # Hz
ACTIVITY_SCALES = (19, 38, 75, 150, 300)  # frames: 0.25 s ... 4 s
ACTIVITY_FLOOR = 0.02  # of the source's typical speed, added before the log so true stillness stays finite
JOINT_COMPONENTS = 8  # leading components whose joint distribution is matched
JOINT_PROJECTIONS = 64
ICA_RESTARTS = 3  # FastICA runs pooled into one overcomplete set of axes
ICA_ITERS = 200
ICA_LAGS = (1, 3)
DYNAMICS_LAG = 3  # frames: the longer of the two velocities in the state-velocity joint
PERFORMANCE_CORES = 4  # the fast cores of an M-series Air: more threads than this, or the slow cores, buy little
SHORTLIST = 32  # candidates the sketch search hands to the exact comparison (see `WindowBank.of`)
CROSS_LAGS = tuple(range(-150, 151, 5))  # frames (+-2 s): how far apart two stems' loud moments may be and still count as related
LOUDNESS_FLOOR = 1e-3  # added to a frame's latent norm before the log, so a silent frame has a finite loudness
RHYTHM_LEVELS = (0.25, 0.5, 0.75)  # quantiles of the leading component at which "active" is told from "resting"
RHYTHM_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)  # frames: 13 ms ... 3.4 s
RHYTHM_SMOOTHING = 19  # frames (0.25 s) averaged before telling active from resting
RHYTHM_SOFTNESS = 0.25  # in the leading component's standard deviations: how gradually "active" switches on
RHYTHM_CROSSING_SCALE = 10.0  # brings crossing rates (a few per hundred frames) to the size of the autocovariances


@dataclass(frozen=True)
class Basis:
    """Latents ~ mean + components @ coefficients."""

    mean: torch.Tensor  # (LATENTS, 1)
    components: torch.Tensor  # (LATENTS, k)

    @classmethod
    def fit(cls, latents: torch.Tensor, k: int, weights: torch.Tensor | None = None) -> Basis:
        """PCA of the frames, each counting in proportion to its weight (all equal when None)."""
        weights = torch.ones(latents.shape[1]) if weights is None else weights
        weights = weights / weights.sum()
        mean = (latents * weights).sum(1, keepdim=True)
        u, _, _ = torch.linalg.svd((latents - mean) * weights.sqrt(), full_matrices=False)
        return cls(mean, u[:, :k])

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.components.T @ (latents - self.mean)

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.mean + self.components @ coefficients


def _distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared error between sorted samples (rows), per unit of the target's energy; `b` is resampled to a's length."""
    if b.shape[-1] != a.shape[-1]:
        b = F.interpolate(b[None], size=a.shape[-1], mode="linear", align_corners=True)[0]
    return (a - b).pow(2).sum() / b.pow(2).sum()


def _log_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared difference between sorted log values (rows); `b` is resampled to a's length. On a log scale a pause
    counts as much as a loud passage, which energy-normalised distances (`_distance`) let slip."""
    if b.shape[-1] != a.shape[-1]:
        b = F.interpolate(b[None], size=a.shape[-1], mode="linear", align_corners=True)[0]
    return (a - b).pow(2).mean()


def _sorted(x: torch.Tensor) -> torch.Tensor:
    return x.sort(-1).values


def _autocovariance(rows: torch.Tensor) -> torch.Tensor:
    """(rows, ACF_LAGS + 1): mean-removed autocovariance at lags 0..ACF_LAGS, by FFT."""
    centred = rows - rows.mean(-1, keepdim=True)
    n = centred.shape[-1]
    power = torch.fft.rfft(centred, n=2 * n).abs().pow(2)
    return torch.fft.irfft(power, n=2 * n)[:, : ACF_LAGS + 1] / n


def _fast_ica(whitened: torch.Tensor, seed: int) -> torch.Tensor:
    """(k, k) orthogonal unmixing matrix for whitened coefficients (k, T): symmetric FastICA with g = tanh."""
    k, n = whitened.shape
    w = torch.linalg.qr(torch.randn(k, k, generator=torch.Generator().manual_seed(seed)))[0]
    for _ in range(ICA_ITERS):
        s = w @ whitened
        w = torch.tanh(s) @ whitened.T / n - (1 - torch.tanh(s).pow(2)).mean(1, keepdim=True) * w
        u, _, vt = torch.linalg.svd(w)
        w = u @ vt
    return w


@dataclass(frozen=True)
class Axes:
    """An overcomplete set of non-Gaussian directions of the source's coefficients: FastICA from several random starts,
    pooled. Heavy-tailed axes (a few rare, large values: attacks, fuzz) are what a random projection almost never lines
    up with and the PCA axes spread across several components."""

    mean: torch.Tensor  # (k, 1)
    spread: torch.Tensor  # (k, 1)
    directions: torch.Tensor  # (ICA_RESTARTS * k, k), on whitened coefficients
    change: torch.Tensor  # (k, 1): the source's standard deviation of a one-frame change, for scaling velocities
    levels: torch.Tensor  # (len(RHYTHM_LEVELS),): the source's quantiles of the whitened leading component

    @classmethod
    def fit(cls, coefficients: torch.Tensor) -> Axes:
        mean, spread = coefficients.mean(1, keepdim=True), coefficients.std(1, keepdim=True)
        whitened = (coefficients - mean) / spread
        change = (coefficients[:, 1:] - coefficients[:, :-1]).std(1, keepdim=True)
        levels = torch.quantile(_smooth(whitened[0]), torch.tensor(RHYTHM_LEVELS))
        return cls(mean, spread, torch.cat([_fast_ica(whitened, seed) for seed in range(ICA_RESTARTS)]), change, levels)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """(ICA_RESTARTS * k, T)."""
        return self.directions @ ((x - self.mean) / self.spread)


@lru_cache(maxsize=None)
def _directions(components: int) -> torch.Tensor:
    """(JOINT_PROJECTIONS, components) fixed random unit directions, the same for source and synthesis."""
    directions = torch.randn(JOINT_PROJECTIONS, components, generator=torch.Generator().manual_seed(0))
    return directions / directions.norm(dim=1, keepdim=True)


def _joint(x: torch.Tensor, spread: torch.Tensor) -> torch.Tensor:
    """(JOINT_PROJECTIONS, T) sorted: the leading components, each in units of the source's spread, projected on random
    directions. A projection mixes components, so matching these distributions matches how the components depend on one
    another (bright only when loud), which linearly uncorrelated PCA axes and their separate histograms cannot say."""
    k = min(JOINT_COMPONENTS, x.shape[0])
    return _sorted(_directions(k) @ (x[:k] / spread[:k]))


def _dynamics(x: torch.Tensor, axes: Axes) -> torch.Tensor:
    """(JOINT_PROJECTIONS, T - DYNAMICS_LAG) sorted: where the leading components are (in units of their spread) together
    with how they are moving (changes over 1 and DYNAMICS_LAG frames, in units of the source's typical change), along
    random projections. Matching these matches velocity as a function of position (a quiet state that stays still, a
    loud one that moves), which separate histograms of positions and of steps cannot say."""
    k = min(JOINT_COMPONENTS, x.shape[0])
    z = (x[:k] - axes.mean[:k]) / axes.spread[:k]
    unit = axes.change[:k]
    state = z[:, :-DYNAMICS_LAG]
    step = (x[:k, 1 : -DYNAMICS_LAG + 1] - x[:k, :-DYNAMICS_LAG]) / unit
    drift = (x[:k, DYNAMICS_LAG:] - x[:k, :-DYNAMICS_LAG]) / (unit * DYNAMICS_LAG**0.5)
    return _sorted(_directions(3 * k) @ torch.cat([state, step, drift]))


def _smooth(signal: torch.Tensor) -> torch.Tensor:
    """(T - RHYTHM_SMOOTHING + 1,): moving average, so the loudness contour is phrases and pauses, not frame jitter."""
    return F.avg_pool1d(signal[None, None], RHYTHM_SMOOTHING, stride=1)[0, 0]


def _rhythm(x: torch.Tensor, axes: Axes) -> torch.Tensor:
    """(levels, 1 + len(RHYTHM_LAGS)): how the stem switches between active and resting, at three levels.

    The leading component tracks loudness (to within a sign, which does not matter here: the levels are symmetric
    quantiles). For each level, "active" is a soft indicator of the component being past it; the row is that
    indicator's crossing rate (how often it starts or stops: pacing) and autocovariance at RHYTHM_LAGS (how long
    stretches and rests last). The histograms and autocovariances of the components themselves cannot say how
    long a pause is, only how often the component takes each value.
    """
    return _pacing(_smooth((x[0] - axes.mean[0]) / axes.spread[0]), axes.levels)


def _pacing(signal: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """(len(levels), 1 + len(RHYTHM_LAGS)): crossing rate and autocovariance of "the smoothed `signal` is past this level"."""
    active = torch.sigmoid((signal[None] - levels[:, None]) / RHYTHM_SOFTNESS)
    crossing = (active[:, 1:] - active[:, :-1]).abs().mean(1, keepdim=True)
    centred = active - active.mean(1, keepdim=True)
    covariance = torch.cat([(centred[:, lag:] * centred[:, :-lag]).mean(1, keepdim=True) for lag in RHYTHM_LAGS], 1)
    return torch.cat([RHYTHM_CROSSING_SCALE * crossing, covariance], 1)


def loudness_curve(latents: torch.Tensor) -> torch.Tensor:
    """Smoothed, standardised log loudness of frames of latents (LATENTS, T): the shape of when a stem is loud or quiet."""
    log = torch.log(latents.norm(dim=0) + LOUDNESS_FLOOR)
    return _smooth((log - log.mean()) / (log.std() + 1e-6))


def _cross_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(len(CROSS_LAGS),): correlation of `a` at time t + lag with `b` at t, over the frames both cover. Positive lags: `a`
    follows `b`."""
    a, b = (s - s.mean() for s in (a, b))
    a, b = a / (a.std() + 1e-6), b / (b.std() + 1e-6)
    n = min(len(a), len(b))
    return torch.stack([(a[lag:n] * b[: n - lag]).mean() if lag >= 0 else (a[: n + lag] * b[-lag:n]).mean() for lag in CROSS_LAGS])


@dataclass(frozen=True)
class Partner:
    """A stem generated earlier in the same sample, and how the stem being generated should move with it."""

    curve: torch.Tensor  # the partner's generated `loudness_curve`
    correlation: torch.Tensor  # (len(CROSS_LAGS),): the source's cross-correlation of the two stems' loudness curves

    def penalty(self, basis: Basis, x: torch.Tensor) -> torch.Tensor:
        return (_cross_correlation(loudness_curve(basis.decode(x)), self.curve) - self.correlation).pow(2).mean()


@dataclass(frozen=True)
class Loudness:
    """How loud a stem is from frame to frame, read from the norm of its decoded latents (which follows the true frame
    loudness at r = 0.8-0.97; the leading PCA component, which `_rhythm` uses, does not): the histogram of log loudness,
    so quiet passages keep their share, and the pacing of loud and quiet (`_pacing`) at levels set by the source."""

    values: torch.Tensor  # (T,) sorted log norms
    pacing: torch.Tensor
    levels: torch.Tensor  # (len(RHYTHM_LEVELS),): the source's quantiles of the smoothed, standardised log loudness
    centre: float
    spread: float

    @classmethod
    def of_source(cls, x: torch.Tensor, basis: Basis) -> Loudness:
        log = torch.log(basis.decode(x).norm(dim=0) + LOUDNESS_FLOOR)
        centre, spread = float(log.mean()), float(log.std())
        levels = torch.quantile(_smooth((log - centre) / spread), torch.tensor(RHYTHM_LEVELS))
        return cls._measured(log, levels, centre, spread)

    def of_synthesis(self, x: torch.Tensor, basis: Basis) -> Loudness:
        """The same statistics of `x`, at this source's levels and scale."""
        return self._measured(torch.log(basis.decode(x).norm(dim=0) + LOUDNESS_FLOOR), self.levels, self.centre, self.spread)

    @classmethod
    def _measured(cls, log: torch.Tensor, levels: torch.Tensor, centre: float, spread: float) -> Loudness:
        return cls(_sorted(log), _pacing(_smooth((log - centre) / spread), levels), levels, centre, spread)

    def distance(self, target: Loudness) -> torch.Tensor:
        """Mean squared gap between the sorted log loudnesses, relative to the target's variance, plus the pacing distance."""
        values = self.values
        reference = target.values
        if reference.shape[-1] != values.shape[-1]:
            reference = F.interpolate(reference[None, None], size=values.shape[-1], mode="linear", align_corners=True)[0, 0]
        return (values - reference).pow(2).mean() / target.values.var() + _distance(self.pacing, target.pacing)


def _band_weights(n: int) -> torch.Tensor:
    """(MOD_BANDS, n // 2 + 1): averaging weights over the FFT bins of each log-spaced modulation band.

    A band too narrow to hold a bin takes the bin nearest its centre, so sources and syntheses of different lengths
    still compare band for band.
    """
    freqs = torch.fft.rfftfreq(n, 1 / FRAME_RATE)
    edges = torch.logspace(math.log10(MOD_RANGE[0]), math.log10(MOD_RANGE[1]), MOD_BANDS + 1)
    member = (freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])
    centre = (edges[:-1] * edges[1:]).sqrt()
    nearest = (freqs[None] - centre[:, None]).abs().argmin(1)
    member[torch.arange(MOD_BANDS), nearest] |= ~member.any(1)
    return member.float() / member.sum(1, keepdim=True)


def _cross_spectrum(x: torch.Tensor) -> torch.Tensor:
    """(MOD_BANDS, c, c): magnitude of the cross-spectral matrix of the energy envelopes of the leading MOD_COMPONENTS
    components and of the whole vector, per band of modulation frequency (0.5-15 Hz: beats and bars, not audio).

    Each envelope is the squared frame-to-frame change, unit variance. The diagonal is each envelope's modulation
    spectrum; off-diagonal cells say which components pulse together at which rate. The cross-spectrum is averaged over
    a band's bins before its magnitude is taken (one bin alone would be rank 1: just the product of two amplitudes).
    """
    change = x[:, 1:] - x[:, :-1]
    envelopes = torch.cat([change[:MOD_COMPONENTS].pow(2), change.pow(2).sum(0, keepdim=True)])
    centred = envelopes - envelopes.mean(-1, keepdim=True)
    unit = centred / (centred.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
    spectrum = torch.fft.rfft(unit) / unit.shape[-1] ** 0.5
    banded = torch.einsum("bf,if,jf->bij", _band_weights(unit.shape[-1]), spectrum, spectrum.conj())
    return (banded.real.pow(2) + banded.imag.pow(2) + 1e-12).sqrt()


@dataclass(frozen=True)
class Statistics:
    values: torch.Tensor  # (k, T) sorted along time
    motion: tuple[torch.Tensor, ...]  # per lag: (k, T - lag), each component's change, sorted
    speed: tuple[torch.Tensor, ...]  # per lag: (1, T - lag), length of the vector's change, sorted
    curvature: tuple[torch.Tensor, ...]  # per lag: (k, T - 2 lag), second differences, sorted
    radius: tuple[torch.Tensor, ...]  # (1, T) sorted: the vector's length, raw and in units of each component's spread
    autocovariance: torch.Tensor  # (k + 1, ACF_LAGS + 1): components, then the speed envelope
    cross_spectrum: torch.Tensor  # (MOD_BANDS, c, c): see `_cross_spectrum`
    activity: tuple[torch.Tensor, ...]  # per ACTIVITY_SCALES: (1, T - scale), log of the speed averaged over that span, sorted
    joint: torch.Tensor  # (JOINT_PROJECTIONS, T): see `_joint`
    ica: torch.Tensor  # (n, T) sorted: values along the `Axes`
    ica_motion: tuple[torch.Tensor, ...]  # per ICA_LAGS: (n, T - lag), change along the `Axes`, sorted
    dynamics: torch.Tensor  # (JOINT_PROJECTIONS, T - DYNAMICS_LAG): see `_dynamics`
    rhythm: torch.Tensor  # (levels, 1 + len(RHYTHM_LAGS)): see `_rhythm`

    @classmethod
    def of(
        cls, x: torch.Tensor, lags: tuple[int, ...], spread: torch.Tensor, axes: Axes,
        activity: float = 1.0, joint: float = 1.0, ica: float = 1.0, dynamics: float = 1.0, rhythm: float = 1.0,
    ) -> Statistics:
        """`spread` (k, 1): the source's standard deviation per component, so synthesis is measured on the same scale.

        A statistic whose weight is 0 is left empty (and skipped by `distance`): sorting it every step is the cost, not the weight."""
        changes = [x[:, lag:] - x[:, :-lag] for lag in lags]
        envelope = (x[:, 1:] - x[:, :-1]).norm(dim=0, keepdim=True)
        along = axes.project(x) if ica else None
        none = torch.zeros(0)
        return cls(
            _sorted(x),
            tuple(_sorted(c) for c in changes),
            tuple(_sorted(c.norm(dim=0, keepdim=True)) for c in changes),
            tuple(_sorted(x[:, 2 * lag :] - 2 * x[:, lag:-lag] + x[:, : -2 * lag]) for lag in CURVATURE_LAGS),
            (_sorted(x.norm(dim=0, keepdim=True)), _sorted((x / spread).norm(dim=0, keepdim=True))),
            _autocovariance(torch.cat([x[:, :-1], envelope])),
            _cross_spectrum(x),
            tuple(
                _sorted(torch.log(F.avg_pool1d(envelope[None], scale, stride=1)[0] + ACTIVITY_FLOOR * axes.change.norm()))
                for scale in ACTIVITY_SCALES
            ) if activity else (),
            _joint(x, spread) if joint else none,
            _sorted(along) if ica else none,
            tuple(_sorted(along[:, lag:] - along[:, :-lag]) for lag in ICA_LAGS) if ica else (),
            _dynamics(x, axes) if dynamics else none,
            _rhythm(x, axes) if rhythm else none,
        )

    def distance(
        self, target: Statistics, modulation: float, activity: float, joint: float, ica: float, dynamics: float, rhythm: float = 0.0
    ) -> torch.Tensor:
        acf = ((self.autocovariance - target.autocovariance).pow(2).sum(-1) / target.autocovariance.pow(2).sum(-1)).mean()
        cross = ((self.cross_spectrum - target.cross_spectrum).pow(2).sum((1, 2)) / target.cross_spectrum.pow(2).sum((1, 2))).mean()
        rate = sum(_log_distance(a, b) for a, b in zip(self.activity, target.activity))
        optional = sum(
            weight * _distance(a, b)
            for weight, a, b in ((rhythm, self.rhythm, target.rhythm), (joint, self.joint, target.joint), (dynamics, self.dynamics, target.dynamics), (ica, self.ica, target.ica))
            if weight
        )
        return (
            modulation * cross
            + optional
            + activity * rate
            + ica * sum(_distance(a, b) for a, b in zip(self.ica_motion, target.ica_motion))
            + _distance(self.values, target.values)
            + sum(_distance(a, b) for a, b in zip(self.motion, target.motion))
            + sum(_distance(a, b) for a, b in zip(self.speed, target.speed))
            + sum(_distance(a, b) for a, b in zip(self.curvature, target.curvature))
            + sum(_distance(a, b) for a, b in zip(self.radius, target.radius))
            + acf
        )


def _windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """(T - window + 1, k * window): every run of `window` consecutive frames, flattened."""
    return x.unfold(1, window, 1).permute(1, 0, 2).flatten(1)


@dataclass(frozen=True)
class WindowBank:
    """Every run of `window` consecutive frames of some clips (never spanning two clips), flattened."""

    runs: torch.Tensor  # (N, k * window)
    window: int
    energy: float  # mean squared norm of a run: the unit distances are measured in
    clip: torch.Tensor  # (N,): which clip each run came from
    basis: torch.Tensor | None = None  # (k * window, dims): the bank's main directions; None = search the runs themselves
    sketch: torch.Tensor | None = None  # (N, dims): the runs along `basis`, a cheap stand-in for the search

    @classmethod
    def of(cls, clips: list[torch.Tensor], window: int, stride: int = 1, dims: int = 0) -> WindowBank:
        """`dims` > 0 and below the run length: nearest runs are looked up in a `dims`-dimensional sketch (the bank's top
        principal directions), then settled exactly among the best `SHORTLIST` candidates."""
        per_clip = [_windows(clip, window)[::stride] for clip in clips]
        runs = torch.cat(per_clip)
        owner = torch.cat([torch.full((len(r),), i) for i, r in enumerate(per_clip)])
        energy = float(runs.pow(2).sum(1).mean())
        if not 0 < dims < runs.shape[1]:
            return cls(runs, window, energy, owner)
        basis = torch.linalg.eigh(runs.T @ runs).eigenvectors[:, -dims:]
        return cls(runs, window, energy, owner, basis, runs @ basis)

    def distance(self, x: torch.Tensor, takes: torch.Tensor | None = None, rows: torch.Tensor | None = None) -> torch.Tensor:
        """(T - window + 1,): squared distance from each run of `x` to the closest run in the bank, per unit of energy.
        window=1 compares single frames; longer windows also demand that the trajectory (attack shape, decay, which
        components move together) occurs in the source. The nearest run is found without gradients; the distance to it
        is what is differentiated (the same gradient, without backpropagating through the whole table).

        `rows`: only the runs of `x` starting at these frames (one distance each, in this order); the rest are never built.
        `takes`: when given, the clip each of those runs may be compared with; the others are out of reach."""
        mine = _windows(x, self.window) if rows is None else x.unfold(1, self.window, 1)[:, rows].permute(1, 0, 2).flatten(1)
        with torch.no_grad():
            far = torch.cdist(mine if self.basis is None else mine @ self.basis, self.runs if self.sketch is None else self.sketch)
            if takes is not None:
                far = far.masked_fill(self.clip[None, :] != takes[:, None], float("inf"))
            nearest = far.argmin(1) if self.sketch is None else self._settled(mine, far.topk(SHORTLIST, largest=False).indices)
        return (mine - self.runs[nearest]).pow(2).sum(1) / self.energy

    def _settled(self, mine: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """(T,): of each run's `candidates` (T, m) indices into the bank, the one exactly closest."""
        exact = (mine[:, None, :] - self.runs[candidates]).pow(2).sum(-1)
        return candidates.gather(1, exact.argmin(1, keepdim=True)).squeeze(1)


def nearest_frame_distance(x: torch.Tensor, frames: torch.Tensor, window: int = 1) -> torch.Tensor:
    """`WindowBank.distance` against the runs of a single clip."""
    return WindowBank.of([frames], window).distance(x)


def source_novelty(frames: torch.Tensor, window: int) -> float:
    """The median of how far a run of `window` source frames is from its closest run elsewhere in the source (runs
    overlapping it excluded), on the scale of `nearest_frame_distance`: the source's own typical unfamiliarity."""
    runs = _windows(frames, window)
    distance = torch.cdist(runs, runs).pow(2)
    index = torch.arange(len(runs))
    distance[(index[:, None] - index[None, :]).abs() < window] = float("inf")
    return float((distance.amin(1) / runs.pow(2).sum(1).mean()).median())


def choose_takes(frames: int, window: int, clips: int, seconds: tuple[float, float], generator: torch.Generator) -> torch.Tensor:
    """(frames - window + 1,): for each run, which of `clips` copies it is compared with. The timeline is cut into stretches
    of random length (uniform in `seconds`) and each stretch gets one copy, so pitch and timbre stay consistent within it
    and the pull only hops between copies at the cuts."""
    lengths = [max(1, round(float(torch.empty(()).uniform_(*seconds, generator=generator)) * FRAME_RATE)) for _ in range(frames)]
    cuts = torch.cumsum(torch.tensor(lengths), 0)
    stretch = torch.searchsorted(cuts, torch.arange(frames - window + 1), right=True)
    return torch.randint(0, clips, (int(stretch.max()) + 1,), generator=generator)[stretch]


def perturbed_latents(source: str, stem: str, count: int, effects: str = "") -> list[torch.Tensor]:
    """EnCodec latents of the first `count` perturbed copies of a stem (made by perturb.py) whose set of effects is exactly
    `effects` (letters, see `effects_of`; any when empty); encoded once, then cached."""
    every = sorted(cache_dir(PERTURB_CACHE, source, stem).glob("*.wav"))
    paths = [p for p in every if not effects or set(effects_of(p)) == set(effects)][:count]
    if len(paths) < count:
        raise SystemExit(f"{source}/{stem}: asked for {count} perturbed copies{f' with effects {effects}' if effects else ''}, found {len(paths)} (run perturb.py)")

    def latents(path) -> torch.Tensor:
        cache = path.with_name(path.stem + ".latents.pt")
        if not cache.exists():
            torch.save(codec.encode(codec.load(path)), cache)
        return torch.load(cache)

    return [latents(p) for p in paths]


def pooled(original: torch.Tensor, variants: list[torch.Tensor], k: int, original_share: float):
    """Arguments of `Basis.fit` for the original plus its copies: the original's frames hold `original_share` of the weight."""
    if not variants:
        return original, k
    frames = torch.cat([original, *variants], 1)
    n = sum(v.shape[1] for v in variants)
    weights = torch.cat([torch.full((original.shape[1],), original_share / original.shape[1]), torch.full((n,), (1 - original_share) / n)])
    return frames, k, weights


def pulled_runs(frames: int, window: int, share: float, generator: torch.Generator) -> torch.Tensor:
    """Indices of the runs of `window` frames that the manifold term pulls toward the source: non-overlapping runs (at a
    random offset), each kept with probability `share`, so the rest of the frames are free to move between the pulled ones."""
    starts = torch.arange(int(torch.randint(0, window, (1,), generator=generator)), frames - window + 1, window)
    kept = starts[torch.rand(len(starts), generator=generator) < share]
    return kept if len(kept) else starts[:1]


NEIGHBOURS = 8  # a jump in the chain start goes to a successor of one of this many nearest frames


def initial_frames(coefficients: torch.Tensor, frames: int, generator: torch.Generator, init: str) -> torch.Tensor:
    """(k, frames) starting point drawn from the source's own frames.

    shuffle  independent draws: the right values, white motion
    chain    a random walk over the source's frames in the manner of video textures: from frame i go to the
             successor of one of the NEIGHBOURS frames nearest to i (a frame that looked like i, and what came next
             there). Motion is already locally smooth, so the optimiser only has to fix the longer timescales.
    """
    n = coefficients.shape[1]
    if init == "shuffle":
        return coefficients[:, torch.randint(0, n, (frames,), generator=generator)]
    distance = torch.cdist(coefficients.T, coefficients.T)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.topk(NEIGHBOURS, largest=False).indices  # (n, NEIGHBOURS)
    picks = torch.randint(0, NEIGHBOURS, (frames,), generator=generator)
    path = [int(torch.randint(0, n, (1,), generator=generator))]
    for pick in picks[1:]:
        path.append(min(int(nearest[path[-1], pick]) + 1, n - 1))
    return coefficients[:, torch.tensor(path)]


@dataclass(frozen=True)
class Lead:
    """Latents the start of a sample should continue from (the tail of the previous clip of a song), and how hard to hold to them."""

    latents: torch.Tensor  # (LATENTS, frames)
    weight: float

    def start(self, basis: Basis, frames: torch.Tensor) -> torch.Tensor:
        """`frames` (k, T) with its first columns replaced by the lead, in `basis` coordinates."""
        return torch.cat([basis.encode(self.latents), frames[:, self.latents.shape[1] :]], 1)

    def penalty(self, basis: Basis, x: torch.Tensor) -> torch.Tensor:
        """How far the decoded start of `x` is from the lead, relative to the lead's own spread."""
        return self.weight * (basis.decode(x[:, : self.latents.shape[1]]) - self.latents).pow(2).mean() / self.latents.var()


@dataclass(frozen=True)
class Band:
    """The manifold term at one window length: which runs of the synthesis are pulled, toward what, and how hard."""

    bank: WindowBank
    ceiling: float  # distances below this cost nothing
    floor: float  # distances below this are pushed up
    pulled: torch.Tensor  # (P,) start frames of the runs of the synthesis that are compared (the rest never are)
    copy: torch.Tensor | None  # (P,) the bank clip each pulled run may be compared with; None = any

    @classmethod
    def of(
        cls,
        coefficients: torch.Tensor,
        pool: list[torch.Tensor] | None,
        frames: int,
        scale: tuple[int, float, float, float],
        takes: tuple[float, float] | None,
        search_dims: int,
        generator: torch.Generator,
    ) -> Band:
        """`scale`: (window, tolerance, floor, share), the last two as in `pulled_runs`; tolerance and floor are in multiples
        of the source's own novelty. The manifold is the pool when given (perturbed copies), else the source itself.

        With `takes`, only the copies some pulled run is assigned to go into the bank, and each run is searched in its own
        copy exactly: 64 copies are rarely all used, and the pulled runs are a few dozen, so this is cheap."""
        window, tolerance, floor, share = scale
        novelty = source_novelty(coefficients, window)
        pulled = pulled_runs(frames, window, share, generator)
        if not pool:
            return cls(WindowBank.of([coefficients], window), tolerance * novelty, floor * novelty, pulled, None)
        if not takes:
            return cls(WindowBank.of(pool, window, stride=2, dims=search_dims), tolerance * novelty, floor * novelty, pulled, None)
        used, copy = choose_takes(frames, window, len(pool), takes, generator)[pulled].unique(return_inverse=True)
        return cls(WindowBank.of([pool[i] for i in used.tolist()], window, stride=2), tolerance * novelty, floor * novelty, pulled, copy)

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        d = self.bank.distance(x, self.copy, self.pulled)
        return ((d - self.ceiling).clamp(min=0) + (self.floor - d).clamp(min=0)).mean()


def synthesise(
    coefficients: torch.Tensor,
    frames: int,
    iters: int,
    seed: int,
    lags: tuple[int, ...],
    manifold: float,
    init: str,
    modulation: float,
    window: int,
    tolerance: float,
    activity: float,
    joint: float,
    ica: float,
    dynamics: float,
    floor_: float,
    long_window: int,
    long_tolerance: float,
    short_share: float = 1.0,
    pool: list[torch.Tensor] | None = None,
    rhythm: float = 0.0,
    takes: tuple[float, float] | None = None,
    search_dims: int = 0,
    basis: Basis | None = None,
    lead: Lead | None = None,
    loudness: float = 0.0,
    cross: float = 0.0,
    partners: tuple[Partner, ...] = (),
) -> torch.Tensor:
    """(k, frames) coefficients with the source's statistics, and frames that stay near the source's own.

    Starts from frames of the source (see `initial_frames`) and moves them by L-BFGS.
    """
    generator = torch.Generator().manual_seed(seed)
    spread = coefficients.std(-1, keepdim=True)
    axes = Axes.fit(coefficients)
    used = (activity, joint, ica, dynamics, rhythm)  # the weights that are not 0: the other statistics are not computed
    target = Statistics.of(coefficients, lags, spread, axes, *used)
    target_loudness = Loudness.of_source(coefficients, basis) if loudness else None
    start = initial_frames(coefficients, frames, generator, init).clone()
    y = (lead.start(basis, start) if lead else start).requires_grad_()
    optimiser = torch.optim.LBFGS([y], lr=1.0, max_iter=iters, history_size=30, line_search_fn="strong_wolfe")

    scales = [(window, tolerance, floor_, short_share)] + ([(long_window, long_tolerance, 0.0, 1.0)] if long_window else [])
    bands = [Band.of(coefficients, pool, frames, scale, takes, search_dims, generator) for scale in scales]

    def objective(x: torch.Tensor) -> torch.Tensor:
        penalty = sum(band.penalty(x) for band in bands)
        held = lead.penalty(basis, x) if lead else 0.0
        loud = loudness * target_loudness.of_synthesis(x, basis).distance(target_loudness) if target_loudness else 0.0
        together = cross * sum(p.penalty(basis, x) for p in partners)
        return Statistics.of(x, lags, spread, axes, *used).distance(target, modulation, activity, joint, ica, dynamics, rhythm) + manifold * penalty + held + loud + together

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = objective(y)
        loss.backward()
        return loss

    first = optimiser.step(closure).item()
    print(f"    objective {first:.3f} -> {objective(y).item():.4f}", flush=True)
    return y.detach()


def loudness(audio: torch.Tensor) -> float:
    """Perceived loudness in LUFS (ITU-R BS.1770: K-weighted, gated so pauses don't count): equal values sound equally loud
    where equal RMS does not, e.g. a steady texture against the same part with gaps."""
    return float(AF.loudness(audio[None], codec.CODEC_RATE))


def stem_loudness(path) -> float:
    """Loudness of a separated stem at its own level (`codec.load` would rescale every stem to the same RMS)."""
    return loudness(decode_mono(path, codec.CODEC_RATE))


def parse_trim(items: list[str]) -> dict[str, float]:
    """['guitar=-4', 'drums=2'] -> {'guitar': -4.0, 'drums': 2.0}"""
    return {name: float(db) for name, db in (item.split("=") for item in items)}


def save(path, audio: torch.Tensor, gain: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, codec.CODEC_RATE, (audio * gain * 32767).round().clamp(-32767, 32767).short().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--stems", nargs="+", help="stems to synthesise (default: all audible)")
    parser.add_argument("--min-db", type=float, default=-25.0, help="skip stems this far below the mix")
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--lags", choices=list(LAG_SETS), default="dense", help="timescales of the motion and speed histograms")
    parser.add_argument("--manifold", type=float, default=0.3, help="weight of the nearest-source-run distance (0 = off)")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="hinge: runs closer to the source than this many times the source's own novelty cost nothing "
                             "(0 = every bit of distance counts, which pulls runs onto the source)")
    parser.add_argument("--modulation", type=float, default=1.0, help="weight of the cross-spectral matrix of component energy envelopes (0 = off)")
    parser.add_argument("--activity", type=float, default=0.0,
                        help="weight of the distribution (on a log scale) of the speed averaged over 0.25 s ... 4 s: how long "
                             "it stays active or still, so phrase and pause lengths, and how many attacks there are (0 = off)")
    parser.add_argument("--joint", type=float, default=0.0,
                        help="weight of the joint distribution of the leading components, matched along random "
                             "projections (0 = off)")
    parser.add_argument("--long-window", type=int, default=0,
                        help="a second, longer window for the manifold term, with its own --long-tolerance as a ceiling only "
                             "(0 = off). Use with a short --window: the short one keeps every piece close to a real piece "
                             "(clean audio, no replayed words), the long one keeps the trajectory plausible over a syllable "
                             "or more (not busy)")
    parser.add_argument("--long-tolerance", type=float, default=1.0,
                        help="ceiling of the long window, in multiples of the source's novelty at that window")
    parser.add_argument("--take-seconds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="with --augment, cut each sample's timeline into stretches of random length in this range (seconds) "
                             "and let each stretch be pulled toward one perturbed copy only, so the copies are not blended")
    parser.add_argument("--search-dims", type=int, default=128,
                        help="with --augment, find nearest runs in a sketch of this many dimensions, then settle among the best few "
                             "exactly (much faster on 64 copies); 0 = exact search")
    parser.add_argument("--loudness", type=float, default=0.0,
                        help="weight of matching the stem's frame loudness (read from the latent norm): its histogram, so quiet "
                             "stretches keep their share, and how it switches between loud and quiet")
    parser.add_argument("--cross", type=float, default=0.0,
                        help="weight of making stems loud and quiet together (or in alternation) as they are in the source: the stems "
                             "are generated one after another in a random order per sample, and each is matched to the loudness "
                             "of those before it")
    parser.add_argument("--rhythm", type=float, default=0.0,
                        help="weight of the pacing of the leading (loudness) component: how often it switches between "
                             "active and resting and how long stretches and rests last, at three levels (0 = off). Every stem")
    parser.add_argument("--augment", type=int, default=0,
                        help="number of perturbed copies of each stem (from perturb.py) to use: the PCA is fitted on the "
                             "original plus the copies, and the manifold term pulls toward the copies only, never the original "
                             "(0 = the original alone is the manifold)")
    parser.add_argument("--augment-effects", default="", choices=["", "P", "C", "S", "PC", "PS", "CS", "PCS"],
                        help="use only the copies whose effects are exactly these (P itch, C omb, S ideband); any when empty")
    parser.add_argument("--original-share", type=float, default=0.5,
                        help="with --augment, the share of the PCA's weight held by the original's frames (so it stays "
                             "reconstructable); the copies share the rest equally")
    parser.add_argument("--short-share", type=float, default=1.0,
                        help="share of the short window's (non-overlapping) runs that are pulled toward the source, chosen "
                             "at random per sample; 1 = every frame. Below 1 the long window can't tighten every piece into a copy")
    parser.add_argument("--floor", type=float, default=0.0,
                        help="the other side of the band: runs closer to the source than this many times the source's own "
                             "novelty are pushed away (0 = off). Real waveform-level matches live below about 0.3. "
                             "Must not exceed --tolerance")
    parser.add_argument("--dynamics", type=float, default=0.0,
                        help="weight of the joint distribution of position and velocity of the leading components, "
                             "along random projections: how fast it moves where (0 = off)")
    parser.add_argument("--ica", type=float, default=0.0,
                        help="weight of the distribution of the values and changes along an overcomplete set of ICA "
                             "axes of the source (0 = off)")
    parser.add_argument("--window", type=int, default=24, help="frames per run compared with the source in the manifold term (1 = single frames)")
    parser.add_argument("--init", choices=["shuffle", "chain"], default="shuffle", help="how the optimisation starts")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--trim", nargs="*", default=[], metavar="STEM=DB",
                        help="turn a stem up or down in the mix, in dB (e.g. guitar=-4); stems are matched to their source's "
                             "loudness (LUFS) first")
    parser.add_argument("--lead-from", type=Path, metavar="DIR",
                        help="continue a previous clip of a song: start from the tail of the latents it saved in DIR "
                             "(<stem>.pt, see --latents-dir) and hold to them for --lead-seconds")
    parser.add_argument("--lead-seconds", type=float, default=2.0)
    parser.add_argument("--lead-weight", type=float, default=5.0, help="weight of holding to the lead, against the statistics")
    parser.add_argument("--latents-dir", type=Path, metavar="DIR",
                        help="also save each stem's generated latents there as <stem>.pt (the next clip's --lead-from)")
    parser.add_argument("--name", help="file name for the sample instead of a fresh uuid (with --n 1)")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--decode", choices=["continuous", "snapped6", "snapped24"], default="continuous",
                        help="decode the latents as they are, or snapped to the codec's quantiser grid at 6 / 24 kbps")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=0,
                        help="samples made at the same time, each in its own process with the cores shared between them "
                             "(0 = all of --n, up to the performance cores): the threads of one process scale poorly")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.floor > args.tolerance:
        parser.error("--floor must not exceed --tolerance")

    stems = stems_of(args.source)
    report = stem_levels(stems)
    kept = [s for s in stems if report[s.name][1] >= args.min_db and (args.stems is None or s.name in args.stems)]
    tag = f"k{args.components}_{args.decode}_{args.lags}_m{args.manifold:g}" + (f"w{args.window}" if args.window > 1 else "") + (f"_x{args.modulation:g}" if args.modulation else "") + (f"_t{args.tolerance:g}" if args.tolerance else "") + (f"_f{args.floor:g}" if args.floor else "") + (f"_g{args.augment}{args.augment_effects}" if args.augment else "") + (f"_s{args.short_share:g}" if args.short_share < 1 else "") + (f"_L{args.long_window}t{args.long_tolerance:g}" if args.long_window else "") + (f"_T{args.take_seconds[0]:g}-{args.take_seconds[1]:g}" if args.take_seconds else "") + (f"_r{args.rhythm:g}" if args.rhythm else "") + (f"_ld{args.loudness:g}" if args.loudness else "") + (f"_cx{args.cross:g}" if args.cross else "") +(f"_a{args.activity:g}" if args.activity else "") + (f"_j{args.joint:g}" if args.joint else "") + (f"_i{args.ica:g}" if args.ica else "") + (f"_d{args.dynamics:g}" if args.dynamics else "") + ("" if args.init == "shuffle" else f"_{args.init}")
    frames = round(args.seconds * codec.CODEC_RATE / codec.HOP)
    trim = parse_trim(args.trim)
    loud = {s.name: stem_loudness(s.path) for s in kept}  # each stem's loudness in the source, which its sample is matched to
    print("  source loudness (LUFS): " + ", ".join(f"{name} {value:.1f}" for name, value in loud.items()), flush=True)
    sources: dict[str, tuple[Basis, torch.Tensor, list[torch.Tensor] | None, torch.Tensor]] = {}  # basis, coefficients, copies, loudness curve
    for stem in kept:
        latents = codec.encode(codec.load(stem.path))
        variants = perturbed_latents(args.source, stem.name, args.augment, args.augment_effects)
        basis = Basis.fit(*pooled(latents, variants, args.components, args.original_share))
        error = lambda b: float((b.decode(b.encode(latents)) - latents).pow(2).sum() / (latents - latents.mean(1, keepdim=True)).pow(2).sum())
        if variants:
            print(f"  {stem.name}: {len(variants)} perturbed copies; original's PCA reconstruction error {error(basis):.3f} "
                  f"(own basis {error(Basis.fit(latents, args.components)):.3f})", flush=True)
        sources[stem.name] = (basis, basis.encode(latents), [basis.encode(v) for v in variants] or None, loudness_curve(latents))

    plan = Plan(args, tag, frames, tuple(kept), loud, trim, sources)
    jobs = min(args.jobs or args.n, args.n)
    if jobs == 1:
        for i in range(args.n):
            render(plan, i)
        return
    with ProcessPoolExecutor(jobs, initializer=torch.set_num_threads, initargs=(max(1, PERFORMANCE_CORES // jobs),)) as workers:
        list(workers.map(partial(render, plan), range(args.n)))


@dataclass(frozen=True)
class Plan:
    """Everything the samples of one run share; each sample is `render`ed from it (in this process or a worker)."""

    args: argparse.Namespace
    tag: str
    frames: int
    kept: tuple  # the stems to make
    loud: dict[str, float]  # each stem's loudness in the source (LUFS)
    trim: dict[str, float]
    sources: dict[str, tuple[Basis, torch.Tensor, list[torch.Tensor] | None, torch.Tensor]]  # basis, coefficients, copies, loudness curve


def render(plan: Plan, i: int) -> None:
    """Make sample `i`: the stems one after another (in a random order with --cross), their mix, and write them out."""
    args, kept, sources = plan.args, plan.kept, plan.sources
    snap = {"continuous": lambda z: z, "snapped6": codec.snap, "snapped24": lambda z: codec.snap(z, 24.0)}[args.decode]
    lead_frames = round(args.lead_seconds * FRAME_RATE)

    def lead_for(stem: str) -> Lead | None:
        path = args.lead_from / f"{stem}.pt" if args.lead_from else None
        return Lead(torch.load(path)[:, -lead_frames:], args.lead_weight) if path and path.exists() else None

    name = f"{args.name or uuid.uuid7().hex}.wav"
    rendered = {}
    generated: dict[str, torch.Tensor] = {}  # loudness curves of the stems made so far in this sample
    order = random.Random(f"{args.seed}/{i}").sample(list(kept), len(kept)) if args.cross else kept  # no stem type always goes first
    for stem in order:
        k = kept.index(stem)
        print(f"  sample {i + 1}/{args.n}  {stem.name}", flush=True)
        basis, coefficients, pool, curve = sources[stem.name]
        partners = tuple(Partner(made, _cross_correlation(curve, sources[other][3])) for other, made in generated.items()) if args.cross else ()
        new = synthesise(coefficients, plan.frames, args.iters, args.seed + 100 * i + k, LAG_SETS[args.lags], args.manifold, args.init, args.modulation, args.window, args.tolerance, args.activity, args.joint, args.ica, args.dynamics, args.floor, args.long_window, args.long_tolerance, args.short_share, pool, args.rhythm, tuple(args.take_seconds) if args.take_seconds else None, args.search_dims, basis, lead_for(stem.name), args.loudness, args.cross, partners)
        generated[stem.name] = loudness_curve(basis.decode(new))
        audio = codec.decode(snap(basis.decode(new)))
        if args.latents_dir:
            args.latents_dir.mkdir(parents=True, exist_ok=True)
            torch.save(basis.decode(new), args.latents_dir / f"{stem.name}.pt")
        rendered[stem.name] = audio * 10 ** ((plan.loud[stem.name] + plan.trim.get(stem.name, 0.0) - loudness(audio)) / 20)
        coefficient_path = step_dir(CODEC_TEXTURE, f"01_generate_{plan.tag}", args.source, stem.name) / name
        coefficient_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new, coefficient_path.with_suffix(".pt"))
    mix = torch.stack(list(rendered.values())).sum(0)
    gain = OUTPUT_PEAK / float(mix.abs().max())
    for stem_name, audio in rendered.items():
        save(step_dir(CODEC_TEXTURE, f"01_generate_{plan.tag}", args.source, stem_name) / name, audio, gain)
    save(step_dir(CODEC_TEXTURE, f"02_mix_{plan.tag}", args.source) / name, mix, gain)
    print(f"  {name}", flush=True)


if __name__ == "__main__":
    main()
