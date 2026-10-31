"""Time-frequency descriptors of audio that do not depend on where in the clip anything happens: a reduced interference
distribution (RID, a member of Cohen's class), cut into overlapping windows, each window summarised by numbers that do not change
if the sound moves within it. What is matched is then the distribution of those numbers over the windows (sorted values) and,
for rhythm, their autocovariance over longer lags. Shift the whole clip and nothing changes except at its edges.

The distribution: the instantaneous autocorrelation z(t+k) z*(t-k) of the analytic signal is smoothed along t with a window that
widens with the lag (the RID kernel, a product kernel H(theta tau): the time and frequency marginals stay exact and most of the
Wigner-Ville distribution's cross-terms go), then Fourier-transformed over the lag. Harmonics come out as sharp lines and attacks
as sharp spikes at the same time, which no spectrogram window can do.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

import codec

RATE = codec.CODEC_RATE // 2  # the analytic signal is low-passed at RATE / 2 (6 kHz) and kept at RATE
LAGS = 128  # lag window (samples at RATE), so the distribution has 2 * LAGS frequency bins, 23 Hz apart
HOP = 8  # samples between times of the distribution (0.67 ms)
BANDS = 48  # log-spaced bands of the window's spectrum, 80 Hz up
BAND_RANGE = (80.0, 5800.0)
CEPSTRUM = 20  # cepstral coefficients (the spectral envelope) kept per window
F0_RANGE = (80.0, 1000.0)  # Hz: where a harmonic spacing is looked for in the cepstrum
FLOOR = 1e-12
FRAME_LEVEL_WINDOW = 64  # samples (2.7 ms): the span of the RMS whose maximum over a frame is the frame's level
LEVEL_RANGE_DB = 60.0  # a frame this far below the loudest counts as being that far below
LEVEL_LAGS = (1, 3, 9)  # frames: the spans over which `LevelStats` takes changes of level
LEVEL_SMOOTHING = 19  # frames (0.25 s) averaged in a level curve, so it is phrases and pauses, not frame jitter
CROSS_LAGS = tuple(range(-150, 151, 5))  # frames (+-2 s): how far apart two stems' loud moments may be and still count as related
HIGH_BANDS = 12  # bands between 6 and 12 kHz, above where the RID looks, whose energy is matched
HIGH_RANGE = (6000.0, 12000.0)


def frame_level(audio: torch.Tensor) -> torch.Tensor:
    """(frames,) dB: how loud the audio is in each codec frame, as the maximum over the frame's samples of its RMS in a
    FRAME_LEVEL_WINDOW-sample window. A frame holding an attack should score as loud as the attack, not as its average (the
    codec's latents do not say: their norm moves about 0.1 dB for every dB of audio level)."""
    power = F.avg_pool1d(audio.pow(2)[None, None], FRAME_LEVEL_WINDOW, stride=1, padding=FRAME_LEVEL_WINDOW // 2, count_include_pad=False)[0, 0]
    level = 10 * torch.log10(power[: audio.numel()].reshape(-1, codec.HOP).amax(1) + 1e-10)
    return level.clamp(min=level.max().detach() - LEVEL_RANGE_DB)  # digital silence would otherwise sit at -100 dB and dominate every statistic


def level_curve(level: torch.Tensor) -> torch.Tensor:
    """Smoothed, standardised frame levels (dB, (T,)): the shape of when a stem is loud or quiet."""
    return F.avg_pool1d(((level - level.mean()) / (level.std() + 1e-6))[None, None], LEVEL_SMOOTHING, stride=1)[0, 0]


def cross_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(len(CROSS_LAGS),): correlation of `a` at time t + lag with `b` at t, over the frames both cover. Positive lags: `a`
    follows `b`."""
    a, b = (s - s.mean() for s in (a, b))
    a, b = a / (a.std() + 1e-6), b / (b.std() + 1e-6)
    n = min(len(a), len(b))
    return torch.stack([(a[lag:n] * b[: n - lag]).mean() if lag >= 0 else (a[: n + lag] * b[-lag:n]).mean() for lag in CROSS_LAGS])


@dataclass(frozen=True)
class Partner:
    """A stem generated earlier in the same sample, and how the stem being generated should move with it (in loudness, on the waveform)."""

    curve: torch.Tensor  # the partner's generated level curve
    correlation: torch.Tensor  # (len(CROSS_LAGS),): the source's cross-correlation of the two stems' level curves

    def penalty(self, audio: torch.Tensor) -> torch.Tensor:
        return (cross_correlation(level_curve(frame_level(audio)), self.curve) - self.correlation).pow(2).mean()


@dataclass(frozen=True)
class LevelStats:
    """The envelope of the audio level (`frame_level`), as sorted values: the histogram of the level around its mean (how peaky: many
    quiet frames and a few loud ones, or even) and of its changes over LEVEL_LAGS frames (signed, so attacks and decays are told
    apart: sharp rises and slow falls)."""

    values: torch.Tensor  # (T,) sorted, minus the mean
    steps: tuple[torch.Tensor, ...]  # per LEVEL_LAGS: (T - lag,) sorted

    @classmethod
    def of(cls, level: torch.Tensor) -> LevelStats:
        """`level` (T,) dB."""
        return cls((level - level.mean()).sort().values, tuple((level[lag:] - level[:-lag]).sort().values for lag in LEVEL_LAGS))

    def distance(self, target: LevelStats) -> torch.Tensor:
        """Squared gap between the sorted values, each set relative to the target's energy."""
        pairs = [(self.values, target.values), *zip(self.steps, target.steps)]
        return sum(_gap(a[:, None], b[:, None]) for a, b in pairs)

    @classmethod
    def from_state(cls, s: dict) -> LevelStats:
        return cls(s["values"], tuple(s["steps"]))


def analytic(audio: torch.Tensor) -> torch.Tensor:
    """(n / 2,) complex: the analytic signal of audio (n,) at the codec rate, low-passed at RATE / 2 and decimated to RATE."""
    n = audio.numel() // 4 * 4
    spectrum = torch.fft.fft(audio[:n])
    return torch.fft.ifft(spectrum[: n // 4], n=n // 2)


def rid(audio: torch.Tensor) -> torch.Tensor:
    """(times, 2 * LAGS) real: the reduced interference distribution of audio, one row every HOP samples at RATE; column m is
    the frequency m * RATE / (4 * LAGS). May be slightly negative."""
    z = analytic(audio)
    padded = F.pad(z, (LAGS, LAGS))
    centres = LAGS + torch.arange(0, z.numel(), HOP)
    lags = torch.arange(LAGS)
    r = padded[centres[:, None] + lags] * padded[centres[:, None] - lags].conj()  # (times, LAGS)
    smoothed = torch.stack([_smooth_lag(r[:, k], k) for k in range(LAGS)], 1)
    full = torch.cat([smoothed, smoothed[:, 1:].flip(1).conj()], 1)  # lags 0..LAGS-1, then -(LAGS-1)..-1
    return torch.fft.fft(full, n=2 * LAGS, dim=1).real


def _smooth_lag(column: torch.Tensor, lag: int) -> torch.Tensor:
    """(times,) complex: R(., lag) smoothed in time with a Hann window of half-width `lag` samples (the RID kernel)."""
    half = lag // HOP
    if half == 0:
        return column
    taps = torch.hann_window(2 * half + 3, periodic=False)[1:-1]
    taps = taps / taps.sum()
    pair = torch.stack([column.real, column.imag])[:, None]
    return torch.complex(*F.conv1d(pair, taps[None, None], padding=half)[:, 0])


@dataclass(frozen=True)
class Descriptors:
    """Per window, numbers that do not move if the sound moves within the window. Each field is (windows, dims)."""

    bands: torch.Tensor  # log energy in BANDS log-spaced bands: where the harmonics and the noise are
    cepstrum: torch.Tensor  # CEPSTRUM coefficients of the log spectrum: its envelope
    harmonicity: torch.Tensor  # (windows, 1): the strongest cepstral peak where a harmonic spacing of F0_RANGE would be
    time: torch.Tensor  # (windows, 4): spread, skewness, kurtosis and crest of the energy over the window's time
    plane: torch.Tensor  # (windows, 2): Renyi entropy (order 3) and flatness of the window's time-frequency plane
    frequency: torch.Tensor  # (windows, 3): centroid and spread of the spectrum, and how far the instantaneous frequency moves

    @classmethod
    def of(cls, audio: torch.Tensor, window: float = 0.096, stride: float = 0.024) -> Descriptors:
        """`window`, `stride` in seconds."""
        return cls.from_plane(rid(audio), HOP / RATE, torch.arange(2 * LAGS) * RATE / (4 * LAGS), window, stride)

    @classmethod
    def from_plane(cls, w: torch.Tensor, hop: float, freqs: torch.Tensor, window: float, stride: float) -> Descriptors:
        """Descriptors of any time-frequency plane w (times, bins) with `hop` seconds between its rows and `freqs` (bins,) in Hz
        (a spectrogram's power works too, which is how the RID is compared with it)."""
        length, step = round(window / hop), round(stride / hop)
        runs = w.unfold(0, length, step).permute(0, 2, 1)  # (windows, length, bins)
        plane = runs.clamp(min=0)
        spectrum = plane.sum(1) + FLOOR  # (windows, bins)
        time = plane.sum(2) + FLOOR  # (windows, length)
        return cls(
            _bands(spectrum, freqs), *_cepstral(spectrum, freqs), _time(time), _plane(plane), _frequency(plane, spectrum, time, freqs),
        )

    def fields(self) -> dict[str, torch.Tensor]:
        return {"bands": self.bands, "cepstrum": self.cepstrum, "harmonicity": self.harmonicity, "time": self.time,
                "plane": self.plane, "frequency": self.frequency}


RHYTHM_LAGS = tuple(range(1, 81))  # windows (24 ms ... 1.9 s): the lags of the medium scale


def _rhythm(d: Descriptors) -> torch.Tensor:
    """(4, len(RHYTHM_LAGS)): autocorrelation of four descriptor trajectories over the windows: how the texture pulses."""
    trajectories = torch.stack([d.plane[:, 0], d.time[:, 0], d.frequency[:, 0], d.bands.mean(1)])
    z = (trajectories - trajectories.mean(1, keepdim=True)) / (trajectories.std(1, keepdim=True) + FLOOR)
    return torch.stack([(z[:, :-lag] * z[:, lag:]).mean(1) for lag in RHYTHM_LAGS], 1)


PATCH_BANDS = 20  # log-spaced bands of the surface the patches are cut from
PATCH_STEP = 15  # rows of the RID per column of the surface (10 ms)
PATCH_STEPS = 40  # columns in a patch (400 ms): long enough to hold an attack and the start of its decay
PATCH_STRIDE = 2  # columns between patches (20 ms)
PATCH_COMPRESSION = 0.3  # the RID's positive part is raised to this power, so a loud partial does not outweigh all the quiet ones
PATCH_DERIVATIVE = False  # patches of the log surface's change over time (signed: rises are attacks, falls are decays) instead of the surface
PATCH_METHOD = "pca"  # "pca" or "nmf": how the patches are decomposed (see `PatchBasis`)
NMF_FIT_ITERS = 300
NMF_INFER_ITERS = 20
NMF_EPS = 1e-9
PATCH_COMPONENTS = 32
PATCH_JOINT = 8  # leading components whose joint distribution (random projections) is matched
PATCH_PROJECTIONS = 64
PATCH_CO_ACTIVITY = 16  # leading components whose energy correlations are matched


@lru_cache(maxsize=None)
def _patch_directions() -> torch.Tensor:
    """(PATCH_PROJECTIONS, PATCH_JOINT): fixed random unit directions, the same for source and synthesis."""
    directions = torch.randn(PATCH_PROJECTIONS, PATCH_JOINT, generator=torch.Generator().manual_seed(2))
    return directions / directions.norm(dim=1, keepdim=True)


def patches_of(w: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """(patches, PATCH_BANDS * PATCH_STEPS): the RID plane w (times, bins) as a band-by-time image (positive part, compressed, pooled to
    PATCH_BANDS log bands and PATCH_STEP-row columns), cut into overlapping patches of PATCH_STEPS columns, each flattened. Unlike the
    window descriptors a patch keeps the order of what happens inside it: a pluck and its reverse are different patches."""
    edges = torch.logspace(math.log10(BAND_RANGE[0]), math.log10(BAND_RANGE[1]), PATCH_BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()  # (PATCH_BANDS, bins)
    surface = (w.clamp(min=0) + 1e-6).pow(PATCH_COMPRESSION) @ member.T  # (times, PATCH_BANDS)
    columns = F.avg_pool1d(surface.T[None], PATCH_STEP, stride=PATCH_STEP)[0]  # (PATCH_BANDS, columns)
    if PATCH_DERIVATIVE:
        columns = torch.log(columns + 1e-3 * columns.mean()).diff(dim=1)
    return columns.unfold(1, PATCH_STEPS, PATCH_STRIDE).permute(1, 0, 2).flatten(1)


@dataclass(frozen=True)
class PatchBasis:
    """The parts of the source's patches, which are a property of the source: a synthesis is described in the same ones. Either the
    principal components ("pca": orthogonal, signed, a patch's score on each) or non-negative parts ("nmf": additive templates such as
    'a harmonic stack that rises fast and falls slowly', a patch's activation of each, sparse: mostly near zero, rarely large)."""

    kind: str
    mean: torch.Tensor  # (D,); zero for nmf
    components: torch.Tensor  # (D, PATCH_COMPONENTS), the nmf parts with unit length
    scale: torch.Tensor  # (PATCH_COMPONENTS,): the source's standard deviation of each score

    @classmethod
    def fit(cls, patches: torch.Tensor) -> PatchBasis:
        if PATCH_METHOD == "nmf":
            return cls._fit_nmf(patches)
        mean = patches.mean(0)
        _, singular, vt = torch.linalg.svd(patches - mean, full_matrices=False)
        k = min(PATCH_COMPONENTS, len(singular))
        return cls("pca", mean, vt[:k].T, singular[:k] / math.sqrt(len(patches)) + 1e-9)

    @classmethod
    def _fit_nmf(cls, patches: torch.Tensor) -> PatchBasis:
        """Lee-Seung multiplicative updates from a seeded random start; the parts are ordered by how much they are used."""
        generator = torch.Generator().manual_seed(3)
        k = min(PATCH_COMPONENTS, len(patches))
        level = float(patches.mean() / k) ** 0.5
        parts = torch.rand(k, patches.shape[1], generator=generator) * level + 1e-3
        activations = torch.rand(len(patches), k, generator=generator) * level + 1e-3
        for _ in range(NMF_FIT_ITERS):
            activations = activations * (patches @ parts.T) / (activations @ parts @ parts.T + NMF_EPS)
            parts = parts * (activations.T @ patches) / (activations.T @ activations @ parts + NMF_EPS)
        length = parts.norm(dim=1, keepdim=True) + NMF_EPS
        parts, activations = parts / length, activations * length.T
        order = activations.mean(0).argsort(descending=True)
        parts = parts[order]
        scores = _nmf_activations(patches, parts.T)
        spread = scores.std(0)
        return cls("nmf", torch.zeros(patches.shape[1]), parts.T, spread.clamp(min=1e-2 * float(spread.max())))

    def scores(self, patches: torch.Tensor) -> torch.Tensor:
        """(patches, components), each component in units of the source's spread."""
        if self.kind == "nmf":
            return _nmf_activations(patches, self.components) / self.scale
        return (patches - self.mean) @ self.components / self.scale


def _nmf_activations(patches: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
    """(patches, k) non-negative activations of the parts (D, k) for patches (N, D): a few multiplicative updates from the clipped
    least-squares solution. Always the same number of steps, for the source and for a synthesis, and differentiable in the patches."""
    gram = parts.T @ parts
    activations = (patches @ torch.linalg.pinv(parts).T).clamp(min=1e-3)
    for _ in range(NMF_INFER_ITERS):
        activations = activations * (patches @ parts) / (activations @ gram + NMF_EPS)
    return activations


@dataclass(frozen=True)
class PatchStatistics:
    """The distributions of the patch components' scores: each alone (marginals, which tell how rare a big attack is), together (sorted
    random projections of the leading ones: what appears with what), and their energies' correlations (which components switch on
    together). Not the source's patches, only these."""

    basis: PatchBasis
    marginals: torch.Tensor  # (patches, components) sorted
    joint: torch.Tensor  # (patches, PATCH_PROJECTIONS) sorted
    co_activity: torch.Tensor  # (c, c): correlation of the scores' magnitudes

    @classmethod
    def of(cls, w: torch.Tensor, freqs: torch.Tensor, basis: PatchBasis | None = None) -> PatchStatistics:
        """`basis`: the source's, when measuring a synthesis; fitted to these patches when None (measuring the source)."""
        patches = patches_of(w, freqs)
        basis = basis or PatchBasis.fit(patches.detach())
        scores = basis.scores(patches)
        k = min(PATCH_JOINT, scores.shape[1])
        c = min(PATCH_CO_ACTIVITY, scores.shape[1])
        return cls(
            basis, scores.sort(0).values, (scores[:, :k] @ _patch_directions()[:, :k].T).sort(0).values, torch.corrcoef(scores[:, :c].abs().T),
        )

    def distance(self, target: PatchStatistics) -> torch.Tensor:
        return (
            _gap(self.marginals, target.marginals) + _gap(self.joint, target.joint)
            + (self.co_activity - target.co_activity).pow(2).mean() / target.co_activity.pow(2).mean()
        )

    @classmethod
    def from_state(cls, s: dict) -> PatchStatistics:
        return cls(PatchBasis(**s["basis"]), s["marginals"], s["joint"], s["co_activity"])


_WEIGHTS_FILE = Path(__file__).with_name("term_weights.json")
TERM_WEIGHTS: dict[str, float] = json.loads(_WEIGHTS_FILE.read_text()) if _WEIGHTS_FILE.exists() else {}  # by term of `Statistics.terms`; see calibrate.py

DYN_BANDS = 12
DYN_LAGS = (1, 3, 9, 27)  # frames of 8 ms: 8, 24, 72 and 216 ms ahead
DYN_RANDOM = 32  # random directions over every band's value and changes together


@lru_cache(maxsize=None)
def _dyn_directions(dims: int) -> torch.Tensor:
    directions = torch.randn(DYN_RANDOM, dims, generator=torch.Generator().manual_seed(4))
    return directions / directions.norm(dim=1, keepdim=True)


def band_levels(audio: torch.Tensor) -> torch.Tensor:
    """(DYN_BANDS, frames): energy per log-spaced band in dB, one frame every 8 ms, floored 70 dB under the loudest cell."""
    power = torch.stft(audio, 1024, 192, window=torch.hann_window(1024), return_complex=True).abs().pow(2)
    freqs = torch.arange(513) * codec.CODEC_RATE / 1024
    edges = torch.logspace(math.log10(BAND_RANGE[0]), math.log10(BAND_RANGE[1]), DYN_BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()
    levels = 10 * torch.log10(member @ power + 1e-10)
    return levels.clamp(min=levels.max().detach() - 70.0)


@dataclass(frozen=True)
class BandDynamics:
    """How each band's loudness moves, as a joint of its value and where it goes next. Per band and per look-ahead (8 ms to 0.2 s): the
    band's level now, its change over the look-ahead, and those two combined at 45 degrees (value + change and value - change), each as
    a sorted histogram over frames; plus random directions over all bands' values and changes together. A decay is a high value with a
    falling change, an attack a low value with a rising one: the reversed signal swaps the signs of the changes, which moves the two
    combined histograms though the value's and the change's separate ones need not move much."""

    mean: torch.Tensor  # (DYN_BANDS, 1): the source's mean level per band
    spread: torch.Tensor  # (DYN_BANDS, 1)
    change: torch.Tensor  # (len(DYN_LAGS), DYN_BANDS, 1): the source's standard deviation of each look-ahead's change
    columns: torch.Tensor  # (frames, DYN_BANDS * (1 + 3 * len(DYN_LAGS)) + DYN_RANDOM) sorted

    @classmethod
    def of(cls, audio: torch.Tensor, like: BandDynamics | None = None) -> BandDynamics:
        levels = band_levels(audio)
        mean = like.mean if like else levels.mean(1, keepdim=True)
        spread = like.spread if like else levels.std(1, keepdim=True) + 1e-6
        reach = max(DYN_LAGS)
        value = (levels - mean) / spread
        changes = [value[:, lag : value.shape[1] - reach + lag] - value[:, : value.shape[1] - reach] for lag in DYN_LAGS]
        change = like.change if like else torch.stack([c.std(1, keepdim=True) + 1e-6 for c in changes])
        now = value[:, : value.shape[1] - reach]
        scaled = [c / s for c, s in zip(changes, change)]
        columns = torch.cat([now, *scaled, *[(now + c) / math.sqrt(2) for c in scaled], *[(now - c) / math.sqrt(2) for c in scaled]])
        everything = torch.cat([now, *scaled])
        columns = torch.cat([columns, _dyn_directions(everything.shape[0]) @ everything])
        return cls(mean, spread, change, columns.T.sort(0).values)

    def groups(self) -> dict[str, torch.Tensor]:
        n, k = DYN_BANDS, len(DYN_LAGS)
        edges = [0, n, n + n * k, n + 2 * n * k, n + 3 * n * k]
        names = ("value", "change", "value + change", "value - change")
        return {**{name: self.columns[:, a:b] for name, a, b in zip(names, edges[:-1], edges[1:])}, "random": self.columns[:, edges[-1] :]}

    def distance(self, target: BandDynamics) -> torch.Tensor:
        return _gap(self.columns, target.columns)

    @classmethod
    def from_state(cls, s: dict) -> BandDynamics:
        return cls(s["mean"], s["spread"], s["change"], s["columns"])


@dataclass(frozen=True)
class Statistics:
    """What is matched in the audio: the descriptors' sorted values over the windows, and the rhythm of the descriptor trajectories.
    None of it is in time order or a piece of the source."""

    descriptors: Descriptors  # each field sorted along the windows
    rhythm: torch.Tensor  # (4, len(RHYTHM_LAGS))
    coherence: torch.Tensor  # (frames, PHASE_BANDS) sorted: see `phase_coherence`
    pulse: torch.Tensor  # (windows, 3) sorted: see `pulse`
    highband: torch.Tensor  # (frames, HIGH_BANDS) sorted: see `high_band`
    level: LevelStats  # the loudness of the audio: its histogram and changes
    note: torch.Tensor  # (windows, NOTE_DIMS) sorted: see `note_descriptors`
    patches: PatchStatistics  # see `PatchStatistics`
    band_dynamics: BandDynamics  # see `BandDynamics`

    @classmethod
    def of(cls, audio: torch.Tensor, like: Statistics | None = None) -> Statistics:
        """`like`: the source's statistics, when measuring a synthesis of it (the patch components must be the source's)."""
        w, freqs = rid(audio), torch.arange(2 * LAGS) * RATE / (4 * LAGS)
        d = Descriptors.from_plane(w, HOP / RATE, freqs, 0.096, 0.024)
        return cls(
            Descriptors(*(f.sort(0).values for f in d.fields().values())), _rhythm(d),
            phase_coherence(audio).sort(0).values, pulse(audio).sort(0).values, high_band(audio).sort(0).values,
            LevelStats.of(frame_level(audio)), note_descriptors(w, HOP / RATE, freqs).sort(0).values,
            PatchStatistics.of(w, freqs, like.patches.basis if like else None),
            BandDynamics.of(audio, like.band_dynamics if like else None),
        )

    def terms(self, target: Statistics) -> dict[str, torch.Tensor]:
        """The gap to `target`, one number per named group of statistics: squared gap between sorted values, per dimension in units of the
        target's spread (the rhythm's is a relative squared gap). `self` may be measured on a synthesis of any length: sorted values
        are compared by quantile, not window for window."""
        out = {f"descriptors/{name}": _gap(x, target.descriptors.fields()[name]) for name, x in self.descriptors.fields().items()}
        out["rhythm"] = (self.rhythm - target.rhythm).pow(2).sum() / target.rhythm.pow(2).sum()
        out["highband"] = _gap(self.highband, target.highband)
        out["level/values"] = _gap(self.level.values[:, None], target.level.values[:, None])
        for lag, a, b in zip(LEVEL_LAGS, self.level.steps, target.level.steps):
            out[f"level/step{lag}"] = _gap(a[:, None], b[:, None])
        out["coherence"] = _gap(self.coherence, target.coherence)
        out["pulse"] = _gap(self.pulse, target.pulse)
        n = NOTE_BANDS
        for name, columns in (("bands", slice(0, n)), ("early/late", slice(n, 2 * n)), ("trends", slice(2 * n, 2 * n + 2)), ("time shape", slice(2 * n + 2, 2 * n + 6))):
            out[f"note/{name}"] = _gap(self.note[:, columns], target.note[:, columns])
        out["patches/marginals"] = _gap(self.patches.marginals, target.patches.marginals)
        out["patches/joint"] = _gap(self.patches.joint, target.patches.joint)
        out["patches/co-activity"] = (self.patches.co_activity - target.patches.co_activity).pow(2).mean() / target.patches.co_activity.pow(2).mean()
        for name, columns in self.band_dynamics.groups().items():
            out[f"dynamics/{name}"] = _gap(columns, target.band_dynamics.groups()[name])
        return out

    def distance(self, target: Statistics, weights: dict[str, float] | None = None) -> torch.Tensor:
        """The weighted sum of `terms`; `weights` (default `TERM_WEIGHTS`) by term name, 1 for a name it lacks."""
        weights = TERM_WEIGHTS if weights is None else weights
        return sum(weights.get(name, 1.0) * gap for name, gap in self.terms(target).items())

    def state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, s: dict) -> Statistics:
        return cls(
            Descriptors(**s["descriptors"]), s["rhythm"], s["coherence"], s["pulse"], s["highband"], LevelStats.from_state(s["level"]),
            s["note"], PatchStatistics.from_state(s["patches"]), BandDynamics.from_state(s["band_dynamics"]),
        )


def _gap(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mean squared gap of sorted values (n, dims), each dimension in units of the reference's spread; the reference is resampled to n."""
    y = reference if len(reference) == len(x) else F.interpolate(reference.T[None], size=len(x), mode="linear", align_corners=True)[0].T
    return ((x - y) / (reference.std(0) + 1e-6)).pow(2).mean()


NOTE_WINDOW = 0.4  # seconds: long enough to hold an attack and the start of its decay
NOTE_STRIDE = 0.1
NOTE_BANDS = 12
NOTE_SEGMENTS = 10  # parts of a window whose energy and centroid are followed to find the trend
NOTE_DIMS = 2 * NOTE_BANDS + 6


def note_descriptors(w: torch.Tensor, hop: float, freqs: torch.Tensor) -> torch.Tensor:
    """(windows, NOTE_DIMS): the plane in windows of NOTE_WINDOW, each described by numbers that depend on the ORDER of what happens inside it.

    The window slides, so every event lands at every position in it: the distribution of these numbers over the windows does not depend on
    where in the clip anything is, even though each number depends on where in its window the energy falls. That is what lets them tell an
    attack followed by a decay from the reverse, which the symmetric descriptors cannot:
      - the log energy of NOTE_BANDS bands over the whole window, and the log ratio of each band's energy in the later half to the earlier
        half (a plucked string is bright first and darkens: the ratio is lowest in the high bands)
      - the slope across the window of the log energy and of the spectral centroid
      - where the energy sits in time (offset of its centroid from the middle) and the spread, skewness and kurtosis of that distribution
    """
    length, step = round(NOTE_WINDOW / hop), round(NOTE_STRIDE / hop)
    length = length // NOTE_SEGMENTS * NOTE_SEGMENTS
    plane = w.unfold(0, length, step).permute(0, 2, 1).clamp(min=0)  # (windows, length, bins)
    edges = torch.logspace(math.log10(BAND_RANGE[0]), math.log10(BAND_RANGE[1]), NOTE_BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()  # (NOTE_BANDS, bins)
    half = length // 2
    early, late = plane[:, :half].sum(1) @ member.T, plane[:, half:].sum(1) @ member.T
    whole = early + late
    floor = 1e-3 * whole.sum(1, keepdim=True)
    bands = torch.log10(whole + FLOOR)
    asymmetry = torch.log10((late + floor) / (early + floor))
    segments = plane.reshape(plane.shape[0], NOTE_SEGMENTS, -1, plane.shape[2]).sum(2)  # (windows, segments, bins)
    energy = segments.sum(2) + FLOOR
    centroid = (segments * freqs).sum(2) / energy
    x = (torch.arange(NOTE_SEGMENTS, dtype=energy.dtype) - (NOTE_SEGMENTS - 1) / 2) / NOTE_SEGMENTS
    slope = lambda y: ((y - y.mean(1, keepdim=True)) * x).sum(1, keepdim=True) / x.pow(2).sum()
    time = plane.sum(2) + FLOOR
    q = time / time.sum(1, keepdim=True)
    t = torch.arange(length, dtype=q.dtype) / length
    mean = (q * t).sum(1, keepdim=True)
    spread = (q * (t - mean).pow(2)).sum(1, keepdim=True).sqrt() + FLOOR
    skew = (q * ((t - mean) / spread).pow(3)).sum(1, keepdim=True)
    kurtosis = (q * ((t - mean) / spread).pow(4)).sum(1, keepdim=True)
    return torch.cat([bands, asymmetry, slope(torch.log10(energy)), slope(centroid / 1000), mean - 0.5, spread, skew, kurtosis], 1)


def high_band(audio: torch.Tensor) -> torch.Tensor:
    """(frames, HIGH_BANDS): log energy per frame in HIGH_BANDS bands between 6 and 12 kHz, where the RID does not look (it is low-passed
    at 6 kHz). Without it a waveform free to move can pour noise into those bands and hear as a sizzle."""
    power = torch.stft(audio, 1024, 192, window=torch.hann_window(1024), return_complex=True).abs().pow(2)
    freqs = torch.arange(513) * codec.CODEC_RATE / 1024
    edges = torch.linspace(HIGH_RANGE[0], HIGH_RANGE[1], HIGH_BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()
    return torch.log10(member @ power + 1e-10).T


PHASE_BANDS = 8
PHASE_RANGE = (100.0, 6000.0)  # Hz
PULSE_WINDOW = 192  # samples at RATE (16 ms): a few pitch periods
PULSE_STRIDE = 96


def phase_coherence(audio: torch.Tensor) -> torch.Tensor:
    """(frames, PHASE_BANDS): how aligned the phase is between neighbouring frequency bins of an STFT, per band and frame: the length of
    the sum of X[f+1] X*[f] over the band divided by the sum of their lengths. A click or a pluck, whose partials start together, has
    one group delay across the band (coherence near 1); partials with unrelated phases, or noise, scatter it (near 0). The magnitude
    spectrum, and so every descriptor above, is the same either way."""
    stft = torch.stft(audio, 1024, 192, window=torch.hann_window(1024), return_complex=True)
    neighbours = stft[1:] * stft[:-1].conj()  # (bins - 1, frames)
    freqs = torch.arange(1, 513) * codec.CODEC_RATE / 1024
    edges = torch.logspace(math.log10(PHASE_RANGE[0]), math.log10(PHASE_RANGE[1]), PHASE_BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()  # (PHASE_BANDS, bins - 1)
    return ((member.to(neighbours.dtype) @ neighbours).abs() / (member @ neighbours.abs()).clamp(min=FLOOR)).T


def pulse(audio: torch.Tensor) -> torch.Tensor:
    """(windows, 3): crest, kurtosis and log spread of the instantaneous power |z(t)|^2 over PULSE_WINDOW samples (16 ms) of the analytic
    signal: whether the energy within a few pitch periods comes in pulses (aligned partials, a pluck) or evenly (noise)."""
    p = analytic(audio).abs().pow(2)
    q = p.unfold(0, PULSE_WINDOW, PULSE_STRIDE)
    q = q / (q.mean(1, keepdim=True) + FLOOR)
    kurtosis = (q - 1).pow(4).mean(1) / ((q - 1).pow(2).mean(1).pow(2) + FLOOR)
    return torch.stack([q.amax(1), kurtosis, torch.log(q + 1e-3).std(1)], 1)


def _bands(spectrum: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    edges = torch.logspace(math.log10(BAND_RANGE[0]), math.log10(BAND_RANGE[1]), BANDS + 1)
    member = ((freqs[None] >= edges[:-1, None]) & (freqs[None] < edges[1:, None])).float()  # (BANDS, bins)
    return torch.log10(spectrum @ member.T + FLOOR)


def _cepstral(spectrum: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cepstrum = torch.fft.irfft(torch.log(spectrum), dim=1)  # quefrency q: a spacing of q bins, i.e. q * 23 Hz, in the spectrum
    spacing = freqs[1] - freqs[0]
    lo, hi = math.ceil(F0_RANGE[0] / spacing), int(F0_RANGE[1] / spacing)
    return cepstrum[:, 1 : 1 + CEPSTRUM], cepstrum[:, lo : hi + 1].amax(1, keepdim=True)


def _time(time: torch.Tensor) -> torch.Tensor:
    """Moments of the energy as a distribution over the window's time (unchanged by moving it within the window)."""
    q = time / time.sum(1, keepdim=True)
    t = torch.arange(time.shape[1], dtype=q.dtype)
    mean = (q * t).sum(1, keepdim=True)
    spread = ((q * (t - mean).pow(2)).sum(1, keepdim=True)).sqrt() + FLOOR
    skew = (q * ((t - mean) / spread).pow(3)).sum(1, keepdim=True)
    kurtosis = (q * ((t - mean) / spread).pow(4)).sum(1, keepdim=True)
    crest = q.amax(1, keepdim=True) * time.shape[1]
    return torch.cat([spread / time.shape[1], skew, kurtosis, crest], 1)


def _plane(plane: torch.Tensor) -> torch.Tensor:
    p = plane / (plane.sum((1, 2), keepdim=True) + FLOOR)
    renyi = -0.5 * torch.log2(p.pow(3).sum((1, 2)) + FLOOR)
    flatness = torch.log((plane / (plane.mean((1, 2), keepdim=True) + FLOOR) + 1e-3)).mean((1, 2))
    return torch.stack([renyi, flatness], 1)


def _frequency(plane: torch.Tensor, spectrum: torch.Tensor, time: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    q = spectrum / spectrum.sum(1, keepdim=True)
    centroid = (q * freqs).sum(1, keepdim=True)
    spread = (q * (freqs - centroid).pow(2)).sum(1, keepdim=True).sqrt()
    instantaneous = (plane * freqs).sum(2) / time  # (windows, length)
    weights = time / time.sum(1, keepdim=True)
    mean = (weights * instantaneous).sum(1, keepdim=True)
    moving = (weights * (instantaneous - mean).pow(2)).sum(1, keepdim=True).sqrt()
    return torch.cat([centroid, spread, moving], 1)
