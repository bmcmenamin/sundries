"""The simplified, unweighted pipeline core: maps, dictionary fit, phase rebuild, rendering.

    separate (Demucs) -> wavelet analysis -> three maps -> shift-invariant non-negative code -> generate -> render

Everything is plain (covariance-style): no perceptual gain, no masking weights, no per-band
z-scoring. The only normalisation is one global scale per map channel.

  map 0  |c|                          amplitude of each band and frame
  map 1  |c| * relative frequency     phase gradient along time (offset from the band centre)
  map 2  |c| * group delay            phase gradient along frequency (samples, against the frame's own time)
  code   a >= 0 on (atom, band, frame / stride); maps ~ sum_k a_k (*) atom_k, a pure sum of parts
  phase  rebuilt from the two gradients by angular synchronisation (`synchronise`): neighbouring cells agree
         on their phase difference, weighted by amplitude. Cross-band agreement is what makes a click a click.
         `detect_phase_mode` picks the time links: 'integrated' keeps them (tonal stems), 'noise' drops them and
         anchors each frame to the phase of white noise (noisy stems: no ringing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ampfreq import FREQ_CLIP, FREQ_SCALE, _radians_per_frame, relative_offset
from ampfreq_code import Operator
from nmf_code import encode, learn_atoms
from paths import NMF_PHASE, Stem, step_dir
from phase_structure import z_scores
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import DECIMATION, Wavelet, centre_frequencies
from wavelet_code import trim

ATOM_BANDS = 16
ATOM_FRAMES = 150
STRIDE = 12
LAM_REL = 0.05
ROUNDS = 12
ENCODE_ITERS = 25
TRAIN_SECONDS = 8.0
STRUCTURE_Z = 3.0  # a band's phase counts as structured above this z against noise-phase surrogates
INTEGRATED_SHARE = 0.3  # a stem keeps its phase if more than this share of its bands are structured
DELAY_CLIP = 200.0  # samples
CHANNEL_GAIN = torch.tensor([1.0, 1.0, 0.4])[:, None, None]  # fit weight: the noisy delay map must not outvote amplitude
NOISE_TIME_LINK = 0.3  # time-link weight in 'noise' mode (1 = as strong as 'integrated'; 0 = frames independent)
SYNC_ITERS = 400
SYNC_ANCHOR = 1e-3  # pull toward the start phase, relative to the mean squared amplitude
TWO_PI = 2 * math.pi


def band_omega(n_samples: int) -> torch.Tensor:
    """(B,) float64 radians per sample of each band's centre bin, the demodulation frequency."""
    centre_bin = (centre_frequencies() / (SAMPLE_RATE / n_samples)).round().double()
    return TWO_PI * centre_bin / n_samples


def _band_steps(n_samples: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B-1, 1) omega steps between neighbouring bands, and (frames,) frame times in samples."""
    omega = band_omega(n_samples)
    return (omega[1:] - omega[:-1])[:, None], torch.arange(frames).double() * DECIMATION


def group_delay(coefficients: torch.Tensor) -> torch.Tensor:
    """(B, W) float32 samples: -d(phase)/d(omega) between neighbouring bands, against each frame's own time.

    An impulse at time t has coefficient phase -omega_b * t; adding back omega_b * (frame time) leaves the
    small offset t - frame time, so the difference across bands is a short delay without wrapping.
    """
    d_omega, frame_time = _band_steps(coefficients.shape[-1] * DECIMATION, coefficients.shape[-1])
    cross = (coefficients[1:] * coefficients[:-1].conj()).to(torch.complex128)
    delay = -(cross * torch.exp(1j * ((d_omega * frame_time) % TWO_PI))).angle() / d_omega
    return torch.cat([delay, delay[-1:]]).float()


def band_links(delay: torch.Tensor) -> torch.Tensor:
    """(B-1, W) complex64: the phasor z(b + 1) / z(b) that a group delay implies (the inverse of `group_delay`)."""
    d_omega, frame_time = _band_steps(delay.shape[-1] * DECIMATION, delay.shape[-1])
    return torch.exp(-1j * ((d_omega * delay.double() + d_omega * frame_time) % TWO_PI)).to(torch.complex64)


def plain_maps(coefficients: torch.Tensor) -> torch.Tensor:
    """(B, W) complex -> (3, B, W): amplitude, and amplitude times frequency offset, and times group delay."""
    amplitude = coefficients.abs()
    delay = group_delay(coefficients).clamp(-DELAY_CLIP, DELAY_CLIP)
    return torch.stack([amplitude, amplitude * relative_offset(coefficients), amplitude * delay])


def synchronise(
    amplitude: torch.Tensor, advance: torch.Tensor, links: torch.Tensor, start: torch.Tensor, time_weight: float
) -> torch.Tensor:
    """(B, W) unit phasors z with z(m + 1) = z(m) e^{j advance} and z(b + 1) = z(b) links, amplitude-weighted.

    Damped power iteration on the Hermitian link matrix (the diagonal shift keeps it positive), started from
    `start` and weakly anchored to it. `advance` is (B, W - 1), `links` is (B - 1, W).
    """
    step = torch.polar(torch.ones_like(advance), advance)
    time_link = time_weight * amplitude[:, 1:] * amplitude[:, :-1]
    band_link = amplitude[1:] * amplitude[:-1]
    degree = torch.zeros_like(amplitude)
    degree[:, :-1] += time_link
    degree[:, 1:] += time_link
    degree[:-1] += band_link
    degree[1:] += band_link
    anchor = SYNC_ANCHOR * amplitude.pow(2).mean() * start

    def update(z: torch.Tensor) -> torch.Tensor:
        pull = degree * z + anchor
        pull[:, :-1] += time_link * z[:, 1:] * step.conj()
        pull[:, 1:] += time_link * z[:, :-1] * step
        pull[:-1] += band_link * z[1:] * links.conj()
        pull[1:] += band_link * z[:-1] * links
        return pull / pull.abs().clamp(min=1e-9)

    z = start
    for _ in range(SYNC_ITERS):
        z = update(z)
    return z


def render_coefficients(maps: torch.Tensor, mode: str, wavelet: Wavelet, seed: int = 0) -> torch.Tensor:
    """(3, B, W) decoded maps -> complex coefficients; `mode` picks the time links (see the module docstring)."""
    amplitude = maps[0].clamp(min=0)
    live = amplitude > 0.02 * amplitude.mean()
    offset = (maps[1] / (amplitude + 1e-3)).clamp(-FREQ_CLIP, FREQ_CLIP) * live
    advance = (offset * FREQ_SCALE * _radians_per_frame())[:, :-1]
    delay = ((maps[2] / (amplitude + 1e-3)).clamp(-DELAY_CLIP, DELAY_CLIP) * live)[:-1]
    if mode == "noise":
        noise = torch.randn(wavelet.n_samples, generator=torch.Generator().manual_seed(seed))
        start, time_weight = torch.polar(torch.ones_like(amplitude), wavelet.analyse(noise).angle()), NOISE_TIME_LINK
    else:
        start = torch.polar(torch.ones_like(amplitude), torch.cat([torch.zeros(amplitude.shape[0], 1), advance.cumsum(-1)], -1))
        time_weight = 1.0
    return torch.polar(amplitude, synchronise(amplitude, advance, band_links(delay), start, time_weight).angle())


def detect_phase_mode(stem: Stem) -> tuple[str, float]:
    """('integrated' | 'noise', share of bands with structured phase).

    Counts bands rather than weighting by energy: a drum stem's loud kick is tonal and holds most of the
    energy, which would label the whole stem tonal and bring back the harmonic ringing on its noisy bands.
    """
    share = float((z_scores(stem)["structure"] > STRUCTURE_Z).float().mean())
    return ("integrated" if share > INTEGRATED_SHARE else "noise"), share


@dataclass(frozen=True)
class Fit:
    atoms: torch.Tensor  # (K, 3, bands, frames)
    scales: torch.Tensor  # (3, 1, 1) channel scales
    lam_rel: float
    stride: int
    phase: str
    share: float
    codes: torch.Tensor  # the stem's own code (K, B, M)


def load_stem(stem: Stem) -> tuple[torch.Tensor, Wavelet]:
    audio = decode_mono(stem.path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    return audio, Wavelet.for_length(audio.numel())


def config_tag(n_atoms: int, atom_frames: int, stride: int) -> str:
    """Folder suffix of one dictionary configuration: atoms, atom length in frames, stride in frames."""
    return f"k{n_atoms}_t{atom_frames}_s{stride}"


def fit_stem(stem: Stem, n_atoms: int, atom_frames: int = ATOM_FRAMES, stride: int = STRIDE, seed: int = 0) -> Fit:
    """Learn (or load) the stem's dictionary and code, and decide its phase mode."""
    directory = step_dir(NMF_PHASE, f"01_dictionary_{config_tag(n_atoms, atom_frames, stride)}", stem.source, stem.name)
    if (directory / "dictionary.pt").exists():
        return Fit(**torch.load(directory / "dictionary.pt"), codes=torch.load(directory / "codes.pt"))

    audio, wavelet = load_stem(stem)
    coefficients = wavelet.analyse(audio)
    maps = plain_maps(coefficients)
    scales = maps.pow(2).mean((1, 2), keepdim=True).sqrt() / CHANNEL_GAIN
    x = maps / scales
    train = trim(audio[: round(TRAIN_SECONDS * SAMPLE_RATE)]).numel() // DECIMATION
    atoms = learn_atoms(
        x[:, :, :train], torch.ones(x.shape[1], train), n_atoms, (ATOM_BANDS, atom_frames), LAM_REL, ROUNDS,
        ENCODE_ITERS, stride, seed, verbose=False,
    )
    operator = Operator.build(atoms, (x.shape[1], x.shape[2]), stride)
    weights = torch.ones(x.shape[1], x.shape[2])
    codes = encode(x, weights, operator, LAM_REL * float(operator.analyse(x).max()), 6 * ENCODE_ITERS)
    mode, share = detect_phase_mode(stem)

    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"atoms": atoms, "scales": scales, "lam_rel": LAM_REL, "stride": stride, "phase": mode, "share": share},
        directory / "dictionary.pt",
    )
    torch.save(codes, directory / "codes.pt")
    write_wav(directory / "recon.wav", wavelet.synthesize(render_coefficients(operator.decode(codes) * scales, mode, wavelet)))
    print(f"{stem.name}: {n_atoms} atoms, {int((codes > 0).sum()) / (x.shape[2] * DECIMATION / SAMPLE_RATE):.0f} spikes/s, "
          f"phase {mode} ({100 * share:.0f}% of bands structured)", flush=True)
    return Fit(atoms, scales, LAM_REL, stride, mode, share, codes)
