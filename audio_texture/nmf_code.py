"""Shift-invariant non-negative sparse coding of perceptually weighted amplitude maps.

    venv/bin/python nmf_code.py walk_1 piano

Why non-negative: signed codes let atoms cancel one another, which produces a fuzz of
small simultaneous spikes and lets a generator build implausible cancellations. Here
every part is added, never subtracted.

Maps (2, band, frame):
  channel 0  P = |c| * g(band): amplitude with a perceptual (half-strength A-weighting) gain
  channel 1  P * relative frequency offset (ampfreq.relative_offset): amplitude-weighted frequency
Atoms are 2-channel patches; channel 0 is non-negative, channel 1 signed; codes are
non-negative. Both channels are linear in the code, so the decoded frequency offset is
channel1 / channel0: the amplitude-weighted average of the active atoms' frequencies.
Shifting an atom along bands is a transposition. The fit minimises a *masking-weighted*
error: errors count relative to the local masker level (louder neighbouring bands and frames
hide them), clipped so quiet bands do not dominate. L1 penalty, FISTA with a non-negative prox.
Writes out/nmf/01_dictionary/<input>/<stem>/{dictionary.pt,phase_policy.pt,recon.wav,recon_integrated.wav,
recon_features_only.wav}; recon.wav uses the per-band phase policy (see phase_policy), recon_integrated.wav
integrates the frequency channel in every band.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ampfreq import FREQ_CLIP, FREQ_SCALE, _radians_per_frame, relative_offset, snr_db
from ampfreq_code import Operator
from paths import NMF, Stem, step_dir
from phase_structure import z_scores
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import DECIMATION, Wavelet, centre_frequencies
from wavelet_code import trim, upsample

EPS = 1e-8
WEIGHT_RANGE = (0.2, 5.0)
FREQUENCY_WEIGHT = 0.5  # channel-1 error weight relative to channel 0


def perceptual_gain() -> torch.Tensor:
    """(B, 1) half-strength A-weighting amplitude gain, peak 1 over the covered bands."""
    f = centre_frequencies()
    ra = 12194**2 * f**4 / ((f**2 + 20.6**2) * ((f**2 + 107.7**2) * (f**2 + 737.9**2)).sqrt() * (f**2 + 12194**2))
    gain = ra.sqrt()
    return (gain / gain.max())[:, None]


def amplitude_maps(coefficients: torch.Tensor) -> torch.Tensor:
    """(B, W) complex -> (2, B, W)."""
    perceptual = coefficients.abs() * perceptual_gain()
    return torch.stack([perceptual, perceptual * relative_offset(coefficients)])


def coefficients_from(maps: torch.Tensor) -> torch.Tensor:
    """(2, B, W) -> (B, W) complex: undo the gain and integrate the frequency into a phase."""
    amplitude = maps[0].clamp(min=0)
    offset = (maps[1] / (amplitude + 1e-3)).clamp(-FREQ_CLIP, FREQ_CLIP) * (amplitude > 0.02 * amplitude.mean())
    advance = offset * FREQ_SCALE * _radians_per_frame()
    return torch.polar(amplitude / perceptual_gain(), torch.cumsum(advance, dim=-1) - advance)


PHASE_Z_THRESHOLD = 3.0


def phase_policy(stem: Stem, directory: Path) -> torch.Tensor:
    """(B,) bool: bands whose phase is measurably structured (neighbour or harmonic locking z above
    PHASE_Z_THRESHOLD against noise-phase surrogates) keep the integrated phase; the rest get noise phase."""
    cache = directory / "phase_policy.pt"
    if cache.exists():
        return torch.load(cache)
    z = z_scores(stem)
    keep = z["structure"] > PHASE_Z_THRESHOLD
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(keep, cache)
    return keep


def hybrid_coefficients(maps: torch.Tensor, keep: torch.Tensor, wavelet: Wavelet, seed: int = 0) -> torch.Tensor:
    """Complex coefficients from decoded maps: integrated phase where `keep`, else the phase of white noise
    seen through the filterbank (slow within each band's bandwidth, so no spectral tilt)."""
    integrated = coefficients_from(maps)
    noise = torch.randn(wavelet.n_samples, generator=torch.Generator().manual_seed(seed))
    return torch.where(keep[:, None], integrated, torch.polar(integrated.abs(), wavelet.analyse(noise).angle()))


def fit_weights(amplitude: torch.Tensor, weighting: str) -> torch.Tensor:
    """Per-cell error weights: 'masking' (relative to the local masker, 0.2-5x) or 'uniform' (plain least squares,
    where a cell counts in proportion to its own amplitude, i.e. covariance rather than relative error)."""
    return error_weights(amplitude) if weighting == "masking" else torch.ones_like(amplitude)


def error_weights(amplitude: torch.Tensor) -> torch.Tensor:
    """(B, W) weights ~ 1 / local masker level: a neighbourhood max over bands (spread) and frames."""
    kernel = torch.tensor([0.2, 0.45, 0.8, 1.0, 0.8, 0.45, 0.2])
    padded = F.pad(amplitude[None, None], (0, 0, 3, 3))[0, 0]
    spread = torch.stack([padded[i : i + amplitude.shape[0]] * kernel[i] for i in range(7)]).amax(0)
    masker = F.max_pool1d(spread[None], 75, stride=1, padding=37)[0]
    weights = 1 / (masker / masker.mean() + 0.1)
    weights = weights.clamp(*WEIGHT_RANGE)
    return weights / weights.mean()


@torch.no_grad()
def weighted_lipschitz(operator: Operator, weights: torch.Tensor, power_iters: int = 8) -> float:
    """Largest eigenvalue of D^T W D by power iteration (x1.1 margin); the bound max(w) * ||D||^2 is far too loose."""
    v = torch.rand(operator.code_shape)
    value = 1.0
    for _ in range(power_iters):
        u = operator.analyse(weights * operator.decode(v))
        value = float(u.norm() / (v.norm() + EPS))
        v = u / (u.norm() + EPS)
    return value * 1.1


@torch.no_grad()
def encode(
    x: torch.Tensor, weights: torch.Tensor, operator: Operator, lam: float, iters: int, init: torch.Tensor | None = None
) -> torch.Tensor:
    """FISTA for 0.5 * sum w (x - D a)^2 + lam * sum a, a >= 0."""
    step = 1 / weighted_lipschitz(operator, weights)
    a = torch.zeros(operator.code_shape) if init is None else init
    y, t = a, 1.0
    for _ in range(iters):
        a_next = (y + step * operator.analyse(weights * (x - operator.decode(y))) - step * lam).clamp(min=0)
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = a_next + (t - 1) / t_next * (a_next - a)
        a, t = a_next, t_next
    return a


def normalise(atoms: torch.Tensor) -> torch.Tensor:
    """Clamp channel 0 to >= 0 and scale each atom to unit norm on channel 0 (channel 1 follows)."""
    atoms = torch.cat([atoms[:, :1].clamp(min=0), atoms[:, 1:]], dim=1)
    return atoms / (atoms[:, :1].pow(2).sum((1, 2, 3), keepdim=True).sqrt() + 1e-8)


def update_atoms(
    x: torch.Tensor, weights: torch.Tensor, codes: torch.Tensor, atoms: torch.Tensor, steps: int, stride: int
) -> torch.Tensor:
    shape = (x.shape[1], x.shape[2])
    atom_shape = (atoms.shape[2], atoms.shape[3])
    from ampfreq_code import _padded

    code_spectra = torch.fft.rfft2(upsample(codes, stride, shape[1]), s=_padded(shape, atom_shape))
    phi = atoms.clone().requires_grad_()
    optimiser = torch.optim.Adam([phi], lr=0.07 / math.sqrt(atoms[0].numel()))
    channel_weight = torch.tensor([1.0] + [FREQUENCY_WEIGHT] * (atoms.shape[1] - 1))[:, None, None]
    for _ in range(steps):
        optimiser.zero_grad()
        residual = x - Operator.build(phi, shape, stride).decode_spectra(code_spectra)
        (channel_weight * weights * residual.pow(2)).sum().backward()
        optimiser.step()
    return normalise(phi.detach())


def revive_dead_atoms(x: torch.Tensor, codes: torch.Tensor, operator: Operator, atoms: torch.Tensor) -> torch.Tensor:
    dead = (codes.sum((-2, -1)) == 0).nonzero().flatten()
    if dead.numel() == 0:
        return atoms
    residual = x - operator.decode(codes)
    ab, at = atoms.shape[2], atoms.shape[3]
    stride = (ab // 2, at // 2)
    energy = F.avg_pool2d(residual[0].clamp(min=0).pow(2)[None, None], (ab, at), stride=stride)[0, 0]
    picks = energy.flatten().topk(min(dead.numel(), energy.numel())).indices.tolist()
    patches = torch.stack([
        residual[:, (i // energy.shape[1]) * stride[0] :][:, :ab, (i % energy.shape[1]) * stride[1] :][..., :at]
        for i in picks
    ])
    return atoms.index_copy(0, dead[: patches.shape[0]], normalise(patches))


def initial_atoms(n_atoms: int, atom_shape: tuple[int, int], seed: int, channels: int = 2) -> torch.Tensor:
    """Channel 0 is the non-negative amplitude template; the rest (frequency, delay, ...) are small and signed."""
    generator = torch.Generator().manual_seed(seed)
    window = torch.hann_window(atom_shape[0] + 2)[1:-1, None] * torch.hann_window(atom_shape[1] + 2)[1:-1][None, :]
    amplitude = torch.rand(n_atoms, 1, *atom_shape, generator=generator) * window
    signed = torch.randn(n_atoms, channels - 1, *atom_shape, generator=generator) * window * 0.1
    return normalise(torch.cat([amplitude, signed], dim=1))


def learn_atoms(
    x: torch.Tensor, weights: torch.Tensor, n_atoms: int, atom_shape: tuple[int, int], lam_rel: float, rounds: int,
    encode_iters: int, stride: int, seed: int, verbose: bool = True,
) -> torch.Tensor:
    shape = (x.shape[1], x.shape[2])
    atoms = initial_atoms(n_atoms, atom_shape, seed, channels=x.shape[0])
    codes = None
    for i in range(rounds):
        operator = Operator.build(atoms, shape, stride)
        lam = lam_rel * float(operator.analyse(weights * x).max())
        codes = encode(x, weights, operator, lam, encode_iters, init=codes)
        atoms = update_atoms(x, weights, codes, revive_dead_atoms(x, codes, operator, atoms), 10, stride)
        if verbose:
            print(f"round {i + 1}/{rounds}  {summarise(x, Operator.build(atoms, shape, stride), codes)}", flush=True)
    return atoms


def summarise(x: torch.Tensor, operator: Operator, codes: torch.Tensor) -> str:
    seconds = x.shape[2] * DECIMATION / SAMPLE_RATE
    estimate = operator.decode(codes)
    return (
        f"amplitude SNR {snr_db(x[0], estimate[0]):5.1f} dB  "
        + "  ".join(f"channel {c} SNR {snr_db(x[c], estimate[c]):5.1f} dB" for c in range(1, x.shape[0]))
        + "  "
        f"{int((codes > 0).sum()) / seconds:7.0f} spikes/s  {int((codes.sum((-2, -1)) > 0).sum())}/{codes.shape[0]} atoms used"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--atom-bands", type=int, default=16)
    parser.add_argument("--atom-frames", type=int, default=150)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--encode-iters", type=int, default=15)
    parser.add_argument("--train-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="step suffix, e.g. k4 -> 01_dictionary_k4")
    parser.add_argument("--weighting", choices=["masking", "uniform"], default="masking", help="fit error weights")
    args = parser.parse_args()

    audio = decode_mono(Stem(args.source, args.stem).path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    wavelet = Wavelet.for_length(audio.numel())
    maps = amplitude_maps(wavelet.analyse(audio))
    scales = maps.pow(2).mean((1, 2), keepdim=True).sqrt()
    x = maps / scales
    weights = fit_weights(x[0], args.weighting)

    out_dir = step_dir(NMF, "01_dictionary" + (f"_{args.tag}" if args.tag else ""), args.source, args.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "recon_features_only.wav", wavelet.synthesize(coefficients_from(maps)))

    frames = trim(audio[: round(args.train_seconds * SAMPLE_RATE)]).numel() // DECIMATION
    atoms = learn_atoms(
        x[:, :, :frames], weights[:, :frames], args.atoms, (args.atom_bands, args.atom_frames), args.lam, args.rounds,
        args.encode_iters, args.stride, args.seed,
    )
    operator = Operator.build(atoms, (x.shape[1], x.shape[2]), args.stride)
    lam = args.lam * float(operator.analyse(weights * x).max())
    codes = encode(x, weights, operator, lam, 6 * args.encode_iters)
    print(f"full stem  {summarise(x, operator, codes)}")

    torch.save(
        {"atoms": atoms, "scales": scales, "lam_rel": args.lam, "stride": args.stride, "weighting": args.weighting},
        out_dir / "dictionary.pt",
    )
    decoded = operator.decode(codes) * scales
    write_wav(out_dir / "recon_integrated.wav", wavelet.synthesize(coefficients_from(decoded)))
    keep = phase_policy(Stem(args.source, args.stem), out_dir)
    write_wav(out_dir / "recon.wav", wavelet.synthesize(hybrid_coefficients(decoded, keep, wavelet)))


if __name__ == "__main__":
    main()
