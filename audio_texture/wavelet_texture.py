"""Texture generation in wavelet sparse-code space.

    venv/bin/python wavelet_texture.py walk_1 --n 2 --stems piano

For each audible stem: learn (or load) its complex wavelet dictionary
(wavelet_code.py), encode the stem, measure statistics of the code, then
optimise a fresh complex code until its statistics match and render it
(decode -> wavelet synthesis). Writes out/wavelet/02_generate_<variant>/<input>/<stem>/<uuid>.wav
and recomposed sums in out/wavelet/03_recompose_<variant>/<input>/<uuid>.wav.

Statistics of a code c[atom, band, frame] (spikes on a strided frame grid):
  magnitudes        per-atom distribution of |c| (includes sparsity)
  band profile      mean |c| per atom and band (register / pitch placement)
  activity          per-atom + total activity over frames, distribution at 50/200/800 ms
  coactivity        which atoms are active together, at 50/200 ms
  map autocorr      2-D autocorrelation of the band x frame activity map
                    (harmonic spacing along bands, note durations along frames)
  phase             complex correlations: frame lag per band and per atom,
                    neighbouring bands, and between atoms at the same position
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from paths import WAVELET, Stem, new_sample_path, step_dir, stems_of
from synthesize import (
    ROUND_ITERS, SAMPLE_RATE, SOURCE_RMS, Target, decode_mono, draw_round, load_audio, losses, measure_target, write_wav,
)
from synthesize_stems import recompose, stem_levels
from histogram_critic import Critic, FeatureConfig, Reference, window_features
from wavelet import N_BANDS, Wavelet
from wavelet_code import Operator, encode, learn_atoms, trim

CRITIC_ROUND_ITERS = 20
ACTIVITY_WINDOWS = (3, 12, 48)
COACTIVITY_WINDOWS = (3, 12)
AC_BANDS = 36
AC_FRAMES = 120
ENCODE_ITERS = 120
EPS = 1e-6


def activity(code: torch.Tensor) -> torch.Tensor:
    """(K + 1, frames): per-atom |c| summed over bands, plus a total row."""
    per_atom = code.abs().sum(1)
    return torch.cat([per_atom, per_atom.sum(0, keepdim=True)])


def pooled(x: torch.Tensor, window: int) -> torch.Tensor:
    return F.avg_pool1d(x[None], window, stride=max(1, window // 2))[0]


def _standardise(x: torch.Tensor) -> torch.Tensor:
    centred = x - x.mean(-1, keepdim=True)
    return centred / (centred.pow(2).mean(-1, keepdim=True) + EPS).sqrt()


def _map_autocorrelation(code: torch.Tensor) -> torch.Tensor:
    activity_map = code.abs().sum(0)
    z = (activity_map - activity_map.mean()) / (activity_map.std() + EPS)
    bands, frames = z.shape
    spectrum = torch.fft.rfft2(z, s=(2 * bands, 2 * frames))
    acf = torch.fft.irfft2(spectrum.real**2 + spectrum.imag**2, s=(2 * bands, 2 * frames))
    rows = torch.arange(AC_BANDS + 1)
    cols = torch.arange(-AC_FRAMES, AC_FRAMES + 1) % (2 * frames)
    return acf[rows][:, cols] / (acf[0, 0] + EPS)


def _phase_statistics(code: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Complex correlations, each normalised so its magnitude is at most 1."""
    power = code.abs().pow(2)
    lagged = code[..., :-1] * code[..., 1:].conj()
    per_band = lagged.sum((0, 2)) / (power.sum((0, 2)) + EPS)
    per_atom = lagged.sum((1, 2)) / (power.sum((1, 2)) + EPS)
    across_bands = (code[:, :-1] * code[:, 1:].conj()).sum((0, 2)) / (
        (power[:, :-1].sum((0, 2)) * power[:, 1:].sum((0, 2))).sqrt() + EPS
    )
    flat = code.flatten(1)
    norms = flat.abs().pow(2).sum(-1).sqrt()
    across_atoms = (flat @ flat.conj().T) / (norms[:, None] * norms[None, :] + EPS)
    return per_band, per_atom, across_bands, across_atoms


def _resample_rows(sorted_rows: torch.Tensor, n: int) -> torch.Tensor:
    if sorted_rows.shape[-1] == n:
        return sorted_rows
    return F.interpolate(sorted_rows[None], size=n, mode="linear", align_corners=True)[0]


def _distribution_error(sorted_a: torch.Tensor, sorted_b: torch.Tensor) -> torch.Tensor:
    target = _resample_rows(sorted_b, sorted_a.shape[-1])
    return ((sorted_a - target).pow(2).mean(-1) / (target.pow(2).mean(-1) + EPS)).mean()


def _matrix_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = (torch.view_as_real(t) if t.is_complex() else t for t in (a, b))
    return (a - b).pow(2).mean() / (b.pow(2).mean() + 1e-2)


@dataclass(frozen=True)
class CodeStats:
    magnitudes: torch.Tensor  # (K, B * M) sorted
    band_profile: torch.Tensor  # (K, B)
    activity: tuple[torch.Tensor, ...]  # per ACTIVITY_WINDOWS: (K + 1, frames), sorted per row
    coactivity: tuple[torch.Tensor, ...]  # per COACTIVITY_WINDOWS: (K + 1, K + 1)
    map_autocorrelation: torch.Tensor
    phase: tuple[torch.Tensor, ...]  # complex

    @classmethod
    def from_code(cls, code: torch.Tensor) -> CodeStats:
        def coactivity(window: int) -> torch.Tensor:
            z = _standardise(pooled(activity(code), window))
            return z @ z.T / z.shape[-1]

        return cls(
            magnitudes=code.abs().flatten(1).sort(-1).values,
            band_profile=code.abs().mean(-1),
            activity=tuple(pooled(activity(code), w).sort(-1).values for w in ACTIVITY_WINDOWS),
            coactivity=tuple(coactivity(w) for w in COACTIVITY_WINDOWS),
            map_autocorrelation=_map_autocorrelation(code),
            phase=_phase_statistics(code),
        )

    def terms(self, target: CodeStats) -> dict[str, torch.Tensor]:
        return {
            "magnitudes": _distribution_error(self.magnitudes, target.magnitudes),
            "band_profile": _matrix_error(self.band_profile, target.band_profile),
            "activity": sum(_distribution_error(a, b) for a, b in zip(self.activity, target.activity)),
            "coactivity": sum(_matrix_error(a, b) for a, b in zip(self.coactivity, target.coactivity)),
            "map_autocorr": _matrix_error(self.map_autocorrelation, target.map_autocorrelation),
            "phase": sum(_matrix_error(a, b) for a, b in zip(self.phase, target.phase)),
        }

    def distance(self, target: CodeStats) -> torch.Tensor:
        return sum(self.terms(target).values())


def generate_code(target: CodeStats, shape: tuple[int, int, int], iters: int, seed: int) -> torch.Tensor:
    """Optimise a dense complex Gaussian start toward `target`; the magnitude term shrinks it to sparsity."""
    generator = torch.Generator().manual_seed(seed)
    rms = target.magnitudes.pow(2).mean(-1).sqrt()[:, None, None, None]
    start = torch.randn(*shape, 2, generator=generator) * rms / 2**0.5
    real = start.requires_grad_()
    optimiser = torch.optim.LBFGS(
        [real], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = CodeStats.from_code(torch.view_as_complex(real)).distance(target)
        loss.backward()
        return loss

    optimiser.step(closure)
    return torch.view_as_complex(real.detach())


def refine_with_audio(
    code: torch.Tensor, code_target: CodeStats, audio_target: Target, render, iters: int, seed: int
) -> torch.Tensor:
    """Polish `code` so its decoded audio also matches the source's spectral statistics and patches.

    `render(complex code) -> waveform at SOURCE_RMS`. The audio terms are the ones used by synthesize.py;
    the code terms stay in so the result keeps its bursts and phase coherence.
    """
    generator = torch.Generator().manual_seed(seed)
    real = torch.view_as_real(code).clone().requires_grad_()
    for _ in range(max(1, iters // ROUND_ITERS)):
        patch_round = draw_round(render(torch.view_as_complex(real)).detach(), audio_target, generator)
        optimiser = torch.optim.LBFGS(
            [real], lr=1.0, max_iter=ROUND_ITERS, max_eval=ROUND_ITERS * 5 // 4, history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            current = torch.view_as_complex(real)
            stats, patches = losses(render(current), audio_target, patch_round)
            loss = CodeStats.from_code(current).distance(code_target) + stats + patches
            loss.backward()
            return loss

        optimiser.step(closure)
    return torch.view_as_complex(real.detach())


def refine_with_critic(
    code: torch.Tensor, code_target: CodeStats, real_features: torch.Tensor, features_of,
    iters: int, weight: float, ridge_rel: float,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Alternate (re)fitting the linear histogram critic with L-BFGS steps that fool it.

    `features_of(complex code)` returns the histogram window features of the rendered audio.
    Returns the code and the critic loss (fresh fit) before and after.
    """
    real = torch.view_as_real(code).clone().requires_grad_()

    def critic_loss() -> float:
        with torch.no_grad():
            generated = features_of(torch.view_as_complex(real))
            return float(Critic.fit(real_features, generated, ridge_rel).loss(generated))

    before = critic_loss()
    for _ in range(max(1, iters // CRITIC_ROUND_ITERS)):
        with torch.no_grad():
            generated = features_of(torch.view_as_complex(real))
            critic = Critic.fit(real_features, generated, ridge_rel)
        optimiser = torch.optim.LBFGS(
            [real], lr=1.0, max_iter=CRITIC_ROUND_ITERS, max_eval=CRITIC_ROUND_ITERS * 5 // 4, history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            current = torch.view_as_complex(real)
            loss = CodeStats.from_code(current).distance(code_target) + weight * critic.loss(
                features_of(current)
            )
            loss.backward()
            return loss

        optimiser.step(closure)
    return torch.view_as_complex(real.detach()), (before, critic_loss())


def load_or_learn(stem: Stem, x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    path = step_dir(WAVELET, "01_dictionary", stem.source, stem.name) / "dictionary.pt"
    if not path.exists():
        lam_rel, stride = 0.05, 24
        train = trim(x[: 8 * SAMPLE_RATE])
        atoms = learn_atoms(
            Wavelet.for_length(train.numel()).analyse(train), 32, (16, 200), lam_rel, 12, 15, stride, seed=0
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"atoms": atoms, "lam_rel": lam_rel, "stride": stride}, path)
    saved = torch.load(path)
    return saved["atoms"], saved["lam_rel"], saved.get("stride", 1)


def source_code(stem: Stem, x: torch.Tensor, atoms: torch.Tensor, lam_rel: float, stride: int) -> torch.Tensor:
    cache = step_dir(WAVELET, "01_dictionary", stem.source, stem.name) / "codes.pt"
    if cache.exists():
        return torch.load(cache)
    coefficients = Wavelet.for_length(x.numel()).analyse(x)
    operator = Operator.build(atoms, tuple(coefficients.shape), stride)
    lam = lam_rel * float(operator.analyse(coefficients).abs().max())
    codes = encode(coefficients, operator, lam, ENCODE_ITERS)
    torch.save(codes, cache)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--stems", nargs="+", help="only these stems (default: all audible)")
    parser.add_argument("--min-db", type=float, default=-25.0)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refine-iters", type=int, default=0, help="extra iterations that also match decoded audio")
    parser.add_argument("--hist-iters", type=int, default=0, help="iterations against the histogram critic")
    parser.add_argument("--hist-weight", type=float, default=1.0, help="critic loss weight vs the code statistics")
    parser.add_argument("--hist-ridge", type=float, default=0.3, help="critic ridge, relative to feature variance")
    parser.add_argument("--hist-group", type=int, default=3, help="bands per histogram group (12 per octave)")
    parser.add_argument("--hist-bins", type=int, default=24, help="bins of 1-D histograms")
    parser.add_argument("--hist-joint-bins", type=int, default=12, help="bins per axis of 2-D histograms")
    parser.add_argument("--variant", help="step suffix (default: hist, refined or code_only by options)")
    args = parser.parse_args()

    stems = stems_of(args.source)
    report = stem_levels(stems)
    kept = [s for s in stems if report[s.name][1] >= args.min_db and (args.stems is None or s.name in args.stems)]
    variant = args.variant or ("hist" if args.hist_iters else "refined" if args.refine_iters else "code_only")
    sample_dirs = {s.name: step_dir(WAVELET, f"02_generate_{variant}", args.source, s.name) for s in kept}
    n_samples = trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel()
    synth_wavelet = Wavelet.for_length(n_samples)

    for s, stem in enumerate(kept):
        x = decode_mono(stem.path)
        x = trim(x / x.pow(2).mean().sqrt())
        atoms, lam_rel, stride = load_or_learn(stem, x)
        target = CodeStats.from_code(source_code(stem, x, atoms, lam_rel, stride))
        operator = Operator.build(atoms, (N_BANDS, synth_wavelet.frames), stride)
        audio_target = (
            measure_target(load_audio(stem.path), n_samples, args.seed, ("envelope", "spectrogram"))
            if args.refine_iters
            else None
        )

        def render(code: torch.Tensor) -> torch.Tensor:
            return SOURCE_RMS * synth_wavelet.synthesize(operator.decode(code))

        def coefficients_of(code: torch.Tensor) -> torch.Tensor:
            return synth_wavelet.analyse(synth_wavelet.synthesize(operator.decode(code)))

        if args.hist_iters:
            with torch.no_grad():
                source_coefficients = Wavelet.for_length(x.numel()).analyse(x)
                reference = Reference.from_coefficients(source_coefficients)
                config = FeatureConfig(args.hist_group, args.hist_bins, args.hist_joint_bins)
                real_features = window_features(source_coefficients, reference, config)
            print(f"{stem.name}: {real_features.shape[1]} histogram features, {real_features.shape[0]} real windows")

            def features_of(code: torch.Tensor) -> torch.Tensor:
                return window_features(coefficients_of(code), reference, config)

        for i in range(args.n):
            seed = args.seed + 1000 * s + i
            code = generate_code(target, operator.code_shape, args.iters, seed)
            if args.refine_iters:
                code = refine_with_audio(code, target, audio_target, render, args.refine_iters, seed)
            critic_report = ""
            if args.hist_iters:
                code, (before, after) = refine_with_critic(
                    code, target, real_features, features_of, args.hist_iters, args.hist_weight, args.hist_ridge
                )
                critic_report = f"  critic {before:.2f} -> {after:.2f}"
            out = new_sample_path(sample_dirs[stem.name])
            write_wav(out, synth_wavelet.synthesize(operator.decode(code)))
            with torch.no_grad():
                terms = {k: round(float(v), 3) for k, v in CodeStats.from_code(code).terms(target).items()}
            print(f"{out}  {terms}{critic_report}", flush=True)

    levels = {s.name: report[s.name][0] for s in kept}
    recompose_dir = step_dir(WAVELET, f"03_recompose_{variant}", args.source)
    for path in recompose(args.source, levels, sample_dirs, recompose_dir, args.n, args.seed):
        print(path)


if __name__ == "__main__":
    main()
