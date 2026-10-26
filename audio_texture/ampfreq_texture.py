"""Texture generation in the sparse code of the amplitude / frequency maps.

    venv/bin/python ampfreq_texture.py walk_1 piano --n 2

Needs out/ampfreq/01_dictionary/<input>/<stem>/dictionary.pt (ampfreq_code.py).
Stage 1 optimises a fresh code against the source code's statistics (wavelet_texture.CodeStats:
magnitude distributions, band profile, burstiness, co-activity, map autocorrelation, correlations).
Stage 2 adds a linear histogram critic (histogram_critic.py) on the *decoded* amplitude and
frequency maps, whose histograms are of level, rise, persistence, spectral slope, and, because the
frequency channel is relative to each band's centre, the frequency agreement between neighbouring
bands and between harmonically related bands (a real partial has the same relative offset everywhere).
Rendering is phase integration of the frequency channel (ampfreq.coefficients_from).
Writes out/ampfreq/02_generate/<input>/<stem>/<uuid>.wav.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from ampfreq import coefficients_from, features
from ampfreq_code import Operator, encode
from histogram_critic import MAX_LAG, STEP, Critic, FeatureConfig, joint, marginal
from paths import AMPFREQ, Stem, new_sample_path, step_dir
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import Wavelet
from wavelet_code import trim
from wavelet_texture import CodeStats, _distribution_error, _matrix_error

CRITIC_ROUND_ITERS = 20
ENCODE_ITERS = 90
LEVEL_RANGE = (-3.0, 3.0)
FREQ_RANGE = (-2.0, 2.0)
HARMONIC_OFFSETS = (12, 19, 24)


@dataclass(frozen=True)
class AmplitudeReference:
    """Per-band mean / std of the amplitude channel of the source, for z-scoring levels."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_maps(cls, maps: torch.Tensor) -> AmplitudeReference:
        return cls(maps[0].mean(-1, keepdim=True), maps[0].std(-1, keepdim=True) + 1e-3)


def map_features(maps: torch.Tensor, reference: AmplitudeReference, config: FeatureConfig) -> torch.Tensor:
    """(windows, p) histogram features of (amplitude, frequency) maps of shape (2, B, W)."""
    n_bands, frames = maps.shape[1], maps.shape[2]
    steps = frames - MAX_LAG
    level = (maps[0] - reference.mean) / reference.std
    frequency = maps[1]

    def at(x: torch.Tensor, lag: int = 0) -> torch.Tensor:
        return x[:, lag : lag + steps : STEP]

    scalars = {
        "level": (at(level), LEVEL_RANGE),
        "rise16": (at(level, 24) - at(level), LEVEL_RANGE),
        "rise100": (at(level, 150) - at(level), LEVEL_RANGE),
        "later50": (at(level, 75), LEVEL_RANGE),
        "later200": (at(level, MAX_LAG), LEVEL_RANGE),
        "slope": (at(level)[1:] - at(level)[:-1], LEVEL_RANGE),
        "freq": (at(frequency), FREQ_RANGE),
        "freq_later50": (at(frequency, 75), FREQ_RANGE),
        "freq_neighbour": (at(frequency)[1:] - at(frequency)[:-1], FREQ_RANGE),
    }
    for offset in HARMONIC_OFFSETS:
        scalars[f"ratio{offset}"] = (at(level)[offset:] - at(level)[:-offset], LEVEL_RANGE)
        scalars[f"freq_off{offset}"] = (at(frequency)[offset:] - at(frequency)[:-offset], FREQ_RANGE)
    pairs = [
        ("level", "freq"), ("level", "rise16"), ("level", "later50"), ("level", "later200"), ("level", "slope"),
        ("freq", "freq_later50"), ("level", "freq_neighbour"), ("level", "freq_off12"), ("level", "ratio12"),
        ("level", "freq_off19"),
    ]
    parts = [marginal(m, r, n_bands, config) for m, r in scalars.values()]
    parts += [joint(scalars[a][0], scalars[b][0], scalars[a][1], scalars[b][1], n_bands, config) for a, b in pairs]
    return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class PopulationStats:
    """Cross-feature sparsity: how many features are active together, and where.

    Per-feature statistics can all match while far too many features are active at the
    same time (a noise floor of small spikes). `soft` activity a^2 / (a^2 + tau^2) is a
    differentiable stand-in for "this spike is on", with tau a fraction of a typical large spike.
    """

    tau: float
    atom_rate: torch.Tensor  # (K,) mean activity per atom
    band_rate: torch.Tensor  # (B,) mean activity per band
    frame_counts: torch.Tensor  # (frames,) sorted: active spikes per code frame
    location_counts: torch.Tensor  # (B * frames,) sorted: atoms active at each band/frame
    adjacency: torch.Tensor  # P(active | active at band +1, +2, frame +1, +3)

    @classmethod
    def from_code(cls, code: torch.Tensor, tau: float) -> PopulationStats:
        soft = code.pow(2) / (code.pow(2) + tau**2)

        def follows(shifted: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            return (shifted * base).sum() / (base.sum() + 1e-6)

        adjacency = torch.stack([
            follows(soft[:, 1:], soft[:, :-1]), follows(soft[:, 2:], soft[:, :-2]),
            follows(soft[..., 1:], soft[..., :-1]), follows(soft[..., 3:], soft[..., :-3]),
        ])
        return cls(
            tau, soft.mean((1, 2)), soft.mean((0, 2)), soft.sum((0, 1)).sort().values[None],
            soft.sum(0).flatten().sort().values[None], adjacency,
        )

    def distance(self, target: PopulationStats) -> torch.Tensor:
        return (
            _matrix_error(self.atom_rate, target.atom_rate)
            + _matrix_error(self.band_rate, target.band_rate)
            + _distribution_error(self.frame_counts, target.frame_counts)
            + _distribution_error(self.location_counts, target.location_counts)
            + _matrix_error(self.adjacency, target.adjacency)
        )


@dataclass(frozen=True)
class Target:
    code: CodeStats
    population: PopulationStats

    @classmethod
    def from_code(cls, code: torch.Tensor) -> Target:
        tau = 0.05 * float(torch.quantile(code.abs().flatten(), 0.999))
        return cls(CodeStats.from_code(code), PopulationStats.from_code(code, tau))

    def distance(self, code: torch.Tensor) -> torch.Tensor:
        return CodeStats.from_code(code).distance(self.code) + PopulationStats.from_code(code, self.population.tau).distance(
            self.population
        )


def _code_of(parameter: torch.Tensor, positive: bool) -> torch.Tensor:
    """The code as a function of the free parameter: squared when it must be non-negative."""
    return parameter * parameter if positive else parameter


def generate_code(target: Target, shape: tuple[int, int, int], iters: int, seed: int, positive: bool = False) -> torch.Tensor:
    """Optimise a dense Gaussian start toward the code statistics; the magnitude term shrinks it to sparsity."""
    generator = torch.Generator().manual_seed(seed)
    rms = target.code.magnitudes.pow(2).mean(-1).sqrt()[:, None, None]
    parameter = (torch.randn(*shape, generator=generator) * (rms.sqrt() if positive else rms)).requires_grad_()
    optimiser = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = target.distance(_code_of(parameter, positive))
        loss.backward()
        return loss

    optimiser.step(closure)
    return _code_of(parameter, positive).detach()


def refine_with_critic(
    code: torch.Tensor, code_target: Target, real_features: torch.Tensor, features_of, iters: int,
    weight: float, ridge_rel: float, positive: bool = False,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Alternate refitting the linear histogram critic with L-BFGS steps that fool it."""
    parameter = (code.sqrt() if positive else code).clone().requires_grad_()

    def critic_loss() -> float:
        with torch.no_grad():
            generated = features_of(_code_of(parameter, positive))
            return float(Critic.fit(real_features, generated, ridge_rel).loss(generated))

    before = critic_loss()
    for _ in range(max(1, iters // CRITIC_ROUND_ITERS)):
        with torch.no_grad():
            critic = Critic.fit(real_features, features_of(_code_of(parameter, positive)), ridge_rel)
        optimiser = torch.optim.LBFGS(
            [parameter], lr=1.0, max_iter=CRITIC_ROUND_ITERS, max_eval=CRITIC_ROUND_ITERS * 5 // 4, history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            current = _code_of(parameter, positive)
            loss = code_target.distance(current) + weight * critic.loss(features_of(current))
            loss.backward()
            return loss

        optimiser.step(closure)
    return _code_of(parameter, positive).detach(), (before, critic_loss())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300, help="stage 1 (code statistics) iterations")
    parser.add_argument("--hist-iters", type=int, default=60, help="stage 2 (histogram critic) iterations")
    parser.add_argument("--hist-weight", type=float, default=1.0)
    parser.add_argument("--hist-ridge", type=float, default=0.3)
    parser.add_argument("--hist-group", type=int, default=4)
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument("--hist-joint-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stem = Stem(args.source, args.stem)
    dictionary_dir = step_dir(AMPFREQ, "01_dictionary", args.source, args.stem)
    saved = torch.load(dictionary_dir / "dictionary.pt")
    atoms, scales, lam_rel, stride = saved["atoms"], saved["scales"], saved["lam_rel"], saved["stride"]

    audio = decode_mono(stem.path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    source_maps = features(Wavelet.for_length(audio.numel()).analyse(audio))
    scaled = source_maps / scales

    cache = dictionary_dir / "codes.pt"
    if cache.exists():
        source_code = torch.load(cache)
    else:
        operator = Operator.build(atoms, (scaled.shape[1], scaled.shape[2]), stride)
        source_code = encode(scaled, operator, lam_rel * float(operator.analyse(scaled).abs().max()), ENCODE_ITERS)
        torch.save(source_code, cache)
    code_target = Target.from_code(source_code)

    n_samples = trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel()
    wavelet = Wavelet.for_length(n_samples)
    operator = Operator.build(atoms, (scaled.shape[1], wavelet.frames), stride)
    reference = AmplitudeReference.from_maps(source_maps)
    config = FeatureConfig(args.hist_group, args.hist_bins, args.hist_joint_bins)

    def features_of(code: torch.Tensor) -> torch.Tensor:
        return map_features(operator.decode(code) * scales, reference, config)

    with torch.no_grad():
        real_features = map_features(source_maps, reference, config)
    print(f"{args.stem}: {real_features.shape[1]} histogram features, {real_features.shape[0]} real windows", flush=True)

    out_dir = step_dir(AMPFREQ, "02_generate", args.source, args.stem)
    for i in range(args.n):
        seed = args.seed + i
        code = generate_code(code_target, operator.code_shape, args.iters, seed)
        report = ""
        if args.hist_iters:
            code, (before, after) = refine_with_critic(
                code, code_target, real_features, features_of, args.hist_iters, args.hist_weight, args.hist_ridge
            )
            report = f"  critic {before:.2f} -> {after:.2f}"
        with torch.no_grad():
            maps = operator.decode(code) * scales
            out = new_sample_path(out_dir)
            write_wav(out, wavelet.synthesize(coefficients_from(maps)))
            terms = {k: round(float(v), 3) for k, v in CodeStats.from_code(code).terms(code_target.code).items()}
            terms["population"] = round(float(PopulationStats.from_code(code, code_target.population.tau).distance(code_target.population)), 3)
        print(f"{out}  {terms}{report}", flush=True)


if __name__ == "__main__":
    main()
