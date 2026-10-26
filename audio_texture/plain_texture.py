"""Texture generation in the plain (unweighted) non-negative code, three maps (amplitude, frequency, group delay).

    venv/bin/python separate.py inputs/walk_1.mp3
    venv/bin/python plain_texture.py walk_1 --stems piano guitar --n 3

Per stem: fit the dictionary and code if missing (plain.py), then optimise a fresh non-negative code
(through a square) until its covariance-style statistics match the stem's own, decode, and render with
the stem's phase mode. Statistics are absolute: no z-scoring, no per-atom or per-band normalisation,
so a quiet atom or band matters exactly as much as it is loud. Terms:
  magnitudes       per-atom distribution of activation values
  band profile     mean activation per atom and band
  activity         per-atom (+ total) distribution of windowed activation at 50 / 200 / 800 ms
  jumps            per-atom (+ total) distribution of decoded-level change over 1 / 2 code steps (attack sharpness)
  coactivity       covariance of windowed activity between atoms (50 / 200 ms)
  cross-atom       covariance of the raw code between atoms
  neighbours       covariance of each atom with itself one / three frames and one / two bands away
  map autocov      2-D autocovariance of the band x frame activity map (harmonic spacing, durations)
  population       how many features are on together, and where (ampfreq_texture.PopulationStats)
Optional (off by default): --critic-iters runs the linear histogram critic of ampfreq_texture on the
decoded maps afterwards.
Writes out/nmf_phase/02_generate_k<K>_t<frames>/<input>/<stem>/<uuid>.{wav,pt}; with several stems, also
out/nmf_phase/03_recompose_k<K>_t<frames>/<input>/<uuid>.wav.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ampfreq_code import Operator
from ampfreq_texture import PopulationStats
from paths import NMF_PHASE, new_sample_path, step_dir, stems_of
from plain import ATOM_FRAMES, STRIDE, config_tag, fit_stem, load_stem, plain_maps, render_coefficients
from synthesize import SAMPLE_RATE, write_wav
from synthesize_stems import recompose, stem_levels
from wavelet import N_BANDS, Wavelet
from wavelet_code import trim

# Time scales are in wavelet frames (1500 per second) and become code steps through the dictionary's stride,
# so the statistics mean the same thing whatever the stride (at least one step).
LEVEL_LAG = 9  # frames (6 ms): the span over which a broadband rise is measured
ATTACK_SHARE = 0.03  # the fastest-rising frames that count as attacks
LOUD_SHARE = 0.2  # a cell is loud above this share of the stem's 99.9th-percentile amplitude
SPREAD_EPS = 1e-4
ACTIVITY_FRAMES = (36, 144, 576)  # 24 / 96 / 384 ms
COACTIVITY_FRAMES = (36, 144)
JUMP_FRAMES = (6, 12, 24)  # 4 / 8 / 16 ms
NEIGHBOUR_FRAMES = (12, 36)
AC_BANDS = 36
AC_FRAMES = 1440  # 960 ms
EPS = 1e-8


def steps(frames: tuple[int, ...], stride: int) -> tuple[int, ...]:
    """Frame counts as code steps, at least one, without repeats."""
    return tuple(dict.fromkeys(max(1, f // stride) for f in frames))


def _error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared error against the target's overall scale (one scale for the whole array)."""
    return (a - b).pow(2).mean() / (b.pow(2).mean() + EPS)


def _distribution_error(sorted_a: torch.Tensor, sorted_b: torch.Tensor) -> torch.Tensor:
    if sorted_b.shape[-1] != sorted_a.shape[-1]:
        sorted_b = F.interpolate(sorted_b[None], size=sorted_a.shape[-1], mode="linear", align_corners=True)[0]
    return _error(sorted_a, sorted_b)


def _covariance(rows: torch.Tensor) -> torch.Tensor:
    centred = rows - rows.mean(-1, keepdim=True)
    return centred @ centred.T / centred.shape[-1]


def _activity(code: torch.Tensor, window: int) -> torch.Tensor:
    """(K + 1, frames): per-atom activation summed over bands, plus a total row, window-averaged."""
    per_atom = code.sum(1)
    rows = torch.cat([per_atom, per_atom.sum(0, keepdim=True)])
    return F.avg_pool1d(rows[None], window, stride=max(1, window // 2))[0]


def attack_profiles(atoms: torch.Tensor, stride: int) -> torch.Tensor:
    """(K, T): each atom's amplitude over time, summed across bands and binned to code steps."""
    profile = atoms[:, 0].sum(1)
    padded = F.pad(profile, (0, -profile.shape[-1] % stride))
    return padded.unflatten(-1, (-1, stride)).sum(-1)


def _jumps(code: torch.Tensor, profiles: torch.Tensor, lag: int) -> torch.Tensor:
    """(K + 1, frames - lag), sorted: change over `lag` code steps in each atom's (+ the total) decoded level.

    The level is the atom's activation convolved with its time profile, so a wide atom turns an activation into
    a rise at its start and a fall at its end: the code's own step-to-step change would miss both.
    """
    per_atom = code.sum(1)
    taps = profiles.shape[-1]
    level = F.conv1d(F.pad(per_atom, (taps - 1, 0))[None], profiles.flip(-1)[:, None], groups=per_atom.shape[0])[0]
    rows = torch.cat([level, level.sum(0, keepdim=True)])
    return (rows[:, lag:] - rows[:, :-lag]).sort(-1).values


def _map_autocovariance(code: torch.Tensor, max_lag: int) -> torch.Tensor:
    activity_map = code.sum(0)
    centred = activity_map - activity_map.mean()
    bands, frames = centred.shape
    spectrum = torch.fft.rfft2(centred, s=(2 * bands, 2 * frames))
    acf = torch.fft.irfft2(spectrum.real**2 + spectrum.imag**2, s=(2 * bands, 2 * frames)) / centred.numel()
    return acf[torch.arange(AC_BANDS + 1)][:, torch.arange(-max_lag, max_lag + 1) % (2 * frames)]


@dataclass(frozen=True)
class CovarianceStats:
    magnitudes: torch.Tensor
    band_profile: torch.Tensor
    activity: tuple[torch.Tensor, ...]
    jumps: tuple[torch.Tensor, ...]
    coactivity: tuple[torch.Tensor, ...]
    cross_atom: torch.Tensor
    neighbours: torch.Tensor
    map_autocovariance: torch.Tensor

    @classmethod
    def from_code(cls, code: torch.Tensor, profiles: torch.Tensor, stride: int) -> CovarianceStats:
        flat = code.flatten(1)
        return cls(
            magnitudes=flat.sort(-1).values,
            band_profile=code.mean(-1),
            activity=tuple(_activity(code, w).sort(-1).values for w in steps(ACTIVITY_FRAMES, stride)),
            jumps=tuple(_jumps(code, profiles, lag) for lag in steps(JUMP_FRAMES, stride)),
            coactivity=tuple(_covariance(_activity(code, w)) for w in steps(COACTIVITY_FRAMES, stride)),
            cross_atom=flat @ flat.T / flat.shape[-1],
            neighbours=torch.stack(
                [(code[..., :-lag] * code[..., lag:]).mean((1, 2)) for lag in steps(NEIGHBOUR_FRAMES, stride)]
                + [(code[:, :-lag] * code[:, lag:]).mean((1, 2)) for lag in (1, 2)],
                dim=-1,
            ),
            map_autocovariance=_map_autocovariance(code, AC_FRAMES // stride),
        )

    def terms(self, target: CovarianceStats) -> dict[str, torch.Tensor]:
        return {
            "magnitudes": _distribution_error(self.magnitudes, target.magnitudes),
            "band_profile": _error(self.band_profile, target.band_profile),
            "activity": sum(_distribution_error(a, b) for a, b in zip(self.activity, target.activity)),
            "jumps": sum(_distribution_error(a, b) for a, b in zip(self.jumps, target.jumps)),
            "coactivity": sum(_error(a, b) for a, b in zip(self.coactivity, target.coactivity)),
            "cross_atom": _error(self.cross_atom, target.cross_atom),
            "neighbours": _error(self.neighbours, target.neighbours),
            "map_autocov": _error(self.map_autocovariance, target.map_autocovariance),
        }


@dataclass(frozen=True)
class DecodedStats:
    """Statistics of the decoded maps (x-units), which the code statistics cannot see: the delay channel.

      delay          delay-channel values in the loud cells
      spread         amplitude-weighted rms delay across bands, per active frame (0 = every band lined up)
      attack_spread  the same, at the frames where the broadband level rises fastest
      attack_rise    those rises (level change over LEVEL_LAG frames)
    """

    floor: float  # amplitude above which a cell is loud
    delay: torch.Tensor  # each (1, n), sorted
    spread: torch.Tensor
    attack_spread: torch.Tensor
    attack_rise: torch.Tensor

    @classmethod
    def from_maps(cls, maps: torch.Tensor, floor: float | None = None) -> DecodedStats:
        amplitude = maps[0].clamp(min=0)
        floor = LOUD_SHARE * float(torch.quantile(amplitude.flatten()[:4_000_000], 0.999)) if floor is None else floor
        level = amplitude.sum(0)
        spread = (maps[2].pow(2).sum(0) / (amplitude.pow(2).sum(0) + SPREAD_EPS)).sqrt()
        rise = level[LEVEL_LAG:] - level[:-LEVEL_LAG]
        attacks = rise.topk(max(1, int(ATTACK_SHARE * rise.numel()))).indices
        active = (level > 0.5 * level.mean()).detach()
        sort = lambda values: values.sort().values[None]  # noqa: E731
        return cls(
            floor, sort(maps[2][amplitude > floor]), sort(spread[active]),
            sort(spread[LEVEL_LAG:][attacks]), sort(rise[attacks]),
        )

    def terms(self, target: DecodedStats) -> dict[str, torch.Tensor]:
        return {
            "delay": _distribution_error(self.delay, target.delay),
            "spread": _distribution_error(self.spread, target.spread),
            "attack_spread": _distribution_error(self.attack_spread, target.attack_spread),
            "attack_rise": _distribution_error(self.attack_rise, target.attack_rise),
        }


@dataclass(frozen=True)
class Target:
    stats: CovarianceStats
    population: PopulationStats
    profiles: torch.Tensor  # (K, T) atom time profiles, see `attack_profiles`
    stride: int
    decoded: DecodedStats

    @classmethod
    def from_code(cls, code: torch.Tensor, profiles: torch.Tensor, stride: int, maps: torch.Tensor) -> Target:
        tau = 0.05 * float(torch.quantile(code.flatten()[code.flatten() > 0][:4_000_000], 0.999))
        return cls(
            CovarianceStats.from_code(code, profiles, stride), PopulationStats.from_code(code, tau), profiles, stride,
            DecodedStats.from_maps(maps),
        )

    def distance(self, code: torch.Tensor, decode: Callable[[torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
        """Code statistics, plus the decoded-map statistics when `decode` (code -> (3, B, W) maps) is given."""
        own = CovarianceStats.from_code(code, self.profiles, self.stride).terms(self.stats)
        if decode is not None:
            own |= DecodedStats.from_maps(decode(code), self.decoded.floor).terms(self.decoded)
        return sum(own.values()) + PopulationStats.from_code(code, self.population.tau).distance(self.population)


def _optimise(parameter: torch.Tensor, target: Target, iters: int, decode: Callable | None) -> None:
    optimiser = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = target.distance(parameter * parameter, decode)
        loss.backward()
        return loss

    optimiser.step(closure)


def generate_code(
    target: Target, shape: tuple[int, int, int], iters: int, seed: int, decode: Callable | None = None, refine_iters: int = 0
) -> torch.Tensor:
    """Optimise a dense start toward the target; the code is the square of the free parameter, so >= 0.

    `iters` steps on the code statistics, then `refine_iters` more that also match the decoded maps (the delay
    channel), which costs a decode per step.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = target.stats.magnitudes.pow(2).mean(-1).sqrt()[:, None, None].sqrt()
    parameter = (torch.randn(*shape, generator=generator) * scale).requires_grad_()
    _optimise(parameter, target, iters, None)
    if decode is not None and refine_iters:
        _optimise(parameter, target, refine_iters, decode)
    return (parameter * parameter).detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--stems", nargs="+", help="stems to generate (default: all audible)")
    parser.add_argument("--min-db", type=float, default=-25.0)
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--atom-frames", type=int, default=ATOM_FRAMES, help="atom length in wavelet frames (1500 per second)")
    parser.add_argument("--stride", type=int, default=STRIDE, help="frames between possible atom positions")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--refine-iters", type=int, default=60, help="extra steps that also match the decoded delay channel (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-iters", type=int, default=0, help="optional histogram-critic stage (slow)")
    parser.add_argument("--critic-weight", type=float, default=1.0)
    args = parser.parse_args()

    stems = stems_of(args.source)
    report = stem_levels(stems)
    kept = [s for s in stems if report[s.name][1] >= args.min_db and (args.stems is None or s.name in args.stems)]
    wavelet = Wavelet.for_length(trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel())
    config = config_tag(args.atoms, args.atom_frames, args.stride)
    sample_dirs = {s.name: step_dir(NMF_PHASE, f"02_generate_{config}", args.source, s.name) for s in kept}

    for k, stem in enumerate(kept):
        fit = fit_stem(stem, args.atoms, args.atom_frames, args.stride)
        own_operator = Operator.build(fit.atoms, (N_BANDS, fit.codes.shape[-1] * fit.stride), fit.stride)
        with torch.no_grad():
            own_maps = own_operator.decode(fit.codes)
        target = Target.from_code(fit.codes, attack_profiles(fit.atoms, fit.stride), fit.stride, own_maps)
        operator = Operator.build(fit.atoms, (N_BANDS, wavelet.frames), fit.stride)
        critic = _critic(stem, fit, operator, args) if args.critic_iters else None
        for i in range(args.n):
            seed = args.seed + 1000 * k + i
            code = generate_code(target, operator.code_shape, args.iters, seed, operator.decode, args.refine_iters)
            if critic is not None:
                code = critic(code, target, seed)
            with torch.no_grad():
                out = new_sample_path(sample_dirs[stem.name])
                decoded = operator.decode(code) * fit.scales
                write_wav(out, wavelet.synthesize(render_coefficients(decoded, fit.phase, wavelet, seed)))
                torch.save(code, out.with_suffix(".pt"))
                terms = {
                    key: round(float(v), 3)
                    for key, v in (
                        CovarianceStats.from_code(code, target.profiles, target.stride).terms(target.stats)
                        | DecodedStats.from_maps(operator.decode(code), target.decoded.floor).terms(target.decoded)
                    ).items()
                }
            print(f"{out}  phase {fit.phase}  {terms}", flush=True)

    if len(kept) > 1:
        levels = {s.name: report[s.name][0] for s in kept}
        for path in recompose(args.source, levels, sample_dirs, step_dir(NMF_PHASE, f"03_recompose_{config}", args.source), args.n, args.seed):
            print(path)


def _critic(stem, fit, operator, args):
    """Optional stage 2: the linear histogram critic on the decoded maps (see ampfreq_texture.py)."""
    from ampfreq_texture import AmplitudeReference, map_features, refine_with_critic
    from histogram_critic import FeatureConfig
    from nmf_texture import critic_view

    audio, source_wavelet = load_stem(stem)
    view = critic_view(plain_maps(source_wavelet.analyse(audio)))
    reference, config = AmplitudeReference.from_maps(view), FeatureConfig(4, 20, 10)
    with torch.no_grad():
        real_features = map_features(view, reference, config)

    def features_of(code: torch.Tensor) -> torch.Tensor:
        return map_features(critic_view(operator.decode(code) * fit.scales), reference, config)

    def refine(code: torch.Tensor, target: Target, seed: int) -> torch.Tensor:
        refined, (before, after) = refine_with_critic(
            code, target, real_features, features_of, args.critic_iters, args.critic_weight, 0.3, positive=True
        )
        print(f"  critic {before:.2f} -> {after:.2f}", flush=True)
        return refined

    return refine


if __name__ == "__main__":
    main()
