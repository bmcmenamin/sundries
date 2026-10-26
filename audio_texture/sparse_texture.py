"""Texture generation in sparse-code space.

    venv/bin/python sparse_texture.py walk_1 --n 3 --stems piano guitar

For each audible stem: learn (or load) its convolutional dictionary, encode the
stem, measure statistics of the code, then optimise a fresh code until its
statistics match and render it with the linear decoder. Writes
out/sparse_raw/02_generate/<input>/<stem>/<uuid>.wav and sums random picks of
the stems' samples into out/sparse_raw/03_recompose/<input>/<uuid>.wav.

Statistics (a sparse code is a point mass at zero plus an amplitude distribution,
so whole distributions are matched rather than skew/kurtosis):
  amplitudes      per-atom distribution of signed amplitude (includes sparsity)
  activity        per-atom (+ total) distribution of windowed |a| at 50/200/800 ms (bursts)
  coactivity      which atoms are active together, at 50/200 ms
  autocorrelation per-atom autocorrelation to 20 ms (periodicity / pitch)
  cross-correlation  lagged correlation between atoms within +-1.3 ms (relative phase)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from paths import SPARSE_RAW, Stem, new_sample_path, step_dir, stems_of
from sparse_code import Dictionary, encode, learn_dictionary
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from synthesize_stems import recompose, stem_levels

ACTIVITY_WINDOWS = (600, 2400, 9600)
COACTIVITY_WINDOWS = (600, 2400)
MAX_AUTOCORR_LAG = 240
CROSS_LAGS = tuple(range(-16, 17, 2))
ENCODE_ITERS = 120
EPS = 1e-6


def windowed_activity(codes: torch.Tensor, window: int) -> torch.Tensor:
    """(K + 1, frames): mean |a| per atom plus a total-activity row."""
    activity = codes.abs()
    activity = torch.cat([activity, activity.sum(0, keepdim=True)])
    return F.avg_pool1d(activity[None], window, stride=window // 2)[0]


def _standardise(x: torch.Tensor) -> torch.Tensor:
    centred = x - x.mean(-1, keepdim=True)
    return centred / (centred.pow(2).mean(-1, keepdim=True) + EPS).sqrt()


def _autocorrelation(codes: torch.Tensor) -> torch.Tensor:
    n = codes.shape[-1]
    spectrum = torch.fft.rfft(codes, n=2 * n)
    acf = torch.fft.irfft(spectrum.real**2 + spectrum.imag**2, n=2 * n)[:, : MAX_AUTOCORR_LAG + 1]
    return acf / (acf[:, :1] + EPS)


def _lagged_product(x: torch.Tensor, lag: int) -> torch.Tensor:
    """(K, K): sum_t x_k[t] * x_l[t + lag]."""
    n = x.shape[-1]
    lo, hi = max(0, -lag), n - max(0, lag)
    return x[:, lo:hi] @ x[:, lo + lag : hi + lag].T


def _cross_correlations(codes: torch.Tensor) -> torch.Tensor:
    normalised = codes / (codes.pow(2).sum(-1, keepdim=True).sqrt() + EPS)
    return torch.stack([_lagged_product(normalised, lag) for lag in CROSS_LAGS])


def _resample_rows(sorted_rows: torch.Tensor, n: int) -> torch.Tensor:
    if sorted_rows.shape[-1] == n:
        return sorted_rows
    return F.interpolate(sorted_rows[None], size=n, mode="linear", align_corners=True)[0]


def _distribution_error(sorted_a: torch.Tensor, sorted_b: torch.Tensor) -> torch.Tensor:
    """Per-row squared quantile error, relative to the target row's scale."""
    target = _resample_rows(sorted_b, sorted_a.shape[-1])
    return ((sorted_a - target).pow(2).mean(-1) / (target.pow(2).mean(-1) + EPS)).mean()


def _matrix_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).mean() / (b.pow(2).mean() + 1e-2)


@dataclass(frozen=True)
class CodeStats:
    amplitudes: torch.Tensor  # (K, N) sorted signed amplitudes
    activity: tuple[torch.Tensor, ...]  # per ACTIVITY_WINDOWS: (K + 1, frames), sorted per row
    coactivity: tuple[torch.Tensor, ...]  # per COACTIVITY_WINDOWS: (K + 1, K + 1)
    autocorrelation: torch.Tensor  # (K, MAX_AUTOCORR_LAG + 1)
    cross_correlation: torch.Tensor  # (len(CROSS_LAGS), K, K)

    @classmethod
    def from_code(cls, codes: torch.Tensor) -> CodeStats:
        def coactivity(window: int) -> torch.Tensor:
            z = _standardise(windowed_activity(codes, window))
            return z @ z.T / z.shape[-1]

        return cls(
            amplitudes=codes.sort(-1).values,
            activity=tuple(windowed_activity(codes, w).sort(-1).values for w in ACTIVITY_WINDOWS),
            coactivity=tuple(coactivity(w) for w in COACTIVITY_WINDOWS),
            autocorrelation=_autocorrelation(codes),
            cross_correlation=_cross_correlations(codes),
        )

    def terms(self, target: CodeStats) -> dict[str, torch.Tensor]:
        return {
            "amplitudes": _distribution_error(self.amplitudes, target.amplitudes),
            "activity": sum(_distribution_error(a, b) for a, b in zip(self.activity, target.activity)),
            "coactivity": sum(_matrix_error(a, b) for a, b in zip(self.coactivity, target.coactivity)),
            "autocorrelation": _matrix_error(self.autocorrelation, target.autocorrelation),
            "cross_correlation": _matrix_error(self.cross_correlation, target.cross_correlation),
        }

    def distance(self, target: CodeStats) -> torch.Tensor:
        return sum(self.terms(target).values())


def generate_code(target: CodeStats, n: int, iters: int, seed: int) -> torch.Tensor:
    """Optimise a dense Gaussian start toward `target`; the sorted-amplitude term shrinks it to sparsity."""
    generator = torch.Generator().manual_seed(seed)
    scale = target.amplitudes.std(-1, keepdim=True)
    codes = (torch.randn(target.amplitudes.shape[0], n, generator=generator) * scale).requires_grad_()
    optimiser = torch.optim.LBFGS(
        [codes], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = CodeStats.from_code(codes).distance(target)
        loss.backward()
        return loss

    optimiser.step(closure)
    return codes.detach()


def load_or_learn(stem: Stem, x: torch.Tensor) -> tuple[Dictionary, float]:
    path = step_dir(SPARSE_RAW, "01_dictionary", stem.source, stem.name) / "dictionary.pt"
    if path.exists():
        saved = torch.load(path)
        return Dictionary(saved["filters"]), saved["lam_rel"]
    lam_rel = 0.05
    dictionary, _ = learn_dictionary(x, n_atoms=48, length=512, lam_rel=lam_rel, rounds=40, encode_iters=30, seed=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"filters": dictionary.filters, "lam_rel": lam_rel}, path)
    return dictionary, lam_rel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--stems", nargs="+", help="only these stems (default: all audible)")
    parser.add_argument("--min-db", type=float, default=-25.0)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stems = stems_of(args.source)
    report = stem_levels(stems)
    kept = [s for s in stems if report[s.name][1] >= args.min_db and (args.stems is None or s.name in args.stems)]
    n = round(args.seconds * SAMPLE_RATE)
    sample_dirs = {s.name: step_dir(SPARSE_RAW, "02_generate", args.source, s.name) for s in kept}

    for k, stem in enumerate(kept):
        x = decode_mono(stem.path)
        x = x / x.pow(2).mean().sqrt()
        dictionary, lam_rel = load_or_learn(stem, x)
        lam = lam_rel * float(dictionary.analyse(x).abs().max())
        target = CodeStats.from_code(encode(x, dictionary, lam, ENCODE_ITERS))
        for i in range(args.n):
            codes = generate_code(target, n, args.iters, args.seed + 1000 * k + i)
            out = new_sample_path(sample_dirs[stem.name])
            write_wav(out, dictionary.decode(codes))
            with torch.no_grad():
                terms = {key: round(float(v), 3) for key, v in CodeStats.from_code(codes).terms(target).items()}
            print(f"{out}  {terms}", flush=True)

    levels = {s.name: report[s.name][0] for s in kept}
    for path in recompose(
        args.source, levels, sample_dirs, step_dir(SPARSE_RAW, "03_recompose", args.source), args.n, args.seed
    ):
        print(path)


if __name__ == "__main__":
    main()
