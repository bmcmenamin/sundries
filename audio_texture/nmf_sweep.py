"""Sweep NMF dictionary sizes and report how clean the activation distribution is.

    venv/bin/python nmf_sweep.py walk_1 piano --sizes 4 8 12 16 24 32 48

For each size K: learn a dictionary (nmf_code.py settings), encode the whole stem, then report
  SNR          amplitude / frequency reconstruction SNR
  spikes/s     non-zero codes per second
  off / mid / on   share of all code positions that are zero or < 2% of the reference level,
                   between 2% and 25%, and >= 25% (reference: the 99th percentile of non-zero codes)
  clean        on / (mid + on): of the active features, how many are clearly on
  eta          Otsu between-class variance share of log-activation (0.64 for a unimodal bell,
                -> 1 when the active values form two separate clumps)
  per frame    mean number of clearly-on features per code frame
The wanted dictionary is the smallest one with high clean, high eta, few mid values and a small
spike rate without losing much SNR.
"""

from __future__ import annotations

import argparse

import torch

from ampfreq_code import Operator
from nmf_code import amplitude_maps, encode, error_weights, learn_atoms, snr_db
from paths import Stem
from synthesize import SAMPLE_RATE, decode_mono
from wavelet import DECIMATION, Wavelet
from wavelet_code import trim

OFF, ON = 0.02, 0.25


def activation_report(codes: torch.Tensor) -> dict[str, float]:
    nonzero = codes[codes > 0]
    reference = torch.quantile(nonzero[: 4_000_000], 0.99)
    level = nonzero / reference
    off = 1 - float((level >= OFF).sum()) / codes.numel()
    on = float((level >= ON).sum()) / codes.numel()
    mid = 1 - off - on
    log_level = torch.log10(level[level >= OFF].clamp(min=OFF))
    grid = torch.linspace(float(log_level.min()), float(log_level.max()), 64)[1:-1]
    between = torch.stack([_between_variance(log_level, t) for t in grid])
    return {
        "off": 100 * off,
        "mid": 100 * mid,
        "on": 100 * on,
        "clean": on / max(mid + on, 1e-9),
        "eta": float(between.max() / log_level.var(unbiased=False)),
        "per_frame": float((codes > ON * reference).sum() / codes.shape[2]),
    }


def _between_variance(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    low, high = values[values < threshold], values[values >= threshold]
    if low.numel() == 0 or high.numel() == 0:
        return torch.tensor(0.0)
    w_low, w_high = low.numel() / values.numel(), high.numel() / values.numel()
    return w_low * w_high * (low.mean() - high.mean()) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("stem")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 48])
    parser.add_argument("--atom-bands", type=int, default=16)
    parser.add_argument("--atom-frames", type=int, default=150)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--encode-iters", type=int, default=20)
    parser.add_argument("--train-seconds", type=float, default=8.0)
    args = parser.parse_args()

    audio = decode_mono(Stem(args.source, args.stem).path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    maps = amplitude_maps(Wavelet.for_length(audio.numel()).analyse(audio))
    x = maps / maps.pow(2).mean((1, 2), keepdim=True).sqrt()
    weights = error_weights(x[0])
    frames = trim(audio[: round(args.train_seconds * SAMPLE_RATE)]).numel() // DECIMATION
    seconds = x.shape[2] * DECIMATION / SAMPLE_RATE

    print(f"{'K':>3} {'ampSNR':>7} {'frqSNR':>7} {'spikes/s':>9} {'off%':>6} {'mid%':>6} {'on%':>6} {'clean':>6} {'eta':>5} {'on/frame':>9}")
    for k in args.sizes:
        atoms = learn_atoms(
            x[:, :, :frames], weights[:, :frames], k, (args.atom_bands, args.atom_frames), args.lam, args.rounds,
            args.encode_iters, args.stride, seed=0, verbose=False,
        )
        operator = Operator.build(atoms, (x.shape[1], x.shape[2]), args.stride)
        codes = encode(x, weights, operator, args.lam * float(operator.analyse(weights * x).max()), 6 * args.encode_iters)
        estimate = operator.decode(codes)
        r = activation_report(codes)
        print(
            f"{k:>3} {snr_db(x[0], estimate[0]):>7.1f} {snr_db(x[1], estimate[1]):>7.1f} {int((codes > 0).sum()) / seconds:>9.0f} "
            f"{r['off']:>6.1f} {r['mid']:>6.1f} {r['on']:>6.1f} {r['clean']:>6.2f} {r['eta']:>5.2f} {r['per_frame']:>9.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
