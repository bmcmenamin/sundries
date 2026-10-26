"""Is the swirl made of low-level ("intermediate") activations? Prune them from the source code and listen.

    venv/bin/python prune_test.py walk_1 piano k4

Loads the cached source code of an NMF dictionary, zeroes every activation below a fraction of the
reference level (99th percentile of the non-zero codes), decodes, renders with the hybrid phase
policy and writes recon_pruned_<percent>.wav next to the dictionary. Prints how much of the active
set and of the energy each threshold removes, and the reconstruction SNR.
"""

from __future__ import annotations

import argparse

import torch

from ampfreq import snr_db
from ampfreq_code import Operator
from nmf_code import amplitude_maps, hybrid_coefficients, phase_policy
from paths import NMF, Stem, step_dir
from synthesize import decode_mono, write_wav
from wavelet import Wavelet
from wavelet_code import trim

FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("stem")
    parser.add_argument("tag", help="dictionary tag, e.g. k4")
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    args = parser.parse_args()

    directory = step_dir(NMF, f"01_dictionary_{args.tag}", args.source, args.stem)
    saved = torch.load(directory / "dictionary.pt")
    codes = torch.load(directory / "codes.pt")
    audio = decode_mono(Stem(args.source, args.stem).path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    wavelet = Wavelet.for_length(audio.numel())
    scaled = amplitude_maps(wavelet.analyse(audio)) / saved["scales"]
    operator = Operator.build(saved["atoms"], (scaled.shape[1], scaled.shape[2]), saved["stride"])
    keep = phase_policy(Stem(args.source, args.stem), directory)
    reference = torch.quantile(codes[codes > 0][:4_000_000], 0.99)

    print(f"{'threshold':>9} {'active kept%':>12} {'energy kept%':>12} {'ampSNR':>7}")
    for fraction in args.fractions:
        pruned = torch.where(codes >= fraction * reference, codes, torch.zeros_like(codes))
        decoded = operator.decode(pruned)
        print(
            f"{fraction:>9.2f} {100 * float((pruned > 0).sum() / (codes > 0).sum()):>12.1f} "
            f"{100 * float(pruned.pow(2).sum() / codes.pow(2).sum()):>12.1f} {snr_db(scaled[0], decoded[0]):>7.1f}"
        )
        write_wav(
            directory / f"recon_pruned_{round(100 * fraction):02d}.wav",
            wavelet.synthesize(hybrid_coefficients(decoded * saved["scales"], keep, wavelet)),
        )


if __name__ == "__main__":
    main()
