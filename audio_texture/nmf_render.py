"""Re-render saved generated codes with a different phase choice.

    venv/bin/python nmf_render.py walk_1 piano --tag k4 --phase integrated

Reads every <uuid>.pt in out/nmf/02_generate<tag>/<input>/<stem>/ (saved by nmf_texture.py) and writes
out/nmf/03_render_<phase><tag>/<input>/<stem>/<uuid>.wav. Phase choices:
  integrated  integrate the frequency channel in every band
  noise       noise-excitation phase in every band
  policy      integrated in bands with measurable phase structure, noise elsewhere (phase_policy)
"""

from __future__ import annotations

import argparse

import torch

from ampfreq_code import Operator
from nmf_code import coefficients_from, hybrid_coefficients, phase_policy
from paths import NMF, Stem, new_sample_path, step_dir
from synthesize import SAMPLE_RATE, write_wav
from wavelet import N_BANDS, Wavelet
from wavelet_code import trim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("stem")
    parser.add_argument("--tag", default="", help="dictionary / generation suffix, e.g. k4")
    parser.add_argument("--phase", choices=["integrated", "noise", "policy"], default="policy")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    directory = step_dir(NMF, "01_dictionary" + suffix, args.source, args.stem)
    saved = torch.load(directory / "dictionary.pt")
    wavelet = Wavelet.for_length(trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel())
    operator = Operator.build(saved["atoms"], (N_BANDS, wavelet.frames), saved["stride"])
    keep = {
        "integrated": torch.ones(N_BANDS, dtype=torch.bool),
        "noise": torch.zeros(N_BANDS, dtype=torch.bool),
        "policy": phase_policy(Stem(args.source, args.stem), directory),
    }[args.phase]

    out_dir = step_dir(NMF, f"03_render_{args.phase}{suffix}", args.source, args.stem)
    for i, path in enumerate(sorted(step_dir(NMF, "02_generate" + suffix, args.source, args.stem).glob("*.pt"))):
        decoded = operator.decode(torch.load(path)) * saved["scales"]
        out = new_sample_path(out_dir)
        write_wav(out, wavelet.synthesize(hybrid_coefficients(decoded, keep, wavelet, seed=i)))
        print(f"{path.name} -> {out}")


if __name__ == "__main__":
    main()
