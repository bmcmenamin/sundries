"""Control: render the *source's own* code with its time blocks shuffled.

    venv/bin/python nmf_control.py walk_1 piano --tag k4 --block-seconds 1.0

If this still has the swirl, the limit is the dictionary / decoder (atoms, K, amplitude fit), not the
generation statistics. If it is clean, the generator is introducing the problem. Writes
out/nmf/03_control_shuffle<tag>/<input>/<stem>/<uuid>.wav, rendered with the phase policy.
"""

from __future__ import annotations

import argparse

import torch

from ampfreq_code import Operator
from nmf_code import hybrid_coefficients, phase_policy
from paths import NMF, Stem, new_sample_path, step_dir
from synthesize import SAMPLE_RATE, write_wav
from wavelet import DECIMATION, Wavelet
from wavelet_code import trim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("stem")
    parser.add_argument("--tag", default="")
    parser.add_argument("--block-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    directory = step_dir(NMF, "01_dictionary" + suffix, args.source, args.stem)
    saved = torch.load(directory / "dictionary.pt")
    codes = torch.load(directory / "codes.pt")
    stride = saved["stride"]
    block = max(1, round(args.block_seconds * SAMPLE_RATE / DECIMATION / stride))
    blocks = list(codes.split(block, dim=-1))
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(args.seed)).tolist()
    shuffled = torch.cat([blocks[i] for i in order], dim=-1)

    frames = shuffled.shape[-1] * stride
    wavelet = Wavelet.for_length(trim(torch.zeros(frames * DECIMATION)).numel())
    operator = Operator.build(saved["atoms"], (codes.shape[1], wavelet.frames), stride)
    decoded = operator.decode(shuffled[..., : operator.code_shape[2]]) * saved["scales"]
    keep = phase_policy(Stem(args.source, args.stem), directory)
    out = new_sample_path(step_dir(NMF, f"03_control_shuffle{suffix}", args.source, args.stem))
    write_wav(out, wavelet.synthesize(hybrid_coefficients(decoded, keep, wavelet, args.seed)))
    print(out)


if __name__ == "__main__":
    main()
