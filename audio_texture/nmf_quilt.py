"""Texture by quilting the source's own sparse code (Efros-Freeman style, in code space).

    venv/bin/python nmf_quilt.py walk_1 piano --tag k4 --n 3 --max-shift 3

Control experiments showed the dictionary and decoder are fine (a block-shuffled copy of the source
code sounds as clean as the reconstruction) and that statistics-matched codes lose the structure of
how atoms are arranged. So generate by stitching real code blocks: each next block is chosen from
the source, among the best matches to the tail of what has been built so far, and cross-faded over
an overlap. Novelty comes from re-ordering, from random transposition (shifting a block along the
band axis, which the shift-invariant dictionary renders as a pitch shift, with the frequency channel
unchanged because it is relative), and from avoiding replaying the source in order.
Writes out/nmf/02_quilt<tag>/<input>/<stem>/<uuid>.wav (+ .pt with the code), rendered with the
per-band phase policy.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ampfreq_code import Operator
from nmf_code import hybrid_coefficients, phase_policy
from paths import NMF, Stem, new_sample_path, step_dir
from synthesize import SAMPLE_RATE, write_wav
from wavelet import DECIMATION, N_BANDS, Wavelet
from wavelet_code import trim


def shift_bands(code: torch.Tensor, shift: int) -> torch.Tensor:
    """Move a (K, B, M) code `shift` bands up (positive) or down, filling with zeros."""
    if shift == 0:
        return code
    if shift > 0:
        return F.pad(code, (0, 0, shift, 0))[:, : code.shape[1]]
    return F.pad(code, (0, 0, 0, -shift))[:, -shift:]


def quilt(
    source: torch.Tensor, frames: int, block: int, overlap: int, max_shift: int, top_k: int, generator: torch.Generator
) -> torch.Tensor:
    """(K, B, frames) code made of overlapping blocks of `source` (K, B, M)."""
    shifts = range(-max_shift, max_shift + 1)
    heads = {s: shift_bands(source, s).unfold(-1, overlap, 1) for s in shifts}  # (K, B, positions, overlap)
    positions = source.shape[-1] - block + 1
    start = int(torch.randint(positions, (1,), generator=generator))
    output = source[..., start : start + block].clone()
    previous = start
    while output.shape[-1] < frames:
        tail = output[..., -overlap:]
        distances = torch.stack(
            [((heads[s][:, :, :positions] - tail[:, :, None]) ** 2).sum((0, 1, 3)) for s in shifts]
        )  # (shifts, positions)
        replay = previous + block - overlap  # where the source itself would continue
        near = (torch.arange(positions) - replay).abs() < overlap
        distances[len(shifts) // 2, near] = float("inf")
        best = distances.flatten().topk(top_k, largest=False).indices
        pick = int(best[int(torch.randint(top_k, (1,), generator=generator))])
        shift_index, start = divmod(pick, positions)
        chosen = shift_bands(source, list(shifts)[shift_index])[..., start : start + block]
        fade = torch.linspace(0, 1, overlap + 2)[1:-1]
        blended = (1 - fade) * tail + fade * chosen[..., :overlap]
        output = torch.cat([output[..., :-overlap], blended, chosen[..., overlap:]], dim=-1)
        previous = start
    return output[..., :frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("stem")
    parser.add_argument("--tag", default="")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--block-seconds", type=float, default=0.5)
    parser.add_argument("--overlap-fraction", type=float, default=0.3)
    parser.add_argument("--max-shift", type=int, default=0, help="random transposition range, in bands (semitones)")
    parser.add_argument("--top-k", type=int, default=5, help="choose among the k best-matching next blocks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    directory = step_dir(NMF, "01_dictionary" + suffix, args.source, args.stem)
    saved = torch.load(directory / "dictionary.pt")
    source_code = torch.load(directory / "codes.pt")
    stride = saved["stride"]
    wavelet = Wavelet.for_length(trim(torch.zeros(round(args.seconds * SAMPLE_RATE))).numel())
    operator = Operator.build(saved["atoms"], (N_BANDS, wavelet.frames), stride)
    keep = phase_policy(Stem(args.source, args.stem), directory)
    seconds_per_frame = stride * DECIMATION / SAMPLE_RATE
    block = max(4, round(args.block_seconds / seconds_per_frame))
    overlap = max(2, round(block * args.overlap_fraction))

    out_dir = step_dir(NMF, f"02_quilt{suffix}", args.source, args.stem)
    for i in range(args.n):
        generator = torch.Generator().manual_seed(args.seed + i)
        code = quilt(source_code, operator.code_shape[2], block, overlap, args.max_shift, args.top_k, generator)
        decoded = operator.decode(code) * saved["scales"]
        out = new_sample_path(out_dir)
        write_wav(out, wavelet.synthesize(hybrid_coefficients(decoded, keep, wavelet, args.seed + i)))
        torch.save(code, out.with_suffix(".pt"))
        print(out)


if __name__ == "__main__":
    main()
