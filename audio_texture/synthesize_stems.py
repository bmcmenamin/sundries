"""Texture-synthesise each audible stem of a separated source, then recompose.

    venv/bin/python separate.py inputs/walk_1.mp3
    venv/bin/python synthesize_stems.py walk_1 --n 3 --jobs 2

Reads out/separation/01_demucs/<input>/<stem>/stem.wav, skips stems quieter than
--min-db relative to the mix, writes out/spectral/02_stem_texture/<input>/<stem>/<uuid>.wav
per stem, and writes out/spectral/03_recompose/<input>/<uuid>.wav: each is a sum of
one randomly chosen sample per stem, every stem rescaled to its original loudness.
Stems are synthesised independently.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from paths import SPECTRAL, Stem, new_sample_path, step_dir, stems_of
from synthesize import SampleTask, add_synthesis_arguments, decode_mono, n_samples_for, run_tasks, write_wav

SEED_STRIDE = 1000  # keeps stems' initial noise independent


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


def stem_levels(stems: list[Stem]) -> dict[str, tuple[float, float]]:
    """Stem name -> (rms, dB relative to the mix of all stems)."""
    decoded = {s.name: decode_mono(s.path) for s in stems}
    mix_rms = rms(torch.stack(list(decoded.values())).sum(0))
    return {name: (float(rms(x)), float(20 * torch.log10(rms(x) / mix_rms + 1e-9))) for name, x in decoded.items()}


def recompose(source: str, levels: dict[str, float], sample_dirs: dict[str, Path], out_dir: Path, n: int, seed: int) -> list[Path]:
    """`n` mixes, each summing one randomly chosen sample per stem at that stem's own loudness."""
    chooser = random.Random(seed)
    paths = []
    for _ in range(n):
        parts = [
            (x := decode_mono(chooser.choice(sorted(sample_dirs[name].glob("*.wav"))))) * (level / rms(x))
            for name, level in levels.items()
        ]
        path = new_sample_path(out_dir)
        write_wav(path, torch.stack(parts).sum(0))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, a folder under out/separation/01_demucs/")
    parser.add_argument("--min-db", type=float, default=-25.0, help="skip stems quieter than this vs the mix")
    parser.add_argument("--stems", nargs="+", help="only these stems (default: all audible)")
    add_synthesis_arguments(parser)
    args = parser.parse_args()

    stems = stems_of(args.source)
    if not stems:
        raise SystemExit(f"no stems for {args.source}; run separate.py first")
    report = stem_levels(stems)
    for name, (_, db) in report.items():
        print(f"{name:8s} {db:6.1f} dB  {'' if db >= args.min_db else 'skipped (too quiet)'}")
    kept = [s for s in stems if report[s.name][1] >= args.min_db and (args.stems is None or s.name in args.stems)]

    sample_dirs = {s.name: step_dir(SPECTRAL, "02_stem_texture", args.source, s.name) for s in kept}
    tasks = [
        SampleTask(
            stem.path, n_samples_for(args.seconds), args.iters, args.seed, args.seed + SEED_STRIDE * k + i,
            tuple(args.sources), new_sample_path(sample_dirs[stem.name]),
        )
        for k, stem in enumerate(kept)
        for i in range(args.n)
    ]
    for line in run_tasks(tasks, args.jobs, args.threads):
        print(line, flush=True)

    levels = {s.name: report[s.name][0] for s in kept}
    for path in recompose(args.source, levels, sample_dirs, step_dir(SPECTRAL, "03_recompose", args.source), args.n, args.seed):
        print(path)


if __name__ == "__main__":
    main()
