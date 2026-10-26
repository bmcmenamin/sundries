"""Generate audio-texture samples from a sound file.

    venv/bin/python synthesize.py input.mp3 --n 5 --seconds 10 --jobs 2

Writes out/spectral/01_texture/<input>/<uuid>.wav (see paths.py for the layout).
For the stem pipeline use separate.py then synthesize_stems.py.

Statistics and a bank of spectrotemporal patches are measured on the source
(over non-overlapping windows), then each sample is a noise waveform optimised
with L-BFGS until its statistics and patch distribution match. Patches are
redrawn every ROUND_ITERS iterations. Samples are independent, so --jobs renders
them in parallel processes. Output is 16-bit mono WAV at 12 kHz.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch
from scipy.io import wavfile

from patch_loss import Features, PatchBank, PatchRound
from paths import SPECTRAL, input_name, new_sample_path, step_dir
from texture_stats import ENV_DECIMATION, SAMPLE_RATE, TextureStats

MAX_ANALYSIS_WINDOWS = 24
ROUND_ITERS = 25
PATCH_WEIGHT = 1.0
SOURCE_RMS = 0.1
OUTPUT_PEAK = 0.9


def decode_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Decode with ffmpeg to mono float at `sample_rate`, at the file's own level."""
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(pcm, dtype="<f4").copy())


def load_audio(path: Path) -> torch.Tensor:
    """`decode_mono`, scaled to SOURCE_RMS."""
    audio = decode_mono(path)
    if audio.numel() == 0 or audio.pow(2).sum() == 0:
        raise SystemExit(f"{path}: no audible signal")
    return audio * (SOURCE_RMS / audio.pow(2).mean().sqrt())


def analysis_windows(audio: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Non-overlapping windows of the synthesis length (or the whole source, if shorter)."""
    length = min(n_samples, audio.numel()) // ENV_DECIMATION * ENV_DECIMATION
    windows = audio.unfold(0, length, length)
    picks = torch.linspace(0, windows.shape[0] - 1, min(windows.shape[0], MAX_ANALYSIS_WINDOWS)).round().long()
    return windows[picks]


@dataclass(frozen=True)
class Target:
    stats: TextureStats
    patches: PatchBank


class Losses(NamedTuple):
    stats: float
    patches: float


def measure_target(audio: torch.Tensor, n_samples: int, seed: int, sources: tuple[str, ...]) -> Target:
    with torch.no_grad():
        features = [Features.from_waveform(w) for w in analysis_windows(audio, n_samples)]
        return Target(
            stats=TextureStats.average(TextureStats.from_envelope(f.envelope) for f in features),
            patches=PatchBank.from_features(features, torch.Generator().manual_seed(seed), sources),
        )


def draw_round(x: torch.Tensor, target: Target, generator: torch.Generator) -> PatchRound:
    with torch.no_grad():
        return PatchRound.draw(target.patches, Features.from_waveform(x), generator)


def losses(x: torch.Tensor, target: Target, patch_round: PatchRound) -> tuple[torch.Tensor, torch.Tensor]:
    features = Features.from_waveform(x)
    return TextureStats.from_envelope(features.envelope).distance(target.stats), patch_round.loss(features)


def evaluate(x: torch.Tensor, target: Target, patch_round: PatchRound) -> Losses:
    with torch.no_grad():
        return Losses(*(float(loss) for loss in losses(x, target, patch_round)))


def optimise_round(x: torch.Tensor, target: Target, patch_round: PatchRound, iters: int) -> None:
    """In-place L-BFGS on `x` against one fixed (deterministic) draw of patches and directions."""
    optimiser = torch.optim.LBFGS(
        [x], lr=1.0, max_iter=iters, max_eval=iters * 5 // 4, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        stats, patches = losses(x, target, patch_round)
        loss = stats + PATCH_WEIGHT * patches
        loss.backward()
        return loss

    optimiser.step(closure)


def synthesize(target: Target, n_samples: int, iters: int, seed: int) -> tuple[torch.Tensor, Losses, Losses]:
    """Optimise noise toward `target`. Returns (waveform, initial losses, final losses)."""
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(n_samples, generator=generator) * SOURCE_RMS).requires_grad_()

    initial = evaluate(x, target, draw_round(x, target, generator))
    for _ in range(max(1, iters // ROUND_ITERS)):
        optimise_round(x, target, draw_round(x, target, generator), ROUND_ITERS)
    final = evaluate(x, target, draw_round(x, target, generator))
    return x.detach(), initial, final


def write_wav(path: Path, x: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    wavfile.write(path, sample_rate, (x / x.abs().max() * OUTPUT_PEAK * 32767).round().short().numpy())


@dataclass(frozen=True)
class SampleTask:
    """Everything a worker process needs; the source is re-measured once per process."""

    input: Path
    n_samples: int
    iters: int
    target_seed: int
    sample_seed: int
    sources: tuple[str, ...]
    output: Path


@lru_cache(maxsize=8)
def _target_for(input: Path, n_samples: int, seed: int, sources: tuple[str, ...]) -> Target:
    return measure_target(load_audio(input), n_samples, seed, sources)


def render_sample(task: SampleTask) -> str:
    target = _target_for(task.input, task.n_samples, task.target_seed, task.sources)
    waveform, initial, final = synthesize(target, task.n_samples, task.iters, task.sample_seed)
    task.output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(task.output, waveform)
    return (
        f"{task.output}  stats {initial.stats:.3f} -> {final.stats:.3f}"
        f"  patches {initial.patches:.3f} -> {final.patches:.3f}"
    )


def _init_worker(threads: int | None) -> None:
    if threads:
        torch.set_num_threads(threads)


def run_tasks(tasks: list[SampleTask], jobs: int, threads: int | None) -> Iterator[str]:
    """Render `tasks` in `jobs` parallel processes, yielding a report line per task, in order."""
    threads = threads or (os.cpu_count() // jobs if jobs > 1 else None)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(threads,)) as pool:
        yield from pool.map(render_sample, tasks)


def add_synthesis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=5, help="number of samples")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=300, help="L-BFGS iterations per sample")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first sample")
    parser.add_argument("--jobs", type=int, default=1, help="samples rendered in parallel processes")
    parser.add_argument("--threads", type=int, help="torch threads per process (default: cores / jobs)")
    parser.add_argument(
        "--sources", nargs="+", default=["envelope", "spectrogram"], choices=["envelope", "spectrogram"],
        help="which feature maps the patch loss uses",
    )


def n_samples_for(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE) // ENV_DECIMATION * ENV_DECIMATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    add_synthesis_arguments(parser)
    args = parser.parse_args()

    out_dir = step_dir(SPECTRAL, "01_texture", input_name(args.input))
    tasks = [
        SampleTask(
            args.input, n_samples_for(args.seconds), args.iters, args.seed, args.seed + i, tuple(args.sources),
            new_sample_path(out_dir),
        )
        for i in range(args.n)
    ]
    for line in run_tasks(tasks, args.jobs, args.threads):
        print(line, flush=True)


if __name__ == "__main__":
    main()
