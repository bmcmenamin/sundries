"""Expand a whole song: a new recording of the same length, stitched from clips generated one after another.

    venv/bin/python song.py inputs/Walk_Of_Life.mp3                  # the whole song
    venv/bin/python song.py inputs/Walk_Of_Life.mp3 --clips 3        # only its first three clips (a quick listen)
    venv/bin/python song.py inputs/Walk_Of_Life.mp3 --trim guitar=-3 -- --iters 300

The song is separated once (separate.py), each stem is cut into clips of about --clip-seconds, and every clip is generated like
a short input (perturb.py, then codec_texture.py). A clip's output runs --overlap seconds into the next clip's time; the next
clip starts from, and is held to, the latents the previous one ended with (`--lead-from`). All the clips are then blended in
codec latent space over their overlaps (a linear crossfade), decoded once, and each stem is given the loudness curve of the
source stem clip by clip, so the song's quiet and loud passages carry over. Anything after `--` goes to codec_texture.py.

Clips are scratch inputs named <song>__c<NN> (stems under out/separation/01_demucs/, latents under cache/song/<song>/<clip>/);
rerunning skips the clips that are done. The result: out/codec_texture/04_song/<song>/<uuid>.wav, and its stems in <uuid>_stems/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

import codec
from codec_texture import FRAME_RATE, loudness, parse_trim, save, stem_loudness
from paths import CODEC_TEXTURE, Stem, cache_dir, input_name, new_sample_path, step_dir, stems_of
from synthesize import OUTPUT_PEAK
from synthesize_stems import stem_levels

SILENT_DB = -60.0  # gain given to a stem where a clip has none of it
MIN_DB = -25.0  # a stem this far below the clip's mix is left out of the clip (codec_texture.py's own default)
DECODE_CHUNK = 30 * round(FRAME_RATE)  # frames decoded at once when rendering the whole song
DECODE_CONTEXT = round(FRAME_RATE)  # frames of neighbouring latents the decoder sees beyond each chunk
HERE = Path(__file__).parent
STEP = "04_song"


@dataclass(frozen=True)
class Clip:
    index: int
    song: str
    start: float  # seconds into the song
    end: float
    last: bool

    @property
    def name(self) -> str:
        return f"{self.song}__c{self.index:02d}"

    @property
    def work(self) -> Path:
        return cache_dir("song", self.song, f"c{self.index:02d}")

    def seconds_generated(self, overlap: float) -> float:
        return self.end - self.start + (0.0 if self.last else overlap)

    def stems(self) -> list[Stem]:
        return stems_of(self.name)


def clips_of(song: str, duration: float, clip_seconds: float, limit: int | None) -> list[Clip]:
    """Equal-length clips of about `clip_seconds` covering the song (the first `limit` of them, when given)."""
    count = max(1, round(duration / clip_seconds))
    edges = [duration * i / count for i in range(count + 1)]
    clips = [Clip(i, song, edges[i], edges[i + 1], i == count - 1) for i in range(count)]
    return clips[:limit] if limit else clips


def slice_stems(song: str, clip: Clip) -> None:
    """Write the clip's part of every separated stem of the song as an input of its own."""
    for stem in stems_of(song):
        rate, audio = wavfile.read(stem.path)
        target = Stem(clip.name, stem.name).path
        target.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(target, rate, audio[round(clip.start * rate) : round(clip.end * rate)])


def kept_stems(clip: Clip) -> list[str]:
    report = stem_levels(clip.stems())
    return [name for name, (_, db) in report.items() if db >= MIN_DB]


def run(*command: str) -> None:
    subprocess.run([sys.executable, *command], check=True, cwd=HERE)


def generate(clip: Clip, previous: Clip | None, overlap: float, copies: int, extra: list[str]) -> None:
    """Perturb and generate one clip (skipped when done), continuing from the previous clip's latents."""
    done = clip.work / "done"
    if done.exists():
        return
    slice_stems(clip.song, clip)
    stems = kept_stems(clip)
    run("perturb.py", clip.name, "--n", str(copies), "--stems", *stems)
    lead = ["--lead-from", str(previous.work), "--lead-seconds", str(overlap)] if previous else []
    run(
        "codec_texture.py", clip.name, "--stems", *stems, "--n", "1", "--name", "song", "--seed", str(10 * clip.index),
        "--seconds", f"{clip.seconds_generated(overlap):.3f}", "--components", "64", "--manifold", "1", "--window", "48",
        "--augment", str(copies), "--take-seconds", "2", "5", "--latents-dir", str(clip.work), *lead, *extra,
    )
    done.write_text("\n".join(stems))


def ramp(length: int, overlap_in: int, overlap_out: int) -> torch.Tensor:
    """(length,): linear fade-in over the first `overlap_in` frames and fade-out over the last `overlap_out`."""
    f = torch.arange(length) + 0.5
    fade_in = (f / overlap_in).clamp(max=1.0) if overlap_in else torch.ones(length)
    fade_out = ((length - f) / overlap_out).clamp(max=1.0) if overlap_out else torch.ones(length)
    return fade_in * fade_out


def timeline(clips: list[Clip], stem: str) -> torch.Tensor | None:
    """(LATENTS, frames): the stem's latents for the whole song, the clips blended over their overlaps; None when no clip has it."""
    present = [(c, torch.load(c.work / f"{stem}.pt")) for c in clips if (c.work / f"{stem}.pt").exists()]
    if not present:
        return None
    starts = {c.index: round(c.start * FRAME_RATE) for c in clips}
    total = max(starts[c.index] + latents.shape[1] for c, latents in present)
    ends = {c.index: starts[c.index] + latents.shape[1] for c, latents in present}
    numerator, weight = torch.zeros(codec.LATENTS, total), torch.zeros(total)
    for c, latents in present:
        length = latents.shape[1]
        overlap_in = max(0, ends[c.index - 1] - starts[c.index]) if c.index - 1 in ends else 0
        overlap_out = max(0, ends[c.index] - starts[c.index + 1]) if not c.last and c.index + 1 in starts else 0
        w = ramp(length, overlap_in, min(overlap_out, length))
        span = slice(starts[c.index], starts[c.index] + length)
        numerator[:, span] += latents * w
        weight[span] += w
    return numerator / weight.clamp(min=1e-8)


def decode_long(latents: torch.Tensor) -> torch.Tensor:
    """`codec.decode` of a whole song, in chunks that see a second of context on each side."""
    pieces = []
    for start in range(0, latents.shape[1], DECODE_CHUNK):
        lo, hi = max(0, start - DECODE_CONTEXT), min(latents.shape[1], start + DECODE_CHUNK + DECODE_CONTEXT)
        audio = codec.decode(latents[:, lo:hi])
        pieces.append(audio[(start - lo) * codec.HOP : (start - lo + min(DECODE_CHUNK, latents.shape[1] - start)) * codec.HOP])
    return torch.cat(pieces)


def loudness_curve(clips: list[Clip], stem: str, audio: torch.Tensor, trim: float) -> torch.Tensor:
    """(len(audio),): per-sample gain taking the stem, clip by clip, to the loudness of the source stem's clip (+ `trim` dB)."""
    centres, decibels = [], []
    for c in clips:
        source = Stem(c.name, stem)
        span = audio[round(c.start * codec.CODEC_RATE) : round(c.end * codec.CODEC_RATE)]
        have = (c.work / f"{stem}.pt").exists() and source.path.exists()
        target, made = (stem_loudness(source.path), loudness(span)) if have and len(span) > 0.4 * codec.CODEC_RATE else (float("-inf"), 0.0)
        centres.append((c.start + c.end) / 2 * codec.CODEC_RATE)
        decibels.append(target - made + trim if np.isfinite(target) and np.isfinite(made) else SILENT_DB)
    return torch.from_numpy(10 ** (np.interp(np.arange(len(audio)), centres, decibels) / 20)).float()


def stitch(song: str, clips: list[Clip], trim: dict[str, float]) -> Path:
    stems = sorted({p.stem for c in clips for p in c.work.glob("*.pt")})
    rendered = {}
    for stem in stems:
        audio = decode_long(timeline(clips, stem))
        rendered[stem] = audio * loudness_curve(clips, stem, audio, trim.get(stem, 0.0))
    mix = torch.stack(list(rendered.values())).sum(0)
    gain = OUTPUT_PEAK / float(mix.abs().max())
    path = new_sample_path(step_dir(CODEC_TEXTURE, STEP, song))
    save(path, mix, gain)
    for stem, audio in rendered.items():
        save(path.with_name(f"{path.stem}_stems") / f"{stem}.wav", audio, gain)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=2.0, help="seconds each clip's output runs into the next clip's time")
    parser.add_argument("--clips", type=int, help="only the first N clips")
    parser.add_argument("--copies", type=int, default=64, help="perturbed copies of each stem clip")
    parser.add_argument("--trim", nargs="*", default=[], metavar="STEM=DB")
    args, extra = parser.parse_known_args()
    extra = [e for e in extra if e != "--"]
    if not any(e == "--iters" for e in extra):
        extra += ["--iters", "120"]  # the objective is 99.7% converged by then (see codec_texture's --iters)

    song = input_name(args.input)
    if not stems_of(song):
        run("separate.py", str(args.input))
    duration = len(codec.load(stems_of(song)[0].path)) / codec.CODEC_RATE  # the codec's own view of the length, trimmed to whole frames
    clips = clips_of(song, duration, args.clip_seconds, args.clips)
    print(f"{song}: {duration:.0f} s, {len(clips)} clips of about {clips[0].end - clips[0].start:.1f} s", flush=True)
    for i, clip in enumerate(clips):
        print(f"clip {i + 1}/{len(clips)}  {clip.start:.1f}-{clip.end:.1f} s", flush=True)
        generate(clip, clips[i - 1] if i else None, args.overlap, args.copies, extra)
    print(stitch(song, clips, parse_trim(args.trim)))


if __name__ == "__main__":
    main()
