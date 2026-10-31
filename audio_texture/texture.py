"""Profile a song once, then sample as many new versions of it as you like.

    venv/bin/python texture.py profile inputs/Walk_Of_Life.mp3                  # -> profiles/walk_of_life.texture
    venv/bin/python texture.py sample profiles/walk_of_life.texture --n 3       # three new versions, each as long as the song (about)
    venv/bin/python texture.py sample profiles/walk_of_life.texture --clips 3   # only its first three clips (a quick listen)

`profile` separates the song (separate.py, once), cuts every stem into clips of about --clip-seconds, and keeps in a single
file what sampling needs (texture_profile.py); nothing else is written. `sample` reads only that file. A version is made
clip after clip: each clip is generated stem by stem in a random order,
runs --overlap seconds into the next clip's time, and the next clip starts from, and is held to, the latents the previous
one ended with. Every clip's length is drawn at random within --jitter of its source clip's, so a version is not exactly as
long as the song. The clips are blended in codec latent space over their overlaps (a linear crossfade), decoded once, and
each stem is given the loudness curve of the source stem clip by clip.

The result: out/codec_texture/04_song/<song>/<uuid>.wav, and its stems in <uuid>_stems/.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import codec
from codec_texture import FRAME_RATE, LAG_SETS, PERFORMANCE_CORES, Lead, Window, loudness, parse_trim, save, synthesise
from paths import CODEC_TEXTURE, PROFILE_ROOT, input_name, new_sample_path, step_dir, stems_of
from synthesize import OUTPUT_PEAK
from texture_profile import ClipProfile, Profile

SILENT_DB = -60.0  # gain given to a stem where a clip has none of it
DECODE_CHUNK = 30 * round(FRAME_RATE)  # frames decoded at once when rendering the whole song
DECODE_CONTEXT = round(FRAME_RATE)  # frames of neighbouring latents the decoder sees beyond each chunk
STEP = "04_song"
HERE = Path(__file__).parent


@dataclass(frozen=True)
class Recipe:
    """How a version is generated. The defaults are the settings approved by ear."""

    iters: int = 120
    lags: str = "half"  # must be the set the profile was measured at (texture_profile.LAGS)
    activity: float = 0.0  # weights of the optional statistics (a profile holds them all; a weight of 0 skips the sorting)
    joint: float = 1.0
    ica: float = 0.0
    dynamics: float = 1.0
    rhythm: float = 0.0
    windowed: float = 1.0
    positions: float = 0.0  # weights of the density terms (density.py): the synthesis kept as likely as the source's own frames
    transitions: float = 0.0
    audio: float = 0.0  # weight of the decoded audio's statistics (tfd.py: time-frequency, phase, loudness), matched in a short refinement stage
    overlap: float = 2.0  # seconds a clip's output runs into the next clip's time
    lead_weight: float = 5.0
    jitter: float = 0.1  # how far a clip's generated length may stray from its source clip's, as a share of it

    @property
    def overlap_frames(self) -> int:
        return round(self.overlap * FRAME_RATE)


@dataclass(frozen=True)
class Slot:
    """Where one clip of a version lands in the new song (in latent frames)."""

    index: int
    profile: ClipProfile
    start: int
    frames: int  # the clip's own stretch of the timeline
    last: bool

    @property
    def end(self) -> int:
        return self.start + self.frames

    def generated(self, overlap: int) -> int:
        """Frames to generate: the own stretch, plus the overlap into the next clip."""
        return self.frames + (0 if self.last else overlap)


def layout(profile: Profile, jitter: float, rng: random.Random) -> list[Slot]:
    """Every clip's stretch of the timeline: its source length, scaled by a random factor within `jitter` of 1."""
    frames = [max(1, round(c.seconds * FRAME_RATE * (1 + jitter * rng.uniform(-1, 1)))) for c in profile.clips]
    starts = list(accumulate(frames, initial=0))
    return [Slot(i, c, starts[i], n, i == len(frames) - 1) for i, (c, n) in enumerate(zip(profile.clips, frames))]


def generate_clip(slot: Slot, previous: dict[str, torch.Tensor] | None, seed: int, recipe: Recipe) -> dict[str, torch.Tensor]:
    """Stem name -> latents (LATENTS, frames) of one clip: the stems one after another in a random order, and the start held
    to the tail of `previous` (the clip before, same layout). Loudness, within a stem and between stems, is a property of the
    waveform (`tfd`), not of the latents."""
    stems = slot.profile.stems
    names = list(stems)
    made: dict[str, torch.Tensor] = {}
    for name in random.Random(f"{seed}/{slot.index}").sample(names, len(names)):  # no stem type always goes first
        stem = stems[name]
        lead = Lead(previous[name][:, -recipe.overlap_frames :], recipe.lead_weight) if previous and name in previous else None
        print(f"  clip {slot.index + 1}  {name}", flush=True)
        new = synthesise(  # no coefficients, no manifold term: the statistics in `stem.target` are all it knows of the source
            None, slot.generated(recipe.overlap_frames), recipe.iters, seed * 100_000 + slot.index * 100 + names.index(name),
            LAG_SETS[recipe.lags], 0.0, "shuffle", 1.0, 0, 0.0, recipe.activity, recipe.joint, recipe.ica, recipe.dynamics, 0.0, 0, 1.0,
            rhythm=recipe.rhythm, basis=stem.basis, lead=lead, target=stem.target, windowed=recipe.windowed,
            positions=recipe.positions, transitions=recipe.transitions, audio=recipe.audio,
        )
        made[name] = stem.basis.decode(new)
    return made


def ramp(length: int, overlap_in: int, overlap_out: int) -> torch.Tensor:
    """(length,): linear fade-in over the first `overlap_in` frames and fade-out over the last `overlap_out`."""
    f = torch.arange(length) + 0.5
    fade_in = (f / overlap_in).clamp(max=1.0) if overlap_in else torch.ones(length)
    fade_out = ((length - f) / overlap_out).clamp(max=1.0) if overlap_out else torch.ones(length)
    return fade_in * fade_out


def timeline(slots: list[Slot], made: list[dict[str, torch.Tensor]], stem: str) -> torch.Tensor | None:
    """(LATENTS, frames): the stem's latents for the whole song, the clips blended over their overlaps; None when no clip has it.
    A stem is as long as the song, silent where no clip has it."""
    present = [(s, m[stem]) for s, m in zip(slots, made) if stem in m]
    if not present:
        return None
    ends = {s.index: s.start + latents.shape[1] for s, latents in present}
    starts = {s.index: s.start for s in slots}
    total = max(slots[-1].end, *ends.values())

    def weighted(slot: Slot, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = latents.shape[1]
        overlap_in = max(0, ends[slot.index - 1] - slot.start) if slot.index - 1 in ends else 0
        overlap_out = max(0, ends[slot.index] - starts[slot.index + 1]) if not slot.last else 0
        w = ramp(length, overlap_in, min(overlap_out, length))
        pad = (slot.start, total - slot.start - length)
        return F.pad(latents * w, pad), F.pad(w, pad)

    pairs = [weighted(s, latents) for s, latents in present]
    return sum(n for n, _ in pairs) / sum(w for _, w in pairs).clamp(min=1e-8)


def decode_long(latents: torch.Tensor) -> torch.Tensor:
    """`codec.decode` of a whole song, in chunks that see a second of context on each side."""
    pieces = []
    for start in range(0, latents.shape[1], DECODE_CHUNK):
        lo, hi = max(0, start - DECODE_CONTEXT), min(latents.shape[1], start + DECODE_CHUNK + DECODE_CONTEXT)
        audio = codec.decode(latents[:, lo:hi])
        pieces.append(audio[(start - lo) * codec.HOP : (start - lo + min(DECODE_CHUNK, latents.shape[1] - start)) * codec.HOP])
    return torch.cat(pieces)


def gain_curve(slots: list[Slot], stem: str, audio: torch.Tensor, trim: float) -> torch.Tensor:
    """(len(audio),): per-sample gain taking the stem, clip by clip, to the loudness of the source stem's clip (+ `trim` dB)."""
    centres, decibels = [], []
    for slot in slots:
        span = audio[slot.start * codec.HOP : slot.end * codec.HOP]
        source = slot.profile.stems.get(stem)
        made = loudness(span) if source and len(span) > 0.4 * codec.CODEC_RATE else float("-inf")
        centres.append((slot.start + slot.end) / 2 * codec.HOP)
        decibels.append(source.lufs - made + trim if source and np.isfinite(made) else SILENT_DB)
    return torch.from_numpy(10 ** (np.interp(np.arange(len(audio)), centres, decibels) / 20)).float()


def render(song: str, slots: list[Slot], made: list[dict[str, torch.Tensor]], trim: dict[str, float]) -> Path:
    """Stitch the clips of one version and write the mix and its stems."""
    stems = sorted({name for m in made for name in m})
    rendered = {}
    for stem in stems:
        audio = decode_long(timeline(slots, made, stem))
        rendered[stem] = audio * gain_curve(slots, stem, audio, trim.get(stem, 0.0))
    mix = torch.stack(list(rendered.values())).sum(0)
    gain = OUTPUT_PEAK / float(mix.abs().max())
    path = new_sample_path(step_dir(CODEC_TEXTURE, STEP, song))
    save(path, mix, gain)
    for stem, audio in rendered.items():
        save(path.with_name(f"{path.stem}_stems") / f"{stem}.wav", audio, gain)
    return path


def make_version(path: Path, recipe: Recipe, trim: dict[str, float], drop: tuple[str, ...], limit: int | None, seed: int, version: int) -> Path:
    """One new version of the profiled song (in this process or a worker), without the stems named in `drop`."""
    profile = Profile.load(path)
    profile = Profile(profile.song, tuple(c.without(drop) for c in profile.clips[:limit]), profile.settings)
    version_seed = seed * 1000 + version
    slots = layout(profile, recipe.jitter, random.Random(f"{version_seed}/layout"))
    print(f"version {version + 1}: {len(slots)} clips, {sum(s.frames for s in slots) / FRAME_RATE:.0f} s "
          f"(the source: {sum(c.seconds for c in profile.clips):.0f} s)", flush=True)
    made: list[dict[str, torch.Tensor]] = []
    for slot in slots:
        print(f"version {version + 1}  clip {slot.index + 1}/{len(slots)}", flush=True)
        made.append(generate_clip(slot, made[-1] if made else None, version_seed, recipe))
    return render(profile.song, slots, made, trim)


def profile_command(args: argparse.Namespace) -> None:
    song = input_name(args.input)
    if not stems_of(song):
        subprocess.run([sys.executable, "separate.py", str(args.input)], check=True, cwd=HERE)
    window = Window(args.window_frames, args.window_projections, args.window_components)
    profile = Profile.build(song, args.clip_seconds, args.copies, args.components, args.original_share, args.seed, args.clips, window, args.independent)
    path = args.output or PROFILE_ROOT / f"{song}.texture"
    profile.save(path)
    print(f"{path}  ({path.stat().st_size / 1e6:.0f} MB, {len(profile.clips)} clips)")


def sample_command(args: argparse.Namespace) -> None:
    recipe = Recipe(
        iters=args.iters, overlap=args.overlap, jitter=args.jitter,
        activity=args.activity, joint=args.joint, ica=args.ica, dynamics=args.dynamics, rhythm=args.rhythm,
        windowed=args.windowed, positions=args.positions, transitions=args.transitions, audio=args.audio,
    )
    trim = parse_trim(args.trim)
    jobs = min(args.jobs or args.n, args.n, PERFORMANCE_CORES)
    one = partial(make_version, args.profile, recipe, trim, tuple(args.drop), args.clips, args.seed)
    if jobs == 1:
        paths = [one(version) for version in range(args.n)]
    else:
        with ProcessPoolExecutor(jobs, initializer=torch.set_num_threads, initargs=(max(1, PERFORMANCE_CORES // jobs),)) as workers:
            paths = list(workers.map(one, range(args.n)))
    print("\n".join(str(p) for p in paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="measure a song once")
    profile.add_argument("input", type=Path, help="the song, e.g. inputs/Walk_Of_Life.mp3")
    profile.add_argument("--output", type=Path, help="where to write it (default: profiles/<input>.texture)")
    profile.add_argument("--clip-seconds", type=float, default=10.0)
    profile.add_argument("--copies", type=int, default=32, help="perturbed copies of every stem clip (they only shape the PCA basis; none is kept)")
    profile.add_argument("--components", type=int, default=64, help="PCA components (with --independent: how many independent components)")
    profile.add_argument("--independent", action="store_true", help="synthesise in the independent components of the PCA basis, not its axes")
    profile.add_argument("--original-share", type=float, default=0.5, help="the share of the PCA's weight held by the original's frames")
    profile.add_argument("--seed", type=int, default=0)
    profile.add_argument("--clips", type=int, help="only the first N clips (a quick test)")
    profile.add_argument("--window-frames", type=int, default=Window.frames, help="frames in a run whose projections the windowed statistic matches")
    profile.add_argument("--window-projections", type=int, default=Window.projections, help="random filters the windowed statistic matches")
    profile.add_argument("--window-components", type=int, default=Window.components, help="leading components the windowed filters span")
    profile.set_defaults(run=profile_command)

    sample = commands.add_parser("sample", help="make new versions of a profiled song")
    sample.add_argument("profile", type=Path, help="a .texture file made by `profile`")
    sample.add_argument("--n", type=int, default=1, help="versions to make")
    sample.add_argument("--jobs", type=int, default=0,
                        help="versions made at the same time, each in its own process with the cores shared between them "
                             "(0 = all of --n, up to the performance cores): the threads of one process scale poorly")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--clips", type=int, help="only the first N clips (a quick listen)")
    sample.add_argument("--iters", type=int, default=Recipe.iters)
    sample.add_argument("--overlap", type=float, default=Recipe.overlap, help="seconds each clip's output runs into the next clip's time")
    sample.add_argument("--jitter", type=float, default=Recipe.jitter,
                        help="each clip's length is its source clip's times a random factor within this of 1 (0 = the song's own length)")
    for name in ("activity", "joint", "ica", "dynamics", "rhythm", "windowed", "positions", "transitions", "audio"):
        sample.add_argument(f"--{name}", type=float, default=getattr(Recipe, name), help=f"weight of the {name} statistics (0 = leave out)")
    sample.add_argument("--drop", nargs="*", default=[], metavar="STEM", help="leave these stems out of the version entirely, e.g. --drop vocals")
    sample.add_argument("--trim", nargs="*", default=[], metavar="STEM=DB", help="turn a stem up or down in the mix, in dB")
    sample.set_defaults(run=sample_command)

    args = parser.parse_args()
    if args.command == "sample" and not 0 <= args.jitter < 1:
        parser.error("--jitter must be in [0, 1)")
    args.run(args)


if __name__ == "__main__":
    main()
