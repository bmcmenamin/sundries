"""A song's profile: everything sampling needs, in one file (profiles/<input>.texture).

    venv/bin/python texture.py profile inputs/Walk_Of_Life.mp3      # once per song
    venv/bin/python texture.py sample profiles/walk_of_life.texture --n 3

The song's separated stems are cut into clips; each stem of each clip is encoded (codec.py), perturbed copies of it are
made and encoded (perturb.py), and a PCA is fitted on the original plus its copies. The profile keeps, per clip and stem,
only statistics: the PCA basis, the `codec_texture.Target` (sorted values, changes, autocovariance and the rest, none in
time order), the pairwise cross-correlations of the stems' audio level curves (for the waveform stage's cross-stem loudness term) and the stem's loudness in the
source. It holds no frames of the original and no perturbed copies, so sampling cannot copy a stretch of the song, and the
manifold term (which needs frames to pull toward) is not part of it. No audio and no intermediate files: sampling needs
only this file.
"""

from __future__ import annotations

import random
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import codec
import tfd
from codec_texture import LAG_SETS, Basis, Target, Window, loudness, pooled
from tfd import cross_correlation, frame_level, level_curve
from paths import stems_of
from perturb import Perturbation
from synthesize import decode_mono

FORMAT = 13
MIN_DB = -25.0  # a stem this far below the clip's mix is left out of the clip
LAGS = LAG_SETS["half"]  # the lags the statistics are measured at; sampling must use the same set
MEASURED = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # the (activity, joint, ica, dynamics, rhythm, windowed) statistics a profile holds; a sample picks which to match


def clip_edges(duration: float, clip_seconds: float) -> list[float]:
    """Boundaries (seconds) of equal-length clips of about `clip_seconds` covering the song."""
    count = max(1, round(duration / clip_seconds))
    return [duration * i / count for i in range(count + 1)]


def levels_db(pieces: dict[str, torch.Tensor]) -> dict[str, float]:
    """Stem name -> dB of its RMS relative to the RMS of the mix of all the stems."""
    rms = lambda x: x.pow(2).mean().sqrt()
    mix = rms(torch.stack(list(pieces.values())).sum(0))
    return {name: float(20 * torch.log10(rms(x) / mix + 1e-9)) for name, x in pieces.items()}


@dataclass(frozen=True)
class StemProfile:
    basis: Basis
    target: Target  # the source's statistics: nothing in it is in time order
    lufs: float  # the stem's loudness in the source
    correlations: dict[str, torch.Tensor]  # each other stem of the clip -> `cross_correlation` of this stem's level curve with theirs (for `tfd.Partner`)

    @classmethod
    def build(
        cls, audio: torch.Tensor, rng: random.Random, copies: int, components: int, original_share: float, window: Window,
        independent: bool,
    ) -> tuple[StemProfile, torch.Tensor]:
        """The stem's profile, without its correlations, and its level curve (from the audio, only for relating the clip's stems to one another).

        `audio`: the stem's part of the song, at its own level, at the codec's rate. The perturbed copies only shape the PCA basis."""
        scaled = codec.normalise(audio)
        latents = codec.encode(scaled)
        variants = [codec.encode(codec.normalise(Perturbation.random(rng).apply(scaled))) for _ in range(copies)]
        basis = Basis.fit(*pooled(latents, variants, components, original_share))
        basis = basis.independent(latents) if independent else basis
        coefficients = basis.encode(latents)
        audio_statistics = tfd.Statistics.of(codec.decode(basis.decode(coefficients)))  # of the decoded reconstruction: the codec's own loss cancels
        target = Target.of(coefficients, LAGS, MEASURED, window, audio_statistics)
        return cls(basis, target, loudness(audio), {}), level_curve(frame_level(scaled))

    def state(self) -> dict:
        return {
            "mean": self.basis.mean, "components": self.basis.components, "unmix": self.basis.unmix, "target": self.target.state(),
            "lufs": self.lufs, "correlations": self.correlations,
        }

    @classmethod
    def from_state(cls, s: dict) -> StemProfile:
        return cls(Basis(s["mean"], s["components"], s["unmix"]), Target.from_state(s["target"]), s["lufs"], s["correlations"])


@dataclass(frozen=True)
class ClipProfile:
    start: float  # seconds into the source song
    end: float
    stems: dict[str, StemProfile]  # the stems audible in this clip

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def without(self, names: Collection[str]) -> ClipProfile:
        return ClipProfile(self.start, self.end, {n: s for n, s in self.stems.items() if n not in names})

    @classmethod
    def build(
        cls, audio: dict[str, torch.Tensor], start: float, end: float, seed: str, copies: int, components: int, original_share: float,
        window: Window, independent: bool,
    ) -> ClipProfile:
        """`audio`: every stem of the whole song, at the codec's rate."""
        pieces = {name: a[round(start * codec.CODEC_RATE) : round(end * codec.CODEC_RATE)] for name, a in audio.items()}
        levels = levels_db(pieces)
        built = {
            name: StemProfile.build(piece, random.Random(f"{seed}/{name}"), copies, components, original_share, window, independent)
            for name, piece in pieces.items()
            if levels[name] >= MIN_DB
        }
        stems = {
            name: replace(stem, correlations={other: cross_correlation(curve, built[other][1]) for other in built if other != name})
            for name, (stem, curve) in built.items()
        }
        return cls(start, end, stems)


@dataclass(frozen=True)
class Profile:
    song: str
    clips: tuple[ClipProfile, ...]
    settings: dict[str, float]  # how it was built: clip_seconds, copies, components, original_share, seed

    @classmethod
    def build(
        cls, song: str, clip_seconds: float = 10.0, copies: int = 32, components: int = 64, original_share: float = 0.5,
        seed: int = 0, limit: int | None = None, window: Window = Window(),
        independent: bool = False,
    ) -> Profile:
        """The profile of a separated song (separate.py); `limit`: only its first clips (a quick test)."""
        audio = {s.name: decode_mono(s.path, codec.CODEC_RATE) for s in stems_of(song)}
        if not audio:
            raise SystemExit(f"no stems for {song}; run separate.py first")
        length = min(len(a) for a in audio.values()) // codec.HOP * codec.HOP
        audio = {name: a[:length] for name, a in audio.items()}
        edges = clip_edges(length / codec.CODEC_RATE, clip_seconds)
        spans = list(zip(edges, edges[1:]))[:limit]
        clips = []
        for i, (start, end) in enumerate(spans):
            print(f"clip {i + 1}/{len(spans)}  {start:.1f}-{end:.1f} s", flush=True)
            clips.append(ClipProfile.build(audio, start, end, f"{seed}/{song}/{i}", copies, components, original_share, window, independent))
        settings = {
            "independent": independent,
            "clip_seconds": clip_seconds, "copies": copies, "components": components, "original_share": original_share, "seed": seed,
            "window_frames": window.frames, "window_projections": window.projections, "window_components": window.components,
        }
        return cls(song, tuple(clips), settings)

    def save(self, path: Path) -> None:
        """Written to a side file first, so a failed write (a full disk) never leaves a half profile at `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": FORMAT, "song": self.song, "settings": self.settings,
            "clips": [{"start": c.start, "end": c.end, "stems": {n: s.state() for n, s in c.stems.items()}} for c in self.clips],
        }
        side = path.with_name(path.name + ".partial")
        torch.save(state, side)
        side.replace(path)

    @classmethod
    def load(cls, path: Path) -> Profile:
        state = torch.load(path, weights_only=True)
        if state["format"] != FORMAT:
            raise SystemExit(f"{path}: profile format {state['format']}, expected {FORMAT}; build it again")
        clips = tuple(
            ClipProfile(c["start"], c["end"], {n: StemProfile.from_state(s) for n, s in c["stems"].items()}) for c in state["clips"]
        )
        return cls(state["song"], clips, state["settings"])
