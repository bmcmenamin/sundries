"""A song's profile: everything sampling needs, in one file (profiles/<input>.texture).

    venv/bin/python texture.py profile inputs/Walk_Of_Life.mp3      # once per song
    venv/bin/python texture.py sample profiles/walk_of_life.texture --n 3

The song's separated stems are cut into clips; each stem of each clip is encoded (codec.py), perturbed copies of it are
made and encoded (perturb.py), and a PCA is fitted on the original plus its copies. The profile keeps, per clip and stem,
what `codec_texture.synthesise` reads: the PCA basis, the original's coefficients (every statistic it matches is
computed from them), the copies' coefficients (what the manifold term pulls toward; half precision), the loudness curve
(for `--cross`) and the stem's loudness in the source. No audio and no intermediate files: sampling needs only this file.
"""

from __future__ import annotations

import random
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import torch

import codec
from codec_texture import Basis, loudness, loudness_curve, pooled
from paths import stems_of
from perturb import Perturbation
from synthesize import decode_mono

FORMAT = 1
MIN_DB = -25.0  # a stem this far below the clip's mix is left out of the clip


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
    coefficients: torch.Tensor  # (k, T): the source's latents in `basis`
    copies: tuple[torch.Tensor, ...]  # each (k, T_i), half precision; lengths differ a little (the pitch shift changes the speed)
    curve: torch.Tensor  # `loudness_curve` of the source's latents
    lufs: float  # the stem's loudness in the source

    @classmethod
    def build(cls, audio: torch.Tensor, rng: random.Random, copies: int, components: int, original_share: float) -> StemProfile:
        """`audio`: the stem's part of the song, at its own level, at the codec's rate."""
        scaled = codec.normalise(audio)
        latents = codec.encode(scaled)
        variants = [codec.encode(codec.normalise(Perturbation.random(rng).apply(scaled))) for _ in range(copies)]
        basis = Basis.fit(*pooled(latents, variants, components, original_share))
        return cls(basis, basis.encode(latents), tuple(basis.encode(v).half() for v in variants), loudness_curve(latents), loudness(audio))

    def pool(self) -> list[torch.Tensor]:
        return [c.float() for c in self.copies]

    def state(self) -> dict:
        return {
            "mean": self.basis.mean, "components": self.basis.components, "coefficients": self.coefficients,
            "copies": list(self.copies), "curve": self.curve, "lufs": self.lufs,
        }

    @classmethod
    def from_state(cls, s: dict) -> StemProfile:
        return cls(Basis(s["mean"], s["components"]), s["coefficients"], tuple(s["copies"]), s["curve"], s["lufs"])


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
    def build(cls, audio: dict[str, torch.Tensor], start: float, end: float, seed: str, copies: int, components: int, original_share: float) -> ClipProfile:
        """`audio`: every stem of the whole song, at the codec's rate."""
        pieces = {name: a[round(start * codec.CODEC_RATE) : round(end * codec.CODEC_RATE)] for name, a in audio.items()}
        levels = levels_db(pieces)
        stems = {
            name: StemProfile.build(piece, random.Random(f"{seed}/{name}"), copies, components, original_share)
            for name, piece in pieces.items()
            if levels[name] >= MIN_DB
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
        seed: int = 0, limit: int | None = None,
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
            clips.append(ClipProfile.build(audio, start, end, f"{seed}/{song}/{i}", copies, components, original_share))
        settings = {"clip_seconds": clip_seconds, "copies": copies, "components": components, "original_share": original_share, "seed": seed}
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
