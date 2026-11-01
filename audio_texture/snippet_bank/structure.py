"""A song's structure as data: segments (time spans) of a few kinds (verse, chorus, ...), a Markov chain over kinds and a duration law per kind.

    Structure(segments).sample(seconds, rng)  ->  the sections of a new song, e.g. kind 0 for 14 s, kind 1 for 22 s, kind 0 for 9 s ...

segment.py makes a Structure from a recording; stitch.py generates the sections and joins them. Nothing here touches audio.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np

SMOOTHING = 0.25  # pseudo-count on every kind -> kind move: a song has ~10-15 segments, far too few to estimate a chain from
MIN_SECONDS = 4.0  # no generated section is shorter
DEFAULT_SIGMA = 0.25  # log-sd of the duration law for a kind seen once
MIN_SIGMA = 0.15


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    kind: int

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Section:
    kind: int
    seconds: float


@dataclass(frozen=True)
class Structure:
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        kinds = {s.kind for s in self.segments}
        if not self.segments or kinds != set(range(len(kinds))):
            raise ValueError("a structure needs at least one segment and kinds numbered 0..K-1")

    @property
    def n_kinds(self) -> int:
        return len({s.kind for s in self.segments})

    def starts(self) -> np.ndarray:
        """Which kind opens a song: the recording's own opening kind, smoothed."""
        counts = np.full(self.n_kinds, SMOOTHING)
        counts[self.segments[0].kind] += 1
        return counts / counts.sum()

    def transitions(self) -> np.ndarray:
        """Row i: where kind i goes next, from the recording's consecutive segments plus SMOOTHING on every pair (so also back to itself)."""
        counts = np.full((self.n_kinds, self.n_kinds), SMOOTHING)
        np.add.at(counts, ([s.kind for s in self.segments[:-1]], [s.kind for s in self.segments[1:]]), 1.0)
        return counts / counts.sum(1, keepdims=True)

    def law(self, kind: int) -> tuple[float, float, float, float]:
        """Log-normal duration law of a kind: (mean of log seconds, sd of log seconds, shortest allowed, longest allowed)."""
        seconds = np.array([s.seconds for s in self.segments if s.kind == kind])
        logs = np.log(seconds)
        sigma = max(float(logs.std()), MIN_SIGMA) if len(seconds) > 1 else DEFAULT_SIGMA
        return float(logs.mean()), sigma, max(MIN_SECONDS, 0.5 * float(seconds.min())), 1.5 * float(seconds.max())

    def walk(self, rng: np.random.Generator) -> Iterator[Section]:
        """An endless random walk over kinds, each with a duration drawn from its law."""
        transitions = self.transitions()

        def section(kind: int) -> Section:
            mu, sigma, lo, hi = self.law(kind)
            return Section(kind, float(np.clip(rng.lognormal(mu, sigma), lo, hi)))

        kind = int(rng.choice(self.n_kinds, p=self.starts()))
        while True:
            yield section(kind)
            kind = int(rng.choice(self.n_kinds, p=transitions[kind]))

    def sample(self, seconds: float, rng: np.random.Generator) -> tuple[Section, ...]:
        """Sections that add up to exactly `seconds`: the walk is cut where it passes `seconds`, the last section absorbs the difference."""
        walk = tuple(islice(self.walk(rng), int(seconds // MIN_SECONDS) + 2))  # enough even if every section were at the minimum
        passed = int(np.searchsorted(np.cumsum([s.seconds for s in walk]), seconds)) + 1
        return fitted(walk[:passed], seconds)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([[s.start, s.end, s.kind] for s in self.segments]))

    @classmethod
    def load(cls, path: Path) -> Structure:
        return cls(tuple(Segment(start, end, int(kind)) for start, end, kind in json.loads(path.read_text())))


def fitted(sections: tuple[Section, ...], seconds: float) -> tuple[Section, ...]:
    """Sections whose durations sum to `seconds`: the last one is trimmed, or dropped into its neighbour if the trim would leave it too short."""
    *head, last = sections
    remaining = seconds - sum(s.seconds for s in head)
    if remaining >= MIN_SECONDS or not head:
        return (*head, Section(last.kind, remaining))
    *rest, previous = head
    return (*rest, Section(previous.kind, previous.seconds + remaining))
