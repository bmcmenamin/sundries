"""A song's structure as data: segments (time spans) of a few kinds (verse, chorus, ...), a Markov chain over kinds and a duration law per kind.

    Structure(segments).sample(seconds, rng)  ->  the sections of a new song, e.g. kind 0 for 14 s, kind 1 for 22 s, kind 0 for 9 s ...
    Structure(segments).pair(rng)             ->  two consecutive sections, 10-20 s each and 30 s at most (what stitch.py generates as one track)

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
PAIR_SHORTEST, PAIR_LONGEST, PAIR_CAP = 10.0, 20.0, 30.0  # a pair of sections, each 10-20 s, 30 s together: MERT's attention memory grows with the square of the length


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

    def pair(
        self, rng: np.random.Generator, compatible: np.ndarray | None = None,
        shortest: float = PAIR_SHORTEST, longest: float = PAIR_LONGEST, cap: float = PAIR_CAP,
    ) -> tuple[Section, Section]:
        """Two consecutive sections: the first kind uniformly (the start distribution would open every track with the intro kind), the second from the first's
        transition row restricted to the kinds `compatible` allows after it (a boolean matrix, [first, second]; the diagonal must be True, so every kind can
        at least follow itself) and renormalised. Each lasts its kind's law clipped to [shortest, longest]; the second is shortened so the pair fits in `cap`,
        but never under `shortest`."""
        allowed = np.ones((self.n_kinds, self.n_kinds), dtype=bool) if compatible is None else compatible
        if allowed.shape != (self.n_kinds,) * 2 or not allowed.diagonal().all():
            raise ValueError("compatible must be a kinds x kinds boolean matrix with a True diagonal")
        transitions = self.transitions() * allowed
        first = int(rng.choice(self.n_kinds))
        second = int(rng.choice(self.n_kinds, p=transitions[first] / transitions[first].sum()))
        return self.timed(first, second, rng, shortest, longest, cap)

    def fade_kind(self) -> int | None:
        """The kind of a closing section that occurs nowhere else in the song (a fade-out or outro), else None."""
        last = self.segments[-1].kind
        return last if sum(s.kind == last for s in self.segments) == 1 else None

    def pairs(
        self, n: int, rng: np.random.Generator, must: set[int], compatible: np.ndarray | None = None, drop: frozenset[int] = frozenset(),
        shortest: float = PAIR_SHORTEST, longest: float = PAIR_LONGEST, cap: float = PAIR_CAP,
    ) -> tuple[tuple[Section, Section], ...]:
        """`n` pairs that each contain at least one `must` kind. Every ordered kind pair (a, b) gets P(a) * T[a -> b] with a uniform over the kinds not in `drop`
        and T the transition rows restricted to `compatible` and without `drop`, renormalised; pairs with no must kind are removed and the rest drawn from.
        So the must kind is forced and everything else (its place, the other kind) is the chain's. Kind pairs are drawn without repeats until they run out."""
        allowed = np.ones((self.n_kinds,) * 2, dtype=bool) if compatible is None else compatible
        if allowed.shape != (self.n_kinds,) * 2 or not allowed.diagonal().all():
            raise ValueError("compatible must be a kinds x kinds boolean matrix with a True diagonal")
        kept = np.array([k not in drop for k in range(self.n_kinds)])
        rows = self.transitions() * allowed * kept[:, None] * kept[None, :]
        rows = rows / np.maximum(rows.sum(1, keepdims=True), 1e-12)
        has_must = np.array([[a in must or b in must for b in range(self.n_kinds)] for a in range(self.n_kinds)])
        weights = (rows * has_must * kept[:, None]).ravel()
        if not weights.sum():
            raise ValueError("no allowed pair contains a must-have kind")
        weights = weights / weights.sum()
        fresh = min(n, int((weights > 0).sum()))
        picks = [*rng.choice(weights.size, size=fresh, replace=False, p=weights), *rng.choice(weights.size, size=n - fresh, p=weights)]
        return tuple(self.timed(*divmod(int(i), self.n_kinds), rng, shortest, longest, cap) for i in picks)

    def timed(self, first: int, second: int, rng: np.random.Generator, shortest: float, longest: float, cap: float) -> tuple[Section, Section]:
        """Durations for a pair of kinds: each from its kind's law clipped to [shortest, longest]; the second shortened to fit `cap`, but never under `shortest`."""
        def seconds(kind: int) -> float:
            mu, sigma, _, _ = self.law(kind)
            return float(np.clip(rng.lognormal(mu, sigma), shortest, longest))

        head = seconds(first)
        return Section(first, head), Section(second, min(seconds(second), max(cap - head, shortest)))

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
