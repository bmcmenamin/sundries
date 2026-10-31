"""Weights for the terms of `tfd.Statistics`, from damage probes on real stems.

    venv/bin/python calibrate.py

For each stem, the gap of every term from the stem's statistics to (a) its own other half: what sampling noise alone does to the term,
and (b) damaged versions of it: time-reversed, random phase, shuffled in 100 ms blocks, shifted up 3 semitones. A term's weight is
how far its best-detected damage stands above its noise floor, in units of that floor:

    weight = max(MIN_USEFUL, log2(largest median ratio of damaged gap to noise gap)) / max(noise floor, MIN_NOISE)   (at most MAX_WEIGHT)

so a term that is noisy between two halves of the same sound hardly counts, and one that is quiet when the sound is the same and loud
when it is damaged counts most. Written to term_weights.json, which tfd.py reads. Rerun after changing the terms.
"""

import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import codec
import tfd
from paths import stems_of
from synthesize import decode_mono

STEMS = (
    "jack_1:guitar", "jack_2:guitar", "jack_3:guitar", "hanging_2:guitar", "walk_2:guitar", "walk_1:guitar",
    "hanging_1:drums", "hanging_2:drums", "hanging_2:piano", "walk_1:piano", "walk_2:other", "walk_2:drums", "walk_2:bass", "jack_1:drums",
)
FRAMES = 750
MIN_USEFUL = 0.5  # a term never counts for less than this many bits of detection
MIN_NOISE = 0.02
MAX_WEIGHT = 100.0
OUT = Path(__file__).parent / "term_weights.json"


def load(spec: str) -> torch.Tensor:
    song, stem = spec.split(":")
    return codec.normalise(decode_mono(next(s.path for s in stems_of(song) if s.name == stem), codec.CODEC_RATE)[: FRAMES * codec.HOP])


def random_phase(x: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True)
    phase = torch.rand(spec.shape, generator=torch.Generator().manual_seed(0)) * 2 * torch.pi
    return codec.normalise(torch.istft(spec.abs() * torch.exp(1j * phase), 1024, 256, window=torch.hann_window(1024), length=len(x)))


def shuffled(x: torch.Tensor, block: int) -> torch.Tensor:
    pieces = x[: len(x) // block * block].reshape(-1, block)
    return codec.normalise(pieces[torch.randperm(len(pieces), generator=torch.Generator().manual_seed(0))].flatten())


def shifted(x: torch.Tensor, semitones: float) -> torch.Tensor:
    return codec.normalise(F.interpolate(x[None, None], scale_factor=2 ** (-semitones / 12), mode="linear")[0, 0])


DAMAGE = {
    "reversed": lambda x: x.flip(0), "random phase": random_phase, "shuffled 100 ms": lambda x: shuffled(x, 2400),
    "+3 semitones": lambda x: shifted(x, 3.0),
}


def terms_of(x: torch.Tensor, target: tfd.Statistics) -> dict[str, float]:
    return {name: float(gap) for name, gap in tfd.Statistics.of(x, like=target).terms(target).items()}


def main() -> None:
    same: dict[str, dict[str, float]] = {}
    damaged: dict[str, dict[str, dict[str, float]]] = {kind: {} for kind in DAMAGE}
    with torch.no_grad():
        for spec in STEMS:
            x = load(spec)
            target = tfd.Statistics.of(x)
            halves = [terms_of(h, target) for h in (x[len(x) // 2 :], x[: len(x) // 2])]
            same[spec] = {name: statistics.mean(h[name] for h in halves) for name in halves[0]}
            for kind, damage in DAMAGE.items():
                damaged[kind][spec] = terms_of(damage(x), target)
            print(spec, file=sys.stderr, flush=True)

    weights, rows = {}, []
    for name in next(iter(same.values())):
        noise = statistics.median(same[s][name] for s in STEMS)
        ratios = [statistics.median(damaged[kind][s][name] / (same[s][name] + 1e-6) for s in STEMS) for kind in DAMAGE]
        useful = max(MIN_USEFUL, torch.log2(torch.tensor(max(ratios))).item())
        weights[name] = round(min(MAX_WEIGHT, useful / max(noise, MIN_NOISE)), 3)
        rows.append((name, noise, ratios, weights[name]))
    OUT.write_text(json.dumps(weights, indent=1))
    print(f"{'term':28s} {'noise':>7s}" + "".join(f"{k:>16s}" for k in DAMAGE) + f"{'weight':>9s}")
    for name, noise, ratios, weight in rows:
        print(f"{name:28s} {noise:7.3f}" + "".join(f"{r:16.1f}" for r in ratios) + f"{weight:9.1f}")

    def total(gaps: dict[str, float]) -> float:
        return sum(weights[n] * g for n, g in gaps.items())

    print("\nweighted total gap, median over stems (same = noise floor)")
    print(f"  {'same':16s} {statistics.median(total(same[s]) for s in STEMS):8.1f}")
    for kind in DAMAGE:
        print(f"  {kind:16s} {statistics.median(total(damaged[kind][s]) for s in STEMS):8.1f}")
    print(f"written to {OUT}")


main()
