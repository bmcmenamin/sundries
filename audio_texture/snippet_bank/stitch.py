"""Generate a new song of any length from a segmented one: a random walk over its section kinds, each section generated with the end of the one before as a
fixed start, all stems on the same sections, joined without a splice.

    stitch.py SONG SECONDS [SEED]       needs  segment.py SONG, segment.py SONG pool  (structure, pseudo-songs SONG__k0.., levels)

Sections   Structure.sample(SECONDS, rng): kinds from the transition chain, durations from each kind's log-normal law (structure.py).
Banks      one per (kind, stem), made on the pseudo-song with the usual fit -> snippets.py (skipped when they exist). Stems: every separated stem except vocals
           that is within STEM_RANGE_DB of the loudest over the whole song; a stem that is that far down in a kind is silent in that kind.
Clamp      section i (i > 0) is generated HANDOFF = TF frames longer than its share, and its first HANDOFF samples plus the ringing-out tail are the previous
           section's last HANDOFF samples and tail (snippet_generate.py CLAMP), in this section's units. They are constant: MERT sees them and only the new
           triggers are optimised around them. Joining keeps part i up to its last HANDOFF samples and lets part i+1 carry on from there, so nothing is
           cut, crossfaded or added twice.
Levels     each part is generated in unit-variance source units and scaled by the rms of that kind's pooled stem (segment.py levels), which restores the stem's
           level in the song and a chorus's level against a verse's. The clamp is rescaled by the ratio of the two levels.
Writes     compare/Y_{SONG}_seed{SEED}.wav (the mix of the stems) and .json (the sections).
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import snippets as sn  # noqa: E402
from segment import pseudo_song  # noqa: E402
from structure import Section, Structure  # noqa: E402

HANDOFF, TAIL = sn.TF * sn.HOP, sn.SNIPPET
STEM_RANGE_DB = 30.0
FRACTION = os.environ.get("FRACTION", "0.6")  # clusters per event, as run_banks.sh
SOURCE_SECONDS = os.environ.get("SOURCE_SECONDS", "60")  # how much of a pooled stem the fit and the bank read
STEPS = os.environ.get("ITERS", "3")  # L-BFGS steps per section


@dataclass(frozen=True)
class Part:
    """One generated section: its audio with the tail still ringing (samples + TAIL long), its length, and the level it is played at."""
    audio: torch.Tensor
    samples: int
    gain: float
    path: Path | None = None  # the saved triggers, for the next section's clamp (None: nothing to clamp onto)

    def scaled(self) -> torch.Tensor:
        return self.audio * self.gain

    def tail(self) -> torch.Tensor:
        return self.scaled()[self.samples - HANDOFF :]


def stitch(parts: list[Part], handoff: int = HANDOFF) -> torch.Tensor:
    """Part i is kept up to its last `handoff` samples when another follows (those are the next part's first), and the last part is kept whole."""
    return torch.cat([p.scaled()[: p.samples - handoff] for p in parts[:-1]] + [parts[-1].scaled()])


def padded(x: torch.Tensor, length: int) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, length - x.numel()))


def db_of(level: float) -> float:
    return 20 * float(np.log10(max(level, 1e-9)))


def stems_in(levels: dict[str, dict[str, float]]) -> list[str]:
    whole = {name: db_of(v) for name, v in levels["song"].items() if name != "vocals"}
    return [name for name, db in whole.items() if db >= max(whole.values()) - STEM_RANGE_DB]


def absent(levels: dict[str, dict[str, float]], kind: int, stem: str) -> bool:
    """A stem that is STEM_RANGE_DB below the kind's loudest stem is not played in that kind."""
    in_kind = levels[f"k{kind}"]
    return db_of(in_kind[stem]) < max(db_of(v) for n, v in in_kind.items() if n != "vocals") - STEM_RANGE_DB


def run(script: str, *args: str, **env: str) -> None:
    done = subprocess.run([sys.executable, str(HERE / script), *args], env={**os.environ, "TF": str(sn.TF), "K": str(sn.K), **env}, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{script} {' '.join(args)} failed:\n{done.stderr[-2000:]}")


def ensure_bank(pseudo: str, stem: str) -> None:
    fit = sn.ATOMS_DIR / f"{pseudo}_{stem}_attack_f0.25{sn.K_TAG}_tf{sn.TF}_sus.pt"
    if not fit.exists():
        run("sustain48.py", pseudo, stem, "attack", str(sn.K), "0.25", TEMPLATE_FRAMES=str(sn.TF), SOURCE_SECONDS=SOURCE_SECONDS)
    if not (sn.ATOMS_DIR / f"{pseudo}_{stem}_bank{sn.TAG}.pt").exists():
        run("snippets.py", pseudo, stem, FRACTION, SOURCE_SECONDS=SOURCE_SECONDS)


def generate_section(song: str, stem: str, sections: list[Section], i: int, levels: dict, seed: int, before: Part | None) -> Part:
    section = sections[i]
    samples = round(section.seconds * sn.RATE) + (HANDOFF if i else 0)
    if absent(levels, section.kind, stem):
        carried = padded(before.tail(), samples + TAIL) if before else torch.zeros(samples + TAIL)  # the previous part's ring-out, so it is not cut
        return Part(carried, samples, 1.0)
    pseudo, level = pseudo_song(song, section.kind), levels[f"k{section.kind}"][stem]
    ensure_bank(pseudo, stem)
    tag = f"{song}_{stem}_seed{seed}_s{i}"
    clamp = {"CLAMP": str(before.path), "CLAMP_GAIN": str(before.gain / level)} if before and before.path else {}
    run("snippet_generate.py", pseudo, stem, STEPS, tag, OUT_SAMPLES=str(samples), SEED=str(seed + i), **clamp)
    path = sn.ATOMS_DIR / f"N_{tag}_triggers.pt"
    return Part(torch.load(path)["audio_ext"], samples, level, path)


def stem_timeline(song: str, stem: str, sections: list[Section], levels: dict, seed: int) -> torch.Tensor:
    parts: list[Part] = []
    for i in range(len(sections)):
        parts.append(generate_section(song, stem, sections, i, levels, seed, parts[-1] if parts else None))
        print(f"  {stem} section {i + 1}/{len(sections)} done", flush=True)
    return stitch(parts)


if __name__ == "__main__":
    song, seconds = sys.argv[1], float(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    structure = Structure.load(sn.ATOMS_DIR / f"{song}_structure.json")
    levels = json.loads((sn.ATOMS_DIR / f"{song}_levels.json").read_text())
    sections = list(structure.sample(seconds, np.random.default_rng(seed)))
    stems = os.environ["STEMS"].split(",") if "STEMS" in os.environ else stems_in(levels)  # STEMS=guitar,drums overrides the level filter
    print(f"{len(sections)} sections: " + " ".join(f"{chr(65 + s.kind)}{s.seconds:.0f}s" for s in sections) + f"; stems {', '.join(stems)}")
    mix = sum(stem_timeline(song, stem, sections, levels, seed) for stem in stems)
    sn.write(sn.OUT / f"Y_{song}_seed{seed}.wav", mix)
    (sn.OUT / f"Y_{song}_seed{seed}.json").write_text(json.dumps({"sections": [[s.kind, s.seconds] for s in sections], "stems": stems}))
    print(f"wrote Y_{song}_seed{seed}.wav ({len(mix) / sn.RATE:.1f} s)")
