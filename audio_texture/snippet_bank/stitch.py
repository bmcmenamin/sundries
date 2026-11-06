"""Generate short tracks from a segmented song: each track is two consecutive sections (10-20 s each, 30 s at most), the second generated with the end of the
first as a fixed start, all stems on the same sections, joined without a splice. A whole song is many small, independent generations: MERT's attention memory
grows with the square of the length, so one long generation stalls a 16 GB machine.

    stitch.py SONG TRACKS [FIRST_SEED]  needs  segment.py SONG, segment.py SONG pool  (structure, pseudo-songs SONG__k0.., levels)
                                        track n (seed FIRST_SEED + n) -> compare/Y_{SONG}_pair{seed}.wav and .json; its intermediates are deleted afterwards (KEEP=1 keeps them)

Sections   Structure.pair(rng): the first kind uniformly, the second from the transition chain without the kinds more than MAX_STEP_DB louder or quieter, durations from each kind's log-normal law (structure.py).
Banks      one per (kind, stem), made on the pseudo-song with the usual fit -> snippets.py (skipped when they exist). Stems: every separated stem except vocals
           that is within STEM_RANGE_DB of the loudest over the whole song; a stem that is that far down in a kind is silent in that kind.
Clamp      section i (i > 0) is generated HANDOFF = TF frames longer than its share, and its first HANDOFF samples plus the ringing-out tail are the previous
           section's last HANDOFF samples and tail (snippet_generate.py CLAMP), in this section's units. They are constant: MERT sees them and only the new
           triggers are optimised around them. Joining keeps part i up to its last HANDOFF samples and lets part i+1 carry on from there, so nothing is
           cut, crossfaded or added twice.
Levels     each part is generated in unit-variance source units and scaled by the rms of that kind's pooled stem (segment.py levels), which restores the stem's
           level in the song and a chorus's level against a verse's. The clamp is rescaled by the ratio of the two levels.
Writes     compare/Y_{SONG}_pair{SEED}.wav (the mix of the stems) and .json (the sections), one per track.
"""
import json
import os
import subprocess
import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mixchain  # noqa: E402
import snippets as sn  # noqa: E402
from segment import pseudo_song  # noqa: E402
from structure import Section, Structure  # noqa: E402

HANDOFF, TAIL = sn.TF * sn.HOP, sn.SNIPPET
STEM_RANGE_DB = 30.0
FRACTION = os.environ.get("FRACTION", "0.6")  # clusters per event, as run_banks.sh
SOURCE_SECONDS = os.environ.get("SOURCE_SECONDS", "60")  # how much of a pooled stem the fit and the bank read
STEPS = os.environ.get("ITERS", "3")  # L-BFGS steps per section
MAX_STEP_DB = 6.0  # no track joins two kinds whose levels differ by more than this
MIX_CHAIN = {"drums": f"compressor 8:1, {mixchain.DRUMS_DB:g} dB", "bass": f"high-pass {mixchain.BASS_HIGHPASS_HZ} Hz, {mixchain.BASS_DB:g} dB", "mix": f"rms {mixchain.MIX_RMS_DBFS:g} dBFS"}  # mixchain.py
SPARSE_HITS = 2 # a stem with fewer events than MIN_EVENTS plays at least this many random hits per section
MIN_EVENTS = 5 # a bank built from fewer events than this is not played (bass is sparse: long notes, 7 events in a 16 s kind)


@dataclass(frozen=True)
class Part:
    """One generated section: its audio with the tail still ringing (samples + TAIL long), its length, and the level it is played at."""
    audio: torch.Tensor
    samples: int
    gain: float
    path: Path | None = None  # the saved triggers, for the next section's clamp (None: nothing to clamp onto)
    skipped: str | None = None  # why a stem that should have played did not (a failed bank or generation); None when it played or was rightly absent
    fallback: str | None = None  # set when the stem played by the sparse fallback (too few events for the statistics), so the track's json says so

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


def kinds_within(levels: dict[str, dict[str, float]], limit_db: float = MAX_STEP_DB) -> np.ndarray:
    """[a, b] is True when kind b is at most `limit_db` louder or quieter than kind a (rms of the non-vocal stems, power-summed): a jump bigger than that
    would be a volume cliff at the join, so Structure.pair does not draw it."""
    kinds = sorted(int(name[1:]) for name in levels if name != "song")
    db = np.array([10 * np.log10(sum(v**2 for s, v in levels[f"k{k}"].items() if s != "vocals")) for k in kinds])
    return np.abs(db[:, None] - db[None, :]) <= limit_db


def absent(levels: dict[str, dict[str, float]], kind: int, stem: str) -> bool:
    """A stem that is STEM_RANGE_DB below the kind's loudest stem is not played in that kind."""
    in_kind = levels[f"k{kind}"]
    return db_of(in_kind[stem]) < max(db_of(v) for n, v in in_kind.items() if n != "vocals") - STEM_RANGE_DB


class StageError(RuntimeError):
    """A pipeline stage (fit, bank, generation) failed; what is lost is one stem in one section, not the whole run."""


def run(script: str, *args: str, **env: str) -> None:
    done = subprocess.run([sys.executable, str(HERE / script), *args], env={**os.environ, "TF": str(sn.TF), "K": str(sn.K), **env}, capture_output=True, text=True)
    if done.returncode:
        raise StageError(f"{script} {' '.join(args)} failed:\n{done.stderr[-2000:]}")


def ensure_bank(pseudo: str, stem: str) -> None:
    fit = sn.ATOMS_DIR / f"{pseudo}_{stem}_attack_f0.25{sn.K_TAG}_tf{sn.TF}_sus.pt"
    if not fit.exists():
        run("sustain48.py", pseudo, stem, "attack", str(sn.K), "0.25", TEMPLATE_FRAMES=str(sn.TF), SOURCE_SECONDS=SOURCE_SECONDS)
    if not (sn.ATOMS_DIR / f"{pseudo}_{stem}_bank{sn.TAG}.pt").exists():
        run("snippets.py", pseudo, stem, FRACTION, SOURCE_SECONDS=SOURCE_SECONDS)


def reason_unusable(pseudo: str, stem: str) -> str | None:
    """Why this (kind, stem) cannot be played, or None when its bank is good. Making the bank is the slow step and a bad one stays bad, so a failure or a
    bank with fewer than MIN_EVENTS events is written next to the fits and not tried again (delete the file to retry)."""
    marker = sn.ATOMS_DIR / f"{pseudo}_{stem}_unusable{sn.TAG}.txt"
    if marker.exists():
        return marker.read_text()
    try:
        ensure_bank(pseudo, stem)
        n_events = len(torch.load(sn.ATOMS_DIR / f"{pseudo}_{stem}_bank{sn.TAG}.pt", weights_only=False)["events"])
        reason = f"only {n_events} events in the fit (need {MIN_EVENTS})" if n_events < MIN_EVENTS else None
    except (StageError, OSError) as error:
        reason = f"no bank: {str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__}"
    if reason:
        marker.write_text(reason)
    return reason


@lru_cache(maxsize=None)
def usable(pseudo: str, stem: str) -> str | None:
    return reason_unusable(pseudo, stem)


def silent(sections: list[Section], i: int, before: Part | None, skipped: str | None = None) -> Part:
    """A stem that does not play in this section: only the previous part's ring-out, so it is not cut."""
    samples = round(sections[i].seconds * sn.RATE) + (HANDOFF if i else 0)
    carried = padded(before.tail(), samples + TAIL) if before else torch.zeros(samples + TAIL)
    return Part(carried, samples, 1.0, skipped=skipped)


def sparse_part(pseudo: str, stem: str, sections: list[Section], i: int, level: float, seed: int, before: Part | None, reason: str) -> Part:
    """A stem whose fit found too few events for any statistics (a bass that plays a few long notes): its own events dropped at random times, ignoring every
    timing and placement statistic. At least SPARSE_HITS of them, more when the source's event rate asks for it; played at the stem's level (the bank's audio is in
    unit-variance source units, so `level` restores the stem's own scale and nothing is pinned to unit rms, which would blow a couple of hits up)."""
    samples = round(sections[i].seconds * sn.RATE) + (HANDOFF if i else 0)
    bank = torch.load(sn.ATOMS_DIR / f"{pseudo}_{stem}_bank{sn.TAG}.pt", weights_only=False)
    rng = np.random.default_rng([seed, i, zlib.crc32(stem.encode())])
    rate = len(bank["events"]) / (bank["audio"].numel() / sn.RATE)  # events per second of source
    hits = max(SPARSE_HITS, round(rate * sections[i].seconds))
    first = HANDOFF if i else 0  # the first HANDOFF samples are the previous section's ring-out
    played = torch.zeros(samples + TAIL)
    for start, event in zip(rng.integers(first, max(samples - sn.SNIPPET, first + 1), hits), rng.integers(0, len(bank["events"]), hits)):
        snippet = bank["atoms"][bank["labels"][event]] * float(bank["gains"][event])
        played[start : start + snippet.numel()] += snippet[: played.numel() - start]
    pinned = played * level / played[first:samples].pow(2).mean().sqrt().clamp_min(1e-8)  # the section's rms is the stem's, as for every generated part
    carried = padded(before.tail(), samples + TAIL) if before else torch.zeros(samples + TAIL)
    return Part(carried + pinned, samples, 1.0, fallback=f"{hits} random hits ({reason})")


def generate_section(song: str, stem: str, sections: list[Section], i: int, levels: dict, seed: int, before: Part | None) -> Part:
    section = sections[i]
    samples = round(section.seconds * sn.RATE) + (HANDOFF if i else 0)
    if absent(levels, section.kind, stem):
        return silent(sections, i, before)
    pseudo, level = pseudo_song(song, section.kind), levels[f"k{section.kind}"][stem]
    if reason := usable(pseudo, stem):
        if reason.startswith("only "):  # the bank exists but has fewer than MIN_EVENTS events
            return sparse_part(pseudo, stem, sections, i, level, seed, before, reason)
        return silent(sections, i, before, skipped=reason)
    tag = f"{song}_{stem}_seed{seed}_s{i}"
    clamp = {"CLAMP": str(before.path), "CLAMP_GAIN": str(before.gain / level)} if before and before.path else {}
    try:
        run("snippet_generate.py", pseudo, stem, STEPS, tag, OUT_SAMPLES=str(samples), SEED=str(10 * seed + i), **clamp)  # distinct per track and section, so no two tracks start from the same random seed events
        path = sn.ATOMS_DIR / f"N_{tag}_triggers.pt"
        return Part(torch.load(path)["audio_ext"], samples, level, path)
    except (StageError, OSError) as error:
        return silent(sections, i, before, skipped=f"generation failed: {str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__}")


def stem_timeline(song: str, stem: str, sections: list[Section], levels: dict, seed: int) -> tuple[torch.Tensor, list[str], list[str]]:
    """The stem's audio over all sections, what was skipped in it (so a skipped stem is reported, never silent) and where the sparse fallback played."""
    parts: list[Part] = []
    for i in range(len(sections)):
        parts.append(generate_section(song, stem, sections, i, levels, seed, parts[-1] if parts else None))
        print(f"  {stem} section {i + 1}/{len(sections)} " + (f"SKIPPED: {parts[-1].skipped}" if parts[-1].skipped else f"sparse fallback: {parts[-1].fallback}" if parts[-1].fallback else "done"), flush=True)
    return (
        stitch(parts),
        [f"{stem} section {i}: {p.skipped}" for i, p in enumerate(parts) if p.skipped],
        [f"{stem} section {i}: {p.fallback}" for i, p in enumerate(parts) if p.fallback],
    )


def discard_intermediates(song: str, seed: int) -> None:
    """The per-section triggers and wavs of one track: only the next section's clamp needs them, and only while the track is being made."""
    for path in (*sn.ATOMS_DIR.glob(f"N_{song}_*_seed{seed}_s*_triggers.pt"), *sn.OUT.glob(f"N_{song}_*_seed{seed}_s*.wav")):
        path.unlink()


def prepare_banks(song: str, sections: list[Section], levels: dict, stems: list[str]) -> None:
    """Every bank the track needs, before any generation: a bank that cannot be made shows up (and is marked) in minutes, not after hours of generation."""
    for section in sections:
        for stem in stems:
            if not absent(levels, section.kind, stem):
                usable(pseudo_song(song, section.kind), stem)


def make_track(song: str, levels: dict, stems: list[str], seed: int, sections: list[Section]) -> None:
    print(f"track {seed}: " + " ".join(f"{chr(65 + s.kind)}{s.seconds:.0f}s" for s in sections) + f"; stems {', '.join(stems)}", flush=True)
    prepare_banks(song, sections, levels, stems)
    timelines = [stem_timeline(song, stem, sections, levels, seed) for stem in stems]
    skipped = [line for _, lines, _ in timelines for line in lines]
    sparse = [line for _, _, lines in timelines for line in lines]
    raw = sum(audio for audio, _, _ in timelines)  # every stem at its own level
    scale = 0.9 / max(float(raw.abs().max()), 1e-9)
    for stem, (audio, _, _) in zip(stems, timelines):  # one wav per stem at a shared scale (json: stem_scale), so a mix can be redone without generating again
        (sn.OUT / "stems").mkdir(parents=True, exist_ok=True)
        sn.wavfile.write(sn.OUT / "stems" / f"Y_{song}_pair{seed}_{stem}.wav", sn.RATE, (audio * scale * 32767).clamp(-32768, 32767).to(torch.int16).numpy())
    mix = mixchain.mix({stem: audio.detach().numpy() for stem, (audio, _, _) in zip(stems, timelines)}, sn.RATE)
    sn.wavfile.write(sn.OUT / f"Y_{song}_pair{seed}.wav", sn.RATE, (mix * 32767).astype(np.int16))
    (sn.OUT / f"Y_{song}_pair{seed}.json").write_text(json.dumps({
        "sections": [[s.kind, s.seconds] for s in sections], "stems": stems, "skipped": skipped, "sparse_fallback": sparse, "mix_chain": MIX_CHAIN, "stem_scale": scale,
    }))
    print(f"wrote Y_{song}_pair{seed}.wav ({len(mix) / sn.RATE:.1f} s)" + (f"; {len(skipped)} stem-sections SKIPPED (see the json)" if skipped else ""), flush=True)
    if not os.environ.get("KEEP"):
        discard_intermediates(song, seed)


if __name__ == "__main__":
    song, tracks = sys.argv[1], int(sys.argv[2])
    first_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    structure = Structure.load(sn.ATOMS_DIR / f"{song}_structure.json")
    levels = json.loads((sn.ATOMS_DIR / f"{song}_levels.json").read_text())
    wanted = os.environ["STEMS"].split(",") if "STEMS" in os.environ else stems_in(levels)  # STEMS=guitar,drums overrides the level filter
    stems = [s for s in wanted if s in levels["song"] and s != "vocals"]  # vocals never; a name the separation did not make is dropped, loudly
    if dropped := set(wanted) - set(stems):
        print(f"ignoring stems with no level entry (or vocals): {', '.join(sorted(dropped))}", flush=True)
    if not stems:
        raise SystemExit(f"{song}: no stem to generate")
    for seed in range(first_seed, first_seed + tracks):
        make_track(song, levels, stems, seed, list(structure.pair(np.random.default_rng(seed), kinds_within(levels))))
