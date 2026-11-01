"""Song structure from MERT: where the sections are, and which of them are the same kind (all the verses, all the choruses).

    segment.py SONG [TYPES]    segment the ORIGINAL recording (vocals in: they tell a verse from a chorus) -> atoms/{SONG}_structure.json
                               TYPES: how many kinds to group the segments into (default: merge segments closer than MERGE)
    segment.py SONG pool       cut the separated stems (out/separation/01_demucs/SONG, from separate.py) by that structure and write one pseudo-song per kind,
                               SONG__k0, SONG__k1, ..., whose stems are that kind's segments as SEPARATE clips (GAP_S of silence between, never faded together;
                               the clip ranges are in atoms/{pseudo}_clips.json); and atoms/{SONG}_levels.json (rms per stem, in the recording's own units,
                               for the whole song and for each kind). The fit and bank run on the pseudo-songs unchanged; snippet_generate.py takes its source
                               statistics clip by clip.

Boundaries   MERT hidden states (mean of LAYERS, z-scored per unit over the song) pooled into BLOCK_S blocks; cosine self-similarity; Foote's checkerboard
             novelty at HALF_BLOCKS; peaks at least MIN_BLOCKS apart. Kinds: average-linkage on the cosine distance of the segments' mean features.
The thresholds are untuned guesses: the script prints the segment map and the distance matrix so they can be judged against what you hear.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from paths import Stem, input_name, stems_of  # noqa: E402
from structure import Segment, Structure  # noqa: E402
from synthesize import decode_mono  # noqa: E402

RATE, MERT_HOP = 24000, 320  # MERT-v1-95M: 24 kHz in, 75 frames per second out
LAYERS = (5, 6, 7, 8)  # mid layers: instrumentation and harmony rather than raw acoustics or the highest abstractions
BLOCK_S, PROMINENCE = 0.5, float(os.environ.get("PROMINENCE", 0.5))  # blocks of 0.5 s; peak prominence in sd of the novelty
HALF_BLOCKS = int(os.environ.get("HALF_BLOCKS", 16))  # the novelty kernel reaches this many blocks (8 s) either side of a boundary
MIN_BLOCKS = int(os.environ.get("MIN_BLOCKS", 24))  # sections at least this many blocks (12 s) long
MERGE = 0.5  # cosine distance under which two segments are one kind
WINDOW_S, CONTEXT_S = 20, 2  # MERT is run on windows this long (plus this much context each side)
GAP_S, EDGE_S = 4.5, 0.01  # silence between pooled clips: longer than any window the trigger statistics look across (96 frames = 4.1 s); click guard at a clip's ends
INPUTS = ROOT / "inputs"
ATOMS_DIR = ROOT / "out/codec_texture/atoms"


def pseudo_song(song: str, kind: int) -> str:
    return f"{song}__k{kind}"


def recording(song: str) -> torch.Tensor:
    return decode_mono(next(p for p in INPUTS.iterdir() if input_name(p) == song), RATE)


# ------------------------------------------------------------------------------------------------------------------------------------ features
def mert_blocks(wave: torch.Tensor, device: str) -> np.ndarray:
    """(blocks, 768): MERT features in BLOCK_S blocks, z-scored per unit. Windows of WINDOW_S with CONTEXT_S of context either side (attention is quadratic)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval().to(device)
    scaled = wave / wave.std()  # one scale for the whole song, so loudness differences between sections survive
    captured: dict[int, torch.Tensor] = {}

    class Done(Exception):
        pass

    def capture(k: int):
        def hook(_m, _i, output):
            captured[k] = output[0] if isinstance(output, tuple) else output
            if k == max(LAYERS):
                raise Done  # the layers above the last one used are not needed

        return hook

    for k in LAYERS:
        model.encoder.layers[k - 1].register_forward_hook(capture(k))

    def window(start: int) -> np.ndarray:
        lo, hi = max(start - CONTEXT_S * RATE, 0), min(start + (WINDOW_S + CONTEXT_S) * RATE, len(scaled))
        with torch.no_grad():
            try:
                model(scaled[lo:hi][None].to(device))
            except Done:
                pass
        frames = torch.stack([captured[k] for k in LAYERS])[:, 0].mean(0).cpu().numpy()
        first = (start - lo) // MERT_HOP
        return frames[first : first + WINDOW_S * RATE // MERT_HOP]

    features = np.concatenate([window(s) for s in range(0, len(scaled), WINDOW_S * RATE)])
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    return blocked(features, MERT_HOP / RATE, BLOCK_S)


def blocked(frames: np.ndarray, frame_s: float, block_s: float) -> np.ndarray:
    index = (np.arange(len(frames)) * frame_s / block_s).astype(int)
    sums = np.zeros((index[-1] + 1, frames.shape[1]))
    np.add.at(sums, index, frames)
    return sums / np.bincount(index)[:, None]


# ------------------------------------------------------------------------------------------------------------------------------- boundaries
def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def checkerboard(half: int) -> np.ndarray:
    """Foote's kernel: +1 where both indices are on the same side of the centre, -1 where they are on different sides, tapered by a Gaussian."""
    r = np.arange(-half, half) + 0.5
    taper = np.exp(-0.5 * (r / (0.5 * half)) ** 2)
    return np.outer(np.sign(r), np.sign(r)) * np.outer(taper, taper)


def novelty(similarity: np.ndarray, half: int) -> np.ndarray:
    """How much the self-similarity looks like two different blocks meeting, at each block."""
    n = len(similarity)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(similarity, half, mode="reflect"), (2 * half, 2 * half))
    return np.einsum("iab,ab->i", windows[np.arange(n), np.arange(n)], checkerboard(half))


def boundaries(curve: np.ndarray) -> np.ndarray:
    """Peak positions (in blocks), at least MIN_BLOCKS apart and from either end."""
    peaks, _ = find_peaks(curve, distance=MIN_BLOCKS, prominence=PROMINENCE * curve.std())
    return peaks[(peaks >= MIN_BLOCKS) & (peaks <= len(curve) - MIN_BLOCKS)]


# ------------------------------------------------------------------------------------------------------------------------------------ kinds
def kinds_of(means: np.ndarray, types: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(kind per segment numbered by first appearance, cosine distance matrix of the segments)."""
    unit = unit_rows(means)
    distance = (1 - unit @ unit.T).clip(0)
    np.fill_diagonal(distance, 0)
    if len(means) < 2:
        return np.zeros(len(means), dtype=int), distance
    tree = linkage(squareform(distance, checks=False), "average")
    labels = fcluster(tree, types, criterion="maxclust") if types else fcluster(tree, MERGE, criterion="distance")
    order = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    return np.array([order[label] for label in labels]), distance


def structure_of(blocks: np.ndarray, types: int | None) -> tuple[Structure, np.ndarray]:
    edges = np.concatenate([[0], boundaries(novelty(unit_rows(blocks) @ unit_rows(blocks).T, HALF_BLOCKS)), [len(blocks)]])
    means = np.stack([blocks[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])])
    kinds, distance = kinds_of(means, types)
    return Structure(tuple(Segment(a * BLOCK_S, b * BLOCK_S, int(k)) for a, b, k in zip(edges[:-1], edges[1:], kinds))), distance


def describe(structure: Structure, distance: np.ndarray) -> str:
    letter = lambda k: chr(ord("A") + k)  # noqa: E731
    rows = [f"  {letter(s.kind)}  {s.start:6.1f} - {s.end:6.1f} s  ({s.seconds:.1f} s)" for s in structure.segments]
    matrix = "\n".join("  " + " ".join(f"{d:.2f}" for d in row) for row in distance)
    return f"{len(structure.segments)} segments, {structure.n_kinds} kinds\n" + "\n".join(rows) + "\ncosine distance between segments:\n" + matrix


# ---------------------------------------------------------------------------------------------------------------------------------- pooling
def guarded(piece: torch.Tensor) -> torch.Tensor:
    """A clip with 10 ms fades at its two ends, only so that cutting it out of a song does not click."""
    n = min(int(EDGE_S * RATE), len(piece) // 2)
    ramp = torch.sin(0.5 * torch.pi * torch.arange(n) / n) ** 2
    return torch.cat([piece[:n] * ramp, piece[n : len(piece) - n], piece[len(piece) - n :] * ramp.flip(0)])


def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def pool(song: str, structure: Structure) -> dict[str, dict[str, float]]:
    """Write the pseudo-songs: each kind's segments as separate clips laid end to end with GAP_S of silence between them (no fades, no overlap), and the clips'
    sample ranges in atoms/{pseudo}_clips.json. Everything that looks across time (events, trigger statistics, MERT) then works clip by clip.
    Returns the levels {"song": {stem: rms}, "k0": {stem: rms}, ...}, measured on the clips only."""
    waves = {stem.name: decode_mono(stem.path, RATE) for stem in stems_of(song)}
    if not waves:
        raise SystemExit(f"no separated stems for {song}: run separate.py first")
    gap = torch.zeros(int(GAP_S * RATE))
    levels: dict[str, dict[str, float]] = {"song": {name: rms(w) for name, w in waves.items()}}
    for kind in range(structure.n_kinds):
        spans = [(int(s.start * RATE), int(s.end * RATE)) for s in structure.segments if s.kind == kind]
        starts = np.cumsum([0] + [b - a + len(gap) for a, b in spans[:-1]])
        (ATOMS_DIR / f"{pseudo_song(song, kind)}_clips.json").write_text(json.dumps([(int(c), int(c) + b - a) for c, (a, b) in zip(starts, spans)]))
        clips = {name: [guarded(w[a:b]) for a, b in spans] for name, w in waves.items()}
        for name, pieces in clips.items():
            path = Stem(pseudo_song(song, kind), name).path
            path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(path, RATE, torch.cat([x for piece in pieces for x in (piece, gap)][:-1]).numpy().astype(np.float32))
        levels[f"k{kind}"] = {name: rms(torch.cat(pieces)) for name, pieces in clips.items()}
    return levels


if __name__ == "__main__":
    song = sys.argv[1]
    structure_path = ATOMS_DIR / f"{song}_structure.json"
    if len(sys.argv) > 2 and sys.argv[2] == "pool":
        levels = pool(song, Structure.load(structure_path))
        (ATOMS_DIR / f"{song}_levels.json").write_text(json.dumps(levels))
        print(f"wrote pseudo-songs {', '.join(pseudo_song(song, k) for k in range(len(levels) - 1))} and {song}_levels.json")
    else:
        types = int(sys.argv[2]) if len(sys.argv) > 2 else None
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        structure, distance = structure_of(mert_blocks(recording(song), device), types)
        structure.save(structure_path)
        print(describe(structure, distance))
        print(f"wrote {structure_path.name}")
