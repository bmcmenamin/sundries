"""Waveform snippet bank from a sparse fit: real audio patches, clustered by magnitude (phase ignored), one medoid kept per cluster.

    snippets.py SONG WHICH [CLUSTER_FRACTION]       WHICH: guitar | mix      e.g.  snippets.py jack_1 guitar 0.4

Needs the fit atoms/{SONG}_{WHICH}_attack_f0.25_tf4_sus.pt (sustain48.py, TEMPLATE_FRAMES=4).
Events      each connected cluster of the fit's trigger map: (template, pitch shift, frame, mass). An event covers frames t..t+3 (170 ms).
Patch       the source's constant-Q magnitude (compressed ^0.3) under the event's template footprint (120 bins x 4 frames, in template coordinates, so
            the same note at another pitch lines up). Phase is not in it at all.
Clusters    average-linkage on cosine distance of patches, cut to CLUSTER_FRACTION x events clusters. The MEDOID (the member closest to the others)
            is kept, never an average: its waveform is one real stretch of the source.
Atom        the medoid event's source audio, 5120 samples (the 4 frames plus 512 either side), Hann-windowed. Played at pitch shift s' it is resampled
            by 2^((s' - s_medoid) / 12) (a shift step is one semitone: 4 bins of 48 per octave).
render(H, atoms) plays a trigger grid H (clusters x pitch offsets -W..W x frames) as the sum of its atoms: linear in H, so it is differentiable.
Running this file writes the bank to atoms/{SONG}_{WHICH}_bank.pt and compare/B_ref_{SONG}_{WHICH}.wav, the source's own arrangement played back through it.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from paths import stems_of  # noqa: E402
from synthesize import decode_mono  # noqa: E402

TF = int(os.environ.get("TF", 4))  # snippet length in frames of 42.7 ms (4 = 170 ms)
K = int(os.environ.get("K", 24))  # templates in the sparse fit (sustain48.py); the fit and the bank are named by it when it is not 24
K_TAG = "" if K == 24 else f"_k{K}"
TAG = K_TAG + ("" if TF == 4 else f"_tf{TF}")  # keeps the file names used with 24 templates and 4-frame snippets
RATE, HOP, STRIDE, TEMPLATE_BINS, PAD, W, COMPRESS = 24000, 1024, 4, 120, 512, 6, 0.3
FADE = 120  # 5 ms
SNIPPET = TF * HOP + 2 * PAD  # 5120 samples
KERNEL = int(np.ceil(SNIPPET * 2 ** (W / 12)))  # the longest atom: an octave down
ATOMS_DIR = ROOT / "out/codec_texture/atoms"
OUT = ROOT / "out/codec_texture/compare"
SEED_S0 = KERNEL // 2 - TF * HOP // 2  # output offset that puts a snippet's centre at its event's centre


def source_audio(song: str, which: str, seconds: float = float(os.environ.get("SOURCE_SECONDS", 10))) -> torch.Tensor:
    """Full-band, unit-variance audio of one named stem, or of the vocal-free mix (which = "mix")."""
    stems = [decode_mono(s.path, RATE) for s in stems_of(song) if (s.name != "vocals" if which == "mix" else s.name == which)]
    n = min(len(s) for s in stems)
    wave = sum(s[:n] for s in stems)[: int(seconds * RATE)]
    return wave / wave.std()


def clips_of(song: str, n: int) -> list[tuple[int, int]]:
    """The sample ranges of a pooled pseudo-song's separate clips (segment.py pool writes them), cut to the n samples read; empty for an ordinary song."""
    path = ATOMS_DIR / f"{song}_clips.json"
    spans = json.loads(path.read_text()) if path.exists() else []
    return [(a, min(b, n)) for a, b in spans if a < n]


def events_of(x0: torch.Tensor) -> np.ndarray:
    """(n, 4) rows (template, shift, frame, mass) from the attack channel (K, shifts, frames), one per connected cluster, centroid-weighted."""
    rows = []
    for k in range(x0.shape[0]):
        a = x0[k].numpy()
        labels, n = ndimage.label(a > 0, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labels == i)
            m = a[ys, xs]
            rows.append((k, int(round((ys * m).sum() / m.sum())), int(round((xs * m).sum() / m.sum())), float(m.sum())))
    return np.array(sorted(rows, key=lambda r: (r[2], r[0])), dtype=float)


def patch_of(v: np.ndarray, shift: int, frame: int) -> np.ndarray:
    out = np.zeros((TEMPLATE_BINS, TF))
    rows = v[shift * STRIDE : shift * STRIDE + TEMPLATE_BINS, frame : frame + TF]
    out[: rows.shape[0], : rows.shape[1]] = rows
    return out


def snippet_of(audio: torch.Tensor, frame: int) -> torch.Tensor:
    start = frame * HOP - PAD
    out = torch.zeros(SNIPPET)
    lo, hi = max(start, 0), min(start + SNIPPET, audio.numel())
    if hi > lo:
        out[lo - start : hi - start] = audio[lo:hi]
    return out * edge_window()


def edge_window() -> torch.Tensor:
    """Flat body with short raised-cosine fades (FADE samples), so a snippet plays at full level instead of dipping like a Hann window."""
    ramp = torch.sin(0.5 * torch.pi * torch.arange(FADE) / FADE) ** 2
    return torch.cat([ramp, torch.ones(SNIPPET - 2 * FADE), ramp.flip(0)])


def resampled(atom: torch.Tensor, offset: int) -> torch.Tensor:
    """The atom played `offset` semitones higher, zero-padded and centred in KERNEL samples."""
    length = int(round(SNIPPET / 2 ** (offset / 12)))
    out = F.interpolate(atom[None, None], size=length, mode="linear", align_corners=False)[0, 0]
    left = (KERNEL - length) // 2
    return F.pad(out, (left, KERNEL - length - left))


def kernels_of(atoms: torch.Tensor) -> torch.Tensor:
    """(clusters * (2W+1), 1, KERNEL): every atom at every pitch offset -W..W."""
    return torch.stack([resampled(a, d - W) for a in atoms for d in range(2 * W + 1)])[:, None]


def render(H: torch.Tensor, kernels: torch.Tensor, samples: int) -> torch.Tensor:
    """H: (clusters, 2W+1, frames) -> (samples,)"""
    out = F.conv_transpose1d(H.reshape(1, -1, H.shape[-1]), kernels, stride=HOP)[0, 0]
    return (out / overlap(H).sqrt())[SEED_S0 : SEED_S0 + samples]


def overlap(H: torch.Tensor) -> torch.Tensor:
    """How many snippets sound at each sample (each trigger counted at most 1, floor 1). Detached: it is a gain schedule, so gradients still
    flow through the snippets themselves, and overlapping uncorrelated snippets sum in energy, hence the sqrt in render."""
    span = torch.zeros(1, 1, KERNEL)
    left = (KERNEL - SNIPPET) // 2
    span[..., left : left + SNIPPET] = 1
    count = F.conv_transpose1d(H.detach().clamp(0, 1).sum((0, 1))[None, None], span, stride=HOP)[0, 0]
    return count.clamp(min=1)


def write(path: Path, wave: torch.Tensor) -> None:
    x = wave.detach().numpy()
    wavfile.write(path, RATE, (x * (0.9 / max(np.abs(x).max(), 1e-9)) * 32767).astype(np.int16))


def build(song: str, which: str, fraction: float) -> dict:
    fit = torch.load(ATOMS_DIR / f"{song}_{which}_attack_f0.25{K_TAG}_tf{TF}_sus.pt")
    v = fit["V"].numpy()
    events = events_of(fit["x"][0])
    audio = source_audio(song, which)
    patches = np.stack([(patch_of(v, int(s), int(t)) ** COMPRESS).ravel() for _, s, t, _ in events])
    patches /= np.linalg.norm(patches, axis=1, keepdims=True).clip(1e-9)
    distance = (1 - patches @ patches.T).clip(0)
    np.fill_diagonal(distance, 0)
    n_clusters = max(1, round(fraction * len(events)))
    labels = fcluster(linkage(squareform(distance, checks=False), "average"), n_clusters, "maxclust") - 1
    medoids = np.array([
        np.nonzero(labels == c)[0][distance[np.ix_(labels == c, labels == c)].sum(1).argmin()] for c in range(labels.max() + 1)
    ])
    atoms = torch.stack([snippet_of(audio, int(events[m, 2])) for m in medoids])
    rms = lambda x: float(x.pow(2).mean().sqrt()) + 1e-9  # noqa: E731
    gains = np.array([rms(snippet_of(audio, int(t))) / rms(atoms[labels[i]]) for i, (_, _, t, _) in enumerate(events)])
    medoid_shift = events[medoids, 1].astype(int)
    offsets = (events[:, 1] - medoid_shift[labels]).astype(int)
    clips = clips_of(song, audio.numel())
    return {
        **({"clips": clips} if clips else {}),
        "events": events, "labels": labels, "medoids": medoids, "atoms": atoms, "medoid_shift": medoid_shift, "offsets": offsets, "gains": gains,
        "audio": audio, "frames": v.shape[1], "distance": distance,
    }


def grid_of(bank: dict) -> torch.Tensor:
    """The source's own arrangement as a trigger grid (clusters, 2W+1, frames); offsets beyond +-W are clipped."""
    H = torch.zeros(len(bank["atoms"]), 2 * W + 1, bank["frames"])
    for (_, _, t, _), c, d, g in zip(bank["events"], bank["labels"], bank["offsets"], bank["gains"]):
        H[c, int(np.clip(d, -W, W)) + W, int(t)] += g
    return H


if __name__ == "__main__":
    song, which = sys.argv[1], sys.argv[2]
    fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    bank = build(song, which, fraction)
    events, labels = bank["events"], bank["labels"]
    sizes = np.bincount(labels)
    print(f"{song} {which}: {len(events)} events -> {len(sizes)} clusters; sizes {sorted(sizes.tolist(), reverse=True)}; singletons {int((sizes == 1).sum())}")
    print(f"pitch offsets from medoid: min {bank['offsets'].min()}, max {bank['offsets'].max()}, |offset| > {W}: {int((np.abs(bank['offsets']) > W).sum())}")
    print(f"mean within-cluster distance {np.mean([bank['distance'][np.ix_(labels == c, labels == c)].mean() for c in range(len(sizes)) if sizes[c] > 1]):.3f}; mean distance overall {bank['distance'].mean():.3f}")
    kernels = kernels_of(bank["atoms"])
    audio = render(grid_of(bank), kernels, bank["audio"].numel())
    print(f"bank playback of the source's arrangement: rms {float(audio.pow(2).mean().sqrt()):.3f} vs source {float(bank['audio'].pow(2).mean().sqrt()):.3f}")
    write(OUT / f"B_ref_{song}_{which}{TAG}.wav", audio)
    write(OUT / f"B_src_{song}_{which}.wav", bank["audio"])
    torch.save({k: v for k, v in bank.items() if k != "distance"}, ATOMS_DIR / f"{song}_{which}_bank{TAG}.pt")
    print("done")
