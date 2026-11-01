"""Rhythm statistics of a trigger grid, resolving sixteenth notes (a frame is 42.7 ms; a sixteenth at ~100-130 bpm is ~3 frames).

    rhythm_stats.py SONG STEM TAG...      gap of each statistic between the source and compare/N_{TAG}_triggers.pt, against a time-shuffled control

All statistics are scale-free (normalised by the trigger mass) and length-free, so a 30 s output is compared with a 10 s source.
  onset    correlogram of total activity, lags 1..24 frames, lightly smoothed (sigma 0.6 frame) so sixteenth-note peaks survive
  pair     joint of (cluster group A, cluster group B, signed lag -8..8 frames): the relative onset/offset of one patch against another, so patches overlap the
           way the source's do. Clusters are pooled into a few spectral groups so the table is a histogram, not the source's exact note sequence
  rhythm   power spectrum of the activity at periods 2..96 frames (85 ms..4 s), log-spaced and smoothed: every metrical level from sixteenth to ~bar, no tempo estimate
  rate     quantiles of the event rate in windows of 1, 2 and 4 s (bursts and rests: phrase breathing)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage

FINE_LAGS, PAIR_LAGS, GROUPS = 24, 8, 6
WINDOWS, QUANTILES = (23, 47, 94), 16  # frames: 1, 2, 4 s
FREQUENCIES = torch.tensor(np.geomspace(1 / 96, 1 / 2, 160), dtype=torch.float32)  # cycles per frame
BANDS = 32


def gauss(sigma: float) -> torch.Tensor:
    r = int(3 * sigma) + 1
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def smooth_last(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = gauss(sigma)
    return F.conv1d(x.reshape(-1, 1, x.shape[-1]), k[None, None], padding=len(k) // 2).reshape(x.shape)


def groups_of(atoms: torch.Tensor, n: int = GROUPS) -> torch.Tensor:
    """Cluster ids -> a few spectral groups (average linkage on log band energies of each medoid snippet)."""
    power = torch.fft.rfft(atoms, dim=-1).abs().pow(2)
    edges = np.unique(np.geomspace(4, power.shape[-1] - 1, BANDS + 1).astype(int))
    bands = np.stack([np.add.reduceat(p, edges[:-1]) for p in power.numpy()])
    z = np.log1p(bands)
    z = z - z.mean(1, keepdims=True)
    labels = fcluster(linkage(z, "average"), min(n, len(z)), "maxclust") - 1
    return torch.tensor(labels, dtype=torch.long)


def rhythm_statistics(A: torch.Tensor, group: torch.Tensor) -> dict[str, torch.Tensor]:
    """A: (clusters, frames) trigger mass; group: (clusters,) group ids."""
    a = A.sum(0)
    T = a.numel()
    total = a.sum().clamp_min(1e-9)
    density = total**2 / T
    onset = torch.stack([(a[: T - lag] * a[lag:]).sum() for lag in range(1, FINE_LAGS + 1)]) / density

    n_groups = int(group.max()) + 1
    G = torch.zeros(n_groups, A.shape[0]).index_put_((group, torch.arange(A.shape[0])), torch.ones(A.shape[0])) @ A
    forward = [G[:, : T - lag] @ G[:, lag:].T for lag in range(PAIR_LAGS + 1)]  # M_lag[a, b] = sum_t G_a[t] G_b[t + lag]
    pair = torch.stack([m.T for m in forward[:0:-1]] + forward) / density  # lags -8..8
    pair = smooth_last(pair.permute(1, 2, 0), 0.6)

    t = torch.arange(T, dtype=torch.float32)
    phase = 2 * torch.pi * FREQUENCIES[:, None] * t[None]
    power = ((torch.cos(phase) @ a) ** 2 + (torch.sin(phase) @ a) ** 2) / a.pow(2).sum().clamp_min(1e-9)
    rhythm = smooth_last(power[None], 1.5)[0]

    mean_rate = a.mean().clamp_min(1e-9)
    levels = torch.linspace(0.05, 0.95, QUANTILES)
    rate = torch.cat([torch.quantile(F.avg_pool1d(a[None, None], w, stride=max(1, w // 8))[0, 0] / mean_rate, levels) for w in WINDOWS])
    return {"onset": smooth_last(onset[None], 0.6)[0], "pair": pair, "rhythm": rhythm, "rate": rate}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snippets as sn

    song, stem, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    bank = torch.load(sn.ATOMS_DIR / f"{song}_{stem}_bank.pt", weights_only=False)
    group = groups_of(bank["atoms"])
    target = rhythm_statistics(sn.grid_of(bank).sum(1), group)
    scale = {k: float(v.pow(2).mean()) + 1e-9 for k, v in target.items()}

    def gap(H: torch.Tensor) -> dict[str, float]:
        stats = rhythm_statistics(H.sum(1), group)
        return {k: float((stats[k] - target[k]).pow(2).mean()) / scale[k] for k in target}

    torch.manual_seed(0)
    print(f"{song} {stem}: groups {np.bincount(group.numpy()).tolist()}; gap = mean square difference / mean square of the source statistic")
    for tag in tags:
        H = torch.load(sn.ATOMS_DIR / f"N_{tag}_triggers.pt")["H"]
        shuffled = H[:, :, torch.randperm(H.shape[-1])]
        print(f"{tag:14s}", "  ".join(f"{k} {v:.3f}" for k, v in gap(H).items()), " | time-shuffled:", "  ".join(f"{k} {v:.3f}" for k, v in gap(shuffled).items()))
