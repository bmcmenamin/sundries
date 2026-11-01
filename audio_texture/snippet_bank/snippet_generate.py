"""Arrange real snippets so that the trigger statistics AND MERT's activation statistics match the source's. Only the triggers are optimised.

    snippet_generate.py SONG WHICH ITERS TAG        env: NO_MERT=1 (trigger statistics only), NO_TRIG=1 (MERT only), SEED, TRIG_WEIGHT (default 1)

Bank: atoms/{SONG}_{WHICH}_bank.pt (snippets.py). Variable: H (clusters x pitch offsets x frames), non-negative, free only on a support of 3x3 cells
(pitch offset +-1, frame +-1) around 1.2 x as many random seed events as the source has; seeds drawn from the source's cluster usage, offsets and gains,
times uniform. Audio = sum of atoms (snippets.render), so gradients reach H from MERT and from the trigger statistics alike.
Trigger statistics (all scale-free, compared as mean-square difference over the source's scale; lag axes smoothed so they are histograms, not fingerprints):
  usage      share of triggers per cluster          offsets    share per pitch offset            pitch   share per absolute pitch
  delay      correlogram of total activity, lags 1..48 frames (smoothed 2 frames)
  interval   pitch interval x delay joint (interval -12..12, lag 1..24; smoothed 1 and 2)
MERT: the all-layer, coarse-to-fine setup of mert_invert.py (per-unit sorted histograms, 256 random projections, projection changes at lags 1, 4, 16; layer k
counts from 60% * k/12 of the run). Writes compare/N_{TAG}.wav (checkpoints every 100 steps), prints a copy check against the source's own events.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel  # noqa: E402

import rhythm_stats as rs  # noqa: E402
import snippets as sn  # noqa: E402

song, which, iters, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
NO_MERT, NO_TRIG, NO_RHYTHM = bool(os.environ.get("NO_MERT")), bool(os.environ.get("NO_TRIG")), bool(os.environ.get("NO_RHYTHM"))
SEED, TRIG_WEIGHT = int(os.environ.get("SEED", 0)), float(os.environ.get("TRIG_WEIGHT", 1.0))
LAYERS, PROJECTIONS, LAGS, RAMP = list(range(13)), 256, (1, 4, 16), 0.15
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

bank = torch.load(sn.ATOMS_DIR / f"{song}_{which}_bank{sn.TAG}.pt", weights_only=False)
events, labels, offsets, gains = bank["events"], bank["labels"], bank["offsets"], bank["gains"]
atoms, medoid_shift, T = bank["atoms"], bank["medoid_shift"], bank["frames"]
C, D, S_ABS = len(atoms), 2 * sn.W + 1, int(medoid_shift.max()) + 2 * sn.W + 1
OUT_SECONDS = float(os.environ.get("OUT_SECONDS", 10))
samples = int(OUT_SECONDS * sn.RATE)  # the output may be longer than the source: every statistic is length-free
T_OUT = samples // sn.HOP + 1
kernels = sn.kernels_of(atoms)
H_src = sn.grid_of(bank)
group = rs.groups_of(atoms)  # spectral groups of the clusters, for the pair-lag joint

# ----------------------------------------------------------------------------------------------------------------- trigger statistics
to_abs = torch.zeros(C * D, S_ABS)
for c in range(C):
    for d in range(D):
        to_abs[c * D + d, max(int(medoid_shift[c]) + d - sn.W, 0)] = 1.0


def gauss(sigma: float) -> torch.Tensor:
    r = int(3 * sigma) + 1
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


G2, G1 = gauss(2.0), gauss(1.0)


def smooth_last(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv1d(x.reshape(-1, 1, x.shape[-1]), kernel[None, None], padding=len(kernel) // 2).reshape(x.shape)


def trigger_statistics(H: torch.Tensor) -> dict[str, torch.Tensor]:
    A = H.sum(1)
    a = A.sum(0)
    total = a.sum().clamp_min(1e-9)
    T = H.shape[-1]
    P = to_abs.T @ H.reshape(C * D, T)  # (absolute pitch, frames)
    lags = torch.arange(1, 49)
    delay = torch.stack([(a[:-int(l)] * a[int(l) :]).sum() for l in lags]) / (total**2 / T)
    joint = torch.stack([
        torch.stack([torch.diagonal(P[:, : -int(l)] @ P[:, int(l) :].T, offset=delta).sum() for delta in range(-12, 13)])
        for l in range(1, 25)
    ]) / (total**2 / T)  # (lag, interval)
    stats = {
        "usage": A.sum(1) / total, "offsets": H.sum((0, 2)) / total, "pitch": P.sum(1) / total,
        "delay": smooth_last(delay[None], G2)[0], "interval": smooth_last(smooth_last(joint, G1).T, G2).T,
    }
    return stats if NO_RHYTHM else {**stats, **rs.rhythm_statistics(A, group)}


with torch.no_grad():
    trig_target = trigger_statistics(H_src)
trig_scale = {k: float(v.pow(2).mean()) + 1e-9 for k, v in trig_target.items()}


def trigger_loss(H: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    stats = trigger_statistics(H)
    parts = {k: (stats[k] - trig_target[k]).pow(2).mean() / trig_scale[k] for k in trig_target}
    return sum(parts.values()) / len(parts), {k: float(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------------------------------------------------------- MERT
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval()
for p in model.parameters():
    p.requires_grad_(False)
captured: dict[int, torch.Tensor] = {}
model.feature_projection.register_forward_hook(lambda _m, _i, o: captured.__setitem__(0, o[0] if isinstance(o, tuple) else o))
for k, layer in enumerate(model.encoder.layers, start=1):
    layer.register_forward_hook(lambda _m, _i, o, k=k: captured.__setitem__(k, o[0] if isinstance(o, tuple) else o))


def activations(wave: torch.Tensor) -> dict[int, torch.Tensor]:
    model(((wave - wave.mean()) / (wave.std() + 1e-8))[None])
    return {k: captured[k][0].T for k in LAYERS}


with torch.no_grad():
    target_acts = activations(bank["audio"])
mean = {k: a.mean(1, keepdim=True) for k, a in target_acts.items()}
std = {k: a.std(1, keepdim=True) + 1e-6 for k, a in target_acts.items()}
projection = {k: F.normalize(torch.randn(PROJECTIONS, a.shape[0]), dim=1) for k, a in target_acts.items()}


def mert_statistics(acts: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, a in acts.items():
        z = (a - mean[k]) / std[k]
        p = projection[k] @ z
        out[f"{k}/units"] = z.sort(1).values
        out[f"{k}/joint"] = p.sort(1).values
        for lag in LAGS:
            out[f"{k}/motion{lag}"] = (p[:, lag:] - p[:, :-lag]).sort(1).values
    return out


with torch.no_grad():
    mert_target = mert_statistics(target_acts)
mert_scale = {k: float(t.pow(2).mean()) + 1e-6 for k, t in mert_target.items()}


def at_length(sorted_values: torch.Tensor, n: int) -> torch.Tensor:
    """A sorted histogram resampled (quantile by quantile) to n values, so a longer output is compared with a shorter source."""
    if sorted_values.shape[-1] == n:
        return sorted_values
    return F.interpolate(sorted_values[None], size=n, mode="linear", align_corners=True)[0]


def layer_weight(k: int, progress: float) -> float:
    start = 0.6 * k / max(LAYERS)
    return min(max((progress - start) / RAMP, 0.0), 1.0) if k > 0 else 1.0


def mert_loss(audio: torch.Tensor, progress: float) -> tuple[torch.Tensor, dict[int, float]]:
    stats = mert_statistics(activations(audio))
    parts = {k: (stats[k] - at_length(mert_target[k], stats[k].shape[-1])).pow(2).mean() / mert_scale[k] for k in mert_target}
    weights = {k: layer_weight(int(k.split("/")[0]), progress) for k in mert_target}
    by_layer = {layer: float(np.mean([float(v) for k, v in parts.items() if k.startswith(f"{layer}/")])) for layer in LAYERS}
    return sum(weights[k] * parts[k] for k in parts) / sum(weights.values()), by_layer


# ------------------------------------------------------------------------------------------------------------------------ variable
n_seed = int(round(1.2 * len(events) * T_OUT / T))
usage = np.bincount(labels, minlength=C) / len(labels)
seed_c = rng.choice(C, n_seed, p=usage)
seed_d = np.array([int(rng.choice(offsets[labels == c])) for c in seed_c]) + sn.W
seed_t = rng.integers(0, T_OUT - sn.TF, n_seed)
seed_g = rng.choice(gains, n_seed)
support = torch.zeros(C, D, T_OUT, dtype=torch.bool)
start = torch.zeros(C, D, T_OUT)
for c, d, t, g in zip(seed_c, seed_d, seed_t, seed_g):
    support[c, max(d - 1, 0) : d + 2, max(t - 1, 0) : t + 2] = True
    start[c, d, t] += g
raw = (start + 0.05 * support.float()).requires_grad_()
mask = support.float()


def triggers() -> torch.Tensor:
    return torch.relu(raw) * mask


optimiser = torch.optim.Adam([raw], lr=0.02)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, iters, eta_min=0.002)
print(f"{song} {which}: {C} clusters, {len(events)} source events, {n_seed} seed events, support {int(mask.sum())} cells; MERT {'off' if NO_MERT else 'on'}, trigger statistics {'off' if NO_TRIG else 'on'}", flush=True)
begin = time.time()
for i in range(iters):
    optimiser.zero_grad()
    H = triggers()
    audio = sn.render(H, kernels, samples)
    total, notes = torch.zeros(()), []
    if not NO_TRIG:
        tl, tparts = trigger_loss(H)
        total = total + TRIG_WEIGHT * tl
        notes.append(f"trig {float(tl):.3f} (" + " ".join(f"{k} {v:.2f}" for k, v in tparts.items()) + ")")
    if not NO_MERT:
        ml, by_layer = mert_loss(audio, i / iters)
        total = total + ml
        notes.append(f"mert {float(ml):.4f}")
    total.backward()
    optimiser.step()
    schedule.step()
    if i % 25 == 0 or i == iters - 1:
        print(f"iter {i:4d}  " + "  ".join(notes) + f"  active cells {int((H > 0.05).sum())}  ({time.time() - begin:.0f} s)", flush=True)
    if (i + 1) % 100 == 0:
        sn.write(sn.OUT / f"N_{tag}.wav", audio)

with torch.no_grad():
    H = triggers()
    audio = sn.render(H, kernels, samples)
sn.write(sn.OUT / f"N_{tag}.wav", audio)
torch.save({"H": H, "raw": raw.detach(), "audio": audio}, sn.ATOMS_DIR / f"N_{tag}_triggers.pt")

# copy check, over the span the source covers: generated events (cells above 30% of the median seed gain) that coincide with a source event of the same cluster within 1 frame
source_cells = {(int(c), int(t) + dt) for c, t in zip(labels, events[:, 2]) for dt in (-1, 0, 1)}
cells = [(c, t) for c, t in (H.sum(1) > 0.3 * float(np.median(gains))).nonzero().tolist() if t < T]
hit = np.mean([(c, t) in source_cells for c, t in cells]) if cells else float("nan")
chance = np.mean([(int(rng.choice(C, p=usage)), int(rng.integers(0, T))) in source_cells for _ in range(2000)])
print(f"generated events in the source's span {len(cells)} (source {len(events)}); share on a source event of the same cluster within 1 frame: {hit:.2f}; chance with usage-matched random times {chance:.2f}")
print("done")
