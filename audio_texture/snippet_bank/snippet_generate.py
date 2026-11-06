"""Arrange real snippets so that the trigger statistics AND MERT's activation statistics match the source's. Only the triggers are optimised.

    snippet_generate.py SONG WHICH ITERS TAG        env: NO_MERT=1 (trigger statistics only), NO_TRIG=1 (MERT only), SEED, TRIG_WEIGHT (default 1),
                                                DEVICE (MERT's; default mps if present), NO_CUT=1 (no layer cutoff),
                                                OPTIM (lbfgs | adam | adam_const; default lbfgs, where ITERS counts steps of up to 30 evaluations: use 3)
                                                OUT_SAMPLES (exact length; overrides OUT_SECONDS), CLAMP (the previous section's N_*_triggers.pt: its last
                                                TF*1024 samples and its tail are this section's fixed start), CLAMP_GAIN (previous units -> this section's)

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
OPTIM = os.environ.get("OPTIM", "lbfgs")  # lbfgs (the default, 3 steps), adam (lr 0.02, cosine), adam_const (lr 0.05)
NO_CUT = bool(os.environ.get("NO_CUT"))  # run MERT through all layers even while the upper ones have no weight yet
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")  # MERT's device; triggers and render stay on the CPU
LAYERS, PROJECTIONS, LAGS, RAMP = list(range(13)), 256, (1, 4, 16), 0.15
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

bank = torch.load(sn.ATOMS_DIR / f"{song}_{which}_bank{sn.TAG}.pt", weights_only=False)
events, labels, offsets, gains = bank["events"], bank["labels"], bank["offsets"], bank["gains"]
atoms, medoid_shift, T = bank["atoms"], bank["medoid_shift"], bank["frames"]
C, D, S_ABS = len(atoms), 2 * sn.W + 1, int(medoid_shift.max()) + 2 * sn.W + 1
OUT_SECONDS = float(os.environ.get("OUT_SECONDS", 10))
samples = int(os.environ.get("OUT_SAMPLES", OUT_SECONDS * sn.RATE))  # the output may be longer than the source: every statistic is length-free
HANDOFF, TAIL = sn.TF * sn.HOP, sn.SNIPPET  # a section's last HANDOFF samples (and the TAIL after its end) are the next section's first
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


clips = bank.get("clips") or [(0, bank["audio"].numel())]  # sample ranges of the source's separate clips (segment.py pool); one clip otherwise
clip_frames = [(a // sn.HOP, min(b // sn.HOP, H_src.shape[-1])) for a, b in clips] if "clips" in bank else [(0, H_src.shape[-1])]


def source_statistics() -> dict[str, torch.Tensor]:
    """The trigger statistics of each clip on its own (no lag looks across a gap between clips), averaged in proportion to the clips' lengths."""
    per_clip = [(trigger_statistics(H_src[..., a:b]), b - a) for a, b in clip_frames]
    total = sum(n for _, n in per_clip)
    return {k: sum(s[k] * n for s, n in per_clip) / total for k in per_clip[0][0]}


with torch.no_grad():
    trig_target = source_statistics()
trig_scale = {k: float(v.pow(2).mean()) + 1e-9 for k, v in trig_target.items()}


def trigger_loss(H: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    stats = trigger_statistics(H)
    parts = {k: (stats[k] - trig_target[k]).pow(2).mean() / trig_scale[k] for k in trig_target}
    return sum(parts.values()) / len(parts), {k: float(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------------------------------------------------------- MERT
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)
captured: dict[int, torch.Tensor] = {}
stop_at = [None]  # the forward pass ends at this layer (layers above it have no weight yet)


class Stop(Exception):
    pass


def capture(k: int):
    def hook(_m, _i, o):
        captured[k] = o[0] if isinstance(o, tuple) else o
        if stop_at[0] == k:
            raise Stop

    return hook


model.feature_projection.register_forward_hook(capture(0))
for k, layer in enumerate(model.encoder.layers, start=1):
    layer.register_forward_hook(capture(k))


def activations(wave: torch.Tensor, depth: int = max(LAYERS)) -> dict[int, torch.Tensor]:
    stop_at[0] = depth if depth < max(LAYERS) else None
    try:
        model(((wave - wave.mean()) / (wave.std() + 1e-8))[None])
    except Stop:
        pass
    return {k: captured[k][0].T for k in range(depth + 1)}


with torch.no_grad():  # one forward pass per clip: a clip is never joined to the next, and a pooled source never needs one long attention window
    target_acts = [activations(bank["audio"][a:b].to(DEVICE)) for a, b in clips]
pooled = {k: torch.cat([c[k] for c in target_acts], 1) for k in LAYERS}
mean = {k: a.mean(1, keepdim=True) for k, a in pooled.items()}
std = {k: a.std(1, keepdim=True) + 1e-6 for k, a in pooled.items()}
projection = {k: F.normalize(torch.randn(PROJECTIONS, a.shape[0]), dim=1).to(DEVICE) for k, a in pooled.items()}


def mert_statistics(clip_acts: list[dict[int, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Sorted histograms pooled over the clips; the changes at a lag are taken inside each clip, never across the gap between two."""
    out = {}
    for k in clip_acts[0]:
        z = [(c[k] - mean[k]) / std[k] for c in clip_acts]
        p = [projection[k] @ x for x in z]
        out[f"{k}/units"] = torch.cat(z, 1).sort(1).values
        out[f"{k}/joint"] = torch.cat(p, 1).sort(1).values
        for lag in LAGS:
            out[f"{k}/motion{lag}"] = torch.cat([x[:, lag:] - x[:, :-lag] for x in p], 1).sort(1).values
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


def active_depth(progress: float) -> int:
    """The highest layer that already counts (all of them with NO_CUT)."""
    return max(LAYERS) if NO_CUT else max(k for k in LAYERS if layer_weight(k, progress) > 0)


def mert_loss(audio: torch.Tensor, progress: float) -> tuple[torch.Tensor, dict[int, float]]:
    stats = mert_statistics([activations(audio.to(DEVICE), active_depth(progress))])
    parts = {k: (stats[k] - at_length(mert_target[k], stats[k].shape[-1])).pow(2).mean() / mert_scale[k] for k in stats}
    weights = {k: layer_weight(int(k.split("/")[0]), progress) for k in stats}
    by_layer = {layer: float(np.mean([float(v) for k, v in parts.items() if k.startswith(f"{layer}/")])) for layer in {int(k.split("/")[0]) for k in stats}}
    return sum(weights[k] * parts[k] for k in parts) / sum(weights.values()), by_layer


# ------------------------------------------------------------------------------------------------------------------------ variable
n_seed = int(round(1.2 * len(events) * T_OUT / T))
usage = np.bincount(labels, minlength=C) / len(labels)
seed_c = rng.choice(C, n_seed, p=usage)
seed_d = np.clip(np.array([int(rng.choice(offsets[labels == c])) for c in seed_c]), -sn.W, sn.W) + sn.W  # beyond +-W clipped, as snippets.grid_of does
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


def clamp_of(path: str | None, gain: float) -> torch.Tensor:
    """The previous section's last HANDOFF samples and its ringing-out tail, in this section's units, laid at the start of a zero waveform. Constant: only
    the triggers are optimised around it, so MERT sees it and the new triggers have to fit onto it."""
    if path is None:
        return torch.zeros(samples + TAIL)
    previous = torch.load(path)
    kept = previous["audio_ext"][previous["samples"] - HANDOFF :] * gain
    return torch.nn.functional.pad(kept, (0, samples + TAIL - kept.numel()))


clamp = clamp_of(os.environ.get("CLAMP"), float(os.environ.get("CLAMP_GAIN", 1.0)))


def fitted(rendered: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(rendered, (0, length - rendered.numel()))  # a little short when the tail asked for runs past the last snippet's end


def played(H: torch.Tensor, length: int) -> torch.Tensor:
    """The render at unit rms over the section's `samples` (MERT is blind to level, so nothing else holds it: unpinned it drifts by orders of magnitude, and stitch.py
    multiplies by the stem's level), plus the clamp, which is already in these units. A longer `length` (the ringing-out tail) keeps the same scale."""
    base = fitted(sn.render(H, kernels, samples), samples)
    scale = 1.0 / base.pow(2).mean().sqrt().clamp_min(1e-8)
    full = base if length == samples else fitted(sn.render(H, kernels, length), length)
    return full * scale + clamp[:length]


def objective(progress: float) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    H = triggers()
    audio = played(H, samples)
    total, notes = torch.zeros(()), []
    if not NO_TRIG:
        tl, tparts = trigger_loss(H)
        total = total + TRIG_WEIGHT * tl
        notes.append(f"trig {float(tl):.3f} (" + " ".join(f"{k} {v:.2f}" for k, v in tparts.items()) + ")")
    if not NO_MERT:
        ml, _ = mert_loss(audio, progress)
        total = total + ml.cpu()
        notes.append(f"mert {float(ml):.4f}")
    return total, notes, H, audio


if OPTIM == "lbfgs":  # iters = number of L-BFGS steps (each up to 30 evaluations); the full objective throughout, no layer schedule
    optimiser = torch.optim.LBFGS([raw], lr=1.0, max_iter=25, max_eval=30, history_size=10, line_search_fn="strong_wolfe")
    schedule = None
else:
    optimiser = torch.optim.Adam([raw], lr=0.05 if OPTIM == "adam_const" else 0.02)
    schedule = None if OPTIM == "adam_const" else torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, iters, eta_min=0.002)
print(f"{song} {which}: {C} clusters, {len(events)} source events, {n_seed} seed events, support {int(mask.sum())} cells; MERT {'off' if NO_MERT else 'on'}, trigger statistics {'off' if NO_TRIG else 'on'}", flush=True)
begin = time.time()
for i in range(iters):
    progress = 1.0 if OPTIM == "lbfgs" else i / iters

    last = []  # the latest evaluation: for Adam the state the step starts from, for L-BFGS the end of its line search

    def closure():
        optimiser.zero_grad()
        last[:] = objective(progress)
        last[0].backward()
        return last[0]

    optimiser.step(closure)
    _, notes, H, audio = last
    if schedule:
        schedule.step()
    if i % 25 == 0 or i == iters - 1 or OPTIM == "lbfgs":
        print(f"iter {i:4d}  " + "  ".join(notes) + f"  active cells {int((H > 0.05).sum())}  ({time.time() - begin:.0f} s)", flush=True)
    if (i + 1) % 100 == 0:
        sn.write(sn.OUT / f"N_{tag}.wav", audio)

with torch.no_grad():
    H = triggers()
    audio = played(H, samples)
    audio_ext = played(H, samples + TAIL)  # with the notes that are still ringing after the end: the next section's clamp is cut from this
sn.write(sn.OUT / f"N_{tag}.wav", audio)
torch.save({"H": H, "raw": raw.detach(), "audio": audio, "audio_ext": audio_ext, "samples": samples}, sn.ATOMS_DIR / f"N_{tag}_triggers.pt")

# copy check, over the span the source covers: generated events (cells above 30% of the median seed gain) that coincide with a source event of the same cluster within 1 frame
source_cells = {(int(c), int(t) + dt) for c, t in zip(labels, events[:, 2]) for dt in (-1, 0, 1)}
cells = [(c, t) for c, t in (H.sum(1) > 0.3 * float(np.median(gains))).nonzero().tolist() if t < T]
hit = np.mean([(c, t) in source_cells for c, t in cells]) if cells else float("nan")
chance = np.mean([(int(rng.choice(C, p=usage)), int(rng.integers(0, T))) in source_cells for _ in range(2000)])
print(f"generated events in the source's span {len(cells)} (source {len(events)}); share on a source event of the same cluster within 1 frame: {hit:.2f}; chance with usage-matched random times {chance:.2f}")
print("done")
