"""Sparse event coding WITH sustain: a trigger carries a duration drawn from a small set; the template's last frame is held for that long.

    sustain48.py SONG INPUT MODE K FRACTION ...      INPUT: guitar | mix      MODE: sustain | attack       e.g.  sustain48.py jack_1 mix sustain 24 0.12

Surface V ~ conv(sum_d H_d, A)  +  conv(G, S)
  H_d : onset triggers of duration class d (K x pitch x frame), sparse, all penalised equally.  DURATIONS are in frames (0 = a plain attack, as before).
  A_k : attack template (120 bins x 10 frames), unit norm x SCALE.      S_k = A_k[:, last frame]: what is held.
  G_k : sum over d of H_d spread over frames [TEMPLATE_FRAMES, TEMPLATE_FRAMES + d) after its onset (a box filter, via cumulative sums).
So a held note is ONE trigger with a duration class instead of a train of retriggers; everything is linear in H, so the same accelerated proximal gradient as
sparse_prox48.py applies. MODE attack keeps only the d = 0 channel (the old model, same code) for a like-for-like comparison.
Reports error, triggers per duration class, attack clusters, share of rendered energy that is sustain. Writes the fit to atoms/ and
compare/S_{hg,wk,jk}_{input}_{mode}_f{fraction}.wav through the sin+noise renderer.
"""
import math
import sys
import time
from pathlib import Path

source = Path(__file__).with_name("render_model.py").read_text()
exec(compile(source[: source.index("for song, short in")], "render_model_functions", "exec"))
from scipy import ndimage  # noqa: E402

import os  # noqa: E402

TEMPLATE_BINS, TEMPLATE_FRAMES, STRIDE = 120, int(os.environ.get("TEMPLATE_FRAMES", 10)), 4  # TEMPLATE_FRAMES: template length in frames of 42.7 ms
DURATIONS = (0, 5, 12, 25, 50)  # frames of 42.7 ms: 0, 0.2, 0.5, 1.1, 2.1 s of hold after the 10-frame attack
COMPRESS, SCALE, EPS = 0.3, 15.0, 1e-2
ROUNDS, H_ITERS, W_STEPS, W_LR = 60, 20, 5, 0.02
ETA_FLOOR, TOLERANCE = 1e-5, 1e-5

song, which, mode = sys.argv[1:4]
K = int(sys.argv[4])
FRACTIONS = [float(x) for x in sys.argv[5:]] or [0.12]
SHORT = {"hanging_2": "hg", "walk_2": "wk", "jack_1": "jk"}[song]
MASK = torch.tensor([1.0 if (mode == "sustain" or d == 0) else 0.0 for d in DURATIONS]).view(-1, 1, 1, 1)


def mix_of(name: str) -> torch.Tensor:
    stems = [decode_mono(s.path, RATE)[: 750 * codec.HOP] for s in stems_of(name) if s.name != "vocals"]  # vocals are left out of every mix
    return band_limit(codec.normalise(sum(stems)).numpy())


def named_stem(name: str, which: str) -> torch.Tensor:
    path = next(s.path for s in stems_of(name) if s.name == which)
    return band_limit(codec.normalise(decode_mono(path, RATE)[: 750 * codec.HOP]).numpy())


wave = mix_of(song) if which == "mix" else named_stem(song, which)
V = cqt(wave)
GOAL = V**COMPRESS
n_shift, n_t = (V.shape[0] + STRIDE - 1) // STRIDE, V.shape[1]
seconds = n_t * POOL / RATE


def template(raw_a: torch.Tensor) -> torch.Tensor:
    a = F.softplus(raw_a)
    return SCALE * a / a.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-9)


def hold(x: torch.Tensor) -> torch.Tensor:
    """Spread each trigger of duration d over frames [TEMPLATE_FRAMES, TEMPLATE_FRAMES + d) after its onset: C[t - TF] - C[t - TF - d] with C the cumulative sum."""
    total = torch.zeros_like(x[0])
    for channel, d in zip(x, DURATIONS):
        if d == 0:
            continue
        c = F.pad(channel.cumsum(-1), (TEMPLATE_FRAMES + d, 0))
        total = total + c[..., d : d + n_t] - c[..., : n_t]
    return total


def parts(x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attack = F.conv_transpose2d(x.sum(0)[None], a[:, None], stride=(STRIDE, 1))[0, 0, : V.shape[0], : V.shape[1]]
    held = a[:, :, -1]  # (K, bins): the last frame of each attack template
    sustain = F.conv_transpose2d(hold(x)[None], held[:, None, :, None], stride=(STRIDE, 1))[0, 0, : V.shape[0], : V.shape[1]]
    return attack, sustain


def smooth(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    attack, sustain = parts(x, a)
    return (((attack + sustain + EPS) ** COMPRESS - GOAL) ** 2).sum()


def fista(x: torch.Tensor, a: torch.Tensor, lam: float, eta: float) -> tuple[torch.Tensor, float]:
    prev, y, t = x.clone(), x.clone(), 1.0
    eta = max(eta, ETA_FLOOR)
    best = float(smooth(x, a)) + lam * float(x.sum())
    for _ in range(H_ITERS):
        y = y.requires_grad_()
        f_y = smooth(y, a)
        (g,) = torch.autograd.grad(f_y, y)
        y = y.detach()
        f_value = float(f_y)
        tolerance = TOLERANCE * abs(f_value)
        while True:
            new = torch.relu(y - eta * (g + lam)) * MASK
            d = new - y
            f_new = float(smooth(new, a))
            if f_new <= f_value + float((g * d).sum()) + float((d * d).sum()) / (2 * eta) + tolerance or eta < 1e-12:
                break
            eta *= 0.5
        total = f_new + lam * float(new.sum())
        if total > best * (1 + TOLERANCE):
            y, t = prev.clone(), 1.0
            continue
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = torch.relu(new + ((t - 1) / t_new) * (new - prev)) * MASK
        prev, t, best = new, t_new, total
        eta *= 1.2
    return prev, eta


def clusters(h: torch.Tensor) -> int:
    return sum(ndimage.label((h[k] > 0).numpy(), structure=np.ones((3, 3)))[1] for k in range(h.shape[0]))


torch.manual_seed(0)
raw_a = torch.rand(K, TEMPLATE_BINS, TEMPLATE_FRAMES).requires_grad_()
x = torch.zeros(len(DURATIONS), K, n_shift, n_t)
zero = torch.zeros_like(x).requires_grad_()
(g0,) = torch.autograd.grad(smooth(zero, template(raw_a).detach()), zero)
lam_max = float((-g0[0]).max())  # set by the plain-attack channel, so FRACTION means the same as in the attack-only runs
print(f"{song} {which} {mode}: V {tuple(V.shape)}, K={K}, {seconds:.1f} s; lambda_max {lam_max:.1f}; start gradient per duration {[round(float((-g0[i]).max()), 0) for i in range(len(DURATIONS))]}", flush=True)

optimiser = torch.optim.Adam([raw_a], lr=W_LR)
eta = 1e-4
for fraction in FRACTIONS:
    lam = fraction * lam_max
    begin = time.time()
    for r in range(ROUNDS):
        x, eta = fista(x, template(raw_a).detach(), lam, eta)
        for _ in range(W_STEPS):
            optimiser.zero_grad()
            smooth(x, template(raw_a)).backward()
            optimiser.step()
        if (r + 1) % 20 == 0:
            with torch.no_grad():
                print(f"  fraction {fraction}: round {r + 1}, fit {float(smooth(x, template(raw_a))):.1f}, entries {int((x > 0).sum())}, eta {eta:.2e}", flush=True)
    with torch.no_grad():
        a = template(raw_a)
        attack, sustain = parts(x, a)
        total = attack + sustain
        error = float((((total + EPS) ** COMPRESS - GOAL) ** 2).sum() / (GOAL**2).sum())
        per_class = [int((x[i] > 0).sum()) for i in range(len(DURATIONS))]
        held = sum(per_class[1:])
        print(
            f"fraction {fraction} (lambda {lam:.1f}): error {error:.4f}  triggers {sum(per_class)} ({sum(per_class) / seconds:.1f}/s)  by hold {dict(zip(DURATIONS, per_class))}  "
            f"held share {held / max(sum(per_class), 1):.2f}  clusters {clusters(x.sum(0))}  sustain energy share {float(sustain.pow(2).sum() / total.pow(2).sum()):.2f}  ({time.time() - begin:.0f} s)",
            flush=True,
        )
        tag = ("" if K == 24 else f"_k{K}") + ("" if TEMPLATE_FRAMES == 10 else f"_tf{TEMPLATE_FRAMES}")  # K = 24, 10 frames keeps the names used so far
        torch.save({"x": x, "a": a, "V": V, "lambda": lam, "durations": DURATIONS}, ATOMS / f"{song}_{which}_{mode}_f{fraction}{tag}_sus.pt")
        audio = render_sinn(total.clamp_min(0))
        save(OUT / f"S_{SHORT}_{which}_{mode}_f{fraction}{tag}.wav", audio, 0.9 / max(float(audio.abs().max()), 1e-9))
print("done")
