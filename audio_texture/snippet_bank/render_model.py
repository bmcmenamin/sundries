"""Render MODEL surfaces (fitted and generated) with the sinusoidal + noise renderer from sinusoid.py.

    render_model.py

Writes compare/M_{name}.wav (sin + noise) for, per song: the sparse fit (f0.25), the dense fit (f0.06), a Poisson-timed generation (seed 0)
and a chain generation (alpha 0.7, seed 0). Prints the auditory-channel flutter of each relative to the real stem.
"""
from pathlib import Path

source = Path(__file__).with_name("sinusoid.py").read_text()
exec(compile(source[: source.index("for prefix, wave in")], "sinusoid_functions", "exec"))

ATOMS = ROOT / "out/codec_texture/atoms"


def fit_surface(song: str, fraction: str) -> torch.Tensor:
    fit = torch.load(ATOMS / f"{song}_guitar_prox48_f{fraction}_fit.pt")
    w, h, v = fit["w"], fit["h"], fit["V"]
    return F.conv_transpose2d(h[None], w[:, None], stride=(4, 1))[0, 0, : v.shape[0], : v.shape[1]].clamp_min(0)


def render_sinn(surface: torch.Tensor) -> torch.Tensor:
    n = surface.shape[1] * POOL
    sin = torch.tensor(synthesise(link_tracks(surface.numpy()), n), dtype=torch.float32)
    scale = float(surface.max()) / float(cqt(sin).max())
    return sin * scale + noise_layer(surface, cqt(sin) * scale, n)


for song, short in (("hanging_2", "hg"), ("jack_1", "jk")):
    real = stem(song).numpy().astype(np.float64)
    base = flutter(real / np.sqrt((real**2).mean()))
    cases = {
        f"{short}_fit_sparse": fit_surface(song, "0.25"),
        f"{short}_fit_dense": fit_surface(song, "0.06"),
        f"{short}_gen_poisson": torch.load(ATOMS / f"surface_G_{song}_f0.25_s0.pt"),
        f"{short}_gen_chain": torch.load(ATOMS / f"surface_C_{song}_f0.06_a0.7_s0.pt"),
    }
    print(song, flush=True)
    for name, surface in cases.items():
        audio = render_sinn(surface)
        show(name, audio.numpy().astype(np.float64), base)
        save(OUT / f"M_{name}.wav", audio, 0.9 / max(float(audio.abs().max()), 1e-9))
print("done")
