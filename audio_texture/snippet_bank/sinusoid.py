"""Sinusoidal renderer for the 48-bin constant-Q surface (McAulay-Quatieri style), with an optional shaped-noise residual.

    sinusoid.py

For the synthetic pluck, hanging_2 guitar and jack_1 guitar (400-3400 Hz band-limited):
  sin  : peaks of each surface frame (parabolic sub-bin frequency/amplitude) are linked into tracks; each track is ONE sinusoid whose
         frequency and amplitude are interpolated between frames and whose phase is the running integral of its frequency.
  sinn : sin plus a noise layer: per-band noise shaped by sqrt(max(surface^2 - rendered surface^2, 0)), i.e. what the sinusoids left out.
Prints an audibility-style measure for these and for the existing gl / lock_tm renders: the envelope of 24 ERB-spaced auditory channels
(400-3400 Hz) is taken, and the variance of its modulation between 15-80 Hz and between 20-30 Hz, relative to the original signal
(energy-weighted over channels, dB re original; 0 = same flutter as the original, positive = added flutter / ray-gun).
Writes compare/{pl,hg,jk}_{sin,sinn}.wav.
"""
import sys  # noqa: F401  (oracle functions are exec'd below and need these names)
from pathlib import Path

source = Path(__file__).with_name("oracle.py").read_text()
exec(compile(source[: source.index("OUT.mkdir(parents=True")], "oracle_functions", "exec"))

from scipy.io import wavfile  # noqa: E402

FLOOR, LINK_BINS, MIN_FRAMES = 1e-2, 1.5, 2
CENTRES = F_MIN * 2.0 ** (np.arange(int(round(BINS * np.log2(F_MAX / F_MIN)))) / BINS)


def frame_peaks(column: np.ndarray, floor: float) -> list[tuple[float, float]]:
    """(frequency Hz, amplitude) of each local maximum of one surface column, refined by a parabola through log magnitude."""
    log = np.log(column + 1e-12)
    inner = np.nonzero((column[1:-1] >= column[:-2]) & (column[1:-1] > column[2:]) & (column[1:-1] > floor))[0] + 1
    out = []
    for i in inner:
        curvature = log[i - 1] - 2 * log[i] + log[i + 1]
        delta = 0.5 * (log[i - 1] - log[i + 1]) / curvature if curvature < 0 else 0.0
        out.append((CENTRES[i] * 2 ** (delta / BINS), float(np.exp(log[i] - 0.25 * (log[i - 1] - log[i + 1]) * delta))))
    return out


def link_tracks(surface: np.ndarray) -> list[list[tuple[int, float, float]]]:
    """Greedy nearest-frequency linking, loudest peaks first. A track is a list of (frame, frequency, amplitude)."""
    tracks, active = [], []
    for t in range(surface.shape[1]):
        peaks = sorted(frame_peaks(surface[:, t], FLOOR * surface.max()), key=lambda p: -p[1])
        taken, now = set(), []
        for f, a in peaks:
            options = [(abs(np.log2(f / tracks[i][-1][1])) * BINS, i) for i in active if i not in taken]
            best = min(options, default=(np.inf, -1))
            if best[0] <= LINK_BINS:
                taken.add(best[1])
                tracks[best[1]].append((t, f, a))
                now.append(best[1])
            else:
                tracks.append([(t, f, a)])
                now.append(len(tracks) - 1)
        active = now
    return [tr for tr in tracks if len(tr) >= MIN_FRAMES]


def synthesise(tracks: list[list[tuple[int, float, float]]], n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for track in tracks:
        frames = np.array([p[0] for p in track])
        centres = frames * POOL + POOL // 2
        start, stop = max(int(centres[0]) - POOL, 0), min(int(centres[-1]) + POOL, n)
        t = np.arange(start, stop)
        freq = np.interp(t, centres, [p[1] for p in track])
        amp = np.interp(t, np.concatenate([[centres[0] - POOL], centres, [centres[-1] + POOL]]), [0.0] + [p[2] for p in track] + [0.0])
        out[start:stop] += 2 * amp * np.sin(rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(freq) / RATE)  # |analytic| of a sine of amplitude A is A/2
    return out


def noise_layer(target: torch.Tensor, rendered: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Per-band noise at the level the sinusoids leave unexplained."""
    torch.manual_seed(seed)
    residual = ((target**2 - rendered**2).clamp_min(0)).sqrt()
    freqs = torch.fft.fftfreq(n, 1 / RATE)
    q = 1 / (2 ** (1 / BINS) - 1)
    white = torch.fft.fft(torch.randn(n))
    out = torch.zeros(n)
    for i, c in enumerate(CENTRES):
        band = torch.fft.ifft(white * torch.exp(-0.5 * ((freqs - c) / (c / q * 0.5)) ** 2))
        gain = residual[i] / band.abs().mean().clamp_min(1e-9)  # band.real has band as its analytic signal
        frames = torch.arange(residual.shape[1]) * POOL + POOL // 2
        envelope = torch.from_numpy(np.interp(np.arange(n), frames.numpy(), gain.numpy()).astype(np.float32))
        out = out + band.real * envelope
    return out


ERB_CENTRES = (10 ** (np.linspace(21.4 * np.log10(4.37 * 0.4 + 1), 21.4 * np.log10(4.37 * 3.4 + 1), 24) / 21.4) - 1) * 1000 / 4.37


def flutter(x: np.ndarray) -> tuple[float, float]:
    """Energy-weighted modulation variance (15-80 Hz, 20-30 Hz) of the auditory-channel envelopes."""
    spectrum, freqs = np.fft.fft(x), np.fft.fftfreq(len(x), 1 / RATE)
    wide, narrow, weight = [], [], []
    for fc in ERB_CENTRES:
        b = 1.019 * 24.7 * (4.37 * fc / 1000 + 1)
        gain = np.where(freqs > 0, (1 + ((freqs - fc) / b) ** 2) ** -2.0, 0.0)
        env = np.abs(np.fft.ifft(spectrum * gain * 2))
        mean = env.mean()
        window = np.hanning(len(env))
        power = np.abs(np.fft.rfft((env / mean - 1) * window)) ** 2 * 2 / (len(env) * (window**2).sum())  # variance per rfft bin
        mod = np.fft.rfftfreq(len(env), 1 / RATE)
        wide.append(power[(mod >= 15) & (mod <= 80)].sum())
        narrow.append(power[(mod >= 20) & (mod <= 30)].sum())
        weight.append(mean**2)
    w = np.array(weight) / np.sum(weight)
    return float(np.sum(w * wide)), float(np.sum(w * narrow))


def load(name: str, n: int) -> np.ndarray:
    _, audio = wavfile.read(OUT / f"{name}.wav")
    audio = audio.astype(np.float64)
    return (audio / max(np.abs(audio).max(), 1))[:n]


def show(tag: str, x: np.ndarray, base: tuple[float, float]) -> None:
    wide, narrow = flutter(x / np.sqrt((x**2).mean()))
    print(f"  {tag:10s} flutter 15-80 Hz {10 * np.log10(wide / base[0]):+6.1f} dB   20-30 Hz {10 * np.log10(narrow / base[1]):+6.1f} dB", flush=True)


for prefix, wave in (("pl", pluck()), ("hg", stem("hanging_2")), ("jk", stem("jack_1"))):
    n = wave.numel()
    target = cqt(wave)
    tracks = link_tracks(target.numpy())
    audio = synthesise(tracks, n)
    sin = torch.tensor(audio, dtype=torch.float32)
    scale = float(target.max()) / float(cqt(sin).max())  # the sinusoids' surface, brought to the target's peak scale, is what the noise tops up
    rendered = cqt(sin) * scale
    mixed = sin * scale + noise_layer(target, rendered, n)
    print(f"{prefix}: {len(tracks)} tracks, {sum(len(t) for t in tracks)} track-frames", flush=True)
    original = wave.numpy().astype(np.float64)
    base = flutter(original / np.sqrt((original**2).mean()))
    for name, x in (("original", original), ("gl", load(f"{prefix}_gl", n)), ("sin", sin.numpy().astype(np.float64)), ("sinn", mixed.numpy().astype(np.float64))) + (
        (("lock_tm", load(f"{prefix}_lock_tm", n)),) if prefix != "pl" else (("lock", load("pl_lock", n)),)
    ):
        show(name, x, base)
    for name, x in (("sin", sin), ("sinn", mixed)):
        save(OUT / f"{prefix}_{name}.wav", x, 0.9 / max(float(x.abs().max()), 1e-9))
print("done")
