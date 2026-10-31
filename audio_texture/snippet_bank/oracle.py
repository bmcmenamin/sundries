"""Oracle: does knowing each partial's true instantaneous frequency (IF) remove the ray-gun that Griffin-Lim leaves?

    oracle.py            runs the synthetic pluck, hanging_2 guitar and jack_1 guitar

Input: 400-3400 Hz band-limited signal. Magnitude always comes from the 48-bin constant-Q surface mapped to a 2048-point STFT (hop 512), as in the
current pipeline; what differs is the phase given to it:
  true     : the signal's own STFT (sanity: should equal the input)
  gl       : Griffin-Lim from random phase, 100 iterations (current renderer)
  lock     : peak-locked phases. Each spectral peak keeps its track: phase advances by that partial's TRUE IF (taken from the real STFT);
             new peaks start at random phase; the bins around a peak follow it with the Hann alternation (pi per bin).
  lock_gl  : lock, then 30 Griffin-Lim iterations
  lock_flat: control; same locking but every peak's IF is its bin centre (no frequency information)
Prints, per file: inconsistency = |STFT(ISTFT(X)) - X| / |X|, and surface error = compressed-magnitude error of the output's own 48-bin surface vs the target.
For the pluck also the sideband / off-harmonic probe from bandlimit_test.py. Writes compare/{pl,hg,jk}_{variant}.wav.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import codec
from codec_texture import save
from paths import stems_of
from synthesize import decode_mono

RATE = codec.CODEC_RATE
BINS, F_MIN, F_MAX, POOL = 48, 350.0, 3500.0, 1024
N_FFT, HOP = 2048, 512
OUT = ROOT / "out/codec_texture/compare"
WINDOW = torch.hann_window(N_FFT)


def band_limit(x: np.ndarray) -> torch.Tensor:
    y = torch.tensor(sosfiltfilt(butter(4, [400, 3400], btype="band", fs=RATE, output="sos"), x.astype(np.float64)).copy(), dtype=torch.float32)
    return y * (codec.TARGET_RMS / float(y.pow(2).mean().sqrt()))


def cqt(wave: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft(wave)
    freqs = torch.fft.fftfreq(wave.numel(), 1 / RATE)
    q = 1 / (2 ** (1 / BINS) - 1)
    rows = (
        F.avg_pool1d(torch.fft.ifft(spectrum * torch.exp(-0.5 * ((freqs - c) / (c / q * 0.5)) ** 2)).abs()[None, None], POOL, POOL)[0, 0]
        for c in F_MIN * 2.0 ** (torch.arange(int(round(BINS * np.log2(F_MAX / F_MIN)))) / BINS)
    )
    v = torch.stack(list(rows))
    return v / v.max()


def to_linear(v: torch.Tensor, frames: int) -> torch.Tensor:
    centres = F_MIN * 2.0 ** (torch.arange(v.shape[0]) / BINS)
    lin = torch.arange(N_FFT // 2 + 1) * RATE / N_FFT
    width = torch.maximum(centres * (2 ** (1 / BINS) - 1), torch.tensor(RATE / N_FFT))
    bank = torch.relu(1 - (lin[None] - centres[:, None]).abs() / width[:, None])
    fine = F.interpolate(v[None], scale_factor=POOL / HOP, mode="linear")[0]
    linear = (bank.T @ fine) / bank.sum(0).clamp_min(1e-6)[:, None]
    return F.pad(linear, (0, max(frames - linear.shape[1], 0)))[:, :frames]


def stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, N_FFT, HOP, window=WINDOW, return_complex=True)


def istft(s: torch.Tensor, n: int) -> torch.Tensor:
    return torch.istft(s, N_FFT, HOP, window=WINDOW, length=n)


def peaks_of(mag: np.ndarray, floor: float) -> np.ndarray:
    inner = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] > mag[2:]) & (mag[1:-1] > floor)
    return np.nonzero(inner)[0] + 1


def lock_phase(mag: np.ndarray, truth: np.ndarray, use_if: bool, seed: int = 0, floor_share: float = 3e-3) -> np.ndarray:
    """Peak-locked phase for magnitude `mag` (bins x frames); `truth` is the real STFT supplying the IF."""
    rng = np.random.default_rng(seed)
    bins, frames = mag.shape
    k_index = np.arange(bins)
    expected = 2 * np.pi * HOP * k_index / N_FFT
    phase, previous = np.zeros((bins, frames)), np.empty(0, dtype=int)
    floor = floor_share * mag.max()
    last = np.zeros(bins)
    for t in range(frames):
        peaks = peaks_of(mag[:, t], floor)
        if len(peaks) == 0:
            previous = peaks
            continue
        dev = np.angle(np.exp(1j * (np.angle(truth[:, t]) - (np.angle(truth[:, t - 1]) if t else 0) - expected))) if t else np.zeros(bins)
        inst = k_index + dev * N_FFT / (2 * np.pi * HOP) if use_if else k_index.astype(float)
        peak_phase = np.empty(len(peaks))
        for i, k in enumerate(peaks):
            near = previous[np.abs(previous - k) <= 1] if len(previous) else previous
            peak_phase[i] = last[near[np.argmin(np.abs(near - k))]] + 2 * np.pi * HOP * inst[k] / N_FFT if len(near) else rng.uniform(0, 2 * np.pi)
        edges = np.concatenate([[0], (peaks[:-1] + peaks[1:]) // 2 + 1, [bins]])
        for i, k in enumerate(peaks):
            span = np.arange(edges[i], edges[i + 1])
            phase[span, t] = peak_phase[i] + np.pi * (span - k)
            last[k] = peak_phase[i]
        previous = peaks
    return phase


def griffin_lim(mag: torch.Tensor, phase: torch.Tensor, n: int, iters: int) -> torch.Tensor:
    s = mag * torch.exp(1j * phase)
    for _ in range(iters):
        s = mag * torch.exp(1j * torch.angle(stft(istft(s, n))))
    return istft(s, n)


def random_gl(mag: torch.Tensor, n: int) -> torch.Tensor:
    torch.manual_seed(0)
    return griffin_lim(mag, torch.rand_like(mag) * 2 * np.pi, n, 100)


def report(tag: str, x: torch.Tensor, mag: torch.Tensor, target: torch.Tensor) -> None:
    s = stft(x)
    inconsistency = float((stft(istft(mag * torch.exp(1j * torch.angle(s)), x.numel())) - mag * torch.exp(1j * torch.angle(s))).abs().norm() / mag.norm())
    got = cqt(x)
    goal = target**0.3
    error = float(((got**0.3 - goal) ** 2).sum() / (goal**2).sum())
    print(f"  {tag:10s} inconsistency {inconsistency:.3f}   surface error {error:.4f}", flush=True)


def probe(x: np.ndarray) -> str:
    worst, off = -200.0, []
    for start in (0.0, 1.2, 2.4):
        seg = x[int((start + 0.1) * RATE) : int((start + 0.9) * RATE)]
        padded = np.zeros(RATE * 4)
        padded[: len(seg)] = seg * np.hanning(len(seg))
        freqs, power = np.fft.rfftfreq(len(padded), 1 / RATE), np.abs(np.fft.rfft(padded)) ** 2
        for k in (2, 3, 4):
            carrier = power[np.abs(freqs - k * 220) <= 2].max()
            for offset in (11.72, 23.44):
                for sign in (-1, 1):
                    worst = max(worst, 10 * np.log10(power[np.abs(freqs - (k * 220 + sign * offset)) <= 1.5].max() / carrier))
        band = (freqs >= 350) & (freqs <= 3500)
        harmonic = np.zeros_like(freqs, dtype=bool)
        for k in range(2, 16):
            harmonic |= np.abs(freqs - k * 220) <= 6
        off.append(10 * np.log10(power[band & ~harmonic].sum() / power[band].sum()))
    return f"sideband {worst:6.1f} dB  off-harmonic {np.mean(off):6.1f} dB"


def pluck() -> torch.Tensor:
    n = 5 * RATE
    t = torch.arange(n) / RATE
    x = torch.zeros(n)
    for start in np.arange(0, 5, 1.2):
        age = (t - start).clamp_min(0)
        x = x + torch.where(t >= start, sum(torch.sin(2 * np.pi * 220 * k * age) / k for k in (1, 2, 3, 4)) * torch.exp(-age / 0.6), torch.zeros(n))
    return band_limit(x.numpy())


def stem(song: str) -> torch.Tensor:
    path = next(s.path for s in stems_of(song) if s.name == "guitar")
    return band_limit(codec.normalise(decode_mono(path, RATE)[: 750 * codec.HOP]).numpy())


OUT.mkdir(parents=True, exist_ok=True)
for prefix, wave in (("pl", pluck()), ("hg", stem("hanging_2")), ("jk", stem("jack_1"))):
    n = wave.numel()
    truth = stft(wave)
    target = cqt(wave)
    mag = to_linear(target, truth.shape[1])
    mag = mag * (truth.abs().norm() / mag.norm())
    truth_np, mag_np = truth.numpy(), mag.numpy()
    print(prefix, flush=True)
    variants = {
        "true": istft(truth, n),
        "gl": random_gl(mag, n),
        "lock": istft(mag * torch.exp(1j * torch.tensor(lock_phase(mag_np, truth_np, True), dtype=torch.float32)), n),
        "lock_flat": istft(mag * torch.exp(1j * torch.tensor(lock_phase(mag_np, truth_np, False), dtype=torch.float32)), n),
    }
    variants["lock_gl"] = griffin_lim(mag, torch.angle(stft(variants["lock"])), n, 30)
    for name, audio in variants.items():
        report(name, audio, mag, target)
        if prefix == "pl":
            print(f"             {probe(audio.numpy().astype(np.float64))}", flush=True)
        save(OUT / f"{prefix}_{name}.wav", audio, 0.9 / max(float(audio.abs().max()), 1e-9))
print("done")
