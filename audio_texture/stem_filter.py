"""Per-stem spectral post-filter: keep a generated stem inside its source stem's spectrum.

    venv/bin/python stem_filter.py nmf 02_generate_k4 hanging_1 drums

Reads every .wav in out/<pipeline>/<step>/<input>/<stem>/ and writes filtered copies to
out/<pipeline>/<step with 02_generate -> 03_postprocess>/<input>/<stem>/<uuid>.wav.
The filter (zero phase) does two things, both measured on the source stem:
  - long-term EQ: gain = sqrt(source spectrum / generated spectrum), smoothed over 1/3 octave
    and limited to +-12 dB, so tilt or a bright/dark skew in the generated stem is corrected;
  - band limits: roll off below / above the frequencies holding the first / last 0.5% of the
    source's energy, so nothing is added outside the stem's own band (which is what piles up
    as "whoosh" when stems are layered).
Loudness is preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from paths import OUT_ROOT, Stem, new_sample_path
from synthesize import SAMPLE_RATE, decode_mono, write_wav

GAIN_LIMIT_DB = 12.0
EDGE_FRACTION = 0.005
SMOOTH_OCTAVES = 1 / 3
N_FFT = 4096


def long_term_spectrum(x: torch.Tensor) -> torch.Tensor:
    """(N_FFT // 2 + 1,) mean power spectrum over Hann frames."""
    frames = torch.stft(x, N_FFT, hop_length=N_FFT // 4, window=torch.hann_window(N_FFT), return_complex=True)
    return (frames.real**2 + frames.imag**2).mean(-1)


def _smooth_log_frequency(power: torch.Tensor) -> torch.Tensor:
    """Smooth over SMOOTH_OCTAVES on a log-frequency axis (bins above 40 Hz), return on the linear axis."""
    freqs = torch.arange(power.numel()) * SAMPLE_RATE / N_FFT
    keep = freqs > 40
    log_f = torch.log2(freqs[keep])
    grid = torch.linspace(float(log_f.min()), float(log_f.max()), 400)
    index = torch.searchsorted(log_f, grid).clamp(1, log_f.numel() - 1)
    t = (grid - log_f[index - 1]) / (log_f[index] - log_f[index - 1])
    on_grid = power[keep][index - 1] * (1 - t) + power[keep][index] * t
    sigma = SMOOTH_OCTAVES / (float(grid[1] - grid[0]) * 2)
    taps = torch.arange(-int(3 * sigma) - 1, int(3 * sigma) + 2)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    padded = torch.nn.functional.pad(on_grid[None, None], (len(taps) // 2, len(taps) // 2), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, (kernel / kernel.sum())[None, None])[0, 0][: on_grid.numel()]
    back = torch.searchsorted(grid, log_f).clamp(1, grid.numel() - 1)
    u = (log_f - grid[back - 1]) / (grid[back] - grid[back - 1])
    result = power.clone()
    result[keep] = smoothed[back - 1] * (1 - u) + smoothed[back] * u
    return result


def band_edges(power: torch.Tensor) -> tuple[float, float]:
    """Frequencies below / above which the source holds EDGE_FRACTION of its energy."""
    cumulative = torch.cumsum(power, 0) / power.sum()
    low = int(torch.searchsorted(cumulative, torch.tensor(EDGE_FRACTION)))
    high = int(torch.searchsorted(cumulative, torch.tensor(1 - EDGE_FRACTION)))
    return low * SAMPLE_RATE / N_FFT, high * SAMPLE_RATE / N_FFT


def filter_gain(source: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """(N_FFT // 2 + 1,) amplitude gain taking the generated stem's spectrum to the source's."""
    source_power, generated_power = long_term_spectrum(source), long_term_spectrum(generated)
    generated_power = generated_power * (source_power.sum() / generated_power.sum())
    limit = 10 ** (GAIN_LIMIT_DB / 20)
    eq = (_smooth_log_frequency(source_power) / (_smooth_log_frequency(generated_power) + 1e-12)).sqrt().clamp(1 / limit, limit)
    low, high = band_edges(source_power)
    freqs = torch.arange(eq.numel()) * SAMPLE_RATE / N_FFT
    octaves_below = torch.log2((freqs + 1e-3) / max(low, 1.0))
    octaves_above = torch.log2((freqs + 1e-3) / max(high, 1.0))
    roll = lambda octaves: 0.5 - 0.5 * torch.cos(torch.pi * (octaves / SMOOTH_OCTAVES).clamp(0, 1))
    return eq * roll(octaves_below) * roll(-octaves_above)


def postprocess(generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Apply the zero-phase gain, then restore the generated stem's RMS."""
    n = generated.numel()
    gain = filter_gain(source, generated)
    grid = torch.linspace(0, SAMPLE_RATE / 2, n // 2 + 1)
    positions = (grid / (SAMPLE_RATE / N_FFT)).clamp(max=gain.numel() - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=gain.numel() - 1)
    fraction = positions - lower
    full_gain = gain[lower] * (1 - fraction) + gain[upper] * fraction
    filtered = torch.fft.irfft(torch.fft.rfft(generated) * full_gain, n=n)
    return filtered * (generated.pow(2).mean().sqrt() / (filtered.pow(2).mean().sqrt() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", help="e.g. nmf")
    parser.add_argument("step", help="generation step, e.g. 02_generate_k4")
    parser.add_argument("source", help="input name, e.g. hanging_1")
    parser.add_argument("stem", help="stem name, e.g. drums")
    args = parser.parse_args()

    source = decode_mono(Stem(args.source, args.stem).path)
    in_dir = OUT_ROOT / args.pipeline / args.step / args.source / args.stem
    out_dir = OUT_ROOT / args.pipeline / args.step.replace("02_generate", "03_postprocess") / args.source / args.stem
    for path in sorted(in_dir.glob("*.wav")):
        out = new_sample_path(out_dir)
        write_wav(out, postprocess(decode_mono(path), source))
        print(f"{path.name} -> {out}")


if __name__ == "__main__":
    main()
