"""Convolutional sparse coding of the amplitude / frequency maps (ampfreq.py).

    venv/bin/python ampfreq_code.py walk_1 piano

Signal: x[c, band, frame] with two channels (amplitude, frequency offset). Atoms
are small real 2-channel patches (band span x time span); a spike a_k[band, frame]
places atom k there with a signed amplitude. Shifting an atom along the band axis
is a transposition (the frequency channel is relative, so it transposes cleanly).
L1 penalty, FISTA. Writes out/ampfreq/01_dictionary/<input>/<stem>/:
  dictionary.pt, recon.wav (from the sparse code), recon_features_only.wav (no coding:
  the ceiling for what this representation can reconstruct).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len

from ampfreq import coefficients_from, features, reconstruct_audio, snr_db
from paths import AMPFREQ, Stem, step_dir
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import DECIMATION, Wavelet
from wavelet_code import shrink, trim, upsample

LIPSCHITZ_MARGIN = 1.05
EPS = 1e-12


def unit_norm(atoms: torch.Tensor) -> torch.Tensor:
    return atoms / (atoms.pow(2).sum((1, 2, 3), keepdim=True).sqrt() + 1e-8)


def _padded(out_shape: tuple[int, int], atom_shape: tuple[int, int]) -> tuple[int, int]:
    return (
        next_fast_len(out_shape[0] + atom_shape[0] - 1, real=True),
        next_fast_len(out_shape[1] + atom_shape[1] - 1, real=True),
    )


@dataclass(frozen=True)
class Operator:
    """Linear map from codes (K, B, ceil(W / stride)) to feature maps (C, B, W), by FFT convolution."""

    spectra: torch.Tensor  # (K, C, pb, pt // 2 + 1)
    size: tuple[int, int]
    atom_shape: tuple[int, int]
    out_shape: tuple[int, int]
    stride: int

    @classmethod
    def build(cls, atoms: torch.Tensor, out_shape: tuple[int, int], stride: int) -> Operator:
        atom_shape = (atoms.shape[2], atoms.shape[3])
        size = _padded(out_shape, atom_shape)
        return cls(torch.fft.rfft2(atoms, s=size), size, atom_shape, out_shape, stride)

    @property
    def code_shape(self) -> tuple[int, int, int]:
        return (self.spectra.shape[0], self.out_shape[0], -(-self.out_shape[1] // self.stride))

    @property
    def _centre(self) -> tuple[int, int]:
        return self.atom_shape[0] // 2, self.atom_shape[1] // 2

    def decode_spectra(self, code_spectra: torch.Tensor) -> torch.Tensor:
        full = torch.fft.irfft2((code_spectra[:, None] * self.spectra).sum(0), s=self.size)
        (cb, ct), (b, w) = self._centre, self.out_shape
        return full[:, cb : cb + b, ct : ct + w]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decode_spectra(torch.fft.rfft2(upsample(codes, self.stride, self.out_shape[1]), s=self.size))

    def analyse(self, x: torch.Tensor) -> torch.Tensor:
        """(C, B, W) -> (K, B, ceil(W / stride)): the adjoint of `decode`."""
        correlation = torch.fft.irfft2(
            (torch.fft.rfft2(x, s=self.size)[None] * self.spectra.conj()).sum(1), s=self.size
        )
        (cb, ct), (b, w) = self._centre, self.out_shape
        return torch.roll(correlation, shifts=(cb, ct), dims=(-2, -1))[:, :b, :w:self.stride]

    def lipschitz(self) -> float:
        return float(self.spectra.abs().pow(2).sum((0, 1)).max()) * LIPSCHITZ_MARGIN


@torch.no_grad()
def encode(x: torch.Tensor, operator: Operator, lam: float, iters: int, init: torch.Tensor | None = None) -> torch.Tensor:
    """FISTA for 0.5 * ||x - D a||^2 + lam * ||a||_1."""
    step = 1 / operator.lipschitz()
    a = torch.zeros(operator.code_shape) if init is None else init
    y, t = a, 1.0
    for _ in range(iters):
        a_next = shrink(y + step * operator.analyse(x - operator.decode(y)), step * lam)
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = a_next + (t - 1) / t_next * (a_next - a)
        a, t = a_next, t_next
    return a


def update_atoms(x: torch.Tensor, codes: torch.Tensor, atoms: torch.Tensor, steps: int, stride: int) -> torch.Tensor:
    out_shape = (x.shape[1], x.shape[2])
    atom_shape = (atoms.shape[2], atoms.shape[3])
    code_spectra = torch.fft.rfft2(upsample(codes, stride, out_shape[1]), s=_padded(out_shape, atom_shape))
    phi = atoms.clone().requires_grad_()
    optimiser = torch.optim.Adam([phi], lr=0.07 / math.sqrt(atoms[0].numel()))
    for _ in range(steps):
        optimiser.zero_grad()
        (x - Operator.build(phi, out_shape, stride).decode_spectra(code_spectra)).pow(2).sum().backward()
        optimiser.step()
    return unit_norm(phi.detach())


def revive_dead_atoms(x: torch.Tensor, codes: torch.Tensor, operator: Operator, atoms: torch.Tensor) -> torch.Tensor:
    """Replace unused atoms with the highest-energy residual patches."""
    dead = (codes.abs().sum((-2, -1)) == 0).nonzero().flatten()
    if dead.numel() == 0:
        return atoms
    residual = x - operator.decode(codes)
    ab, at = atoms.shape[2], atoms.shape[3]
    stride = (ab // 2, at // 2)
    energy = F.avg_pool2d(residual.pow(2).sum(0)[None, None], (ab, at), stride=stride)[0, 0]
    top = energy.flatten().topk(min(dead.numel(), energy.numel())).indices.tolist()
    patches = torch.stack(
        [residual[:, (i // energy.shape[1]) * stride[0] :][:, :ab, (i % energy.shape[1]) * stride[1] :][..., :at] for i in top]
    )
    return atoms.index_copy(0, dead[: patches.shape[0]], unit_norm(patches))


def initial_atoms(n_atoms: int, channels: int, atom_shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n_atoms, channels, *atom_shape, generator=generator)
    window = torch.hann_window(atom_shape[0] + 2)[1:-1, None] * torch.hann_window(atom_shape[1] + 2)[1:-1][None, :]
    return unit_norm(noise * window)


def learn_atoms(
    x: torch.Tensor, n_atoms: int, atom_shape: tuple[int, int], lam_rel: float, rounds: int, encode_iters: int,
    stride: int, seed: int,
) -> torch.Tensor:
    atoms = initial_atoms(n_atoms, x.shape[0], atom_shape, seed)
    codes = None
    for i in range(rounds):
        operator = Operator.build(atoms, (x.shape[1], x.shape[2]), stride)
        lam = lam_rel * float(operator.analyse(x).abs().max())
        codes = encode(x, operator, lam, encode_iters, init=codes)
        atoms = update_atoms(x, codes, revive_dead_atoms(x, codes, operator, atoms), steps=10, stride=stride)
        print(f"round {i + 1}/{rounds}  {summarise(x, Operator.build(atoms, (x.shape[1], x.shape[2]), stride), codes)}", flush=True)
    return atoms


def summarise(x: torch.Tensor, operator: Operator, codes: torch.Tensor) -> str:
    seconds = x.shape[2] * DECIMATION / SAMPLE_RATE
    estimate = operator.decode(codes)
    return (
        f"amplitude SNR {snr_db(x[0], estimate[0]):5.1f} dB  frequency SNR {snr_db(x[1], estimate[1]):5.1f} dB  "
        f"{int((codes != 0).sum()) / seconds:7.0f} spikes/s  {int((codes.abs().sum((-2, -1)) > 0).sum())}/{codes.shape[0]} atoms used"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--atom-bands", type=int, default=16)
    parser.add_argument("--atom-frames", type=int, default=150, help="frames at 1500 per second")
    parser.add_argument("--stride", type=int, default=12, help="spikes only on every Nth frame")
    parser.add_argument("--lam", type=float, default=0.05, help="L1 weight relative to max|D^T x|")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--encode-iters", type=int, default=15)
    parser.add_argument("--train-seconds", type=float, default=8.0)
    parser.add_argument("--recon-iters", type=int, default=0, help="waveform-consistency iterations (0 = integrate only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stem = Stem(args.source, args.stem)
    audio = decode_mono(stem.path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    wavelet = Wavelet.for_length(audio.numel())
    maps = features(wavelet.analyse(audio))
    scales = maps.pow(2).mean((1, 2), keepdim=True).sqrt()
    x = maps / scales

    out_dir = step_dir(AMPFREQ, "01_dictionary", args.source, args.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "recon_features_only.wav", _render(maps, wavelet, args.recon_iters))

    train_frames = trim(audio[: round(args.train_seconds * SAMPLE_RATE)]).numel() // DECIMATION
    atoms = learn_atoms(
        x[:, :, :train_frames], args.atoms, (args.atom_bands, args.atom_frames), args.lam, args.rounds,
        args.encode_iters, args.stride, args.seed,
    )
    operator = Operator.build(atoms, (x.shape[1], x.shape[2]), args.stride)
    lam = args.lam * float(operator.analyse(x).abs().max())
    codes = encode(x, operator, lam, 6 * args.encode_iters)
    print(f"full stem  {summarise(x, operator, codes)}")

    torch.save({"atoms": atoms, "scales": scales, "lam_rel": args.lam, "stride": args.stride}, out_dir / "dictionary.pt")
    write_wav(out_dir / "recon.wav", _render(operator.decode(codes) * scales, wavelet, args.recon_iters))


def _render(maps: torch.Tensor, wavelet: Wavelet, iters: int) -> torch.Tensor:
    if iters:
        return reconstruct_audio(maps, wavelet, iters)
    return wavelet.synthesize(coefficients_from(maps))


if __name__ == "__main__":
    main()
