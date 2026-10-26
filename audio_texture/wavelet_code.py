"""Convolutional sparse coding of wavelet coefficients (complex, group-sparse).

    venv/bin/python wavelet_code.py walk_1 piano

The stem is analysed with wavelet.py into complex baseband coefficients
c[band, frame]. Atoms are small complex 2-D patches (a band span x a time span);
a spike a_k[b, t] places atom k at band b, frame t with a complex amplitude, so
shifting along the band axis is a pitch shift and the complex amplitude is a
phase rotation. The L1 penalty acts on |a| (complex soft-threshold), so one atom
serves every phase. Writes out/wavelet/01_dictionary/<input>/<stem>/{dictionary.pt,recon.wav}.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len

from paths import WAVELET, Stem, step_dir
from synthesize import SAMPLE_RATE, decode_mono, write_wav
from wavelet import DECIMATION, Wavelet

LIPSCHITZ_MARGIN = 1.05
EPS = 1e-12


def shrink(a: torch.Tensor, threshold: float) -> torch.Tensor:
    """Complex soft-threshold: shrink |a| toward zero, keep the phase."""
    return a * (1 - threshold / (a.abs() + EPS)).clamp(min=0)


def unit_norm(atoms: torch.Tensor) -> torch.Tensor:
    return atoms / (atoms.abs().pow(2).sum((-2, -1), keepdim=True).sqrt() + 1e-8)


def _padded(out_shape: tuple[int, int], atom_shape: tuple[int, int]) -> tuple[int, int]:
    return next_fast_len(out_shape[0] + atom_shape[0] - 1), next_fast_len(out_shape[1] + atom_shape[1] - 1)


def upsample(codes: torch.Tensor, stride: int, frames: int) -> torch.Tensor:
    """Place codes on every `stride`-th frame of a (K, B, frames) grid."""
    full = torch.zeros(*codes.shape[:-1], frames, dtype=codes.dtype)
    full[..., ::stride] = codes
    return full


@dataclass(frozen=True)
class Operator:
    """The linear map from codes (K, B, ceil(W / stride)) to coefficients (B, W), via FFT convolution.

    Spikes sit on every `stride`-th frame (all bands), which keeps the code from
    filling with near-duplicate neighbours that the coherent dictionary invites.
    """

    spectra: torch.Tensor  # (K, pb, pt) FFT of the zero-padded atoms
    atom_shape: tuple[int, int]
    out_shape: tuple[int, int]
    stride: int = 1

    @classmethod
    def build(cls, atoms: torch.Tensor, out_shape: tuple[int, int], stride: int = 1) -> Operator:
        atom_shape = (atoms.shape[1], atoms.shape[2])
        return cls(torch.fft.fft2(atoms, s=_padded(out_shape, atom_shape)), atom_shape, out_shape, stride)

    @property
    def code_shape(self) -> tuple[int, int, int]:
        return (self.spectra.shape[0], self.out_shape[0], -(-self.out_shape[1] // self.stride))

    @property
    def _size(self) -> tuple[int, int]:
        return self.spectra.shape[-2], self.spectra.shape[-1]

    @property
    def _centre(self) -> tuple[int, int]:
        return self.atom_shape[0] // 2, self.atom_shape[1] // 2

    def decode_spectra(self, code_spectra: torch.Tensor) -> torch.Tensor:
        """Like `decode`, from the FFT of the codes (lets callers reuse it)."""
        full = torch.fft.ifft2((code_spectra * self.spectra).sum(0))
        (cb, ct), (b, w) = self._centre, self.out_shape
        return full[cb : cb + b, ct : ct + w]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decode_spectra(torch.fft.fft2(upsample(codes, self.stride, self.out_shape[1]), s=self._size))

    def analyse(self, coefficients: torch.Tensor) -> torch.Tensor:
        """(B, W) -> (K, B, ceil(W / stride)): correlation with every atom; the adjoint of `decode`."""
        correlation = torch.fft.ifft2(torch.fft.fft2(coefficients, s=self._size)[None] * self.spectra.conj())
        (cb, ct), (b, w) = self._centre, self.out_shape
        return torch.roll(correlation, shifts=(cb, ct), dims=(-2, -1))[:, :b, :w:self.stride]

    def lipschitz(self) -> float:
        return float(self.spectra.abs().pow(2).sum(0).max()) * LIPSCHITZ_MARGIN


@torch.no_grad()
def encode(
    coefficients: torch.Tensor, operator: Operator, lam: float, iters: int, init: torch.Tensor | None = None
) -> torch.Tensor:
    """FISTA for 0.5 * ||c - D a||^2 + lam * sum |a|."""
    step = 1 / operator.lipschitz()
    a = torch.zeros(operator.code_shape, dtype=coefficients.dtype) if init is None else init
    y, t = a, 1.0
    for _ in range(iters):
        a_next = shrink(y + step * operator.analyse(coefficients - operator.decode(y)), step * lam)
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = a_next + (t - 1) / t_next * (a_next - a)
        a, t = a_next, t_next
    return a


def update_atoms(
    coefficients: torch.Tensor, codes: torch.Tensor, atoms: torch.Tensor, steps: int, stride: int
) -> torch.Tensor:
    """A few Adam steps on the reconstruction error with the codes fixed, then renormalise."""
    out_shape = (coefficients.shape[0], coefficients.shape[1])
    atom_shape = (atoms.shape[1], atoms.shape[2])
    code_spectra = torch.fft.fft2(upsample(codes, stride, out_shape[1]), s=_padded(out_shape, atom_shape))
    phi = atoms.clone().requires_grad_()
    optimiser = torch.optim.Adam([phi], lr=0.07 / math.sqrt(atom_shape[0] * atom_shape[1]))
    for _ in range(steps):
        optimiser.zero_grad()
        residual = coefficients - Operator.build(phi, out_shape, stride).decode_spectra(code_spectra)
        residual.abs().pow(2).sum().backward()
        optimiser.step()
    return unit_norm(phi.detach())


def revive_dead_atoms(coefficients: torch.Tensor, codes: torch.Tensor, operator: Operator, atoms: torch.Tensor) -> torch.Tensor:
    """Replace unused atoms with the highest-energy residual patches."""
    dead = (codes.abs().sum((-2, -1)) == 0).nonzero().flatten()
    if dead.numel() == 0:
        return atoms
    residual = coefficients - operator.decode(codes)
    ab, at = atoms.shape[1], atoms.shape[2]
    energy = F.avg_pool2d(residual.abs().pow(2)[None, None], (ab, at), stride=(ab // 2, at // 2))[0, 0].flatten()
    top = energy.topk(min(dead.numel(), energy.numel())).indices
    columns = F.avg_pool2d(residual.abs().pow(2)[None, None], (ab, at), stride=(ab // 2, at // 2)).shape[-1]
    patches = torch.stack(
        [residual[(i // columns) * (ab // 2) : (i // columns) * (ab // 2) + ab, (i % columns) * (at // 2) : (i % columns) * (at // 2) + at] for i in top.tolist()]
    )
    return atoms.index_copy(0, dead[: patches.shape[0]], unit_norm(patches))


def initial_atoms(n_atoms: int, atom_shape: tuple[int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.complex(
        torch.randn(n_atoms, *atom_shape, generator=generator), torch.randn(n_atoms, *atom_shape, generator=generator)
    )
    window = torch.hann_window(atom_shape[0] + 2)[1:-1, None] * torch.hann_window(atom_shape[1] + 2)[1:-1][None, :]
    return unit_norm(noise * window)


def learn_atoms(
    coefficients: torch.Tensor, n_atoms: int, atom_shape: tuple[int, int], lam_rel: float, rounds: int,
    encode_iters: int, stride: int, seed: int,
) -> torch.Tensor:
    atoms = initial_atoms(n_atoms, atom_shape, seed)
    codes = None
    for i in range(rounds):
        operator = Operator.build(atoms, tuple(coefficients.shape), stride)
        lam = lam_rel * float(operator.analyse(coefficients).abs().max())
        codes = encode(coefficients, operator, lam, encode_iters, init=codes)
        atoms = update_atoms(
            coefficients, codes, revive_dead_atoms(coefficients, codes, operator, atoms), steps=10, stride=stride
        )
        print(f"round {i + 1}/{rounds}  {summarise(coefficients, Operator.build(atoms, tuple(coefficients.shape), stride), codes)}", flush=True)
    return atoms


def snr_db(target: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(10 * torch.log10(target.abs().pow(2).sum() / ((target - estimate).abs().pow(2).sum() + EPS)))


def summarise(coefficients: torch.Tensor, operator: Operator, codes: torch.Tensor, seconds: float | None = None) -> str:
    seconds = seconds or coefficients.shape[1] * DECIMATION / SAMPLE_RATE
    nonzero = int((codes != 0).sum())
    used = int((codes.abs().sum((-2, -1)) > 0).sum())
    return (
        f"coef SNR {snr_db(coefficients, operator.decode(codes)):5.1f} dB  {nonzero / seconds:7.0f} spikes/s  "
        f"{used}/{codes.shape[0]} atoms used"
    )


def trim(x: torch.Tensor) -> torch.Tensor:
    return x[: x.numel() // (2 * DECIMATION) * (2 * DECIMATION)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--atom-bands", type=int, default=24, help="atom span in bands (12 per octave)")
    parser.add_argument("--atom-frames", type=int, default=600, help="atom span in frames (1500 per second)")
    parser.add_argument("--stride", type=int, default=1, help="spikes only on every Nth frame")
    parser.add_argument("--lam", type=float, default=0.05, help="L1 weight relative to max|D^T c|")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--encode-iters", type=int, default=20)
    parser.add_argument("--train-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    x = decode_mono(Stem(args.source, args.stem).path)
    x = x / x.pow(2).mean().sqrt()
    train = trim(x[: round(args.train_seconds * SAMPLE_RATE)])
    atoms = learn_atoms(
        Wavelet.for_length(train.numel()).analyse(train), args.atoms, (args.atom_bands, args.atom_frames),
        args.lam, args.rounds, args.encode_iters, args.stride, args.seed,
    )

    x = trim(x)
    wavelet = Wavelet.for_length(x.numel())
    coefficients = wavelet.analyse(x)
    operator = Operator.build(atoms, tuple(coefficients.shape), args.stride)
    lam = args.lam * float(operator.analyse(coefficients).abs().max())
    codes = encode(coefficients, operator, lam, 6 * args.encode_iters)
    rebuilt = wavelet.synthesize(operator.decode(codes))
    print(f"full stem  {summarise(coefficients, operator, codes)}  waveform SNR {snr_db(wavelet.covered(x), rebuilt):.1f} dB")

    out_dir = step_dir(WAVELET, "01_dictionary", args.source, args.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"atoms": atoms, "lam_rel": args.lam, "stride": args.stride}, out_dir / "dictionary.pt")
    write_wav(out_dir / "recon.wav", rebuilt)


if __name__ == "__main__":
    main()
