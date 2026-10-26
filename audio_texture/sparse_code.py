"""Convolutional sparse coding of a mono stem (after Smith & Lewicki 2006).

    venv/bin/python sparse_code.py walk_1 piano

A waveform is modelled as x[t] ~ sum_k sum_tau a_k[tau] * phi_k[t - tau + L//2]:
each nonzero a_k[tau] is a "spike" placing atom k at time tau. The atoms are
learned from the stem itself; the code a is found by FISTA with an L1 penalty.
The decoder is linear, so a regenerated code renders phase-coherent audio with
no phase retrieval. Writes out/sparse_raw/01_dictionary/<input>/<stem>/{dictionary.pt,recon.wav}.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from paths import SPARSE_RAW, Stem, step_dir
from synthesize import SAMPLE_RATE, decode_mono, write_wav

LIPSCHITZ_FFT = 8192
LIPSCHITZ_MARGIN = 1.05


def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return x.sign() * (x.abs() - threshold).clamp(min=0)


def unit_norm(filters: torch.Tensor) -> torch.Tensor:
    return filters / (filters.norm(dim=-1, keepdim=True) + 1e-8)


@dataclass(frozen=True)
class Dictionary:
    filters: torch.Tensor  # (K, L), unit-norm rows

    @property
    def n_atoms(self) -> int:
        return self.filters.shape[0]

    @property
    def length(self) -> int:
        return self.filters.shape[1]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """(K, T) spikes -> (T,) waveform. Spike at tau puts the atom's centre at tau."""
        centre = self.length // 2
        full = F.conv_transpose1d(codes[None], self.filters[:, None, :])[0, 0]
        return full[centre : centre + codes.shape[-1]]

    def analyse(self, x: torch.Tensor) -> torch.Tensor:
        """(T,) -> (K, T): correlation with every atom; the adjoint of `decode`."""
        centre = self.length // 2
        padded = F.pad(x, (centre, self.length - 1 - centre))
        return F.conv1d(padded[None, None], self.filters[:, None, :])[0]

    def lipschitz(self) -> float:
        """Upper bound on ||D||^2 for the convolution operator."""
        spectrum = torch.fft.rfft(self.filters, n=LIPSCHITZ_FFT).abs().pow(2).sum(0)
        return float(spectrum.max()) * LIPSCHITZ_MARGIN

    def peak_frequencies(self) -> torch.Tensor:
        spectrum = torch.fft.rfft(self.filters, n=LIPSCHITZ_FFT).abs()
        return spectrum.argmax(-1) * SAMPLE_RATE / LIPSCHITZ_FFT


@torch.no_grad()
def encode(x: torch.Tensor, dictionary: Dictionary, lam: float, iters: int, init: torch.Tensor | None = None) -> torch.Tensor:
    """FISTA for 0.5 * ||x - D a||^2 + lam * ||a||_1."""
    step = 1 / dictionary.lipschitz()
    a = torch.zeros(dictionary.n_atoms, x.shape[-1]) if init is None else init
    y, t = a, 1.0
    for _ in range(iters):
        a_next = soft_threshold(y + step * dictionary.analyse(x - dictionary.decode(y)), step * lam)
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = a_next + (t - 1) / t_next * (a_next - a)
        a, t = a_next, t_next
    return a


def update_filters(x: torch.Tensor, codes: torch.Tensor, filters: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    """A few Adam steps on the reconstruction error, codes fixed, then renormalise."""
    phi = filters.clone().requires_grad_()
    optimiser = torch.optim.Adam([phi], lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        (x - Dictionary(phi).decode(codes)).pow(2).sum().backward()
        optimiser.step()
    return unit_norm(phi.detach())


def revive_dead_atoms(x: torch.Tensor, codes: torch.Tensor, dictionary: Dictionary) -> torch.Tensor:
    """Replace unused atoms with the highest-energy residual segments."""
    dead = (codes.abs().sum(-1) == 0).nonzero().flatten()
    if dead.numel() == 0:
        return dictionary.filters
    residual = x - dictionary.decode(codes)
    energy = F.avg_pool1d(residual.pow(2)[None, None], dictionary.length, stride=dictionary.length // 2)[0, 0]
    starts = energy.topk(dead.numel()).indices * (dictionary.length // 2)
    segments = torch.stack([residual[s : s + dictionary.length] for s in starts])
    return dictionary.filters.index_copy(0, dead, unit_norm(segments))


def learn_dictionary(
    x: torch.Tensor, n_atoms: int, length: int, lam_rel: float, rounds: int, encode_iters: int, seed: int
) -> tuple[Dictionary, torch.Tensor]:
    """Alternate FISTA coding with filter updates. `lam_rel` scales the L1 weight to max|D^T x|."""
    generator = torch.Generator().manual_seed(seed)
    dictionary = Dictionary(unit_norm(torch.randn(n_atoms, length, generator=generator)))
    codes = None
    for i in range(rounds):
        lam = lam_rel * float(dictionary.analyse(x).abs().max())
        codes = encode(x, dictionary, lam, encode_iters, init=codes)
        filters = revive_dead_atoms(x, codes, dictionary)
        dictionary = Dictionary(update_filters(x, codes, filters, steps=10, lr=3e-3))
        print(f"round {i + 1}/{rounds}  {summarise(x, dictionary, codes)}", flush=True)
    lam = lam_rel * float(dictionary.analyse(x).abs().max())
    return dictionary, encode(x, dictionary, lam, 4 * encode_iters, init=codes)


def snr_db(x: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(10 * torch.log10(x.pow(2).sum() / ((x - estimate).pow(2).sum() + 1e-12)))


def summarise(x: torch.Tensor, dictionary: Dictionary, codes: torch.Tensor) -> str:
    spikes = (codes != 0).sum()
    return (
        f"SNR {snr_db(x, dictionary.decode(codes)):5.1f} dB  "
        f"{float(spikes) / (x.numel() / SAMPLE_RATE):7.0f} spikes/s  "
        f"{int((codes.abs().sum(-1) > 0).sum())}/{dictionary.n_atoms} atoms used"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stem", help="stem name, e.g. piano")
    parser.add_argument("--atoms", type=int, default=48)
    parser.add_argument("--length", type=int, default=512, help="atom length in samples")
    parser.add_argument("--lam", type=float, default=0.1, help="L1 weight relative to max|D^T x|")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--encode-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    x = decode_mono(Stem(args.source, args.stem).path)
    x = x / x.pow(2).mean().sqrt()
    dictionary, codes = learn_dictionary(x, args.atoms, args.length, args.lam, args.rounds, args.encode_iters, args.seed)

    out_dir = step_dir(SPARSE_RAW, "01_dictionary", args.source, args.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"filters": dictionary.filters, "lam_rel": args.lam}, out_dir / "dictionary.pt")
    write_wav(out_dir / "recon.wav", dictionary.decode(codes))
    print(f"final  {summarise(x, dictionary, codes)}")
    print("atom peak frequencies (Hz):", [int(f) for f in dictionary.peak_frequencies().sort().values])


if __name__ == "__main__":
    main()
