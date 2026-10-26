"""Does a stem's phase carry structure? A surrogate test, per band.

    venv/bin/python phase_structure.py walk_1 piano guitar --other hanging_1 drums vocals

For every band of the wavelet analysis three amplitude-weighted phase concentrations are
measured (R = |sum z| / sum |z|, 0 = random phase, 1 = perfectly locked):
  advance    frame-to-frame phase advance within a band          -> steady partials (a stable frequency)
  neighbour  phase difference between adjacent bands (remodulated) -> clicks / onsets (consistent group delay)
  harmonic   phase of band b+d against h * phase of band b, best over harmonic offsets
                                                                  -> phase-locked partials of one note
The reference is the same measurement on surrogates with the stem's own amplitude pattern but
phases taken from white noise seen through the same filterbank (what a noise-phase render does),
so both the filters' phase correlations and the stem's burstiness are accounted for.
z = (R - mean_surrogate) / std_surrogate over several draws. Large z: the stem's phase is
predictable and worth preserving; z near 0 (or negative): it is indistinguishable from noise
phase, so noise phase loses nothing.
Prints the median z per octave of bands.
"""

from __future__ import annotations

import argparse

import torch

from histogram_critic import HARMONICS, carrier
from paths import Stem
from synthesize import decode_mono
from wavelet import BANDS_PER_OCTAVE, Wavelet, centre_frequencies
from wavelet_code import trim

NOISE_DRAWS = 16


def _concentration(z: torch.Tensor) -> torch.Tensor:
    """(bands,) amplitude-weighted resultant length of the phasors z, summed over frames."""
    return z.sum(-1).abs() / (z.abs().sum(-1) + 1e-12)


def phase_scores(coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-band concentrations; bands without a partner band (top of the range) get 0."""
    s = coefficients * carrier(coefficients.shape[1] * 8)
    bands = coefficients.shape[0]
    advance = _concentration(coefficients[:, 1:] * coefficients[:, :-1].conj())
    neighbour = torch.zeros(bands)
    neighbour[:-1] = _concentration(s[1:] * s[:-1].conj())
    harmonic = torch.zeros(bands)
    for offset, h in HARMONICS:
        r = torch.zeros(bands)
        r[:-offset] = _concentration(s[offset:] * s[:-offset].conj() ** h)
        harmonic = torch.maximum(harmonic, r)
    # the combined score gets its own null: the max of several statistics is biased upward by chance
    return {"advance": advance, "neighbour": neighbour, "harmonic": harmonic, "structure": torch.maximum(neighbour, harmonic)}


def surrogate_reference(
    coefficients: torch.Tensor, wavelet: Wavelet, seed: int = 0
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Scores of surrogates that keep the stem's amplitude pattern but take their phases from filtered noise.

    Matching the amplitude envelope matters: a bursty stem concentrates its weight in few frames, which
    raises every concentration by chance, so plain white noise would be too easy a null.
    """
    generator = torch.Generator().manual_seed(seed)

    def surrogate() -> dict[str, torch.Tensor]:
        noise_phase = wavelet.analyse(torch.randn(wavelet.n_samples, generator=generator)).angle()
        rebuilt = wavelet.synthesize(torch.polar(coefficients.abs(), noise_phase))
        return phase_scores(wavelet.analyse(rebuilt))

    draws = [surrogate() for _ in range(NOISE_DRAWS)]
    return {k: (torch.stack([d[k] for d in draws]).mean(0), torch.stack([d[k] for d in draws]).std(0) + 1e-6) for k in draws[0]}


def z_scores(stem: Stem) -> dict[str, torch.Tensor]:
    audio = decode_mono(stem.path)
    audio = trim(audio / audio.pow(2).mean().sqrt())
    wavelet = Wavelet.for_length(audio.numel())
    coefficients = wavelet.analyse(audio)
    scores = phase_scores(coefficients)
    reference = surrogate_reference(coefficients, wavelet)
    return {k: (scores[k] - reference[k][0]) / reference[k][1] for k in scores}


def per_octave(z: torch.Tensor) -> list[float]:
    usable = z[: (z.numel() // BANDS_PER_OCTAVE) * BANDS_PER_OCTAVE].reshape(-1, BANDS_PER_OCTAVE)
    return usable.median(dim=1).values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input name, e.g. walk_1")
    parser.add_argument("stems", nargs="+", help="stems of that input")
    parser.add_argument("--other", nargs="+", default=[], help="a second input followed by its stems")
    args = parser.parse_args()

    jobs = [(args.source, name) for name in args.stems]
    if args.other:
        jobs += [(args.other[0], name) for name in args.other[1:]]
    octaves = [f"{centre_frequencies()[i * BANDS_PER_OCTAVE]:.0f}Hz" for i in range(5)]
    print(f"median z per octave (starting at {', '.join(octaves)})")
    for source, name in jobs:
        z = z_scores(Stem(source, name))
        for kind in ("advance", "neighbour", "harmonic"):
            print(f"{source}/{name:8s} {kind:10s} " + " ".join(f"{v:7.1f}" for v in per_octave(z[kind])))


if __name__ == "__main__":
    main()
