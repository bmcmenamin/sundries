"""Perturbed copies of a stem: material that sounds like the stem but is not its waveform. A library, used when a song
is profiled (texture_profile.py); the copies are made and encoded in memory, never written out.

Each variant stacks up to three effects, every one driven by its own slow sine envelope so the stem glides between a lightly
and a strongly modified version of itself, never reaching the original (the envelope has a floor):

  pitch     tape-style, so pitch and speed move together: a constant shift of up to a quarter of a semitone (out of tune, not
            another key), plus a wobble of up to 0.15 of a semitone as the envelope swings
  comb      the stem plus a copy of itself delayed by a fraction of a millisecond to a few, the delay swept by the envelope
            (a flanger / phaser sweep)
  sideband  single-sideband frequency shift by 10-60 Hz, swept by the envelope, partly mixed in: the inharmonic,
            shortwave-radio sound. No reverb, which would smear the room the later atmospherics are meant to supply.

The variants are encoded and added to the manifold the generator pulls toward (see codec_texture.py).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

import codec

RATE = codec.CODEC_RATE
PERIOD_RANGE = (1.0, 12.0)  # seconds, drawn log-uniformly
PITCH_RANGE = (-0.25, 0.25)  # constant shifts, in semitones, any value between: out of tune, not another key
WOBBLE_RANGE = (0.05, 0.15)  # semitones
COMB_GAIN_RANGE = (0.05, 0.15)
COMB_SWEEP_MS = (0.5, 2.0)  # how far the delay is swept above its low end
SHIFT_HZ_RANGE = (10.0, 60.0)
SHIFT_MIX_RANGE = (0.3, 0.7)
EFFECT_PROBABILITY = 0.6  # per effect; at least one is always on


@dataclass(frozen=True)
class Envelope:
    """A sine between `floor` and 1: how strongly an effect acts at time t."""

    period: float
    phase: float
    floor: float

    @classmethod
    def random(cls, rng: random.Random, floor: float) -> Envelope:
        low, high = (math.log(p) for p in PERIOD_RANGE)
        return cls(math.exp(rng.uniform(low, high)), rng.uniform(0, 2 * math.pi), floor)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        return self.floor + (1 - self.floor) * (0.5 + 0.5 * torch.sin(2 * math.pi * t / self.period + self.phase))


def sample_at(audio: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of `audio` at fractional sample positions (clamped to its ends)."""
    positions = positions.clamp(0, len(audio) - 1)
    low = positions.floor().long()
    high = (low + 1).clamp(max=len(audio) - 1)
    return audio[low] + (positions - low) * (audio[high] - audio[low])


def analytic(audio: torch.Tensor) -> torch.Tensor:
    """The audio plus i times its Hilbert transform (positive frequencies only)."""
    n = len(audio)
    keep = torch.zeros(n)
    keep[0] = 1
    keep[1 : (n + 1) // 2] = 2
    if n % 2 == 0:
        keep[n // 2] = 1
    return torch.fft.ifft(torch.fft.fft(audio) * keep)


@dataclass(frozen=True)
class Perturbation:
    """Zero for an amount means that effect is off."""

    semitones: float  # constant pitch shift over the whole file (a quarter of a semitone at most), signed, not restricted to whole semitones
    wobble: float  # peak deviation from it, in semitones, as the sine envelope swings (under a half step); 0 = pitch off
    pitch: Envelope
    comb_gain: float  # signed
    comb_delay_ms: tuple[float, float]  # swept between these
    comb: Envelope
    shift_hz: float  # peak frequency shift, signed
    shift_mix: float  # share of the shifted signal in the output
    shift: Envelope

    @classmethod
    def random(cls, rng: random.Random, floor: float = 0.25) -> Perturbation:
        on = [rng.random() < EFFECT_PROBABILITY for _ in range(3)]
        on = on if any(on) else [True, rng.random() < 0.5, rng.random() < 0.5]
        on = [True, True, False] if on == [False, True, False] else on  # a comb alone leaves the waveform correlated (~0.8)
        sign = lambda: rng.choice((-1, 1))
        delay_low = rng.uniform(0.2, 1.5)
        return cls(
            rng.uniform(*PITCH_RANGE) if on[0] else 0.0,
            rng.uniform(*WOBBLE_RANGE) if on[0] else 0.0,
            Envelope.random(rng, 0.0),
            sign() * rng.uniform(*COMB_GAIN_RANGE) if on[1] else 0.0,
            (delay_low, delay_low + rng.uniform(*COMB_SWEEP_MS)),
            Envelope.random(rng, floor),
            sign() * rng.uniform(*SHIFT_HZ_RANGE) if on[2] else 0.0,
            rng.uniform(*SHIFT_MIX_RANGE),
            Envelope.random(rng, floor),
        )

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """The same length (up to the rate change), at the loudness of the input."""
        t = torch.arange(len(audio)) / RATE
        sideband = self._sideband(audio, t) if self.shift_hz else audio
        combed = self._comb(sideband, t) if self.comb_gain else sideband
        out = self._pitch(combed) if self.wobble else combed
        return out * (audio.pow(2).mean().sqrt() / out.pow(2).mean().sqrt().clamp(min=1e-9))

    def _sideband(self, audio: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phase = 2 * math.pi * torch.cumsum(self.shift_hz * self.shift.at(t), 0) / RATE
        shifted = (analytic(audio) * torch.polar(torch.ones_like(phase), phase)).real
        return (1 - self.shift_mix) * audio + self.shift_mix * shifted

    def _comb(self, audio: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        low, high = self.comb_delay_ms
        delay = (low + (high - low) * self.comb.at(t)) * 1e-3 * RATE
        return audio + self.comb_gain * sample_at(audio, torch.arange(len(audio)) - delay)

    def _pitch(self, audio: torch.Tensor) -> torch.Tensor:
        n = int(len(audio) * 1.1)  # the slowest rate (about -0.35 semitones) needs about 1.02x as many output samples
        out_t = torch.arange(n) / RATE
        rate = torch.exp2((self.semitones + self.wobble * (2 * self.pitch.at(out_t) - 1)) / 12)
        positions = torch.cumsum(rate, 0)
        return sample_at(audio, positions[positions < len(audio) - 1])


