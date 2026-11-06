"""The mix chain: per-stem effects and the output level, applied to the stems of a generated track (chosen by ear on generated tracks).

    drums   heavy compression (8:1, threshold 20 dB under the stem's peak, fast attack), rms restored, then -15 dB
    bass    2nd-order high-pass at 60 Hz (less boom), then -9 dB
    others  untouched
    mix     the sum, set to MIX_RMS_DBFS rms (peak-capped at PEAK_CAP)

The generated drums and bass sit hot because the vocals that sat above them in the song are not played, hence the cuts; the cuts and the compression
lower the average level, hence the fixed rms. Every function takes audio shaped (n,) or (channels, n).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

DRUMS_DB, BASS_DB = -15.0, -9.0
BASS_HIGHPASS_HZ = 60
MIX_RMS_DBFS = -23.0
PEAK_CAP = 0.95


def gain(db: float) -> float:
    return 10 ** (db / 20)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt((audio ** 2).mean()))


@dataclass(frozen=True)
class Compressor:
    """Feed-forward, channel-linked, hard knee. The threshold is relative to the stem's own peak, so it needs no knowledge of the stem's scale; the makeup restores the stem's rms."""
    ratio: float
    below_peak_db: float
    attack_ms: float
    release_ms: float

    def smoothed(self, level: np.ndarray, rate: int) -> np.ndarray:
        """One-pole smoother, quick to rise (attack) and slow to fall (release): inherently sequential."""
        up, down = (np.exp(-1 / (rate * ms / 1000)) for ms in (self.attack_ms, self.release_ms))
        out, state = [], 0.0
        for x in level.tolist():
            k = up if x > state else down
            state = k * state + (1 - k) * x
            out.append(state)
        return np.array(out)

    def __call__(self, audio: np.ndarray, rate: int) -> np.ndarray:
        peak = float(np.abs(audio).max())
        if peak == 0:
            return audio
        detector = np.abs(audio).reshape(-1, audio.shape[-1]).max(axis=0)
        level_db = 20 * np.log10(np.maximum(self.smoothed(detector, rate), 1e-9))
        reduction_db = np.maximum(level_db - (20 * np.log10(peak) - self.below_peak_db), 0) * (1 - 1 / self.ratio)
        squashed = audio * gain(-reduction_db)
        return squashed * (rms(audio) / rms(squashed))


DRUM_COMPRESSOR = Compressor(ratio=8, below_peak_db=20, attack_ms=0.3, release_ms=100)


def voiced(stem: str, audio: np.ndarray, rate: int) -> np.ndarray:
    match stem:
        case "drums":
            return DRUM_COMPRESSOR(audio, rate) * gain(DRUMS_DB)
        case "bass":
            return sosfilt(butter(2, BASS_HIGHPASS_HZ, "highpass", fs=rate, output="sos"), audio, axis=-1) * gain(BASS_DB)
        case _:
            return audio


def at_level(audio: np.ndarray) -> np.ndarray:
    """Set the rms to MIX_RMS_DBFS, then scale down if that would push the peak over PEAK_CAP."""
    leveled = audio * gain(MIX_RMS_DBFS) / max(rms(audio), 1e-9)
    return leveled * min(1.0, PEAK_CAP / max(float(np.abs(leveled).max()), 1e-9))


def mix(stems: dict[str, np.ndarray], rate: int) -> np.ndarray:
    return at_level(sum(voiced(stem, audio, rate) for stem, audio in stems.items()))
