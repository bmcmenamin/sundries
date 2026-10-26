"""Split a recording into instrument stems with Demucs.

    venv/bin/python separate.py inputs/walk_1.mp3

Writes out/separation/01_demucs/<input>/<stem>/stem.wav at the original level
(44.1 kHz stereo float32), so that later steps can match each stem's loudness.
Default model htdemucs_6s: drums, bass, other, vocals, guitar, piano.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from scipy.io import wavfile

from paths import Stem, input_name

DEMUCS_RATE = 44_100


def load_stereo(path: Path) -> torch.Tensor:
    """(2, n) float32 at DEMUCS_RATE."""
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(DEMUCS_RATE), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(pcm, dtype="<f4").reshape(-1, 2).T.copy())


def separate(wav: torch.Tensor, model_name: str) -> dict[str, torch.Tensor]:
    """Stem name -> (2, n) waveform, in the input's own units."""
    model = get_model(model_name)
    model.eval()
    reference = wav.mean(0)
    normalised = (wav - reference.mean()) / reference.std()
    with torch.no_grad():
        stems = apply_model(model, normalised[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
    stems = stems * reference.std() + reference.mean()
    return dict(zip(model.sources, stems))


def rms_db(x: torch.Tensor, reference: torch.Tensor) -> float:
    return float(20 * torch.log10(x.pow(2).mean().sqrt() / reference.pow(2).mean().sqrt() + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="htdemucs_6s")
    args = parser.parse_args()

    wav = load_stereo(args.input)
    for name, audio in separate(wav, args.model).items():
        path = Stem(input_name(args.input), name).path
        path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(path, DEMUCS_RATE, audio.T.contiguous().numpy())
        print(f"{path}  {rms_db(audio, wav):6.1f} dB vs mix", flush=True)


if __name__ == "__main__":
    main()
