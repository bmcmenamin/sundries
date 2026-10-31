"""Gain-adjusted mix of generated stems.

    mix_stems.py SONG SHORT STEM...        reads atoms/N_{SHORT}_{STEM}_30_triggers.pt, writes compare/X_{SHORT}_mix30.wav (+ X_{SHORT}_src_mix.wav, the source's own mix)

Each generated stem is scaled to unit rms, then to its stem's share of the source mix: rms(stem) / rms(vocal-free mix), measured on the source's 10 s.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import snippets as sn  # noqa: E402
from paths import stems_of  # noqa: E402
from synthesize import decode_mono  # noqa: E402

song, short, names = sys.argv[1], sys.argv[2], sys.argv[3:]
rms = lambda x: float(x.pow(2).mean().sqrt()) + 1e-12  # noqa: E731
source = {s.name: decode_mono(s.path, sn.RATE)[: 10 * sn.RATE] for s in stems_of(song)}
mix = sum(v for k, v in source.items() if k != "vocals")
share = {name: rms(source[name]) / rms(mix) for name in names}
generated = {name: torch.load(sn.ATOMS_DIR / f"N_{short}_{name}_30_triggers.pt")["audio"] for name in names}
print("stem share of the source mix (dB):", {k: round(20 * float(torch.log10(torch.tensor(v))), 1) for k, v in share.items()})
out = sum(generated[name] / rms(generated[name]) * share[name] for name in names)
sn.write(sn.OUT / f"X_{short}_mix30.wav", out)
sn.write(sn.OUT / f"X_{short}_src_mix.wav", sum(source[name] for name in names))
