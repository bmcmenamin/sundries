"""Rebuild generated mixes with every stem matched to its source stem's loudness (LUFS), from the stems already written.

    venv/bin/python remix.py                      # every tag, every input
    venv/bin/python remix.py --source walk_1 --trim guitar=-3
    venv/bin/python remix.py --tag k64_continuous_dense_m1w48_x1_g64_T2-5 --trim guitar=-3 piano=1

Rewrites the stems in out/codec_texture/01_generate_<tag>/ and the mix in 02_mix_<tag>/ in place, with one shared gain
so that the loudest peak of the mix sits at OUTPUT_PEAK. The generated material itself is untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import torch
from scipy.io import wavfile

from codec_texture import loudness, parse_trim, save, stem_loudness
from paths import CODEC_TEXTURE, OUT_ROOT, Stem, step_dir
from synthesize import OUTPUT_PEAK

ROOT = OUT_ROOT / CODEC_TEXTURE


def read(path: Path) -> torch.Tensor:
    return torch.from_numpy(wavfile.read(path)[1].astype("float32") / 32768)


def matched(stem_wav: Path, target: float, trim: float) -> torch.Tensor:
    """The stem at the source's loudness `target` (LUFS) plus `trim` dB."""
    audio = read(stem_wav)
    return audio * 10 ** ((target + trim - loudness(audio)) / 20)


@lru_cache(maxsize=None)
def source_loudness(source: str, stem: str) -> float:
    return stem_loudness(Stem(source, stem).path)


def remix(tag: str, source: str, name: str, trim: dict[str, float]) -> None:
    stems = {
        wav.parent.name: matched(wav, source_loudness(source, wav.parent.name), trim.get(wav.parent.name, 0.0))
        for wav in sorted(step_dir(CODEC_TEXTURE, f"01_generate_{tag}", source).glob(f"*/{name}"))
    }
    mix = torch.stack(list(stems.values())).sum(0)
    gain = OUTPUT_PEAK / float(mix.abs().max())
    for stem_name, audio in stems.items():
        save(step_dir(CODEC_TEXTURE, f"01_generate_{tag}", source, stem_name) / name, audio, gain)
    save(step_dir(CODEC_TEXTURE, f"02_mix_{tag}", source) / name, mix, gain)
    print(f"  {source}/{tag}/{name[-10:-4]} (LUFS): " + ", ".join(f"{k} {loudness(v * gain):.1f}" for k, v in stems.items()), flush=True)


def mixes(tag_filter: str | None, source_filter: str | None) -> Iterator[tuple[str, str, str]]:
    """(tag, input, file name) of every written mix."""
    for wav in sorted(ROOT.glob("02_mix_*/*/*.wav")):
        tag, source = wav.parent.parent.name.removeprefix("02_mix_"), wav.parent.name
        if tag_filter in (None, tag) and source_filter in (None, source):
            yield tag, source, wav.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="only this generation tag (default: all)")
    parser.add_argument("--source", help="only this input (default: all)")
    parser.add_argument("--trim", nargs="*", default=[], metavar="STEM=DB", help="turn a stem up or down, in dB")
    args = parser.parse_args()
    trim = parse_trim(args.trim)

    for tag, source, name in mixes(args.tag, args.source):
        remix(tag, source, name, trim)


if __name__ == "__main__":
    main()
