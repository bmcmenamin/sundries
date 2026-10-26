"""Output layout, shared by every script.

    out/{pipeline}/{NN_step}/{input}/{stem?}/{file}

- `NN_step`: zero-padded step number first, so a lexical sort is processing order.
- `input`: the input file name without extension, lowercased, spaces -> underscores.
- `stem`: only for steps that run per stem.
- Generated samples are unordered, so they get UUIDv7 names (time-ordered: a lexical
  sort is creation order; needs Python 3.14+). Deterministic
  artifacts (a separated stem, a dictionary) keep fixed names.

Intermediates that can be regenerated live apart from the outputs, under cache/{kind}/{input}/{stem?}/ (see `cache_dir`):
  perturb      <uuid>.{wav,json,latents.pt}   perturbed copies of a stem (perturb.py) and their EnCodec latents

Pipelines and their steps:
  separation   01_demucs                      <input>/<stem>/stem.wav
  spectral     01_texture                     <input>/<uuid>.wav               (whole mix)
               02_stem_texture                <input>/<stem>/<uuid>.wav
               03_recompose                   <input>/<uuid>.wav
  sparse_raw   01_dictionary                  <input>/<stem>/{dictionary.pt,recon.wav}
               02_generate                    <input>/<stem>/<uuid>.wav
               03_recompose                   <input>/<uuid>.wav
  wavelet      01_dictionary                  <input>/<stem>/{dictionary.pt,codes.pt,recon.wav}
               02_generate_<variant>          <input>/<stem>/<uuid>.wav
               03_recompose_<variant>         <input>/<uuid>.wav
               (variant: code_only, refined = also matched on decoded audio, hist = histogram critic)
  ampfreq      01_dictionary                  <input>/<stem>/{dictionary.pt,codes.pt,recon.wav,recon_features_only.wav}
               02_generate                    <input>/<stem>/<uuid>.wav
  nmf_phase    01_dictionary_k<K>_t<T>_s<S>  <input>/<stem>/{dictionary.pt,codes.pt,recon.wav}     (the simplified main path:
               02_generate_k<K>_t<T>_s<S>     <input>/<stem>/<uuid>.{wav,pt}                         3 maps, phase rebuilt
               03_recompose_k<K>_t<T>_s<S>    <input>/<uuid>.wav                                     from its gradients;
                                              K atoms, T atom frames, S stride in frames)
  event_vae    01_train_z<Z>_b<beta>_p<post>   <input>/<stem>/model.pt                      (a VAE over the stem's events)
               02_generate_<tag>              <input>/<stem>/<uuid>.wav                    (10 s, new timeline)
               03_resynthesis_<tag>           <input>/<stem>/<uuid>.wav                    (own onsets, sampled events)
  codec_texture 01_generate_k<K>_<decode>      <input>/<stem>/<uuid>.wav        (EnCodec latents; one <uuid> per set of stems)
                02_mix_k<K>_<decode>           <input>/<uuid>.wav
  nmf_plain    same steps as nmf_phase, with 2 maps and integrated / noise phase (superseded)
  nmf          01_dictionary                  <input>/<stem>/{dictionary.pt,codes.pt,recon.wav,recon_features_only.wav}
               02_generate                    <input>/<stem>/<uuid>.wav
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

OUT_ROOT = Path(__file__).parent / "out"
CACHE_ROOT = Path(__file__).parent / "cache"

SEPARATION = "separation"
SPECTRAL = "spectral"
SPARSE_RAW = "sparse_raw"
WAVELET = "wavelet"
AMPFREQ = "ampfreq"
NMF = "nmf"
NMF_PLAIN = "nmf_plain"
NMF_PHASE = "nmf_phase"
EVENT_VAE = "event_vae"
CODEC_TEXTURE = "codec_texture"

DEMUCS_STEP = "01_demucs"
STEM_FILE = "stem.wav"


def input_name(path: Path | str) -> str:
    """'Walk Of Life.mp3' -> 'walk_of_life'."""
    return Path(path).stem.lower().replace(" ", "_")


def step_dir(pipeline: str, step: str, source: str, stem: str | None = None) -> Path:
    base = OUT_ROOT / pipeline / step / source
    return base / stem if stem else base


def cache_dir(kind: str, source: str, stem: str | None = None) -> Path:
    """Intermediate material that can be regenerated and is not for listening to: cache/{kind}/{input}/{stem?}/."""
    base = CACHE_ROOT / kind / source
    return base / stem if stem else base


def new_sample_path(directory: Path) -> Path:
    """A fresh .wav path in `directory` (created if needed), named by a UUIDv7 so names sort by creation time."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid.uuid7().hex}.wav"


@dataclass(frozen=True)
class Stem:
    source: str
    name: str

    @property
    def path(self) -> Path:
        return step_dir(SEPARATION, DEMUCS_STEP, self.source, self.name) / STEM_FILE


def stems_of(source: str) -> list[Stem]:
    root = step_dir(SEPARATION, DEMUCS_STEP, source)
    return [Stem(source, p.parent.name) for p in sorted(root.glob(f"*/{STEM_FILE}"))]


def stem_of_path(path: Path) -> Stem:
    """The `Stem` a path under out/separation/01_demucs/<input>/<stem>/ belongs to."""
    return Stem(path.parent.parent.name, path.parent.name)
