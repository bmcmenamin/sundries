"""Build a page for comparing generated samples.

    venv/bin/python listen.py          # writes out/codec_texture/compare.html; open it in a browser

One section per input: the original and its stems, then one row per variant (labelled with the settings that differ between
variants) with that variant's samples side by side, each with a "stems" fold. Starting any player stops the rest. Each input
folds away by clicking its name (and all at once with the buttons on top); the folded set is remembered by the browser. The
page links to the audio files where they are, so keep it next to `out/` (it is regenerated, not edited; run again after new
samples appear).
"""

from __future__ import annotations

import json
import os
import re
import wave
from dataclasses import dataclass
from html import escape
from pathlib import Path

from paths import CODEC_TEXTURE, DEMUCS_STEP, OUT_ROOT, SEPARATION, STEM_FILE, cache_dir, input_name

ROOT = OUT_ROOT / CODEC_TEXTURE
PAGE = ROOT / "compare.html"
INPUTS = Path(__file__).parent / "inputs"
MIX_TAG = re.compile(
    r"^02_mix_(?P<tag>k(?P<components>\d+)_.+_m(?P<manifold>[\d.]+?)(?:w(?P<window>\d+))?"
    r"(?:_x(?P<modulation>[\d.]+))?(?:_t(?P<tolerance>[\d.]+))?(?:_f(?P<floor>[\d.]+))?(?:_g(?P<augment>\d+)(?P<effects>[PCS]*))?(?:_s(?P<share>[\d.]+))?(?:_L(?P<long_window>\d+)t(?P<long_tolerance>[\d.]+))?(?:_T(?P<takes>[\d.]+-[\d.]+))?(?:_r(?P<rhythm>[\d.]+))?(?:_a(?P<activity>[\d.]+))?(?:_j(?P<joint>[\d.]+))?"
    r"(?:_i(?P<ica>[\d.]+))?(?:_d(?P<dynamics>[\d.]+))?)$"
)
FIELDS = (  # (attribute, name shown in a row label)
    ("components", "k"),
    ("manifold", "m"),
    ("window", "w"),
    ("modulation", "mod"),
    ("tolerance", "hinge"),
    ("floor", "floor"),
    ("augment", "copies"),
    ("effects", "effects"),
    ("share", "share"),
    ("long_window", "long w"),
    ("long_tolerance", "long hinge"),
    ("takes", "takes s"),
    ("rhythm", "rhythm"),
    ("activity", "act"),
    ("joint", "joint"),
    ("ica", "ica"),
    ("dynamics", "dyn"),
)


@dataclass(frozen=True)
class Variant:
    tag: str
    components: int
    manifold: float
    window: int
    modulation: float
    tolerance: float
    floor: float
    augment: int
    effects: str
    share: float
    long_window: int
    long_tolerance: float
    takes: str
    rhythm: float
    activity: float
    joint: float
    ica: float
    dynamics: float

    @classmethod
    def from_mix_folder(cls, folder: Path) -> Variant | None:
        match = MIX_TAG.match(folder.name)
        return match and cls(
            match["tag"],
            int(match["components"]),
            float(match["manifold"]),
            int(match["window"] or 1),
            float(match["modulation"] or 0),
            float(match["tolerance"] or 0),
            float(match["floor"] or 0),
            int(match["augment"] or 0),
            match["effects"] or "",
            float(match["share"] or 1),
            int(match["long_window"] or 0),
            float(match["long_tolerance"] or 0),
            match["takes"] or "",
            float(match["rhythm"] or 0),
            float(match["activity"] or 0),
            float(match["joint"] or 0),
            float(match["ica"] or 0),
            float(match["dynamics"] or 0),
        )

    @property
    def key(self) -> tuple[float, ...]:
        return tuple(getattr(self, attribute) for attribute, _ in FIELDS)

    def label(self, varying: tuple[tuple[str, str], ...]) -> str:
        return " · ".join(f"{name} {value if isinstance(value, str) else format(value, 'g')}" for attribute, name in varying if (value := getattr(self, attribute)) is not None)

    def label_html(self, varying: tuple[tuple[str, str], ...]) -> str:
        """One line per varying setting (a dash when unset), so rows line up and read down a column."""
        return "".join(
            f'<div class="kv"><span>{escape(name)}</span><b>{value if isinstance(value, str) else format(value, "g")}</b></div>'
            if (value := getattr(self, attribute))
            else f'<div class="kv"><span>{escape(name)}</span><b>–</b></div>'
            for attribute, name in varying
        )


@dataclass(frozen=True)
class Sample:
    name: str
    mix: Path
    stems: tuple[tuple[str, Path], ...]

    @classmethod
    def of(cls, variant: Variant, source: str, mix: Path) -> Sample:
        generated = ROOT / f"01_generate_{variant.tag}" / source
        stems = tuple((p.parent.name, p) for p in sorted(generated.glob(f"*/{mix.name}")))
        return cls(mix.stem[:8], mix, stems)

    @property
    def seconds(self) -> float:
        with wave.open(str(self.mix)) as w:
            return w.getnframes() / w.getframerate()


def href(path: Path) -> str:
    return escape(os.path.relpath(path, ROOT))


def player(path: Path, label: str) -> str:
    return f'<figure><audio controls preload="none" src="{href(path)}"></audio><figcaption>{escape(label)}</figcaption></figure>'


def sample_html(sample: Sample, label: str) -> str:
    stems = "".join(player(path, name) for name, path in sample.stems)
    fold = f"<details><summary>stems</summary>{stems}</details>" if stems else ""
    return f'<div class="sample">{player(sample.mix, f"{sample.seconds:.0f} s · {sample.name}")}{fold}</div>'


def row_html(variant: Variant, source: str, varying: tuple[tuple[str, str], ...]) -> str:
    label = variant.label(varying) or variant.tag
    mixes = sorted((ROOT / f"02_mix_{variant.tag}" / source).glob("*.wav"))
    cells = "".join(sample_html(Sample.of(variant, source, mix), label) for mix in mixes)
    return f'<tr><th title="{escape(variant.tag)}">{variant.label_html(varying) or escape(label)}</th><td><div class="samples">{cells}</div></td></tr>'


def reference_html(source: str) -> str:
    originals = [p for p in sorted(INPUTS.glob("*")) if input_name(p) == source]
    stems = sorted((OUT_ROOT / SEPARATION / DEMUCS_STEP / source).glob(f"*/{STEM_FILE}"))
    parts = [player(p, "original") for p in originals] + [player(p, p.parent.name) for p in stems]
    return f'<div class="reference">{"".join(parts)}</div>'


def perturbed_html(source: str) -> str:
    """The perturbed copies of each stem (perturb.py), folded, each labelled with its effects and their sine periods."""
    folders = sorted(cache_dir("perturb", source).glob("*/"))

    def caption(path: Path) -> str:
        p = json.loads(path.with_suffix(".json").read_text())
        effects = [
            f"pitch {p['semitones']:+.2f} st ±{p['wobble']:.2f} / {p['pitch']['period']:.0f} s" if p["wobble"] else "",
            f"comb {p['comb_gain']:+.1f} / {p['comb']['period']:.0f} s" if p["comb_gain"] else "",
            f"shift {p['shift_hz']:+.0f} Hz ×{p['shift_mix']:.1f} / {p['shift']['period']:.0f} s" if p["shift_hz"] else "",
        ]
        return " · ".join(e for e in effects if e)

    def stem_fold(folder: Path) -> str:
        wavs = sorted(folder.glob("*.wav"))
        players = "".join(player(p, caption(p)) for p in wavs)
        return f'<details><summary>{escape(folder.name)} ({len(wavs)})</summary><div class="reference">{players}</div></details>'

    return f'<details><summary>perturbed copies</summary>{"".join(map(stem_fold, folders))}</details>' if folders else ""


def varying_fields(variants: list[Variant]) -> tuple[tuple[str, str], ...]:
    """The settings that differ between variants (the components alone when nothing does)."""
    varying = tuple((a, n) for a, n in FIELDS if len({getattr(v, a) for v in variants}) > 1)
    return varying or FIELDS[:1]


def section_html(source: str, variants: list[Variant], varying: tuple[tuple[str, str], ...]) -> str:
    present = sorted((v for v in variants if (ROOT / f"02_mix_{v.tag}" / source).is_dir()), key=lambda v: v.key)
    varying = varying_fields(present)  # only what differs among this input's rows
    rows = "".join(row_html(v, source, varying) for v in present)
    return (
        f'<details class="input" data-input="{escape(source)}" open><summary><h2>{escape(source)}</h2></summary>'
        f"{reference_html(source)}{perturbed_html(source)}<table>{rows}</table></details>"
    )


STYLE = """
:root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#dcdcd4; --card:#fff; --accent:#2a6fdb; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececE6; --muted:#9a9a92; --line:#33332f; --card:#1e1e1c; --accent:#6aa3ff; } }
body { margin:0; padding:0 16px 24px; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }
h1 { font-size:20px; margin:16px 0 4px; } h2 { font-size:17px; margin:24px 0 8px; } p { color:var(--muted); margin:0 0 12px; }
details.input { overflow-x:auto; border-top:1px solid var(--line); margin:0; padding-bottom:8px; }
details.input > summary { cursor:pointer; list-style-position:inside; color:var(--fg); }
details.input > summary h2 { display:inline; }
.bar { display:flex; gap:8px; margin-bottom:8px; }
button { font:inherit; font-size:13px; padding:4px 10px; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; cursor:pointer; }
table { border-collapse:collapse; width:100%; }
th, td { border:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; background:var(--card); }
th { font-weight:400; font-size:12px; width:1%; white-space:nowrap; }
.kv { display:flex; justify-content:space-between; gap:12px; color:var(--muted); } .kv b { color:var(--fg); font-variant-numeric:tabular-nums; }
.samples { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:12px; }
figure { margin:0 0 6px; } figcaption { font-size:12px; color:var(--muted); }
audio { width:100%; min-width:200px; height:32px; }
details { margin:0 0 6px; } summary { font-size:12px; color:var(--muted); cursor:pointer; }
.reference { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:8px; margin-bottom:12px; }
"""

SCRIPT = """
document.addEventListener('play', e => document.querySelectorAll('audio').forEach(a => { if (a !== e.target) a.pause(); }), true);

const inputs = [...document.querySelectorAll('details.input')];
const KEY = 'codec-texture-collapsed';
const read = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(inputs.filter(d => !d.open).map(d => d.dataset.input))); } catch {} };
const collapsed = read();
inputs.forEach(d => { d.open = !collapsed.includes(d.dataset.input); d.addEventListener('toggle', save); });
document.getElementById('collapse').onclick = () => { inputs.forEach(d => d.open = false); save(); };
document.getElementById('expand').onclick = () => { inputs.forEach(d => d.open = true); save(); };
"""


def main() -> None:
    variants = [v for v in map(Variant.from_mix_folder, sorted(ROOT.glob("02_mix_*"))) if v]
    sources = sorted({p.name for v in variants for p in (ROOT / f"02_mix_{v.tag}").iterdir() if p.is_dir()})
    varying = varying_fields(variants)
    sections = "".join(section_html(source, variants, varying) for source in sources)
    PAGE.write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Codec texture samples</title><style>{STYLE}</style></head><body>"
        f"<h1>Codec texture samples</h1><p>One row per variant, labelled with the settings that differ.</p>"
        f"<div class='bar'><button id='collapse'>Collapse all</button><button id='expand'>Expand all</button></div>"
        f"{sections}<script>{SCRIPT}</script></body></html>"
    )
    print(f"{PAGE}  ({len(variants)} variants, {len(sources)} inputs)")


if __name__ == "__main__":
    main()
