"""A local page for listening to each song's segments (the ORIGINAL recordings) and flagging the ones that must appear in generations.

    review_page.py        -> out/codec_texture/segments.html    (every atoms/{song}_structure.json that exists; open it in a browser)

Nothing is copied: the page plays the mp3s in inputs/ by time range, so it stays small and the recordings stay where they are. Flags are kept in the browser
(localStorage) and "Export must-haves" downloads musthaves.json: {song: [{"start", "end", "kind"}, ...]}, for the generation step to upweight.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paths import input_name  # noqa: E402

from structure import Structure  # noqa: E402

INPUTS = ROOT / "inputs"
ATOMS_DIR = ROOT / "out/codec_texture/atoms"
PAGE = ROOT / "out/codec_texture/segments.html"


@dataclass(frozen=True)
class SongRow:
    name: str
    title: str
    audio: str  # url of the recording, relative to the page
    segments: list[list[float]]  # [start, end, kind]

    def as_json(self) -> dict:
        return {"name": self.name, "title": self.title, "audio": self.audio, "segments": self.segments}


UP = "/".join([".."] * len(PAGE.parent.relative_to(ROOT).parts))  # from the page's folder back to the project root


def song_row(name: str, recording: Path) -> SongRow:
    structure = Structure.load(ATOMS_DIR / f"{name}_structure.json")
    return SongRow(
        name=name,
        title=recording.stem.replace("_", " "),
        audio=f"{UP}/inputs/{quote(recording.name)}",
        segments=[[s.start, s.end, s.kind] for s in structure.segments],
    )


def song_rows() -> list[SongRow]:
    recordings = {input_name(p): p for p in INPUTS.glob("*.mp3")}
    names = (p.name.removesuffix("_structure.json") for p in ATOMS_DIR.glob("*_structure.json"))
    return sorted((song_row(n, recordings[n]) for n in names if n in recordings), key=lambda r: r.title.lower())


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segments</title>
<style>
  :root { --bg:#fafaf8; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --accent:#2b59c3; --star:#c98a00; --playing:#e9effc;
          --k0:#4e79a7; --k1:#f28e2b; --k2:#59a14f; --k3:#b07aa1; --k4:#e15759; --k5:#76b7b2; --k6:#9c755f; --k7:#bab0ac; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161615; --card:#1f1f1d; --ink:#ecebe6; --muted:#9a9992; --line:#33332f; --accent:#7aa2f7; --star:#f0b429; --playing:#232c44; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.4 -apple-system, system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0 8px 0 0; }
  header .count { color:var(--muted); font-size:14px; margin-right:auto; }
  button { font:inherit; color:var(--ink); background:var(--card); border:1px solid var(--line); border-radius:8px; padding:6px 12px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  main { max-width:860px; margin:0 auto; padding:16px; }
  .song { background:var(--card); border:1px solid var(--line); border-radius:12px; margin:0 0 16px; padding:14px 16px; }
  .song h2 { font-size:16px; margin:0 0 2px; }
  .song .meta { color:var(--muted); font-size:13px; margin-bottom:10px; }
  .bar { display:flex; height:22px; border-radius:6px; overflow:hidden; margin-bottom:10px; gap:1px; }
  .bar div { display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:600; cursor:pointer; min-width:0; }
  .bar div.flag { outline:2px solid var(--star); outline-offset:-2px; }
  .row { display:grid; grid-template-columns:28px 36px 1fr auto 40px; gap:10px; align-items:center; padding:6px 8px; border-radius:8px; }
  .row.playing { background:var(--playing); }
  .chip { width:26px; height:26px; border-radius:6px; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; font-size:13px; }
  .time { color:var(--muted); font-variant-numeric:tabular-nums; font-size:13px; }
  .track { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .track i { display:block; height:100%; width:0; background:var(--accent); }
  .star { border:none; background:none; font-size:20px; color:var(--muted); padding:0 4px; }
  .star.on { color:var(--star); }
  .play { width:36px; padding:4px 0; }
  @media (max-width:520px) { .row { grid-template-columns:26px 36px 1fr 36px; } .row .time { display:none; } }
</style>
</head>
<body>
<header>
  <h1>Segments</h1>
  <span class="count" id="count"></span>
  <button id="export">Export must-haves</button>
  <button id="clear">Clear flags</button>
</header>
<main id="songs"></main>
<script>
const SONGS = /*DATA*/;
const KEY = "segment-musthaves";
const flags = new Set(JSON.parse((() => { try { return localStorage.getItem(KEY) } catch (e) { return null } })() || "[]"));
const save = () => { try { localStorage.setItem(KEY, JSON.stringify([...flags])) } catch (e) {} };
const id = (song, seg) => `${song.name}|${seg[0].toFixed(1)}|${seg[1].toFixed(1)}`;
const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const letter = k => String.fromCharCode(65 + k);
const color = k => `var(--k${k % 8})`;

const audio = new Audio();
audio.preload = "auto";
let current = null; // {song, seg, row, fill}

function stop() {
  audio.pause();
  if (current) { current.row.classList.remove("playing"); current.fill.style.width = "0"; current.button.textContent = "▶"; }
  current = null;
}

function play(song, seg, row, fill, button) {
  const same = current && current.seg === seg;
  stop();
  if (same) return;
  const start = () => { audio.currentTime = seg[0]; audio.play(); };
  const src = new URL(song.audio, location.href).href;
  if (audio.src !== src) { audio.src = src; audio.addEventListener("loadedmetadata", start, { once: true }); } else start();
  current = { song, seg, row, fill, button };
  row.classList.add("playing");
  button.textContent = "■";
}

setInterval(() => {
  if (!current) return;
  const [a, b] = current.seg;
  current.fill.style.width = `${Math.max(0, Math.min(1, (audio.currentTime - a) / (b - a))) * 100}%`;
  if (audio.currentTime >= b || audio.ended) stop();
}, 60);

function render() {
  const main = document.getElementById("songs");
  SONGS.forEach(song => {
    const card = document.createElement("section");
    card.className = "song";
    const total = song.segments.at(-1)[1];
    const kinds = new Set(song.segments.map(s => s[2])).size;
    card.innerHTML = `<h2></h2><div class="meta">${song.segments.length} segments, ${kinds} kinds, ${clock(total)}</div><div class="bar"></div><div class="rows"></div>`;
    card.querySelector("h2").textContent = song.title;
    const bar = card.querySelector(".bar"), rows = card.querySelector(".rows");
    song.segments.forEach(seg => {
      const [a, b, k] = seg;
      const cell = document.createElement("div");
      cell.style.cssText = `flex:${b - a};background:${color(k)}`;
      cell.textContent = letter(k);
      cell.title = `${clock(a)}–${clock(b)}`;
      bar.append(cell);

      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<button class="play">▶</button><div class="chip" style="background:${color(k)}">${letter(k)}</div><div class="track"><i></i></div><span class="time">${clock(a)}–${clock(b)} · ${Math.round(b - a)} s</span><button class="star" title="must have">★</button>`;
      const button = row.querySelector(".play"), fill = row.querySelector("i"), star = row.querySelector(".star");
      const sync = () => { star.classList.toggle("on", flags.has(id(song, seg))); cell.classList.toggle("flag", flags.has(id(song, seg))); };
      button.onclick = () => play(song, seg, row, fill, button);
      cell.onclick = () => play(song, seg, row, fill, button);
      star.onclick = () => { const key = id(song, seg); flags.has(key) ? flags.delete(key) : flags.add(key); save(); sync(); count(); };
      sync();
      rows.append(row);
    });
    main.append(card);
  });
  count();
}

function count() {
  document.getElementById("count").textContent = `${flags.size} must-have${flags.size === 1 ? "" : "s"} flagged, ${SONGS.length} songs`;
}

document.getElementById("export").onclick = () => {
  const out = {};
  SONGS.forEach(song => song.segments.filter(seg => flags.has(id(song, seg))).forEach(seg => {
    (out[song.name] ||= []).push({ start: seg[0], end: seg[1], kind: seg[2] });
  }));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  link.download = "musthaves.json";
  link.click();
};
document.getElementById("clear").onclick = () => { if (flags.size && confirm("Clear all flags?")) { flags.clear(); save(); location.reload(); } };
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    rows = song_rows()
    PAGE.write_text(TEMPLATE.replace("/*DATA*/", json.dumps([r.as_json() for r in rows])))
    print(f"wrote {PAGE} ({len(rows)} songs, {sum(len(r.segments) for r in rows)} segments)")
