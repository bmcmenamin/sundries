#!/bin/zsh
# Per song, stem after stem: sparse fit (4-frame templates) -> snippet bank -> 30 s trigger optimisation. Songs run in parallel.
# usage: run_banks.sh SONG SHORT STEM...
SCRATCH="${0:A:h}"
VENV="${SCRATCH:h}/venv"
ATOMS="${SCRATCH:h}/out/codec_texture/atoms"
song=$1
short=$2
shift 2
for stem in "$@"; do
  fit="$ATOMS/${song}_${stem}_attack_f0.25_tf4_sus.pt"
  if [[ ! -f "$fit" ]]; then
    TEMPLATE_FRAMES=4 "$VENV/bin/python" "$SCRATCH/sustain48.py" "$song" "$stem" attack 24 0.25 > "$SCRATCH/fit_${short}_${stem}.log" 2>&1
  fi
  "$VENV/bin/python" "$SCRATCH/snippets.py" "$song" "$stem" 0.4 > "$SCRATCH/bank_${short}_${stem}.log" 2>&1
  OUT_SECONDS=30 "$VENV/bin/python" "$SCRATCH/snippet_generate.py" "$song" "$stem" 200 "${short}_${stem}_30" > "$SCRATCH/gen_${short}_${stem}.log" 2>&1
  echo "finished $song $stem"
done
