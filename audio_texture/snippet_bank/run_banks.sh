#!/bin/zsh
# Per song, stem after stem: sparse fit -> snippet bank -> 30 s trigger optimisation. Songs run in parallel.
# usage: [TF=4] [FRACTION=0.4] [SUFFIX=_r] [LOGS=dir] run_banks.sh SONG SHORT STEM...      outputs are tagged N_{SHORT}_{STEM}_30{SUFFIX}
# TF: snippet length in frames of 42.7 ms (4 = 170 ms; 12 = 0.5 s). FRACTION: clusters per event in the bank.
HERE="${0:A:h}"
VENV="${HERE:h}/venv"
ATOMS="${HERE:h}/out/codec_texture/atoms"
LOGS="${LOGS:-$HERE}"
export TF="${TF:-4}"
FRACTION="${FRACTION:-0.4}"
song=$1
short=$2
shift 2
for stem in "$@"; do
  fit="$ATOMS/${song}_${stem}_attack_f0.25_tf${TF}_sus.pt"
  if [[ ! -f "$fit" ]]; then
    TEMPLATE_FRAMES="$TF" "$VENV/bin/python" "$HERE/sustain48.py" "$song" "$stem" attack 24 0.25 > "$LOGS/fit_${short}_${stem}_tf${TF}.log" 2>&1
  fi
  "$VENV/bin/python" "$HERE/snippets.py" "$song" "$stem" "$FRACTION" > "$LOGS/bank_${short}_${stem}_tf${TF}.log" 2>&1
  OUT_SECONDS=30 "$VENV/bin/python" "$HERE/snippet_generate.py" "$song" "$stem" 200 "${short}_${stem}_30${SUFFIX}" > "$LOGS/gen_${short}_${stem}${SUFFIX}.log" 2>&1
  echo "finished $song $stem"
done
