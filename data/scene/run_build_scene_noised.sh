#!/bin/bash
# Build scene_play + scene_play_noised + scene_play_mix_noised from the pinned OGBench archive.
#
#   data/scene/run_build_scene_noised.sh [NUM_BLOCKS]
#
# NUM_BLOCKS parallel workers round-robin over $NG GPUs. The block count does NOT affect the output:
# workers take contiguous slices of the canonical episode order and the merge concatenates them in
# index order, so any NUM_BLOCKS yields the same state content hash (verified in the test suite).
#
#   NG=<n>              GPUs to round-robin over (default 3)
#   STABLEWM_HOME=<p>   dataset root (root .env.example)
#   BLOCK_DIR=<p>       where per-worker block files land (default $STABLEWM_HOME/datasets/_scene_build)
#   KINDS="play noised" which halves to build
#   PLAY_OUT/NOISED_OUT/MIX_OUT   output paths (default scene_play[_noised|_mix_noised].h5)
set -uo pipefail
W=${1:-36}
NG=${NG:-3}
PY=".venv/bin/python"
SCRIPT="data/scene/build_scene_noised.py"
if [ -z "${STABLEWM_HOME:-}" ]; then
    STABLEWM_HOME=$(grep -h '^STABLEWM_HOME=' .env .env.example 2>/dev/null | head -1 | cut -d= -f2-)
fi
export STABLEWM_HOME MUJOCO_GL=egl
ROOT="$STABLEWM_HOME/datasets"
BLOCK_DIR=${BLOCK_DIR:-$ROOT/_scene_build}
PREP="$BLOCK_DIR/prep.npz"
KINDS=${KINDS:-"play noised"}
PLAY_OUT=${PLAY_OUT:-$ROOT/scene_play.h5}
NOISED_OUT=${NOISED_OUT:-$ROOT/scene_play_noised.h5}
MIX_OUT=${MIX_OUT:-$ROOT/scene_play_mix_noised.h5}
mkdir -p "$BLOCK_DIR"

echo "=== prepare (source hash-checked once; workers never touch ogbench) ==="
[ -f "$PREP" ] || $PY $SCRIPT prepare --prep "$PREP" || { echo "PREPARE FAILED"; exit 1; }

# Gate the whole build on this machine being able to reproduce the reference simulation. Costs a
# minute; a mismatch discovered after a full render costs a day.
echo "=== preflight ==="
$PY $SCRIPT preflight --prep "$PREP" --gpu 0 || { echo "PREFLIGHT FAILED -- refusing to build"; exit 1; }
$PY $SCRIPT selftest  --prep "$PREP" --gpu 0 || { echo "SELFTEST FAILED -- refusing to build"; exit 1; }

for KIND in $KINDS; do
  echo "=== render $KIND: $W blocks over $NG GPUs ==="
  pids=()
  for K in $(seq 0 $((W - 1))); do
    $PY $SCRIPT render --kind "$KIND" --block "$K" --num-blocks "$W" --gpu $((K % NG)) \
        --prep "$PREP" --block-dir "$BLOCK_DIR" > "$BLOCK_DIR/${KIND}_block_$K.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
  if [ "$fail" -gt 0 ]; then
    echo "$fail $KIND WORKERS FAILED -- see $BLOCK_DIR/${KIND}_block_*.log"
    grep -hiE "error|traceback|egl|out of memory" "$BLOCK_DIR/${KIND}_block_"*.log | tail -20
    exit 1
  fi

  echo "=== merge $KIND ==="
  # merge refuses to run on a missing block rather than silently writing a short dataset
  OUT=$( [ "$KIND" = play ] && echo "$PLAY_OUT" || echo "$NOISED_OUT" )
  $PY $SCRIPT merge --kind "$KIND" --num-blocks "$W" --block-dir "$BLOCK_DIR" --out "$OUT" \
      || { echo "MERGE FAILED"; exit 1; }
done

if [ "$KINDS" = "play noised" ]; then
  echo "=== mix ==="
  $PY $SCRIPT mix --play "$PLAY_OUT" --noised "$NOISED_OUT" --out "$MIX_OUT" \
      || { echo "MIX FAILED"; exit 1; }
  $PY $SCRIPT verify --out "$MIX_OUT" || exit 1
fi
echo "=== DONE ==="
echo "Block files under $BLOCK_DIR can be deleted once the merged files verify."
