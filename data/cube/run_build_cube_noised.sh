#!/bin/bash
# Build cube_play.h5 (marker-free re-render) and cube_play_noised.h5 (failure data) with
# data/cube/build_cube_noised.py: W render workers pinned one per core over GPU_IDS, then a
# chunk-verbatim merge per kind and a manifest verify. Run from the repo root on a node whose
# $STABLEWM_HOME/datasets/_cube_build/prep.npz exists (`build_cube_noised.py prepare` needs
# ~/.ogbench, so prepare runs where home is mounted).
#   W=32 GPU_IDS="2 3" CORES="$(seq 16 31) $(seq 48 63)" data/cube/run_build_cube_noised.sh
set -uo pipefail
W=${W:-32}
GPU_IDS=(${GPU_IDS:-0})
CORES=(${CORES:-})
PY=${PY:-.venv/bin/python}
SCRIPT="data/cube/build_cube_noised.py"
export MUJOCO_GL=egl
ROOT="$STABLEWM_HOME/datasets"
BLOCK_DIR=${BLOCK_DIR:-$ROOT/_cube_build}
PREP="$BLOCK_DIR/prep.npz"
KINDS=${KINDS:-"play noised"}
PLAY_OUT=${PLAY_OUT:-$ROOT/cube_play.h5}
NOISED_OUT=${NOISED_OUT:-$ROOT/cube_play_noised.h5}
[ -f "$PREP" ] || { echo "no $PREP -- run: $SCRIPT prepare (on a node that mounts ~/.ogbench)"; exit 1; }
pin(){ local k=$1; if [ ${#CORES[@]} -gt 0 ]; then echo "taskset -c ${CORES[$((k % ${#CORES[@]}))]}"; fi; }
echo "=== preflight on gpu ${GPU_IDS[0]} ==="
$(pin 0) $PY $SCRIPT preflight --prep "$PREP" --gpu "${GPU_IDS[0]}" || { echo "PREFLIGHT FAILED -- refusing to build"; exit 1; }
for KIND in $KINDS; do
  echo "=== render $KIND: $W blocks over gpus ${GPU_IDS[*]} $(date '+%T') ==="
  pids=()
  for K in $(seq 0 $((W - 1))); do
    G=${GPU_IDS[$((K % ${#GPU_IDS[@]}))]}
    $(pin $K) $PY $SCRIPT render --kind "$KIND" --block "$K" --num-blocks "$W" --gpu "$G" \
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
  echo "=== merge $KIND $(date '+%T') ==="
  OUT=$( [ "$KIND" = play ] && echo "$PLAY_OUT" || echo "$NOISED_OUT" )
  $PY $SCRIPT merge --kind "$KIND" --num-blocks "$W" --block-dir "$BLOCK_DIR" --out "$OUT" \
      || { echo "MERGE FAILED"; exit 1; }
  $PY $SCRIPT verify --kind "$KIND" --out "$OUT" || exit 1
done
echo "=== DONE $(date '+%T') ==="
echo "Block files under $BLOCK_DIR can be deleted once the merged files verify."
