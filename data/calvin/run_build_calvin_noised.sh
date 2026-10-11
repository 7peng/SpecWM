#!/bin/bash
# Render calvin_abc_d_noised on one node: W simulator workers (the CALVIN venv) pinned one per core
# over GPU_IDS, then merge + mix in the training venv. Run from the repo root on the node; prepare first
# (build_calvin_noised.py prepare, training venv). Blocks land under $CALVIN_STAGE_DIR/_calvin_build.
#   W=32 GPU_IDS="2 3" CORES="$(seq 16 31) $(seq 48 63)" data/calvin/run_build_calvin_noised.sh
set -uo pipefail
W=${W:-32}; GPU_IDS=(${GPU_IDS:-0}); CORES=(${CORES:-})
PY=${PY:-.venv/bin/python}; CPY=${CALVIN_VENV:?}/bin/python
S=data/calvin/build_calvin_noised.py
STAGE=${CALVIN_STAGE_DIR:?}; B=$STAGE/_calvin_build; mkdir -p "$B"
PREP=${PREP:-$STABLEWM_HOME/datasets/_calvin_build/prep.npz}
export CALVIN_USE_EGL=1 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
pin(){ local k=$1; if [ ${#CORES[@]} -gt 0 ]; then echo "taskset -c ${CORES[$((k % ${#CORES[@]}))]}"; fi; }
echo "=== render: $W blocks over gpus ${GPU_IDS[*]} $(date '+%T') ==="
pids=()
for K in $(seq 0 $((W - 1))); do
  G=${GPU_IDS[$((K % ${#GPU_IDS[@]}))]}
  $(pin $K) $CPY $S render --block $K --num-blocks $W --gpu $G --prep "$PREP" --block-dir "$B" > "$B/noised_block_$K.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
if [ "$fail" -gt 0 ]; then echo "$fail WORKERS FAILED -- see $B/noised_block_*.log"; grep -hiE "error|traceback" "$B"/noised_block_*.log | tail -20; exit 1; fi
echo "=== merge $(date '+%T') ==="; $PY $S merge --num-blocks $W --block-dir "$B" --out "$STAGE/calvin_abc_d_noised.h5" || { echo "MERGE FAILED"; exit 1; }
echo "=== mix $(date '+%T') ==="; $PY $S mix --play "$STAGE/calvin_abc_d.h5" --noised "$STAGE/calvin_abc_d_noised.h5" --out "$STAGE/calvin_abc_d_mix_noised.h5" || { echo "MIX FAILED"; exit 1; }
echo "=== DONE $(date '+%T') ==="; ls -la "$STAGE"
