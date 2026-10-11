#!/usr/bin/env bash
# Run every standalone test script and report pass/fail/skip. Tests are plain scripts that exit
# nonzero on failure, so this is the whole protocol. $1 = report file, $2 = gpu (default 3).
set -u
REPORT=${1:?usage: run_all.sh <report-file> [gpu]}
GPU=${2:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/.venv/bin/python
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONPATH=src TOKENIZERS_PARALLELISM=false
# CAP THE MATH THREADS. Unbounded, the CPU-heavy probe tests take every core on the box --
# measured at ~56 of 64, load average 112 -- which starves the dataloader workers of a training
# run sharing the node. A rank held off the GPU long enough trips the NCCL watchdog and kills the
# job, which has happened here before. THREADS overrides the cap when the box is idle.
export OMP_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8} \
       OPENBLAS_NUM_THREADS=${THREADS:-8} NUMEXPR_NUM_THREADS=${THREADS:-8}
# one writer per report: two concurrent runs both truncate it and interleave their lines, which
# reads as a short clean report rather than as the collision it is
exec 9>"$REPORT.lock"
flock -n 9 || { echo "another run_all.sh is already writing $REPORT"; exit 2; }
: > "$REPORT"
pass=0; fail=0
for t in tests/test_*.py tests/smoke_*.py; do
  [ -f "$t" ] || continue
  out=$(timeout 900 $PY "$t" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "PASS $t" >> "$REPORT"; pass=$((pass+1))
  else
    echo "FAIL $t (rc=$rc)" >> "$REPORT"
    echo "$out" | tail -6 | sed 's/^/     | /' >> "$REPORT"
    fail=$((fail+1))
  fi
done
echo "TOTAL pass=$pass fail=$fail" >> "$REPORT"
tail -1 "$REPORT"
