#!/usr/bin/env bash
# Train one paper run from paper/RUNS.tsv on one GPU:  scripts/reproduce/train.sh <run_id> [gpu]
# The row names the entry point, recipe and overrides (stop-gradient arms); seeds 2 and 3 pass `seed=<n>`.
set -euo pipefail
RUN=${1:?run_id from paper/RUNS.tsv}; GPU=${2:-0}
cd "$(dirname "$0")/../.."
read -r env arm seed entry recipe run offset role overrides < <(awk -v r="$RUN" -F'\t' '$6==r' paper/RUNS.tsv)
[ -n "${recipe:-}" ] || { echo "no row for $RUN in paper/RUNS.tsv" >&2; exit 1; }
ov=(); [ "$seed" != 1 ] && ov+=("seed=$seed"); [ -n "${overrides:-}" ] && ov+=($overrides)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU python src/$entry.py --config-name=$recipe run_id=$run "${ov[@]}"
