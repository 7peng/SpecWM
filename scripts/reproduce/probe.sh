#!/usr/bin/env bash
# Held-out linear probe of one run's final checkpoint (results/paper/PROBING_PROTOCOL.md).
#   scripts/reproduce/probe.sh <run_id> [gpu]
set -euo pipefail
RUN=${1:?run_id}; GPU=${2:-0}
cd "$(dirname "$0")/../.."
read -r env arm seed entry recipe run offset role overrides < <(awk -v r="$RUN" -F'\t' '$6==r' paper/RUNS.tsv)
OUT=results/paper/data/probe_heldout_unified_$env.csv
[ "$env" = calvin ] && [ "$role" != lewm ] && OUT=results/paper/data/probe_heldout_paper_calvin.csv
CUDA_VISIBLE_DEVICES=$GPU python results/paper/scripts/probe_unified.py --protocol heldout --env $env --runs $RUN --final --out $OUT
