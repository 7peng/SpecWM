#!/usr/bin/env bash
# Fingerprint every paper recipe (tests/paper_recipes.txt) into $1 from the checkout this script lives in. GPU $2 (default 3).
set -u
OUT=${1:?usage: regression_paper.sh <out-dir> [gpu]}; GPU=${2:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd); PY=$ROOT/.venv/bin/python
cd $ROOT || exit 1
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
grep -vE '^#|^\s*$' tests/paper_recipes.txt | while read -r entry recipe extra; do
  name=$recipe; [ -n "$extra" ] && name="$recipe+sg"
  echo "=== $name ($(date +%H:%M:%S))"
  timeout 1200 $PY tests/regression_baseline.py --entry "$entry" --recipe "$recipe" --name "$name" --out "$OUT" --extra $extra 2>&1 | grep -E "^\[regr\]|Error|error:" | tail -n 3
done
echo "=== done $(date +%H:%M:%S)"
