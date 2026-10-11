#!/usr/bin/env bash
# CEM-score one run at the paper protocol: every 4k snapshot and the final checkpoint, one cell at a time.
#   scripts/reproduce/score.sh <run_id> [gpu]      (offset from paper/RUNS.tsv; cells already scored are skipped)
set -euo pipefail
RUN=${1:?run_id}; GPU=${2:-0}
cd "$(dirname "$0")/../.."; CK=${STABLEWM_HOME:?}/checkpoints/lewm/$RUN
read -r env arm seed entry recipe run offset role overrides < <(awk -v r="$RUN" -F'\t' '$6==r' paper/RUNS.tsv)
scored(){ grep -qE "'success_rate': [0-9.]+" results/$1/O$offset.txt 2>/dev/null; }
[ -f $CK/config.json ] || python scripts/write_config_json.py "$(ls -d outputs/*/*/ | while read d; do grep -q "run_id=$RUN\$" $d/.hydra/overrides.yaml 2>/dev/null && echo $d; done | tail -1)" $CK
for s in $(seq 4000 4000 32000); do
  [ -f $CK/step_snaps/step_$s.pt ] || continue
  [ -f $CK/weights_step_$s.pt ] || python -c "import torch; torch.save(torch.load('$CK/step_snaps/step_$s.pt', map_location='cpu')['model'], '$CK/weights_step_$s.pt')"
  scored $RUN-s$s || MUJOCO_GL=egl python scripts/eval_cem.py $RUN-s$s lewm/$RUN/weights_step_$s.pt --gpus $GPU --offsets $offset
done
F=$(ls $CK/weights_epoch_*.pt | sed 's/.*weights_epoch_//; s/.pt//' | sort -n | tail -n 1)
scored $RUN-final || MUJOCO_GL=egl python scripts/eval_cem.py $RUN-final lewm/$RUN/weights_epoch_$F.pt --gpus $GPU --offsets $offset
