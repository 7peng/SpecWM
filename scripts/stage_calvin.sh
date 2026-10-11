#!/usr/bin/env bash
# Stage the CALVIN inputs on the node's local disk: the play file, the late-onset mixture (built from the play file
# and the noised file if not yet present under $STABLEWM_HOME) and the probe pack.
#   scripts/stage_calvin.sh            # from the repo root, with .env loaded (set -a; source .env; set +a)
set -euo pipefail
: "${STABLEWM_HOME:?}" "${CALVIN_STAGE_DIR:?}" "${CALVIN_PROBE_PACK_SRC:?}" "${CALVIN_PROBE_PACK_DIR:?}"
mkdir -p "$CALVIN_STAGE_DIR" "$CALVIN_PROBE_PACK_DIR"
cp -n "$STABLEWM_HOME/calvin_abc_d.h5" "$CALVIN_STAGE_DIR/"
if [ -f "$STABLEWM_HOME/datasets/calvin_abc_d_mix_lateonset.h5" ]; then
  cp -n "$STABLEWM_HOME/datasets/calvin_abc_d_mix_lateonset.h5" "$CALVIN_STAGE_DIR/"
else
  python data/calvin/build_calvin_lateonset.py mix --play "$CALVIN_STAGE_DIR/calvin_abc_d.h5" \
    --noised "$STABLEWM_HOME/calvin_abc_d_noised_lateonset.h5" --out "$CALVIN_STAGE_DIR/calvin_abc_d_mix_lateonset.h5"
fi
cp -rn "$CALVIN_PROBE_PACK_SRC/." "$CALVIN_PROBE_PACK_DIR/"
echo "staged: $(ls "$CALVIN_STAGE_DIR")"
