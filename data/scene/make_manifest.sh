#!/usr/bin/env bash
# Regenerate data_manifest.txt (sha256 + size + rows) for the pinned datasets.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -z "${STABLEWM_HOME:-}" ]; then
    STABLEWM_HOME=$(grep -h '^STABLEWM_HOME=' "$ROOT/.env" "$ROOT/.env.example" 2>/dev/null | head -1 | cut -d= -f2-)
fi
cd "$STABLEWM_HOME/datasets"
OUT="$ROOT/data/scene/data_manifest.txt"
FILES="scene_play.h5 scene_offset_tasks_O25.h5 scene_offset_tasks_O50.h5 \
scene_offset_tasks_O75.h5 scene_offset_tasks_O100.h5"
{
  echo "# generated $(date -Is)"
  for f in $FILES; do
    ROWS=$("$ROOT/.venv/bin/python" -c \
      "import h5py; print(h5py.File('$f','r')['pixels'].shape[0])")
    echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c%s "$f")  $ROWS  $f"
  done
} > "$OUT"
cat "$OUT"
