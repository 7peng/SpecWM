#!/usr/bin/env bash
# Download + extract one CALVIN split (default ABC_D, 517 GB zip) into $CALVIN_ROOT.
# Resumable (wget -c), sha256-checked against the official manifest, idempotent:
# completed steps are skipped on re-run and the zip is deleted after extraction.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -z "${CALVIN_ROOT:-}" ]; then
    CALVIN_ROOT=$(grep -h '^CALVIN_ROOT=' "$ROOT/.env" "$ROOT/.env.example" 2>/dev/null | head -1 | cut -d= -f2-)
fi
SPLIT=${SPLIT:-ABC_D}          # ABC_D | D_D | ABCD_D
NEED_GB=${NEED_GB:-1100}       # zip + extracted copy coexist until the zip is removed
BASE=http://calvin.cs.uni-freiburg.de/dataset
ZIP="task_${SPLIT}.zip"

mkdir -p "$CALVIN_ROOT"
cd "$CALVIN_ROOT"

if [ -e "task_${SPLIT}/.extracted" ]; then
    echo "task_${SPLIT} already extracted -- nothing to do"
    exit 0
fi

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "need ${NEED_GB}G free for download+extract, have ${FREE_GB}G -- not starting" >&2
    exit 1
fi

if [ ! -e ".${ZIP}.verified" ]; then
    REMOTE=$(curl -sI "$BASE/$ZIP" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
    # aria2c with 16 connections when available: measured 14.7 MB/s single-stream vs 108 MB/s
    # at -x16 on this host, which is 10h vs 1.4h for the ABC_D zip. wget -c is the fallback.
    if command -v aria2c > /dev/null; then
        aria2c -c -x16 -s16 -k 100M --file-allocation=none \
               --summary-interval=60 -d . -o "$ZIP" "$BASE/$ZIP"
    else
        wget -c "$BASE/$ZIP"
    fi
    # du, not stat: with parallel range writes the apparent size is misleading until the
    # transfer completes, and stat -c%s reports the sparse extent rather than bytes written.
    [ "$(du -B1 --apparent-size "$ZIP" | cut -f1)" = "$REMOTE" ] \
        || { echo "size mismatch for $ZIP" >&2; exit 1; }
    wget -qN "$BASE/sha256sum.txt"
    sha256sum --check --ignore-missing sha256sum.txt
    touch ".${ZIP}.verified"
fi

# 7z, NOT Info-ZIP unzip: unzip 6.00 mishandles this ZIP64 archive and exits 0 having
# written a fraction of it (measured: 1,255 of ~2M frames, no metadata), after which the
# script used to stamp .extracted and delete the zip. Verify the count before either.
# -aos: skip files already on disk, so this doubles as a repair pass for a partial tree
7z x -bd -y -aos -mmt=4 -o"$CALVIN_ROOT" "$ZIP" > "extract_${SPLIT}.log" 2>&1

# 7z restores the archive's ORIGINAL mtimes (2021 for CALVIN). A scratch cleanup policy keyed on
# mtime then reads the whole tree as five years stale and deletes it -- measured 2026-08-24,
# 226,655 files lost at up to 630/min before the mtimes were refreshed. Stamp them to now.
echo "refreshing mtimes so the tree does not read as stale to scratch cleanup..."
find "task_${SPLIT}" -print0 | xargs -0 -P 6 -n 400 touch -m -c

N=$(find "task_${SPLIT}" -name 'episode_*.npz' | wc -l)
echo "extracted $N episode files"
[ "$N" -ge 100000 ] || { echo "only $N frames extracted -- refusing to mark done or delete the zip" >&2; exit 1; }
for f in ep_start_end_ids.npy statistics.yaml scene_info.npy; do
    find "task_${SPLIT}" -name "$f" | grep -q . || { echo "missing $f -- extraction incomplete" >&2; exit 1; }
done
touch "task_${SPLIT}/.extracted"
rm -f "$ZIP"
echo "extracted to $CALVIN_ROOT/task_${SPLIT} ($N frames)"
