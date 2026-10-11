# stage_data.sh — copy HDF5 dataset(s) from networked NFS to fast node-local disk.
#
# Both the home and scratch directories use networked storage, so the
# dataloader's per-step random HDF5 reads cross the network. This stages the .h5
# onto the GPU node's local /tmp (ext4) and points LOCAL_DATASET_DIR there, so
# train.py reads locally instead of over NFS.
#
# USAGE — run ON THE GPU COMPUTE NODE (inside your srun/sbatch alloc); /tmp is node-local:
#   source stage_data.sh                                  # stage 'scene_play', export LOCAL_DATASET_DIR
#   source stage_data.sh scene_play                       # explicit
#   source stage_data.sh scene_play other_dataset         # stage several
#   source stage_data.sh && python train.py --config-name=lewm data=scene ...   # stage then train
#
#   bash stage_data.sh        # also works, but only copies + prints the export
#                             # (a child process can't set env in your shell — prefer `source`)
#
# Override defaults via env before calling:
#   SRC_DATASET_DIR   (default: $STABLEWM_HOME)   source root
#   STAGE_DEST_ROOT   (default: /tmp/$USER/.stable-wm)             node-local dest root
#
# Idempotent: a dataset whose local copy already matches the source size is skipped.
# Dataset names carry no .h5 suffix (they match the Hydra data configs); the file is
# resolved as <root>/datasets/<name>.h5, the same layout swm.data.HDF5Dataset expects.

# Detect sourced vs executed so we can return (not exit) when sourced, which would
# otherwise kill the caller's interactive shell. Deliberately NO `set -e` here.
if (return 0 2>/dev/null); then _STAGE_SOURCED=1; else _STAGE_SOURCED=0; fi

: "${USER:=$(id -un)}"
_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -z "${SCENE_STAGE_DIR:-}" ]; then
    SCENE_STAGE_DIR=$(grep -h '^SCENE_STAGE_DIR=' "$_REPO/.env" "$_REPO/.env.example" 2>/dev/null | head -1 | cut -d= -f2-)
fi
if [ -z "${STABLEWM_HOME:-}" ]; then
    STABLEWM_HOME=$(grep -h '^STABLEWM_HOME=' "$_REPO/.env" "$_REPO/.env.example" 2>/dev/null | head -1 | cut -d= -f2-)
fi
SRC_ROOT="${SRC_DATASET_DIR:-$STABLEWM_HOME}"
DEST_ROOT="${STAGE_DEST_ROOT:-/tmp/$USER/.stable-wm}"

# human-readable byte size (falls back to raw bytes if numfmt is unavailable)
_stage_hr() { numfmt --to=iec "$1" 2>/dev/null || echo "$1 B"; }

_stage_main() {
    local names=("$@")
    [ "${#names[@]}" -eq 0 ] && names=("scene_play")

    mkdir -p "$DEST_ROOT/datasets" || {
        echo "stage_data: ERROR: cannot create $DEST_ROOT/datasets" >&2; return 1; }

    local name src dst alt src_size avail tmp
    for name in "${names[@]}"; do
        src="$SRC_ROOT/datasets/$name.h5"
        if [ ! -f "$src" ]; then                       # fall back to the home root
            alt="$HOME/.stable-wm/datasets/$name.h5"
            if [ -f "$alt" ]; then
                src="$alt"
            else
                echo "stage_data: ERROR: source not found for '$name':" >&2
                echo "             $SRC_ROOT/datasets/$name.h5 (also tried $alt)" >&2
                return 1
            fi
        fi
        dst="$DEST_ROOT/datasets/$name.h5"
        mkdir -p "$(dirname "$dst")"

        src_size=$(stat -c %s "$src")
        if [ -f "$dst" ] && [ "$(stat -c %s "$dst" 2>/dev/null)" = "$src_size" ]; then
            echo "stage_data: '$name' already staged ($(_stage_hr "$src_size")) -> skip"
            continue
        fi

        # free-space guard on the destination filesystem
        avail=$(df -P "$DEST_ROOT" | awk 'NR==2{print $4*1024}')
        if [ -n "$avail" ] && [ "$avail" -lt "$src_size" ]; then
            echo "stage_data: ERROR: not enough space on $DEST_ROOT" >&2
            echo "             need $(_stage_hr "$src_size"), have $(_stage_hr "$avail")" >&2
            return 1
        fi

        echo "stage_data: copying '$name' ($(_stage_hr "$src_size")) from $src ..."
        tmp="$dst.partial.$$"                           # temp + atomic mv: no truncated file on interrupt
        if command -v rsync >/dev/null 2>&1; then
            rsync -a --inplace "$src" "$tmp" || {
                echo "stage_data: ERROR: rsync failed for '$name'" >&2; rm -f "$tmp"; return 1; }
        else
            cp -f "$src" "$tmp" || {
                echo "stage_data: ERROR: cp failed for '$name'" >&2; rm -f "$tmp"; return 1; }
        fi
        mv -f "$tmp" "$dst" || {
            echo "stage_data: ERROR: could not finalize $dst" >&2; rm -f "$tmp"; return 1; }
        echo "stage_data: staged '$name' -> $dst"
        # configs that pin an absolute path (data/scene_explore.yaml) read SCENE_STAGE_DIR,
        # not LOCAL_DATASET_DIR -- mirror the file there so both mechanisms see it
        if [ -n "${SCENE_STAGE_DIR:-}" ] && [ "${name#scene}" != "$name" ]; then
            mkdir -p "$SCENE_STAGE_DIR"
            [ -e "$SCENE_STAGE_DIR/$name.h5" ] || ln -s "$dst" "$SCENE_STAGE_DIR/$name.h5"
            echo "stage_data: linked '$name' -> $SCENE_STAGE_DIR/$name.h5"
        fi
    done
    return 0
}

if _stage_main "$@"; then
    export LOCAL_DATASET_DIR="$DEST_ROOT"
    echo "stage_data: LOCAL_DATASET_DIR=$LOCAL_DATASET_DIR"
    if [ "$_STAGE_SOURCED" != 1 ]; then
        echo "stage_data: NOTE — you ran this in a subshell, so the export did NOT reach your shell."
        echo "stage_data:        Either 'source stage_data.sh', or run: export LOCAL_DATASET_DIR=$DEST_ROOT"
    fi
else
    _stage_rc=$?
    if [ "$_STAGE_SOURCED" = 1 ]; then return "$_stage_rc"; else exit "$_stage_rc"; fi
fi
