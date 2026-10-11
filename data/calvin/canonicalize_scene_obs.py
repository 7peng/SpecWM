"""Permute an already-built CALVIN pack's scene_obs blocks into canonical red, blue, pink order.

`roster.py` holds the why: calvin_env lists the three movable blocks in a different order in
three of the four scenes, so scene_obs[6:9] is red only in scenes B and D. Packs built before
2026-09-04 carry the simulator's per-scene order.

    python data/calvin/canonicalize_scene_obs.py --h5 $CALVIN_STAGE_DIR/calvin_abc_d.h5 --check
    python data/calvin/canonicalize_scene_obs.py --h5 $CALVIN_STAGE_DIR/calvin_abc_d.h5

Only the 24-column scene_obs dataset is rewritten (172 MB of a 216 GB pack); pixels are not
touched. The write is one assignment of an array built in RAM, guarded by an attribute so a
migrated pack is never permuted twice and an interrupted one refuses to run without --force.

THE GATE, on the language annotations and independent of the configs: for every scene and every
annotation window naming one block colour, the slot that moved most between the window's first
and last frame must be that colour's canonical slot (red 0, blue 1, pink 2). Before migration
that holds per scene under the simulator's own order; after it, it must hold under the canonical
one at the same rate. --check reports both without writing.
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_env  # noqa: E402

import roster as R  # noqa: E402

load_env()

IN_PROGRESS = "roster_migration_in_progress"


def moved_slot_agreement(scene_obs, scene_id, root, split="training"):
    """(agree, total) per scene: does the colour named by the annotation move in its canonical slot?

    Returns {scene_letter: (n_agree, n_windows)} plus the pooled pair.
    """
    ann_path = Path(root) / "task_ABC_D" / split / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    # annotation frame ids are absolute; the pack's row 0 is the smallest id of the split
    lo = int(np.load(Path(root) / "task_ABC_D" / split / "ep_start_end_ids.npy")[:, 0].min())
    xyz = scene_obs[:, R.BLOCK_START:].reshape(len(scene_obs), len(R.CANONICAL), R.BLOCK_DIM)[..., :3]
    sid_to_letter = {v: k for k, v in R.SCENE_IDS.items()}

    per = {}
    for task, (a, b) in zip(ann["language"]["task"], ann["info"]["indx"]):
        named = [c for c in R.CANONICAL if c in task]
        if len(named) != 1:
            continue
        a, b = int(a) - lo, int(b) - lo
        if a < 0 or b >= len(scene_obs):
            continue
        letter = sid_to_letter.get(int(scene_id[a]))
        if letter is None:
            continue
        moved = int(np.linalg.norm(xyz[b] - xyz[a], axis=-1).argmax())
        agree, total = per.get(letter, (0, 0))
        per[letter] = (agree + (moved == R.CANONICAL.index(named[0])), total + 1)
    pooled = (sum(v[0] for v in per.values()), sum(v[1] for v in per.values()))
    return per, pooled


def report(tag, per, pooled):
    order = sorted(per)
    cells = "  ".join(f"{k} {a}/{t} ({a / t:.3f})" for k in order for a, t in [per[k]])
    print(f"[{tag}] {cells}   pooled {pooled[0]}/{pooled[1]} ({pooled[0] / pooled[1]:.4f})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=f"{os.environ['CALVIN_STAGE_DIR']}"
                                    "/calvin_abc_d.h5")
    ap.add_argument("--root", default=os.environ.get("CALVIN_ROOT"))
    ap.add_argument("--split", default="training")
    ap.add_argument("--check", action="store_true", help="report the gate, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="proceed even though a previous migration did not finish")
    ap.add_argument("--min-agreement", type=float, default=0.95)
    args = ap.parse_args()

    R.verify_against_configs(os.environ["CALVIN_ENV_SRC"])
    print(f"[roster] scene yamls agree with SCENE_ROSTER {R.SCENE_ROSTER}", flush=True)

    with h5py.File(args.h5, "r+" if not args.check else "r") as f:
        d = f["scene_obs"]
        stamped = d.attrs.get("roster")
        pending = bool(d.attrs.get(IN_PROGRESS, 0))
        print(f"[roster] {args.h5}: scene_obs {d.shape} {d.dtype}, "
              f"roster attr = {stamped!r}, migration_in_progress = {pending}", flush=True)

        scene_id = f["scene_id"][:].reshape(-1)
        raw = d[:]
        per_raw, pooled_raw = moved_slot_agreement(raw, scene_id, args.root, args.split)
        report("as stored ", per_raw, pooled_raw)

        canon = R.canonicalize(raw, scene_id)
        per_new, pooled_new = moved_slot_agreement(canon, scene_id, args.root, args.split)
        report("canonical ", per_new, pooled_new)

        changed = int((raw != canon).any(1).sum())
        print(f"[roster] permutation changes {changed:,} of {len(raw):,} rows "
              f"({changed / len(raw):.1%})", flush=True)

        if stamped == R.roster_attr() and not args.force:
            print("[roster] already canonical -- nothing to do", flush=True)
            return
        if pending and not args.force:
            raise SystemExit("[roster] a previous migration did not finish; the column may be "
                             "half-written. Rebuild the pack, or re-run with --force if you know "
                             "the column is still the simulator's order.")
        if args.check:
            print("[roster] --check: no write", flush=True)
            return

        rate = pooled_new[0] / pooled_new[1]
        if rate < args.min_agreement:
            raise SystemExit(f"[roster] GATE FAILED: canonical agreement {rate:.4f} is below "
                             f"{args.min_agreement}; not writing")

        d.attrs[IN_PROGRESS] = 1
        f.flush()
        d[...] = canon
        f.flush()
        d.attrs["roster"] = R.roster_attr()
        d.attrs["roster_source"] = "canonicalize_scene_obs.py"
        d.attrs[IN_PROGRESS] = 0
        f.flush()
        print(f"[roster] wrote canonical scene_obs and stamped roster={R.roster_attr()!r}",
              flush=True)


if __name__ == "__main__":
    main()
