"""Build a NON-TRIVIAL CALVIN goal-offset eval set -- data/scene/make_scene_offset_tasks.py for CALVIN.

Samples (s, s+O) play-frame pairs from the canonical play pack whose goal is NOT already met at the
start under data/calvin/goal_predicate.py, and keeps each as one episode of O+1 frames: init = frame
0, goal = frame O, the dataset panel shows the expert's own path. Columns are what
config/eval/calvin.yaml's callables read (robot_obs, scene_obs, scene_id) plus the pack's 200x200
pixels for the goal image and the reference video; `action` is zeros as in the scene sets (the
planner's action statistics must come from the TRAINING file, never from a task set).

    python data/calvin/make_calvin_offset_tasks.py --offset 50 --n 100

Runs where the pack is staged locally; writes to $STABLEWM_HOME/datasets.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_env  # noqa: E402
import goal_predicate as GP  # noqa: E402
import roster as R  # noqa: E402

load_env()
COLS = ["pixels", "robot_obs", "scene_obs", "scene_id", "action", "step_idx"]
EVAL_RES = 224   # swm sizes its `pixels` info buffer from the task set's frames and then writes the
                 # env's render into it, so the two must agree: the render is the pack's 200x200
                 # resized to the eval's 224 by MegaWrapper (PIL bilinear), and so is this.


def resize_frames(px):
    return np.stack([np.asarray(Image.fromarray(f).resize((EVAL_RES, EVAL_RES), Image.BILINEAR)) for f in px])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pack", default=f"{os.environ['CALVIN_STAGE_DIR']}/calvin_abc_d.h5")
    ap.add_argument("--out-dir", default=f"{os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable-wm'))}/datasets")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    O = a.offset
    R.assert_canonical(a.pack, context="make_calvin_offset_tasks")
    out = Path(a.out_dir) / f"{a.name or f'calvin_offset_tasks_O{O}'}.h5"
    rng = np.random.default_rng(a.seed)
    eps, factors_hist, tries = [], {}, 0
    with h5py.File(a.pack, "r") as f:
        ep_off, ep_len = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
        so_all = f["scene_obs"][:]                       # 172 MB, read once
        while len(eps) < a.n and tries < 200000:
            tries += 1
            e = int(rng.integers(len(ep_len))); L = int(ep_len[e])
            if L < O + 1:
                continue
            s = int(ep_off[e]) + int(rng.integers(0, L - O)); g = s + O
            if GP.at_goal(so_all[s], so_all[g]):
                continue                                  # a freebie: nothing to manipulate
            moved = GP.moved_factors(so_all[s], so_all[g])
            for m in moved:
                factors_hist[m] = factors_hist.get(m, 0) + 1
            sl = slice(s, g + 1)
            eps.append({"pixels": resize_frames(f["pixels"][sl]), "robot_obs": f["robot_obs"][sl].astype(np.float32),
                        "scene_obs": so_all[sl].astype(np.float32),
                        "scene_id": f["scene_id"][sl].reshape(-1).astype(np.int64),
                        "action": np.zeros((O + 1, 7), np.float32),
                        "step_idx": np.arange(O + 1, dtype=np.int64), "src_frame": s})
        print(f"offset {O}: {len(eps)} non-trivial pairs of {tries} tries; moved factors {factors_hist}")
    assert len(eps) == a.n, f"only {len(eps)} pairs found"
    flat = {c: np.concatenate([ep[c] for ep in eps]) for c in COLS}
    ep_len_out = np.full(len(eps), O + 1, np.int32)
    ep_offset_out = np.concatenate([[0], np.cumsum(ep_len_out)[:-1]]).astype(np.int64)
    ep_idx_out = np.repeat(np.arange(len(eps)), ep_len_out).astype(np.int32)
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as o:
        o.create_dataset("ep_len", data=ep_len_out)
        o.create_dataset("ep_offset", data=ep_offset_out)
        o.create_dataset("ep_idx", data=ep_idx_out)
        for c, v in flat.items():
            if c == "pixels":
                o.create_dataset(c, data=v, chunks=(16, *v.shape[1:]), **blosc)
            else:
                o.create_dataset(c, data=v)
        o["scene_obs"].attrs["roster"] = R.roster_attr()
        o.attrs["offset"] = O; o.attrs["seed"] = a.seed; o.attrs["source"] = str(a.pack)
        o.attrs["tol"] = json.dumps(GP.TOL); o.attrs["moved_factors_hist"] = json.dumps(factors_hist)
        o.attrs["pixels_resize"] = f"pack 200x200 -> {EVAL_RES}x{EVAL_RES}, PIL bilinear (MegaWrapper's path)"
        # per-episode provenance goes in an ATTRIBUTE: swm's HDF5Dataset reads every dataset as a
        # per-frame column, and a 100-row dataset fancy-indexed by frame rows raises IndexError
        o.attrs["src_frames"] = np.array([ep["src_frame"] for ep in eps], np.int64)
    print(f"wrote {out}  ({len(eps)} episodes x {O + 1} frames)")


if __name__ == "__main__":
    main()
