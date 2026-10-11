"""Pack CALVIN task_ABC_D into the flat HDF5 schema the training pipeline reads.

CALVIN ships one .npz per frame -- 1,795,045 of them for training -- each holding rgb_static,
rgb_gripper, depths, actions, rel_actions, robot_obs and scene_obs. One file per frame is
unusable as a corpus: a clip read is num_steps file opens on NFS. This packs the columns the
objective needs into the same flat layout scene_play.h5 uses (per-frame arrays plus ep_idx /
step_idx / ep_len / ep_offset), which is what swm.data.HDF5Dataset expects.

    python data/calvin/build_calvin_h5.py --split training --out $CALVIN_STAGE_DIR

UNCOMPRESSED, and written node-local by default. The pixel block is 215 GB for training; /tmp is
7 TB of NVMe here, and an uncompressed random read costs no CPU at train time, which matters for
a corpus read at random clip starts by several dataloader workers. The cost is that /tmp does not
survive an allocation change, so this has to be rerun on a new node (~1 h). Pass --out to a
scratch path for a persistent copy.

COLUMNS, and why each:
  pixels      rgb_static at its native 200x200. NOT resized here -- the loader's img_size does
              that, and the probe suite renders at 200 for the same reason, so both traverse the
              same path. Resizing at pack time would bake one choice in.
  action      rel_actions (7): CALVIN's relative end-effector command, the standard policy target.
  scene_obs   (24): [0:6] slider, drawer, button, switch, lightbulb, green_light, then 3 blocks x
              (x,y,z,euler). THE privileged column -- fixed 3-block roster, so a kernel over it
              needs no set machinery and no category weighting. The blocks are PERMUTED INTO
              canonical red, blue, pink order here (see roster.py): the simulator writes them in
              its scene config's order, which differs in three of the four scenes, so the raw
              column's slot 0 is pink in scene A and blue in scene C. `scene_obs.attrs["roster"]`
              records that this pack is canonical.
  robot_obs   (15): proprioception, for the covariate side of any excess-R2 read.
  ep_idx      episode index; ep_len / ep_offset give the clip table its episode boundaries.
  step_idx    position within the episode.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import roster as R
from utils import load_env

load_env()

H = W = 200
COLS = {"action": (7, np.float32), "scene_obs": (24, np.float32), "robot_obs": (15, np.float32)}


def read_block(args):
    """One contiguous run of frame ids -> the packed columns for it."""
    root, ids = args
    n = len(ids)
    px = np.empty((n, H, W, 3), np.uint8)
    out = {k: np.empty((n, d), t) for k, (d, t) in COLS.items()}
    for i, fid in enumerate(ids):
        z = np.load(f"{root}/episode_{fid:07d}.npz")
        px[i] = z["rgb_static"]
        out["action"][i] = z["rel_actions"]
        out["scene_obs"][i] = z["scene_obs"]
        out["robot_obs"][i] = z["robot_obs"]
    return px, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training", choices=("training", "validation"))
    ap.add_argument("--task", default="task_ABC_D")
    ap.add_argument("--root", default=os.environ.get(
        "CALVIN_ROOT", os.path.expanduser("~/.stable-wm/calvin")))
    ap.add_argument("--out", default=os.environ["CALVIN_STAGE_DIR"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--block", type=int, default=2000, help="frames per worker task")
    args = ap.parse_args()

    # the block permutation is a claim about calvin_env's scene configs; check it when reachable
    if os.environ.get("CALVIN_ENV_SRC") and Path(os.environ["CALVIN_ENV_SRC"]).exists():
        R.verify_against_configs(os.environ["CALVIN_ENV_SRC"])
        print(f"roster: scene yamls agree with {R.SCENE_ROSTER}", flush=True)

    src = Path(args.root) / args.task / args.split
    se = np.load(src / "ep_start_end_ids.npy")
    # episode boundaries are INCLUSIVE frame ids and are not contiguous across episodes in the
    # validation split, so the frame list is built per episode rather than as one arange
    eps = [np.arange(int(a), int(b) + 1) for a, b in se]
    ep_len = np.array([len(e) for e in eps], np.int32)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len)])[:-1].astype(np.int64)
    ids = np.concatenate(eps)
    N = len(ids)
    ep_idx = np.repeat(np.arange(len(eps), dtype=np.int32), ep_len)
    step_idx = np.concatenate([np.arange(l, dtype=np.int64) for l in ep_len])
    # scene A/B/C per frame, from the split's scene_info (inclusive frame-id ranges)
    scene_info = np.load(src / "scene_info.npy", allow_pickle=True).item()
    scene_id = np.full((N, 1), -1, np.int64)
    for k, name in enumerate(("calvin_scene_A", "calvin_scene_B", "calvin_scene_C", "calvin_scene_D")):
        if name in scene_info:
            a, b = scene_info[name]
            scene_id[(ids >= a) & (ids <= b)] = k
    assert (scene_id >= 0).all(), "frames outside every scene range"
    missing = [f for f in ids[:: max(1, N // 2000)]
               if not (src / f"episode_{int(f):07d}.npz").exists()]
    assert not missing, f"{len(missing)} sampled frames absent, e.g. {missing[:3]}"
    print(f"{args.split}: {len(eps)} episodes, {N:,} frames "
          f"({N * H * W * 3 / 1e9:.0f} GB of pixels)", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    dst = Path(args.out) / f"calvin_abc_d{'' if args.split == 'training' else '_val'}.h5"
    blocks = [(str(src), ids[i:i + args.block]) for i in range(0, N, args.block)]

    t0 = time.time()
    with h5py.File(dst, "w") as f:
        # ONE frame per chunk: the loader reads a clip's kept rows one by one at frameskip > 1
        # (utils.RowReadHDF5Dataset); a multi-frame chunk makes every clip read its whole raw span
        # (82 MB to keep 4 MB at frameskip 20). rechunk_calvin_h5.py converts an older pack.
        dpx = f.create_dataset("pixels", (N, H, W, 3), np.uint8, chunks=(1, H, W, 3))
        dc = {k: f.create_dataset(k, (N, d), t) for k, (d, t) in COLS.items()}
        f.create_dataset("ep_idx", data=ep_idx)
        f.create_dataset("step_idx", data=step_idx)
        f.create_dataset("scene_id", data=scene_id)
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        at = 0
        with ProcessPoolExecutor(args.workers) as pool:
            for bi, (px, cols) in enumerate(pool.map(read_block, blocks, chunksize=1)):
                n = len(px)
                dpx[at:at + n] = px
                # the blocks leave the simulator in the scene's own order; store them red,
                # blue, pink so a slot means the same object in every scene (roster.py)
                cols["scene_obs"] = R.canonicalize(cols["scene_obs"], scene_id[at:at + n, 0])
                for k, v in cols.items():
                    dc[k][at:at + n] = v
                at += n
                if bi % 50 == 0 or at == N:
                    el = time.time() - t0
                    print(f"  {at:>9,}/{N:,}  {at / N:5.1%}  {at / max(el, 1e-9):,.0f} fr/s  "
                          f"eta {(N - at) / max(at / el, 1e-9) / 60:5.1f} min", flush=True)
        assert at == N, f"wrote {at} of {N}"
        # stamped only once the column is complete, so a half-written pack never claims it
        dc["scene_obs"].attrs["roster"] = R.roster_attr()
        dc["scene_obs"].attrs["roster_source"] = "build_calvin_h5.py"
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.0f} GB) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
