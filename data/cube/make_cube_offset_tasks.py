"""Build the cube goal-offset eval set: (s, s+O) play-frame pairs from cube_play.h5 where the cube
is NOT already at its goal, each kept pair stored as the full (O+1)-frame expert sub-trajectory.
The cube analogue of data/scene/make_scene_offset_tasks.py (same sampling, same margin = the env's
4 cm success tolerance, one object factor instead of four), read by config/eval/cube.yaml's
callables: set_state from frame 0, set_target_pos from the goal frame's block_pos.
    python data/cube/make_cube_offset_tasks.py --offset 25 --n 100
"""
import argparse

import h5py
import hdf5plugin  # noqa: F401  (Blosc codec for reading/writing pixels)
import numpy as np
from stable_worldmodel.data.utils import get_cache_dir

CUBE = slice(14, 17)            # cube-single qpos layout (object_joint_0), same indices as scene
COLS = ["pixels", "qpos", "qvel", "block_pos", "action", "step_idx"]


def episode_cols(px, q, qv):
    return {
        "pixels": px,
        "qpos": q.astype(np.float32), "qvel": qv.astype(np.float32),
        "block_pos": q[:, CUBE].astype(np.float32),
        "action": np.zeros((len(px), 5), np.float32),
        "step_idx": np.arange(len(px), dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True, help="temporal gap O between init and goal")
    ap.add_argument("--n", type=int, default=100, help="non-trivial pairs to collect")
    ap.add_argument("--margin", type=float, default=0.04,
                    help="goal counts as non-trivial if the cube differs by > this (the success tol)")
    ap.add_argument("--source", default="cube_play", help="play file under $STABLEWM_HOME/datasets")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    O = args.offset
    root = get_cache_dir(sub_folder="datasets")
    out = root / f"{args.name or f'cube_offset_tasks_O{O}'}.h5"
    rng = np.random.default_rng(args.seed)

    episodes = []
    with h5py.File(root / f"{args.source}.h5", "r") as f:
        ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
        qpos = f["qpos"]
        tries = 0
        while len(episodes) < args.n and tries < 50000:
            tries += 1
            e = rng.integers(len(ep_len)); L = int(ep_len[e])
            if L < O + 1:
                continue
            s = int(ep_off[e]) + int(rng.integers(0, L - O))
            g = s + O
            if np.linalg.norm(qpos[s][CUBE] - qpos[g][CUBE]) <= args.margin:
                continue                                  # a freebie: the cube is already at goal
            sl = slice(s, g + 1)
            episodes.append(episode_cols(f["pixels"][sl], qpos[sl], f["qvel"][sl]))
        print(f"offset {O}: collected {len(episodes)} non-trivial pairs (of {tries} tries)")
    if len(episodes) < args.n:
        raise SystemExit(f"only {len(episodes)} < {args.n} non-trivial pairs -- not writing a short set")

    flat = {c: np.concatenate([ep[c] for ep in episodes]) for c in COLS}
    ep_len_out = np.array([len(ep["pixels"]) for ep in episodes], np.int32)
    ep_offset_out = np.concatenate([[0], np.cumsum(ep_len_out)[:-1]]).astype(np.int64)
    ep_idx_out = np.repeat(np.arange(len(episodes)), ep_len_out).astype(np.int32)
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(out, "w") as o:
        o.create_dataset("ep_len", data=ep_len_out)
        o.create_dataset("ep_offset", data=ep_offset_out)
        o.create_dataset("ep_idx", data=ep_idx_out)
        for c, v in flat.items():
            if c == "pixels":
                o.create_dataset(c, data=v, chunks=(16, *v.shape[1:]), **blosc)
            else:
                o.create_dataset(c, data=v)
    moved = np.array([np.linalg.norm(ep["block_pos"][-1] - ep["block_pos"][0]) for ep in episodes])
    print(f"wrote {out}  ({len(episodes)} episodes x {O + 1} frames); cube displacement "
          f"p10/p50/p90 = {np.percentile(moved, 10):.3f}/{np.percentile(moved, 50):.3f}/{np.percentile(moved, 90):.3f} m")


if __name__ == "__main__":
    main()
