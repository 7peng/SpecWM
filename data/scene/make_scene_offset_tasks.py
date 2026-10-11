"""Build a NON-TRIVIAL scene goal-offset eval set (full init->goal sub-trajectories).

scene_offset.h5 uses long windows where eval.py samples arbitrary starts, so most
goals are already satisfied at the start (no manipulation needed) and success is
inflated. This instead samples (s, s+O) play-frame PAIRS where the goal is NOT
already achieved at the start, so every goal demands a real manipulation.

Each kept pair becomes one episode = the full expert sub-trajectory frames s..s+O
(O+1 frames). Eval uses goal_offset_steps=O: init = frame 0, goal = frame O, and
the "dataset" reference panel shows the real path (not a 2-frame snap). The
cube/drawer/window/button targets are read from the goal frame by
config/eval/scene.yaml's callables.

    python scripts/make_scene_offset_tasks.py --offset 50 --n 40
"""

import argparse

import h5py
import hdf5plugin  # noqa: F401  (Blosc codec for reading/writing pixels)
import numpy as np

from stable_worldmodel.data.utils import get_cache_dir

CUBE = slice(14, 17)            # scene qpos layout (object_joint_0), from env probe
DRAWER, WINDOW = 23, 24         # drawer_slide / window_slide qpos
COLS = ["pixels", "qpos", "qvel", "button_state_0", "button_state_1",
        "block_pos", "drawer_pos", "window_pos", "action", "step_idx"]


def already_solved(q0, qO, b0, bO, margin):
    """True if the goal object-config is already met at the start (a freebie)."""
    return (np.linalg.norm(q0[CUBE] - qO[CUBE]) <= margin
            and abs(q0[DRAWER] - qO[DRAWER]) <= margin
            and abs(q0[WINDOW] - qO[WINDOW]) <= margin
            and np.array_equal(b0, bO))


def episode_cols(px, q, qv, bs):
    """Per-frame columns for one (O+1)-frame sub-trajectory."""
    return {
        "pixels": px,
        "qpos": q.astype(np.float32), "qvel": qv.astype(np.float32),
        "button_state_0": bs[:, 0].astype(np.int64), "button_state_1": bs[:, 1].astype(np.int64),
        "block_pos": q[:, CUBE].astype(np.float32),
        "drawer_pos": q[:, DRAWER].astype(np.float32), "window_pos": q[:, WINDOW].astype(np.float32),
        "action": np.zeros((len(px), 5), np.float32),
        "step_idx": np.arange(len(px), dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True, help="temporal gap O between init and goal")
    ap.add_argument("--n", type=int, default=40, help="non-trivial pairs to collect (>= num_eval+1)")
    ap.add_argument("--margin", type=float, default=0.04,
                    help="goal counts as non-trivial if an object differs by > this (success tol)")
    ap.add_argument("--require-cube", type=float, default=0.0,
                    help="if >0, keep ONLY pairs whose cube moves more than this (m) — builds a "
                         "cube-MANIPULATION benchmark instead of the (arm-dominated) random-offset set")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    O = args.offset
    out = get_cache_dir(sub_folder="datasets") / f"{args.name or f'scene_offset_tasks_O{O}'}.h5"
    rng = np.random.default_rng(args.seed)
    src = get_cache_dir(sub_folder="datasets") / "scene_play.h5"

    episodes = []  # each = dict of (O+1)-frame columns for one sub-trajectory
    with h5py.File(src, "r") as f:
        ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
        qpos, btn = f["qpos"], f["button_states"]
        tries = 0
        while len(episodes) < args.n and tries < 50000:
            tries += 1
            e = rng.integers(len(ep_len)); L = int(ep_len[e])
            if L < O + 1:
                continue
            s = int(ep_off[e]) + int(rng.integers(0, L - O))
            g = s + O
            if already_solved(qpos[s], qpos[g], btn[s], btn[g], args.margin):
                continue
            if args.require_cube > 0 and \
                    np.linalg.norm(qpos[s][CUBE] - qpos[g][CUBE]) <= args.require_cube:
                continue  # cube-manipulation set: drop pairs where the cube barely moves
            sl = slice(s, g + 1)  # full sub-trajectory s..s+O (inclusive)
            episodes.append(episode_cols(f["pixels"][sl], qpos[sl], f["qvel"][sl], btn[sl]))
        print(f"offset {O}: collected {len(episodes)} non-trivial pairs (of {tries} tries)")

    flat = {c: np.concatenate([ep[c] for ep in episodes]) for c in COLS}
    ep_len_out = np.array([len(ep["pixels"]) for ep in episodes], np.int32)  # all O+1
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
    print(f"wrote {out}  ({len(episodes)} episodes x {O + 1} frames)")


if __name__ == "__main__":
    main()
