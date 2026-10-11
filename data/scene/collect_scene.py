"""Render OGBench scene-play into the cube_single_expert.h5 schema at 224x224.

OGBench's `scene-play-v0` is a state-only dataset (1.001M transitions, 2000
episodes). We replay each transition through the swm SceneEnv (`set_state` from
the stored qpos/qvel/button_states), render the front camera at 224, and stream
the result to a single HDF5 file with the same flat per-step + ep_len/ep_offset
layout the training/eval pipelines expect.

Pixels are far too large to buffer in memory (1M x 224x224x3 uint8 ~ 150 GB), so
episodes are written incrementally to resizable, Blosc-compressed datasets.

    python scripts/collect_scene.py --test                # 3 eps -> /tmp, round-trips
    python scripts/collect_scene.py --episodes 1000       # first run
    python scripts/collect_scene.py                       # all 2000 episodes
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import tempfile
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import ogbench
from tqdm import tqdm

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

# columns written per step (model uses pixels+action; the rest support eval
# goal-setting / analysis and dataset stats)
PER_STEP = ["pixels", "action", "observation", "qpos", "qvel", "button_states", "step_idx"]


def episode_slices(terminals):
    """Yield (start, stop) per episode, delimited by terminals == 1."""
    ends = np.flatnonzero(terminals > 0)
    start = 0
    for e in ends:
        yield start, int(e) + 1
        start = int(e) + 1


def render_episode(env, ds, sl):
    """Set state + render every frame in dataset slice `sl`; return episode dict."""
    uw = env.unwrapped
    s0, s1 = sl
    qpos, qvel = ds["qpos"][s0:s1], ds["qvel"][s0:s1]
    btn, act, obs = ds["button_states"][s0:s1], ds["actions"][s0:s1], ds["observations"][s0:s1]
    pixels = np.empty((s1 - s0, 224, 224, 3), np.uint8)
    for t in range(s1 - s0):
        uw.set_state(qpos[t], qvel[t], button_state_0=int(btn[t, 0]), button_state_1=int(btn[t, 1]))
        pixels[t] = np.asarray(uw.render(), np.uint8)
    return {
        "pixels": pixels,
        "action": act.astype(np.float32),
        "observation": obs.astype(np.float32),
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "button_states": btn.astype(np.int64),
        "step_idx": np.arange(s1 - s0, dtype=np.int64),
    }


def stream_write(path, episodes, n_episodes):
    """Stream episode dicts into one HDF5 file (resizable + Blosc on pixels)."""
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep_len, ptr, dsets = [], 0, {}
    with h5py.File(path, "w", libver="latest") as f:
        for ep in tqdm(episodes, total=n_episodes, desc="render"):
            n = len(ep["action"])
            if not dsets:
                for col in PER_STEP:
                    per = np.asarray(ep[col]).shape[1:]
                    kw = dict(chunks=(min(64, n), *per), **blosc) if col == "pixels" else {}
                    dsets[col] = f.create_dataset(
                        col, shape=(0, *per), maxshape=(None, *per),
                        dtype=np.asarray(ep[col]).dtype, **kw)
            for col in PER_STEP:
                dsets[col].resize(ptr + n, axis=0)
                dsets[col][ptr:ptr + n] = ep[col]
            ep_len.append(n)
            ptr += n
        ep_len = np.asarray(ep_len, np.int32)
        ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_idx", data=np.repeat(np.arange(len(ep_len)), ep_len).astype(np.int32))
    return ep_len


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=None, help="cap episodes (default: all)")
    ap.add_argument("--min-len", type=int, default=10,
                    help="drop episodes shorter than this (filters compact-format padding frames)")
    ap.add_argument("--dataset", default="scene-play-v0")
    ap.add_argument("--output", type=Path, default=None,
                    help="defaults to <cache>/datasets/scene_play.h5")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        args.episodes, args.output = 3, Path(tempfile.gettempdir()) / "scene_test.h5"
    if args.output is None:
        args.output = get_cache_dir(sub_folder="datasets") / "scene_play.h5"

    train_ds, _ = ogbench.make_env_and_datasets(
        args.dataset, dataset_only=True, add_info=True, compact_dataset=True)
    term = np.asarray(train_ds["terminals"])
    slices = [(a, b) for a, b in episode_slices(term) if b - a >= args.min_len]
    if args.episodes is not None:
        slices = slices[:args.episodes]
    print(f"{len(slices)} episodes, {sum(b - a for a, b in slices)} frames -> {args.output}")

    env = SceneEnv(ob_type="states", width=224, height=224)
    env.reset()
    eps = (render_episode(env, train_ds, sl) for sl in slices)
    ep_len = stream_write(args.output, eps, len(slices))
    env.close()

    print(f"\nwrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
    print(f"  episodes={len(ep_len)}  frames={int(ep_len.sum())}  "
          f"ep_len min/med/max={ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}")

    if args.test:
        from stable_worldmodel.data import HDF5Dataset
        ds = HDF5Dataset(path=str(args.output), keys_to_load=["pixels", "action", "observation"])
        print(f"round-trip OK: cols={ds.column_names} "
              f"rows={len(ds.get_col_data('action'))} "
              f"episodes={len(np.unique(ds.get_col_data('ep_idx')))}")


if __name__ == "__main__":
    main()
