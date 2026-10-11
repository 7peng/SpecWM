"""Render the front-camera depth field of every frame of a Scene HDF5 file.

Each frame is restored with set_state(qpos, qvel, button_states) and rendered at 224x224 with the
camera that drew the stored pixels. Output is one float16 .npy per episode, (n_frames, 224*224), in
metres, clipped to [0, D_CLIP]. build_scene_raw_depth_h5.py assembles the shards into an HDF5 file.

The cube's goal marker is a mocap body whose position set_state does not restore; its geom is moved
to a geom group the scene option does not draw, so it is absent from the depth buffer as it is from
the stored pixels.

    MUJOCO_GL=egl python data/scene/render_scene_depth.py --src $SCENE_STAGE_DIR/scene_play_mix_lateonset.h5 \
        --out-dir <shard-dir> --workers 16
"""
import argparse
import fcntl
import multiprocessing as mp
import os
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

CAM = "front_pixels"
RES = 224
D_CLIP = 2.0
HIDE_GROUP = 5
EGL_LOCK = os.path.join(tempfile.gettempdir(), "scene_depth_egl_init.lock")

_WENV = None


def _winit():
    """One SceneEnv per worker, with the goal marker hidden; renderer construction is serialised."""
    global _WENV
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

    _WENV = SceneEnv(ob_type="states", width=RES, height=RES)
    _WENV.reset()
    uw = _WENV.unwrapped
    m, d = uw._model, uw._data
    assert int((np.asarray(m.geom_group) == HIDE_GROUP).sum()) == 0
    assert int(np.asarray(uw._scene_option.geomgroup)[HIDE_GROUP]) == 0
    tgt = np.concatenate([np.asarray(x) for x in uw._cube_target_geom_ids_list]).astype(int)
    m.geom_group[tgt] = HIDE_GROUP
    for mid in uw._cube_target_mocap_ids:
        d.mocap_pos[mid] = (0.0, 0.0, -5.0)
    with open(EGL_LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        uw._initialize_renderer()


def _render_episode(job):
    ep, src, out_dir = job
    uw = _WENV.unwrapped
    with h5py.File(src, "r") as f:
        off, ln = int(f["ep_offset"][ep]), int(f["ep_len"][ep])
        Q = np.asarray(f["qpos"][off:off + ln], np.float64)
        V = np.asarray(f["qvel"][off:off + ln], np.float64)
        Bt = np.asarray(f["button_states"][off:off + ln], np.int64)
    Y = np.empty((ln, RES * RES), np.float16)
    t0 = time.perf_counter()
    for t in range(ln):
        uw.set_state(Q[t], V[t], button_state_0=int(Bt[t, 0]), button_state_1=int(Bt[t, 1]))
        D = np.asarray(uw.render(camera=CAM, depth=True), np.float32)
        if not np.isfinite(D).all():
            raise ValueError(f"ep {ep} t {t}: non-finite depth")
        Y[t] = np.clip(D, 0.0, D_CLIP).ravel()
    _atomic_save(os.path.join(out_dir, f"ep{ep:04d}.npy"), Y)
    return ep, ln, (time.perf_counter() - t0) / ln * 1e3


def _atomic_save(path, arr):
    tmp = f"{path[:-4]}.tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _done(path, n):
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape == (n, RES * RES) and a.dtype == np.float16
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.src, "r") as f:
        ep_len = np.asarray(f["ep_len"])
    episodes = range(len(ep_len)) if args.episodes is None else args.episodes
    todo = [ep for ep in episodes if not _done(os.path.join(args.out_dir, f"ep{ep:04d}.npy"), int(ep_len[ep]))]
    print(f"[render] {len(todo)} episodes to render, workers={args.workers}, out={args.out_dir}", flush=True)
    if not todo:
        return
    jobs = [(ep, args.src, args.out_dir) for ep in todo]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(min(args.workers, len(jobs)), initializer=_winit) as pool:
        for i, (ep, ln, ms) in enumerate(pool.imap_unordered(_render_episode, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] ep{ep:04d} n={ln} {ms:.2f} ms/frame "
                      f"| {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
