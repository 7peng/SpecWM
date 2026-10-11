"""Bit-reproducible build of the OGBench CUBE datasets: cube_play (marker-free re-render of the
expert play) and cube_play_noised (the failure data), on the scene builder's machinery.

cube-single is the scene env without drawer, window and buttons, and this module is
data/scene/build_scene_noised.py with exactly those differences: CubeEnv for SceneEnv, no button
column anywhere (source table, reset, rendered episode), its own SPEC (source archive, output
names) and its own goldens. Every mechanism -- contiguous episode blocks, index-keyed per-episode
noise streams, chunk-verbatim merge, the two content chains, preflight/selftest -- is IMPORTED from
the scene builder rather than copied, so the two builders cannot drift; the scene module's
env-specific globals are rebound below, which is the whole port.

WHY REBUILD. The shipped cube_play.h5 (legacy/scripts/collect_cube.py, 2026-06-17) was rendered
under the env's visualize_info default (True): ManipSpaceEnv.reset() draws cur_task_id from the
UNSEEDED global np.random and paints a translucent goal cube at that task's target position -- the
same ghost scene_play carried before its v3 rebuild. Every pixel of that file, and every model
trained on it, saw a goal marker the state columns do not record, at a position that changed
between render processes. SPEC pins task_id AND turns the marker off, exactly as scene-noised-v3.

Noised dynamics are the scene recipe verbatim: episode e is STEPPED from its recorded start state
under a_t = clip(expert_a_t + sigma*z, -1, 1), sigma = sigmas[e % 3], z from default_rng(1000 + e);
the frame is captured BEFORE the action is applied.

Pipeline (see run_build_cube_noised.sh):
    build_cube_noised.py prepare                     # on a node that mounts ~/.ogbench (login)
    build_cube_noised.py preflight [--emit]          # --emit once on the reference machine
    build_cube_noised.py selftest
    build_cube_noised.py render --kind play|noised --block K --num-blocks W --gpu G
    build_cube_noised.py merge  --kind play|noised --num-blocks W
    build_cube_noised.py verify --kind play|noised [--deep]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scene"))
import build_scene_noised as B  # noqa: E402  (the machinery; env-specific names rebound below)

SPEC = {
    "version": "cube-noised-v1",
    "source_dataset": "cube-single-play-v0",
    "source_sha256": "80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e",
    "min_episode_len": 10,          # drops compact-format padding rows (1001 -> 1000 frames)
    "max_episodes": None,           # None = every episode passing min_episode_len (1000)
    # --- noised dynamics (scene-noised-v3 verbatim) ---------------------------------------
    "sigmas": [0.0, 0.3, 0.6],      # episode e gets sigmas[e % 3]
    "noise_seed": 1000,             # per-episode stream = default_rng(noise_seed + e)
    # --- render ----------------------------------------------------------------------------
    "width": 224,
    "height": 224,
    "camera": "front_pixels",
    "offsamples": 4,                # MSAA as shipped (pixels +/-1 LSB across machines)
    "supersample": 1,
    "visualize_info": False,        # THE FIX: no goal marker can reach the pixels
    "task_id": 1,                   # pins the otherwise-unseeded task draw
    "columns": ["pixels", "action", "observation", "qpos", "qvel", "step_idx"],
    "pixel_chunk": 50,              # divides the 1000-frame episode -> chunk-aligned blocks
    "blosc": {"cname": "lz4", "clevel": 5, "shuffle": "SHUFFLE"},
}

# Recorded by `preflight --emit` on the reference GPU (L40S); a build elsewhere must
# reproduce the STATE digests exactly and the pixel means within PIXEL_MEAN_TOL.
GOLDEN = {
    "play_ep0": {"state": "422154ca7df98136b0353dcac4c6dafae6bb376cb7a4c27e60a7807e9c9ba0ba", "pixel_mean": 58.709074502},
    "play_ep500": {"state": "b4e8be9ac58025197a28018e56538fc095864e890899ae2c4a31ac6d22c06e8b", "pixel_mean": 59.868479840},
    "noised_ep1": {"state": "fbc4ea69628d7d275efd4fd28f4e784d701df9e350968493e0f65cd86fc678e3", "pixel_mean": 61.061858480},
}

GOLDEN_PROBES = [("play_ep0", "play", 0), ("play_ep500", "play", 500), ("noised_ep1", "noised", 1)]


# ---------------------------------------------------------------------------------------------
# source + episode table (no button column)
# ---------------------------------------------------------------------------------------------
def prepare(prep_path, spec=SPEC, verify_source=True):
    """Read the pinned OGBench archive once and freeze the episode table workers replay."""
    import ogbench
    from ogbench.utils import DEFAULT_DATASET_DIR
    src = Path(DEFAULT_DATASET_DIR).expanduser() / f"{spec['source_dataset']}.npz"
    ds, _ = ogbench.make_env_and_datasets(
        spec["source_dataset"], dataset_only=True, add_info=True, compact_dataset=True)
    if verify_source:
        if not src.exists():
            raise SystemExit(f"source archive not found at {src}")
        got = B.sha256_file(src)
        if got != spec["source_sha256"]:
            raise SystemExit(
                f"SOURCE MISMATCH\n  {src}\n  expected {spec['source_sha256']}\n  got      {got}\n"
                "The upstream OGBench archive is not the one this SPEC was pinned to.")
        print(f"[prepare] source sha256 OK ({src})", flush=True)
    term = np.asarray(ds["terminals"])
    ends = np.flatnonzero(term > 0)
    slices, start = [], 0
    for e in ends:
        if int(e) + 1 - start >= spec["min_episode_len"]:
            slices.append((start, int(e) + 1))
        start = int(e) + 1
    if spec["max_episodes"] is not None:
        slices = slices[: spec["max_episodes"]]
    ep_start = np.array([a for a, _ in slices], np.int64)
    ep_len = np.array([b - a for a, b in slices], np.int64)
    sig = np.array([spec["sigmas"][i % len(spec["sigmas"])] for i in range(len(slices))], np.float32)
    qpos, qvel = (np.asarray(ds[k]) for k in ("qpos", "qvel"))
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    # native dtypes: these arrays go straight to set_state
    np.savez(
        prep_path,
        actions=ds["actions"], observations=ds["observations"],
        qpos_all=qpos, qvel_all=qvel,
        ep_start=ep_start, ep_len=ep_len, ep_sigma=sig,
        spec_digest=np.array(B.spec_digest(spec)),
    )
    print(f"[prepare] {len(slices)} episodes, {int(ep_len.sum())} frames, "
          f"sigma split {[int((sig == s).sum()) for s in spec['sigmas']]} -> {prep_path}", flush=True)


# ---------------------------------------------------------------------------------------------
# environment + rollout
# ---------------------------------------------------------------------------------------------
def make_env(spec=SPEC, gpu=0):
    """CubeEnv (single cube) pinned to the SPEC's render configuration."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
    f = spec["supersample"]
    env = CubeEnv(env_type="single", ob_type="states", width=spec["width"] * f,
                  height=spec["height"] * f, visualize_info=spec["visualize_info"])
    env.reset(options={"task_id": spec["task_id"]})     # pins the otherwise-unseeded task draw
    uw = env.unwrapped
    uw.model.vis.quality.offsamples = spec["offsamples"]
    if uw._renderer is not None:                        # MjrContext caches offsamples at creation
        uw._renderer.close()
        uw._renderer = None
    return env


def reset_sim(uw, qpos, qvel):
    """A sim state that depends on nothing but its arguments (see the scene builder's note on the
    solver warm start, the clock and cfrc_ext)."""
    import mujoco
    uw.data.qacc_warmstart[:] = 0.0
    uw.data.time = 0.0
    uw.set_state(qpos, qvel)
    mujoco.mj_rnePostConstraint(uw.model, uw.data)


def render_play(env, d, e, spec=SPEC):
    """Expert episode: TELEPORT to each recorded state and render. No stepping, no noise."""
    uw = env.unwrapped
    s, n = int(d["ep_start"][e]), int(d["ep_len"][e])
    qpos, qvel = d["qpos_all"][s:s + n], d["qvel_all"][s:s + n]
    px = np.empty((n, spec["height"], spec["width"], 3), np.uint8)
    for t in range(n):
        reset_sim(uw, qpos[t], qvel[t])
        px[t] = B.grab(uw, spec)
    return {"pixels": px,
            "action": d["actions"][s:s + n].astype(np.float32),
            "observation": d["observations"][s:s + n].astype(np.float32),
            "qpos": qpos.astype(np.float32), "qvel": qvel.astype(np.float32),
            "step_idx": np.arange(n, dtype=np.int64)}


def render_noised(env, d, e, spec=SPEC):
    """Noised episode: STEP from the recorded start under expert action + sigma*N(0,1); the frame
    is captured BEFORE the action is applied (frame[t] = s_t, action[t] = the action taken at s_t)."""
    uw = env.unwrapped
    s, n = int(d["ep_start"][e]), int(d["ep_len"][e])
    sig = float(d["ep_sigma"][e])
    actions = d["actions"]
    adim = actions.shape[1]
    rng = B.episode_rng(e, spec)
    reset_sim(uw, d["qpos_all"][s], d["qvel_all"][s])
    nq, nv = d["qpos_all"].shape[1], d["qvel_all"].shape[1]
    px = np.empty((n, spec["height"], spec["width"], 3), np.uint8)
    QP = np.empty((n, nq), np.float32)
    QV = np.empty((n, nv), np.float32)
    AC = np.empty((n, adim), np.float32)
    OB = []
    ob = np.asarray(uw.compute_observation(), np.float32)
    for t in range(n):
        px[t] = B.grab(uw, spec)
        QP[t] = uw.data.qpos[:nq]
        QV[t] = uw.data.qvel[:nv]
        OB.append(ob)
        a = np.clip(actions[s + t] + sig * rng.standard_normal(adim).astype(np.float32), -1.0, 1.0)
        AC[t] = a
        ob = np.asarray(env.step(a)[0], np.float32)
    return {"pixels": px, "action": AC, "observation": np.asarray(OB, np.float32),
            "qpos": QP, "qvel": QV, "step_idx": np.arange(n, dtype=np.int64)}


# THE PORT: the scene module's env-specific globals, looked up at call time by render_block,
# preflight and selftest, now name the cube versions. Everything else runs unchanged.
B.make_env = make_env
B.reset_sim = reset_sim
B.render_play, B.render_noised = render_play, render_noised
B.RENDERERS = {"play": render_play, "noised": render_noised}
B.GOLDEN = GOLDEN
B.GOLDEN_PROBES = GOLDEN_PROBES


# ---------------------------------------------------------------------------------------------
def main():
    from stable_worldmodel.data.utils import get_cache_dir

    root = get_cache_dir(sub_folder="datasets")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "render", "merge", "mix", "verify", "preflight", "selftest", "spec"])
    ap.add_argument("--kind", choices=[*B.KINDS, "mix"], default="noised")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--num-blocks", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--deep", action="store_true", help="verify: re-read pixels instead of trusting stored digests")
    ap.add_argument("--emit", action="store_true", help="preflight: print GOLDEN block to paste into this file")
    ap.add_argument("--repeats", type=int, default=2, help="preflight: replays per probe episode")
    ap.add_argument("--no-verify-source", action="store_true")
    ap.add_argument("--prep", type=Path, default=root / "_cube_build/prep.npz")
    ap.add_argument("--block-dir", type=Path, default=root / "_cube_build")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--play", type=Path, default=None, help="mix: expert input (default cube_play.h5)")
    ap.add_argument("--noised", type=Path, default=None, help="mix: noised input (default cube_play_noised.h5)")
    a = ap.parse_args()

    default_out = {"play": root / "cube_play.h5", "noised": root / "cube_play_noised.h5",
                   "mix": root / "cube_play_mix_noised.h5"}
    out = a.out or default_out[a.kind]

    if a.cmd == "spec":
        print(json.dumps({"spec": SPEC, "spec_digest": B.spec_digest(SPEC), "golden": GOLDEN,
                          "environment": B.env_fingerprint()}, indent=1))
    elif a.cmd == "prepare":
        prepare(a.prep, SPEC, verify_source=not a.no_verify_source)
    elif a.cmd == "preflight":
        B.preflight(a.prep, a.gpu, emit=a.emit, repeats=a.repeats, spec=SPEC)
    elif a.cmd == "selftest":
        B.selftest(a.prep, a.gpu, spec=SPEC)
    elif a.cmd == "render":
        if a.block is None:
            raise SystemExit("--block is required")
        B.render_block(a.kind, a.block, a.num_blocks, a.gpu, a.prep,
                       a.block_dir / f"{a.kind}_block_{a.block}.h5", spec=SPEC)
    elif a.cmd == "merge":
        B.merge(a.kind, a.block_dir, a.num_blocks, out, spec=SPEC)
    elif a.cmd == "mix":
        B.mix(a.play or default_out["play"], a.noised or default_out["noised"],
              a.out or default_out["mix"], spec=SPEC)
    elif a.cmd == "verify":
        raise SystemExit(0 if B.verify(out, deep=a.deep, spec=SPEC) else 1)


if __name__ == "__main__":
    main()
