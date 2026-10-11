"""scene_play_noised_lateonset.h5: the scene play episodes re-executed with LATE-ONSET action noise,
the cube protocol (data/cube/build_cube_lateonset.py) on scene. Each episode replays the recorded
actions EXACTLY until a per-episode onset frame t0, drawn uniformly from [onset.low_frac,
onset.high_frac] of the episode, and adds sigma*N(0,1) to the expert action from t0 on. Episode e
gets sigmas[e % 2] and its own stream default_rng(noise_seed + e); t0 is the stream's first draw.
The episode table comes from scene_play.h5; the render machinery is build_scene_noised unchanged.

    python data/scene/build_scene_lateonset.py prepare --play $SCENE_STAGE_DIR/scene_play.h5
    python data/scene/build_scene_lateonset.py render --block K --num-blocks W --gpu G
    python data/scene/build_scene_lateonset.py merge --num-blocks W --out <noised.h5>
    python data/scene/build_scene_lateonset.py mix --play <play.h5> --noised <noised.h5> --out <mix.h5>
"""
import argparse
import os
import json
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene_noised as B  # noqa: E402

SPEC = dict(B.SPEC)
SPEC.update({
    "version": "scene-noised-v4-lateonset",
    "source_dataset": "scene_play.h5 (scene-play-v0 episodes 0-999)",
    "sigmas": [0.3, 0.6],           # episode e gets sigmas[e % 2]; no sigma-0 duplicates of play
    "noise_seed": 2000,
    "onset": {"low_frac": 0.05, "high_frac": 0.9},
})


def prepare(prep_path, play_h5, spec=SPEC):
    with h5py.File(play_h5, "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        d = dict(actions=f["action"][:], observations=f["observation"][:], qpos_all=f["qpos"][:], qvel_all=f["qvel"][:], btn_all=f["button_states"][:])
    n_ep = len(ln)
    sig = np.array([spec["sigmas"][e % len(spec["sigmas"])] for e in range(n_ep)], np.float32)
    onset = np.array([int(B.episode_rng(e, spec).integers(int(spec["onset"]["low_frac"] * ln[e]), int(spec["onset"]["high_frac"] * ln[e]) + 1)) for e in range(n_ep)], np.int64)
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prep_path, ep_start=off.astype(np.int64), ep_len=ln.astype(np.int64), ep_sigma=sig, ep_onset=onset,
             spec_digest=np.array(B.spec_digest(spec)), **d)
    print(f"[prepare] {n_ep} episodes, {int(ln.sum())} frames, sigma split {[int((sig == s).sum()) for s in spec['sigmas']]}, "
          f"onset median {int(np.median(onset))} -> {prep_path}", flush=True)


def render_noised(env, d, e, spec=SPEC):
    """Replay the recorded actions exactly until the episode's onset frame, then expert action +
    sigma*N(0,1); the frame is captured before the action is applied, as in build_scene_noised."""
    uw = env.unwrapped
    s, n = int(d["ep_start"][e]), int(d["ep_len"][e])
    sig, t0 = float(d["ep_sigma"][e]), int(d["ep_onset"][e])
    actions = d["actions"]; adim = actions.shape[1]
    rng = B.episode_rng(e, spec); rng.integers(0, 1)          # the onset was the stream's first draw
    B.reset_sim(uw, d["qpos_all"][s], d["qvel_all"][s], d["btn_all"][s])
    nq, nv = d["qpos_all"].shape[1], d["qvel_all"].shape[1]
    px = np.empty((n, spec["height"], spec["width"], 3), np.uint8)
    QP = np.empty((n, nq), np.float32); QV = np.empty((n, nv), np.float32); BT = np.empty((n, d["btn_all"].shape[1]), np.int64)
    AC = np.empty((n, adim), np.float32); OB = []
    ob = np.asarray(uw.compute_observation(), np.float32)
    for t in range(n):
        px[t] = B.grab(uw, spec); QP[t] = uw.data.qpos[:nq]; QV[t] = uw.data.qvel[:nv]; BT[t] = np.asarray(uw._cur_button_states, np.int64); OB.append(ob)
        a = actions[s + t].astype(np.float32)
        if t >= t0:
            a = np.clip(a + sig * rng.standard_normal(adim).astype(np.float32), -1.0, 1.0)
        AC[t] = a
        ob = np.asarray(env.step(a)[0], np.float32)
    return {"pixels": px, "action": AC, "observation": np.asarray(OB, np.float32), "qpos": QP, "qvel": QV, "button_states": BT,
            "step_idx": np.arange(n, dtype=np.int64)}


B.RENDERERS = {"play": B.render_play, "noised": render_noised}
B.GOLDEN, B.GOLDEN_PROBES = {}, []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "render", "merge", "mix", "verify", "smoke", "spec"])
    ap.add_argument("--block", type=int, default=None); ap.add_argument("--num-blocks", type=int, default=1); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prep", type=Path, default=Path(os.environ["SCENE_STAGE_DIR"]) / "_lateonset/prep.npz"); ap.add_argument("--block-dir", type=Path, default=Path(os.environ["SCENE_STAGE_DIR"]) / "_lateonset")
    ap.add_argument("--play", type=Path, default=Path(os.environ["SCENE_STAGE_DIR"]) / "scene_play.h5"); ap.add_argument("--noised", type=Path, default=None); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--episode", type=int, default=1, help="smoke: episode to render")
    a = ap.parse_args()
    if a.cmd == "spec":
        print(json.dumps({"spec": SPEC, "spec_digest": B.spec_digest(SPEC)}, indent=1))
    elif a.cmd == "prepare":
        prepare(a.prep, a.play, SPEC)
    elif a.cmd == "smoke":
        d = np.load(a.prep); env = B.make_env(SPEC, a.gpu); t = time.time(); ep = render_noised(env, d, a.episode, SPEC)
        x = ep["qpos"][:, 14:17]; t0 = int(d["ep_onset"][a.episode]); path = np.linalg.norm(x[1:] - x[:-1], axis=1)
        print(f"episode {a.episode}: {len(x)} frames in {time.time() - t:.0f}s; sigma {float(d['ep_sigma'][a.episode])}, onset {t0}; "
              f"cube path before onset {100 * path[:t0].sum():.0f} cm, after {100 * path[t0:].sum():.0f} cm")
    elif a.cmd == "render":
        B.render_block("noised", a.block, a.num_blocks, a.gpu, a.prep, a.block_dir / f"noised_block_{a.block}.h5", spec=SPEC)
    elif a.cmd == "merge":
        B.merge("noised", a.block_dir, a.num_blocks, a.out, spec=SPEC)
    elif a.cmd == "mix":
        B.mix(a.play, a.noised, a.out, spec=SPEC)
    elif a.cmd == "verify":
        raise SystemExit(0 if B.verify(a.out, deep=False, spec=SPEC) else 1)


if __name__ == "__main__":
    main()
