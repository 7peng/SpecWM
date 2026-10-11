"""calvin_abc_d_noised_lateonset.h5: the CALVIN segments re-stepped with LATE-ONSET action noise, the
cube protocol (data/cube/build_cube_lateonset.py) on CALVIN. Each 1000-frame segment is reset to its
recorded start state, replays the recorded rel_actions EXACTLY until a per-segment onset frame t0,
drawn uniformly from [onset.low_frac, onset.high_frac] of the segment, and adds sigma*N(0,1) on the
six motion dims from t0 on (the gripper as recorded). Segment s gets sigmas[s % 2] and its own stream
default_rng(noise_seed + s); t0 is the stream's first draw. Segment table, env factory, merge and mix
are build_calvin_noised unchanged.

    python data/calvin/build_calvin_lateonset.py prepare
    $CALVIN_VENV/bin/python data/calvin/build_calvin_lateonset.py render --block K --num-blocks W --gpu G
    python data/calvin/build_calvin_lateonset.py merge --num-blocks W --out <noised.h5>
    python data/calvin/build_calvin_lateonset.py mix --play <play.h5> --noised <noised.h5> --out <mix.h5>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_calvin_noised as C  # noqa: E402

SPEC = dict(C.SPEC)
SPEC.update({
    "version": "calvin-noised-v2-lateonset",
    "sigmas": [0.3, 0.6],           # segment s gets sigmas[s % 2]; no sigma-0 replays
    "noise_seed": 2000,
    "onset": {"low_frac": 0.05, "high_frac": 0.9},
})


def segment_rng(s, spec=SPEC):
    return np.random.default_rng(spec["noise_seed"] + int(s))


def prepare(pack, raw_root, prep_path, spec=SPEC):
    C.prepare(pack, raw_root, prep_path, spec)
    d = dict(np.load(prep_path))
    S = len(d["seg_len"])
    d["seg_sigma"] = np.array([spec["sigmas"][s % len(spec["sigmas"])] for s in range(S)], np.float32)
    d["seg_onset"] = np.array([int(segment_rng(s, spec).integers(int(spec["onset"]["low_frac"] * d["seg_len"][s]), int(spec["onset"]["high_frac"] * d["seg_len"][s]) + 1)) for s in range(S)], np.int64)
    np.savez(prep_path, **d)
    print(f"[prepare] sigma split {[int((d['seg_sigma'] == s).sum()) for s in spec['sigmas']]}, onset median {int(np.median(d['seg_onset']))} -> {prep_path}", flush=True)


def render_block(block, num_blocks, gpu, prep_path, out_dir, spec=SPEC):
    """build_calvin_noised.render_block with the onset gate: exact replay until t0, noise after."""
    for v in ("CUDA_VISIBLE_DEVICES", "EGL_DEVICE", "EGL_VISIBLE_DEVICES"):
        os.environ[v] = str(gpu)
    os.environ.setdefault("CALVIN_USE_EGL", "1" if spec["use_egl"] else "0")
    from make_calvin_suite import build_env, CALVIN_SRC
    d = np.load(prep_path)
    idx = C.blocks_for(len(d["seg_len"]), num_blocks)[block]
    n_tot = int(d["seg_len"][idx].sum())
    out = Path(out_dir) / f"noised_block_{block}"; out.mkdir(parents=True, exist_ok=True)
    px = np.lib.format.open_memmap(out / "pixels.npy", "w+", np.uint8, (n_tot, spec["H"], spec["W"], 3))
    AC = np.empty((n_tot, 7), np.float32); RO = np.empty((n_tot, 15), np.float32); SO = np.empty((n_tot, 24), np.float32)
    SI = np.empty((n_tot,), np.int64); ST = np.empty((n_tot,), np.int64)
    envs, letters = {}, "ABCD"
    digests, ptr, t_start = [], 0, time.time()
    for i, s in enumerate(idx):
        st, n, sid, sig, t0 = int(d["seg_start"][s]), int(d["seg_len"][s]), int(d["scene_id"][d["seg_start"][s]]), float(d["seg_sigma"][s]), int(d["seg_onset"][s])
        if sid not in envs:
            envs[sid] = build_env(letters[sid], str(CALVIN_SRC / "data"))
        env = envs[sid]
        rng = segment_rng(s, spec); rng.integers(0, 1)        # the onset was the stream's first draw
        env.reset(robot_obs=d["seg_robot_obs0"][s], scene_obs=d["seg_scene_obs0"][s])
        obs = env.get_obs()
        a_seg = np.empty((n, 7), np.float32)
        for t in range(n):
            px[ptr + t] = obs["rgb_obs"]["rgb_static"]
            RO[ptr + t] = obs["robot_obs"]; SO[ptr + t] = obs["scene_obs"]
            a = np.array(d["actions"][st + t], np.float32)
            if t >= t0:
                a[spec["noised_dims"]] = np.clip(a[spec["noised_dims"]] + sig * rng.standard_normal(len(spec["noised_dims"])).astype(np.float32), -1.0, 1.0)
            a_seg[t] = a
            obs, _, _, _ = env.step({"action": a.astype(np.float64), "type": "cartesian_rel"})
        AC[ptr:ptr + n] = a_seg; SI[ptr:ptr + n] = sid; ST[ptr:ptr + n] = np.arange(n)
        digests.append(C.state_digest(action=a_seg, robot_obs=RO[ptr:ptr + n], scene_obs=SO[ptr:ptr + n]))
        ptr += n
        if i % 5 == 0:
            print(f"[noised block {block}] {i + 1}/{len(idx)} segs, {ptr} frames ({time.time() - t_start:.0f}s, {ptr / max(time.time() - t_start, 1e-9):.0f} fps)", flush=True)
    px.flush(); del px
    for name, arr in (("action", AC), ("robot_obs", RO), ("scene_obs_raw", SO), ("scene_id", SI), ("step_idx", ST)):
        np.save(out / f"{name}.npy", arr)
    meta = {"block": block, "num_blocks": num_blocks, "segments": [int(s) for s in idx],
            "seg_len": [int(d["seg_len"][s]) for s in idx], "state_digests": digests, "spec_digest": C.spec_digest(spec)}
    (out / "meta.json").write_text(json.dumps(meta))
    for e in envs.values():
        e.close()
    print(f"[noised block {block}/{num_blocks}] {len(idx)} segs {ptr} frames -> {out} ({time.time() - t_start:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "render", "merge", "mix", "spec"])
    ap.add_argument("--pack", default=os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "calvin_abc_d.h5"))
    ap.add_argument("--raw", default=os.path.join(os.environ.get("CALVIN_ROOT", os.path.expanduser("~/.stable-wm/calvin")), "task_ABC_D/training"))
    ap.add_argument("--prep", type=Path, default=Path(os.environ["CALVIN_STAGE_DIR"]) / "_lateonset" / "prep.npz")
    ap.add_argument("--block-dir", type=Path, default=Path(os.environ["CALVIN_STAGE_DIR"]) / "_lateonset")
    ap.add_argument("--block", type=int, default=None); ap.add_argument("--num-blocks", type=int, default=1); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--play", type=Path, default=None); ap.add_argument("--noised", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "spec":
        print(json.dumps({"spec": SPEC, "spec_digest": C.spec_digest(SPEC)}, indent=1))
    elif a.cmd == "prepare":
        prepare(a.pack, a.raw, a.prep, SPEC)
    elif a.cmd == "render":
        render_block(a.block, a.num_blocks, a.gpu, a.prep, a.block_dir, SPEC)
    elif a.cmd == "merge":
        C.merge(a.block_dir, a.num_blocks, a.out, spec=SPEC)
    elif a.cmd == "mix":
        C.mix(a.play, a.noised, a.out, spec=SPEC)


if __name__ == "__main__":
    main()
