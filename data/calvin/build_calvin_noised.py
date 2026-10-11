"""calvin_abc_d_noised: the scene/cube noised recipe on CALVIN ABC (the failure data).

Every training frame of the pack (data/calvin/build_calvin_h5.py) is re-visited in the CALVIN
simulator: the recorded episodes are cut into SEGMENTS of `segment_len` frames (tails >= `min_tail`
kept), each segment is reset to its recorded start state (robot_obs + raw scene_obs from the
episode npz) and then STEPPED under the recorded relative action plus Gaussian noise,
a_t[:6] = clip(rel_action_t[:6] + sigma * z, -1, 1), sigma = sigmas[segment % 3], z from
default_rng(noise_seed + segment). The gripper command (dim 6, +-1) is left as recorded: noising a
binary command flips it, which is chattering, not the error the recipe means. Frames are captured
BEFORE the action is applied, so frame[t] is s_t and action[t] the action taken at s_t. Output is
the pack's own schema (pixels 200x200 static camera, action, robot_obs, canonical scene_obs,
scene_id, ep_* tables; 1-frame chunks, uncompressed) so every CALVIN recipe reads it unchanged,
one segment = one episode.

Fidelity, measured 2026-09-05 (calvin_replay_probe.py): the reset render matches the stored frame
to 0.2 grey levels; an open-loop replay of 300 recorded actions stays within ~2 grey levels and
5 mm end-effector drift; ~120 fps per worker.

Two venvs: `prepare`, `merge`, `mix` run in the training venv (h5py); `render` runs in
$CALVIN_VENV (the simulator's venv, no h5py) and writes .npy blocks. Pipeline (run_build_calvin_noised.sh):
    build_calvin_noised.py prepare                          # pack + episode npz -> prep.npz
    $CALVIN_VENV/bin/python build_calvin_noised.py render --block K --num-blocks W --gpu G
    build_calvin_noised.py merge --num-blocks W             # -> calvin_abc_d_noised.h5 (+ manifest)
    build_calvin_noised.py mix                              # play ++ noised -> calvin_abc_d_mix_noised.h5
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roster import canonicalize, CANONICAL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

SPEC = {
    "version": "calvin-noised-v1",
    "source_pack": "calvin_abc_d",           # the training-split pack the segments come from
    "segment_len": 1000,
    "min_tail": 171,                         # a 35-frame window at frameskip 5 spans 171 frames
    "sigmas": [0.0, 0.3, 0.6],               # segment s gets sigmas[s % 3]
    "noise_seed": 1000,
    "noised_dims": [0, 1, 2, 3, 4, 5],       # gripper (6) as recorded
    "camera": "static",
    "use_egl": True,
    "H": 200, "W": 200,
}
COLS = ["pixels", "action", "robot_obs", "scene_obs", "scene_id", "step_idx"]


def spec_digest(spec=SPEC):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_digest(**arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def blocks_for(n, num_blocks):
    return [b for b in np.array_split(np.arange(n), num_blocks) if len(b)]


# ---------------------------------------------------------------------------------------------
def prepare(pack, raw_root, prep_path, spec=SPEC):
    """Segment table + the small columns, read from the pack once; raw start states from the npz."""
    import h5py
    with h5py.File(pack, "r") as f:
        ep_len, ep_off = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
        actions = f["action"][:].astype(np.float32)
        scene_id = f["scene_id"][:, 0].astype(np.int64)
        N = f["pixels"].shape[0]
    seg_start, seg_len, seg_ep = [], [], []
    for e, (o, L) in enumerate(zip(ep_off, ep_len)):
        for k in range(0, int(L), spec["segment_len"]):
            n = min(spec["segment_len"], int(L) - k)
            if n < spec["segment_len"] and n < spec["min_tail"]:
                continue
            seg_start.append(o + k); seg_len.append(n); seg_ep.append(e)
    seg_start, seg_len, seg_ep = (np.array(x, np.int64) for x in (seg_start, seg_len, seg_ep))
    S = len(seg_start)
    sig = np.array([spec["sigmas"][s % len(spec["sigmas"])] for s in range(S)], np.float32)
    # raw (simulator-order) start states: the pack's scene_obs is canonicalized, the env wants its own order
    raw = Path(raw_root)
    so0 = np.empty((S, 24), np.float64); ro0 = np.empty((S, 15), np.float64)
    for i, s in enumerate(seg_start):                    # pack frame index == raw episode index (training split, in order)
        d = np.load(raw / f"episode_{int(s):07d}.npz")
        so0[i], ro0[i] = d["scene_obs"], d["robot_obs"]
        if i % 200 == 0:
            print(f"[prepare] start states {i}/{S}", flush=True)
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prep_path, actions=actions, scene_id=scene_id, seg_start=seg_start, seg_len=seg_len, seg_ep=seg_ep,
             seg_sigma=sig, seg_scene_obs0=so0, seg_robot_obs0=ro0, spec_digest=np.array(spec_digest(spec)))
    print(f"[prepare] {S} segments ({int((seg_len == spec['segment_len']).sum())} full + {int((seg_len < spec['segment_len']).sum())} tails), "
          f"{int(seg_len.sum())} of {N} frames, sigma split {[int((sig == s).sum()) for s in spec['sigmas']]} -> {prep_path}", flush=True)


# ---------------------------------------------------------------------------------------------
def render_block(block, num_blocks, gpu, prep_path, out_dir, spec=SPEC):
    """One contiguous block of segments -> out_dir/noised_block_K/{pixels,...}.npy (simulator venv)."""
    for v in ("CUDA_VISIBLE_DEVICES", "EGL_DEVICE", "EGL_VISIBLE_DEVICES"):
        os.environ[v] = str(gpu)
    os.environ.setdefault("CALVIN_USE_EGL", "1" if spec["use_egl"] else "0")
    from make_calvin_suite import build_env, CALVIN_SRC          # the suite's env factory: EGL, static+gripper cams
    d = np.load(prep_path)
    idx = blocks_for(len(d["seg_len"]), num_blocks)[block]
    n_tot = int(d["seg_len"][idx].sum())
    out = Path(out_dir) / f"noised_block_{block}"; out.mkdir(parents=True, exist_ok=True)
    px = np.lib.format.open_memmap(out / "pixels.npy", "w+", np.uint8, (n_tot, spec["H"], spec["W"], 3))
    AC = np.empty((n_tot, 7), np.float32); RO = np.empty((n_tot, 15), np.float32); SO = np.empty((n_tot, 24), np.float32)
    SI = np.empty((n_tot,), np.int64); ST = np.empty((n_tot,), np.int64)
    envs, letters = {}, "ABCD"
    digests, ptr, t0 = [], 0, time.time()
    for i, s in enumerate(idx):
        st, n, sid, sig = int(d["seg_start"][s]), int(d["seg_len"][s]), int(d["scene_id"][d["seg_start"][s]]), float(d["seg_sigma"][s])
        if sid not in envs:
            envs[sid] = build_env(letters[sid], str(CALVIN_SRC / "data"))
        env = envs[sid]
        rng = np.random.default_rng(spec["noise_seed"] + int(s))
        env.reset(robot_obs=d["seg_robot_obs0"][s], scene_obs=d["seg_scene_obs0"][s])
        obs = env.get_obs()
        a_seg = np.empty((n, 7), np.float32)
        for t in range(n):
            px[ptr + t] = obs["rgb_obs"]["rgb_static"]
            RO[ptr + t] = obs["robot_obs"]; SO[ptr + t] = obs["scene_obs"]
            a = np.array(d["actions"][st + t], np.float32)
            a[spec["noised_dims"]] = np.clip(a[spec["noised_dims"]] + sig * rng.standard_normal(len(spec["noised_dims"])).astype(np.float32), -1.0, 1.0)
            a_seg[t] = a
            obs, _, _, _ = env.step({"action": a.astype(np.float64), "type": "cartesian_rel"})
        AC[ptr:ptr + n] = a_seg; SI[ptr:ptr + n] = sid; ST[ptr:ptr + n] = np.arange(n)
        digests.append(state_digest(action=a_seg, robot_obs=RO[ptr:ptr + n], scene_obs=SO[ptr:ptr + n]))
        ptr += n
        if i % 5 == 0:
            print(f"[noised block {block}] {i + 1}/{len(idx)} segs, {ptr} frames ({time.time() - t0:.0f}s, {ptr / max(time.time() - t0, 1e-9):.0f} fps)", flush=True)
    px.flush(); del px
    for name, arr in (("action", AC), ("robot_obs", RO), ("scene_obs_raw", SO), ("scene_id", SI), ("step_idx", ST)):
        np.save(out / f"{name}.npy", arr)
    meta = {"block": block, "num_blocks": num_blocks, "segments": [int(s) for s in idx],
            "seg_len": [int(d["seg_len"][s]) for s in idx], "state_digests": digests, "spec_digest": spec_digest(spec)}
    (out / "meta.json").write_text(json.dumps(meta))
    for e in envs.values():
        e.close()
    print(f"[noised block {block}/{num_blocks}] {len(idx)} segs {ptr} frames -> {out} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------
def _write_ep_tables(f, ep_len):
    ep_len = np.asarray(ep_len, np.int32)
    f.create_dataset("ep_len", data=ep_len)
    f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64))
    f.create_dataset("ep_idx", data=np.repeat(np.arange(len(ep_len), dtype=np.int32), ep_len))


def merge(block_dir, num_blocks, output, spec=SPEC, slab=2048):
    import h5py
    import roster as R
    parts = [Path(block_dir) / f"noised_block_{k}" for k in range(num_blocks)]
    missing = [p.name for p in parts if not (p / "meta.json").exists()]
    if missing:
        raise SystemExit(f"MISSING BLOCKS: {missing} -- refusing to merge a partial dataset")
    metas = [json.loads((p / "meta.json").read_text()) for p in parts]
    ep_len = np.concatenate([m["seg_len"] for m in metas]).astype(np.int32)
    digests = sum((m["state_digests"] for m in metas), [])
    N = int(ep_len.sum()); H, W = spec["H"], spec["W"]
    t0 = time.time()
    with h5py.File(str(output) + ".partial", "w") as f:
        dpx = f.create_dataset("pixels", (N, H, W, 3), np.uint8, chunks=(1, H, W, 3))
        dc = {"action": f.create_dataset("action", (N, 7), np.float32), "robot_obs": f.create_dataset("robot_obs", (N, 15), np.float32),
              "scene_obs": f.create_dataset("scene_obs", (N, 24), np.float32), "scene_id": f.create_dataset("scene_id", (N, 1), np.int64),
              "step_idx": f.create_dataset("step_idx", (N,), np.int64)}
        at = 0
        for p in parts:
            px = np.load(p / "pixels.npy", mmap_mode="r"); n = px.shape[0]
            for c0 in range(0, n, slab):
                dpx[at + c0:at + min(c0 + slab, n)] = px[c0:c0 + slab]
            sid = np.load(p / "scene_id.npy")
            dc["scene_obs"][at:at + n] = R.canonicalize(np.load(p / "scene_obs_raw.npy"), sid)
            dc["scene_id"][at:at + n] = sid[:, None]
            for k in ("action", "robot_obs", "step_idx"):
                dc[k][at:at + n] = np.load(p / f"{k}.npy")
            at += n
            print(f"[merge] {p.name} -> {at:,}/{N:,} frames ({time.time() - t0:.0f}s)", flush=True)
        assert at == N
        _write_ep_tables(f, ep_len)
        f["scene_obs"].attrs["roster"] = R.roster_attr()
        f.create_dataset("episode_state_digest", data=np.array(digests, dtype="S64"))
    os.replace(str(output) + ".partial", output)
    write_manifest(output, "noised", ep_len, digests, spec)


def _read_slab(args):
    path, c0, slab = args
    import h5py
    with h5py.File(path, "r") as h:
        return h["pixels"][c0:c0 + slab]


def mix(play, noised, output, spec=SPEC, slab=2048, workers=8):
    """play ++ noised [++ more] as ONE file, the pack schema; episode tables rebuilt for the concatenation.
    `noised` is one path or a list of paths (play ++ noised ++ transplant is the three-way mix)."""
    import h5py
    from contextlib import ExitStack
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    others = [noised] if isinstance(noised, (str, os.PathLike)) else list(noised)
    with ExitStack() as st:
        a = st.enter_context(h5py.File(play, "r"))
        srcs = [a] + [st.enter_context(h5py.File(p, "r")) for p in others]
        N = sum(s["pixels"].shape[0] for s in srcs); H, W = spec["H"], spec["W"]
        ep_len = np.concatenate([s["ep_len"][:] for s in srcs]).astype(np.int32)
        with h5py.File(str(output) + ".partial", "w") as f:
            dpx = f.create_dataset("pixels", (N, H, W, 3), np.uint8, chunks=(1, H, W, 3))
            dc = {k: f.create_dataset(k, (N, *a[k].shape[1:]), a[k].dtype) for k in ("action", "robot_obs", "scene_obs", "scene_id", "step_idx")}
            at = 0
            for path, src in zip([play] + others, srcs):
                n = src["pixels"].shape[0]
                # slab reads run in `workers` processes (one file handle each): a single reader is capped at
                # ~0.45 GB/s on this filesystem, eight readers reach ~2 GB/s; the writer stays single and in order
                with ProcessPoolExecutor(workers) as ex:
                    for k, (c0, buf) in enumerate(zip(range(0, n, slab), ex.map(_read_slab, ((path, c0, slab) for c0 in range(0, n, slab))))):
                        dpx[at + c0:at + c0 + len(buf)] = buf
                        if k % 100 == 0:
                            print(f"[mix] {at + c0:,}/{N:,} frames ({time.time() - t0:.0f}s)", flush=True)
                for k in dc:
                    v = src[k][:]
                    if k == "scene_obs" and not src["scene_obs"].attrs.get("roster"):
                        v = canonicalize(v, src["scene_id"][:].reshape(-1))   # a pack in simulator order
                    dc[k][at:at + n] = v
                at += n
            _write_ep_tables(f, ep_len)
            f["scene_obs"].attrs["roster"] = ",".join(CANONICAL)
    os.replace(str(output) + ".partial", output)
    write_manifest(output, "mix", ep_len, [], spec, file_hash=False)


def write_manifest(output, kind, ep_len, digests, spec, file_hash=False):
    h = hashlib.sha256()
    for d in digests:
        h.update(bytes.fromhex(d))
    man = {"kind": kind, "spec_version": spec["version"], "spec_digest": spec_digest(spec), "spec": spec,
           "episodes": int(len(ep_len)), "frames": int(np.asarray(ep_len).sum()),
           "content_sha256_state": h.hexdigest() if digests else None, "episode_state_digests": digests,
           "file_bytes": Path(output).stat().st_size}
    Path(str(output) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[manifest] {output}: episodes={man['episodes']} frames={man['frames']} bytes={man['file_bytes'] / 1e9:.1f} GB state {man['content_sha256_state']}", flush=True)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "render", "merge", "mix", "spec"])
    ap.add_argument("--pack", default=os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "calvin_abc_d.h5"), help="the training pack (full)")
    ap.add_argument("--raw", default=os.path.join(os.environ.get("CALVIN_ROOT", os.path.expanduser("~/.stable-wm/calvin")), "task_ABC_D/training"))
    ap.add_argument("--prep", type=Path, default=Path(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))) / "datasets/_calvin_build/prep.npz")
    ap.add_argument("--block-dir", type=Path, default=Path(os.environ["CALVIN_STAGE_DIR"]) / "_calvin_build")
    ap.add_argument("--block", type=int, default=None); ap.add_argument("--num-blocks", type=int, default=1); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--play", type=Path, default=None, help="mix: the full play pack (default $CALVIN_STAGE_DIR/calvin_abc_d.h5)")
    ap.add_argument("--noised", type=Path, default=None)
    a = ap.parse_args()
    stage = Path(os.environ["CALVIN_STAGE_DIR"])
    if a.cmd == "spec":
        print(json.dumps({"spec": SPEC, "spec_digest": spec_digest()}, indent=1))
    elif a.cmd == "prepare":
        prepare(a.pack, a.raw, a.prep)
    elif a.cmd == "render":
        if a.block is None:
            raise SystemExit("--block is required")
        render_block(a.block, a.num_blocks, a.gpu, a.prep, a.block_dir)
    elif a.cmd == "merge":
        merge(a.block_dir, a.num_blocks, a.out or stage / "calvin_abc_d_noised.h5")
    elif a.cmd == "mix":
        mix(a.play or stage / "calvin_abc_d.h5", a.noised or stage / "calvin_abc_d_noised.h5", a.out or stage / "calvin_abc_d_mix_noised.h5")


if __name__ == "__main__":
    main()
