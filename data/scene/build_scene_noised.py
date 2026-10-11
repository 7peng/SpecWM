"""Bit-reproducible rebuild of the scene datasets: scene_play, scene_play_noised, scene_play_mix_noised.

Everything the output depends on is declared in SPEC below. Given the same SPEC and the same pinned
source archive, two independent clusters produce byte-identical column content -- verified by a
content hash that is computed during the render and committed to a manifest, NOT by trusting the
HDF5 container bytes (those legitimately vary with h5py/HDF5/blosc build).

Three properties had to be engineered in; none came for free (see REPRODUCIBILITY.md):

  1. STATE IS EXACT, PIXELS ARE NOT.  Everything with a seed is pinned, and the state columns come
     out bit-identical on any machine -- the manifest carries a `content_sha256_state` chain that a
     rebuild must reproduce exactly. Rendering has no seed: MuJoCo's 4x MSAA resolves multisamples in
     a schedule-dependent order, so replaying an episode flips +/-1 LSB on a third of its frames.
     SPEC keeps MSAA (v1-identical pixels) and the artifact is distributed as a file, so pixel
     identity comes from the download, checked by `content_sha256_full` + `file_sha256`. `preflight`
     gates a build on state-exactness plus a pixel tolerance. Note it probes EPISODES: re-rendering a
     single frozen state was stable for 250 consecutive renders even under MSAA, so it proves nothing.
     For exactly reproducible pixels at 10-30x the render cost, set offsamples=0 and supersample=4.
  2. NO UNSEEDED GOAL-GHOST.  ManipSpaceEnv.reset() draws cur_task_id from the UNSEEDED global
     np.random, and with mode='task' + visualize_info=True that paints a translucent goal cube whose
     position set_state never touches -- so the draw silently persists for a whole worker process.
     v1 therefore carried a ghost that changed between shards. SPEC pins task_id AND turns
     visualize_info off, so the draw cannot reach the pixels by any path.
  3. ORDER INDEPENDENCE.  Workers take CONTIGUOUS blocks of the canonical episode order and the merge
     concatenates blocks in index order, so the output row order is identical for any worker count.
     reset_sim() additionally scrubs the sim state that set_state leaves behind -- solver warm start,
     clock, and above all cfrc_ext, whose staleness leaked the previous episode's contact forces into
     the next episode's first observation. `selftest` checks an episode against one rendered by a
     process that has rendered nothing else; that check is what caught the leak.

Pipeline (see run_build_scene_noised.sh for the driver):

    build_scene_noised.py preflight                     # 10s: does THIS machine render the goldens?
    build_scene_noised.py prepare                       # source npz -> prep.npz (episode table)
    build_scene_noised.py render --kind noised --block K --num-blocks W --gpu G
    build_scene_noised.py merge   --kind noised --num-blocks W
    build_scene_noised.py mix --play A.h5 --noised B.h5 --out mix.h5
    build_scene_noised.py verify --out mix.h5           # manifest check (add --deep to re-read pixels)
    build_scene_noised.py selftest                      # order/isolation independence of the state
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

# ---------------------------------------------------------------------------------------------
# SPEC -- the complete declaration of the artifact. Changing ANY field changes the output; bump
# `version` when you do, so a stale manifest can never be mistaken for a matching one.
# ---------------------------------------------------------------------------------------------
SPEC = {
    "version": "scene-noised-v3",
    # --- source ---------------------------------------------------------------------------
    "source_dataset": "scene-play-v0",
    "source_sha256": "66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e748316e507e89f4a9afb87c341",
    "min_episode_len": 10,          # drops compact-format padding rows
    "max_episodes": None,           # None = every episode passing min_episode_len (1000)
    # --- noised dynamics ------------------------------------------------------------------
    # Episode e is STEPPED from its recorded start state with a_t = clip(expert_a_t + sigma*z, -1, 1),
    # so the trajectory diverges from the expert and the action stops being pi(state).
    "sigmas": [0.0, 0.3, 0.6],      # episode e gets sigmas[e % 3]
    "noise_seed": 1000,             # per-episode stream = default_rng(noise_seed + e), as in v1
    # --- rendering ------------------------------------------------------------------------
    "width": 224,
    "height": 224,
    "camera": "front_pixels",
    # MuJoCo's default 4x MSAA, exactly as v1 rendered -- pixels come out identical to the shipped
    # files (measured 0/150528 differing subpixels). It is the one stage that is NOT bit-reproducible:
    # the GPU resolves the multisample buffer in a schedule-dependent order and float addition is not
    # associative, so replaying an episode differs on up to a third of its frames by +/-1 LSB. That is
    # fine here because the ARTIFACT is distributed (HuggingFace), not re-rendered -- hash identity
    # comes from downloading the file. Everything that can be pinned is: see the two digest chains in
    # write_manifest. To trade render time for exact pixel reproducibility instead, set offsamples=0
    # and supersample=2 or 4 (deterministic in every episode-level trial; 4.9 / 12.7 GPU-hours).
    "offsamples": 4,
    "supersample": 1,
    # No goal-ghost. v1 rendered OGBench's translucent goal-cube overlay into every frame at a
    # position drawn from the unseeded global np.random -- see REPRODUCIBILITY.md. It is a debug
    # visualisation with no role in this data, and it is painted in the manipulated cube's own colour,
    # so it put a second cube-coloured object at a fixed image position in every frame. task_id still
    # has to be pinned because reset() would otherwise draw it, but with visualize_info False it no
    # longer reaches the pixels.
    "visualize_info": False,
    "task_id": 1,
    # --- container ------------------------------------------------------------------------
    "columns": ["pixels", "action", "observation", "qpos", "qvel", "button_states", "step_idx"],
    "pixel_chunk": 50,             # divides the 1000-frame episode -> blocks are chunk-aligned,
                                   # which lets merge/mix copy compressed chunks verbatim
    "blosc": {"cname": "lz4", "clevel": 5, "shuffle": "SHUFFLE"},
}

# Reference fingerprints for three probe episodes (first PREFLIGHT_FRAMES frames), emitted by
# `preflight --emit` on an NVIDIA L40S / driver 595.71.05 under MUJOCO_GL=egl. `state` is an exact
# hash of the simulation columns; `pixel_mean` is a tolerance-checked summary, because MSAA makes an
# exact pixel hash unreproducible by design. A machine failing either cannot produce this artifact.
GOLDEN = {
    "play_ep0": {"state": "f3e6a41882f6240045a15da5d3b5a34618c4f7112a5f2d65fd452cd11d33b014", "pixel_mean": 91.181940458},
    "play_ep500": {"state": "f45e495502ff7a86027a583317da8313c6f8eca10de7a85820e1f7b338c712c1", "pixel_mean": 85.243574617},
    "noised_ep1": {"state": "000b0f2eb226fb814bd67b5ce76afa20ab30d6d415f88d1e6d76b83f880133e1", "pixel_mean": 86.863671654},
}

# The shipped v1 files (scene-noised-v1, collected by scripts/collect_scene_noised.py before this
# rewrite) carried the unseeded ghost. Recovered for the record by re-rendering stored states under
# each candidate task_id: scene_play.h5 was task 1 throughout; scene_play_noised.h5's 36 shards drew
# the sequence below. Kept only so v1-trained checkpoints remain interpretable -- do not rebuild v1.
V1_GHOST_TASK_IDS = [1, 4, 2, 1, 3, 2, 3, 3, 3, 4, 4, 4, 2, 4, 2, 4, 1, 4, 3, 3,
                     3, 4, 1, 3, 4, 3, 3, 1, 4, 3, 1, 4, 4, 4, 2, 2]

KINDS = ("play", "noised")


# ---------------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------------
def spec_digest(spec=SPEC):
    """Hash of the SPEC itself; any field change gives a different artifact identity."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def episode_digest(ep, columns):
    """Content hash of ONE episode: column name + dtype + shape + raw bytes, in declared order.

    Hashing per episode (rather than per file) is what makes a cross-cluster mismatch debuggable --
    the digest lists tell you exactly which episode diverged instead of just that the file differs.
    """
    h = hashlib.sha256()
    for col in columns:
        a = np.ascontiguousarray(ep[col])
        h.update(col.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_columns(spec=SPEC):
    """The columns a rebuild must reproduce bit-exactly: everything except the rendered pixels."""
    return [c for c in spec["columns"] if c != "pixels"]


def chain_digest(digests):
    """Fold ordered per-episode digests into one content hash for the dataset."""
    h = hashlib.sha256()
    for d in digests:
        h.update(bytes.fromhex(d))
    return h.hexdigest()


def env_fingerprint():
    """Recorded in the manifest so a mismatch can be attributed instead of guessed at."""
    fp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "hdf5plugin": hdf5plugin.version,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
    }
    try:
        import mujoco

        fp["mujoco"] = mujoco.__version__
    except Exception:
        pass
    try:
        import ogbench

        fp["ogbench"] = getattr(ogbench, "__version__", "unknown")
    except Exception:
        pass
    try:
        fp["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
    except Exception:
        fp["gpu"] = "unknown"
    return fp


# ---------------------------------------------------------------------------------------------
# source + episode table
# ---------------------------------------------------------------------------------------------
def prepare(prep_path, spec=SPEC, verify_source=True):
    """Read the pinned OGBench archive once and freeze the episode table workers will replay.

    Workers never touch ogbench: they load prep.npz, so the episode->row mapping cannot drift
    between them and the source archive is read (and hash-checked) exactly once.
    """
    import ogbench
    from ogbench.utils import DEFAULT_DATASET_DIR

    src = Path(DEFAULT_DATASET_DIR).expanduser() / f"{spec['source_dataset']}.npz"
    ds, _ = ogbench.make_env_and_datasets(
        spec["source_dataset"], dataset_only=True, add_info=True, compact_dataset=True)
    if verify_source:
        if not src.exists():
            raise SystemExit(f"source archive not found at {src}")
        got = sha256_file(src)
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
    qpos, qvel, btn = (np.asarray(ds[k]) for k in ("qpos", "qvel", "button_states"))

    prep_path.parent.mkdir(parents=True, exist_ok=True)
    # Stored in the source's NATIVE dtypes: these arrays are fed straight to set_state, so any cast
    # here would silently perturb the states the pixels are rendered from.
    np.savez(
        prep_path,
        actions=ds["actions"], observations=ds["observations"],
        qpos_all=qpos, qvel_all=qvel, btn_all=btn.astype(np.int64),
        ep_start=ep_start, ep_len=ep_len, ep_sigma=sig,
        spec_digest=np.array(spec_digest(spec)),
    )
    print(f"[prepare] {len(slices)} episodes, {int(ep_len.sum())} frames, "
          f"sigma split {[int((sig == s).sum()) for s in spec['sigmas']]} -> {prep_path}", flush=True)


def blocks_for(n_eps, num_blocks):
    """Canonical episode order 0..n-1 cut into CONTIGUOUS blocks.

    Contiguous (not strided) is the whole point: merging blocks in index order reproduces the
    canonical order for ANY num_blocks, so a 4-worker run and a 36-worker run agree byte for byte.
    """
    return [b for b in np.array_split(np.arange(n_eps), num_blocks) if len(b)]


# ---------------------------------------------------------------------------------------------
# environment + rollout
# ---------------------------------------------------------------------------------------------
def make_env(spec=SPEC, gpu=0):
    """SceneEnv pinned to the SPEC's render configuration (at the supersampled resolution)."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

    f = spec["supersample"]
    env = SceneEnv(ob_type="states", width=spec["width"] * f, height=spec["height"] * f,
                   visualize_info=spec["visualize_info"])
    env.reset(options={"task_id": spec["task_id"]})     # pins the otherwise-unseeded task draw
    uw = env.unwrapped
    uw.model.vis.quality.offsamples = spec["offsamples"]
    if uw._renderer is not None:                        # MjrContext caches offsamples at creation
        uw._renderer.close()
        uw._renderer = None
    return env


def box_downsample(px, f):
    """Integer box filter (f*H, f*W, 3) -> (H, W, 3), round-half-up. Fixed summation order in
    uint32, so the result is identical on any machine regardless of SIMD width or thread count."""
    if f == 1:
        return px
    h, w, c = px.shape
    acc = px.reshape(h // f, f, w // f, f, c).astype(np.uint32).sum(axis=(1, 3))
    return ((acc + (f * f) // 2) // (f * f)).astype(np.uint8)


def grab(uw, spec=SPEC):
    """One SPEC-conformant frame: render supersampled, box-filter to the stored resolution."""
    return box_downsample(np.asarray(uw.render(camera=spec["camera"]), np.uint8), spec["supersample"])


def reset_sim(uw, qpos, qvel, btn):
    """Put the sim in a state that depends on nothing but its arguments.

    set_state restores qpos/qvel only; the constraint solver's warm start and the sim clock survive
    it, which would make an episode a function of whatever the worker rendered before it.
    """
    import mujoco

    uw.data.qacc_warmstart[:] = 0.0
    uw.data.time = 0.0
    uw.set_state(qpos, qvel, button_state_0=int(btn[0]), button_state_1=int(btn[1]))
    # set_state ends at mj_forward, which does NOT compute cfrc_ext -- only mj_rnePostConstraint
    # does, and that runs inside step(). Without this call the first frame's observation carries the
    # PREVIOUS episode's contact forces through proprio/gripper_contact, which made the episode
    # depend on what the worker rendered before it (measured: frame 0 of an episode following a
    # grasp differed by up to 0.98). Recomputing here makes the observation a function of the state.
    mujoco.mj_rnePostConstraint(uw.model, uw.data)
    uw._prev_button_states = np.asarray(uw._cur_button_states).copy()


def episode_rng(ep_index, spec=SPEC):
    """Index-addressable noise stream per episode, identical to v1's `default_rng(1000 + e)`.

    Keyed on the episode index, not on worker or arrival order, so the realization is a property of
    the episode alone. NumPy's stream-compatibility policy makes PCG64-from-int reproducible across
    versions, so this yields the same actions v1 drew.
    """
    return np.random.default_rng(spec["noise_seed"] + int(ep_index))


def render_play(env, d, e, spec=SPEC):
    """Expert episode: TELEPORT to each recorded state and render. No stepping, no noise."""
    uw = env.unwrapped
    s, n = int(d["ep_start"][e]), int(d["ep_len"][e])
    qpos, qvel = d["qpos_all"][s:s + n], d["qvel_all"][s:s + n]
    btn = d["btn_all"][s:s + n]
    px = np.empty((n, spec["height"], spec["width"], 3), np.uint8)
    for t in range(n):
        reset_sim(uw, qpos[t], qvel[t], btn[t])
        px[t] = grab(uw, spec)
    return {"pixels": px,
            "action": d["actions"][s:s + n].astype(np.float32),
            "observation": d["observations"][s:s + n].astype(np.float32),
            "qpos": qpos.astype(np.float32), "qvel": qvel.astype(np.float32),
            "button_states": btn.astype(np.int64),
            "step_idx": np.arange(n, dtype=np.int64)}


def render_noised(env, d, e, spec=SPEC):
    """Noised episode: STEP from the recorded start state under expert action + sigma*N(0,1).

    The frame is captured BEFORE the action is applied, so frame[t] is s_t and action[t] is the
    action taken AT s_t. Rendering after the step would shift the pair by one and leak the action
    into qvel, letting the state trivially predict the action.
    """
    uw = env.unwrapped
    s, n = int(d["ep_start"][e]), int(d["ep_len"][e])
    sig = float(d["ep_sigma"][e])
    actions = d["actions"]
    adim = actions.shape[1]
    rng = episode_rng(e, spec)
    reset_sim(uw, d["qpos_all"][s], d["qvel_all"][s], d["btn_all"][s])

    nq, nv = d["qpos_all"].shape[1], d["qvel_all"].shape[1]
    px = np.empty((n, spec["height"], spec["width"], 3), np.uint8)
    QP = np.empty((n, nq), np.float32)
    QV = np.empty((n, nv), np.float32)
    BT = np.empty((n, d["btn_all"].shape[1]), np.int64)
    AC = np.empty((n, adim), np.float32)
    OB = []
    ob = np.asarray(uw.compute_observation(), np.float32)
    for t in range(n):
        px[t] = grab(uw, spec)
        QP[t] = uw.data.qpos[:nq]
        QV[t] = uw.data.qvel[:nv]
        BT[t] = np.asarray(uw._cur_button_states, np.int64)
        OB.append(ob)
        a = np.clip(actions[s + t] + sig * rng.standard_normal(adim).astype(np.float32), -1.0, 1.0)
        AC[t] = a
        ob = np.asarray(env.step(a)[0], np.float32)
    return {"pixels": px, "action": AC, "observation": np.asarray(OB, np.float32),
            "qpos": QP, "qvel": QV, "button_states": BT,
            "step_idx": np.arange(n, dtype=np.int64)}


RENDERERS = {"play": render_play, "noised": render_noised}


# ---------------------------------------------------------------------------------------------
# HDF5 io
# ---------------------------------------------------------------------------------------------
def _blosc(spec=SPEC):
    b = spec["blosc"]
    return hdf5plugin.Blosc(cname=b["cname"], clevel=b["clevel"],
                            shuffle=getattr(hdf5plugin.Blosc, b["shuffle"]))


def stream_write(path, episodes, spec=SPEC):
    """Stream episode dicts into one resizable HDF5 -> (ep_len, full digests, state digests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = spec["columns"]
    ep_len, digests, states, ptr, dsets = [], [], [], 0, {}
    with h5py.File(path, "w", libver="latest") as f:
        for ep in episodes:
            n = len(ep["action"])
            if not dsets:
                for col in cols:
                    per = np.asarray(ep[col]).shape[1:]
                    kw = (dict(chunks=(min(spec["pixel_chunk"], n), *per), **_blosc(spec))
                          if col == "pixels" else {})
                    dsets[col] = f.create_dataset(col, shape=(0, *per), maxshape=(None, *per),
                                                  dtype=np.asarray(ep[col]).dtype, **kw)
            for col in cols:
                dsets[col].resize(ptr + n, axis=0)
                dsets[col][ptr:ptr + n] = ep[col]
            digests.append(episode_digest(ep, cols))
            states.append(episode_digest(ep, state_columns(spec)))
            ep_len.append(n)
            ptr += n
        ep_len = np.asarray(ep_len, np.int32)
        _write_ep_tables(f, ep_len, digests, states)
    return ep_len, digests, states


def _write_ep_tables(f, ep_len, digests, states):
    f.create_dataset("ep_len", data=ep_len)
    f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64))
    f.create_dataset("ep_idx", data=np.repeat(np.arange(len(ep_len)), ep_len).astype(np.int32))
    f.create_dataset("episode_digest", data=np.array(digests, dtype="S64"))
    f.create_dataset("episode_state_digest", data=np.array(states, dtype="S64"))


def render_block(kind, block, num_blocks, gpu, prep_path, out_path, spec=SPEC):
    d = np.load(prep_path)
    idx = blocks_for(len(d["ep_len"]), num_blocks)[block]
    env = make_env(spec, gpu)
    fn = RENDERERS[kind]
    t0 = time.time()

    def gen():
        for i, e in enumerate(idx):
            yield fn(env, d, int(e), spec)
            if i % 10 == 0:
                print(f"[{kind} block {block}] {i + 1}/{len(idx)} eps "
                      f"({time.time() - t0:.0f}s)", flush=True)

    ep_len, digests, states = stream_write(out_path, gen(), spec)
    env.close()
    meta = {"kind": kind, "block": block, "num_blocks": num_blocks,
            "episodes": [int(e) for e in idx], "digests": digests, "state_digests": states,
            "spec_digest": spec_digest(spec)}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[{kind} block {block}/{num_blocks}] {len(idx)} eps {int(ep_len.sum())} frames "
          f"-> {out_path} ({time.time() - t0:.0f}s)", flush=True)


def _copy_into(fo, dsets, fi, ptr, cols, spec, rows=4096):
    """Append one source file's columns at row `ptr`.

    When the destination chunking matches the source, the append lands on a chunk boundary, and the
    chunk spans the column FULL WIDTH, the compressed pixel chunks are copied VERBATIM -- no
    decompress/recompress, which is otherwise by far the dominant cost of merging 100+ GB. SPEC's
    pixel_chunk divides the episode length so this fast path always applies to `pixels`; the
    elementwise path stays as a correct fallback.

    THE FULL-WIDTH GUARD IS LOAD-BEARING (fixed 2026-09-07, found in the probe numbers). The loop
    below walks the chunk grid along ROWS only, at column offset 0. h5py auto-chunks a narrow 2-d
    column by BOTH axes -- qpos (N, 25) becomes (256, 7), so its chunks live at column offsets
    0/7/14/21 -- and the three offsets past the first were never written. They read back as fill,
    i.e. ZERO, which is a perfectly plausible state vector: `scene_play_noised`, `cube_play` and
    `cube_play_noised` shipped with qpos/qvel/observation identically zero past the arm for
    episodes 0-255 (every block whose row count is a multiple of the 256-row chunk, which a 32-way
    split makes the first eight), and both mix files inherited it. Every cube_xyz / drawer /
    window probe read on those files was scored against a zero label for a quarter of its rows.
    Restricting the fast path to full-width chunks costs nothing: `pixels` is chunked
    (50, 224, 224, 3), the whole trailing extent, so it still takes it.
    """
    n = fi[cols[0]].shape[0]
    if not dsets:
        for col in cols:
            per = fi[col].shape[1:]
            kw = dict(chunks=fi[col].chunks, **_blosc(spec)) if col == "pixels" else {}
            dsets[col] = fo.create_dataset(col, shape=(0, *per), maxshape=(None, *per),
                                           dtype=fi[col].dtype, **kw)
    for col in cols:
        dst, src = dsets[col], fi[col]
        dst.resize(ptr + n, axis=0)
        cs = src.chunks[0] if src.chunks else None
        full_width = bool(src.chunks) and tuple(src.chunks[1:]) == tuple(src.shape[1:])
        if cs and full_width and dst.chunks == src.chunks and ptr % cs == 0 and n % cs == 0:
            zeros = (0,) * (src.ndim - 1)
            for c0 in range(0, n, cs):
                mask, raw = src.id.read_direct_chunk((c0, *zeros))
                dst.id.write_direct_chunk((ptr + c0, *zeros), raw, filter_mask=mask)
            _copy_into.verbatim += 1
        else:
            _copy_into.elementwise += 1
            for c0 in range(0, n, rows):                # pixels are far too big for one read
                c1 = min(c0 + rows, n)
                dst[ptr + c0:ptr + c1] = src[c0:c1]
    return n


def _copy_report(tag):
    v, e = _copy_into.verbatim, _copy_into.elementwise
    print(f"[{tag}] column copies: {v} verbatim (compressed chunks, no re-encode), {e} elementwise",
          flush=True)
    _copy_into.verbatim = _copy_into.elementwise = 0


_copy_into.verbatim = _copy_into.elementwise = 0


def merge(kind, block_dir, num_blocks, output, spec=SPEC):
    """Concatenate blocks IN INDEX ORDER -> canonical episode order for any num_blocks."""
    parts = [block_dir / f"{kind}_block_{k}.h5" for k in range(num_blocks)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:                                          # a silent skip here would quietly truncate
        raise SystemExit(f"MISSING BLOCKS: {missing} -- refusing to merge a partial dataset")
    ep_len_all, digests, states, ptr, dsets = [], [], [], 0, {}
    t0 = time.time()
    with h5py.File(str(output) + ".partial", "w", libver="latest") as fo:
        for p in parts:
            with h5py.File(p, "r") as fi:
                ptr += _copy_into(fo, dsets, fi, ptr, spec["columns"], spec)
                ep_len_all.append(np.asarray(fi["ep_len"]))
                digests += [x.decode() for x in np.asarray(fi["episode_digest"])]
                states += [x.decode() for x in np.asarray(fi["episode_state_digest"])]
            print(f"[merge {kind}] {p.name} -> {ptr} frames ({time.time() - t0:.0f}s)", flush=True)
        el = np.concatenate(ep_len_all).astype(np.int32)
        _write_ep_tables(fo, el, digests, states)
    os.replace(str(output) + ".partial", output)
    _copy_report(f"merge {kind}")
    write_manifest(output, kind, el, digests, states, spec)


def mix(play, noised, output, spec=SPEC):
    """scene_play ++ scene_play_noised as ONE physical file (a VDS would read over NFS and defeat
    node-local staging). Episode tables are rebuilt for the concatenation."""
    ep_len_all, digests, states, ptr, dsets = [], [], [], 0, {}
    t0 = time.time()
    with h5py.File(str(output) + ".partial", "w", libver="latest") as fo:
        for p in (play, noised):
            with h5py.File(p, "r") as fi:
                ptr += _copy_into(fo, dsets, fi, ptr, spec["columns"], spec)
                ep_len_all.append(np.asarray(fi["ep_len"]))
                digests += [x.decode() for x in np.asarray(fi["episode_digest"])]
                states += [x.decode() for x in np.asarray(fi["episode_state_digest"])]
            print(f"[mix] {Path(p).name} -> {ptr} frames ({time.time() - t0:.0f}s)", flush=True)
        el = np.concatenate(ep_len_all).astype(np.int32)
        _write_ep_tables(fo, el, digests, states)
    os.replace(str(output) + ".partial", output)
    _copy_report("mix")
    write_manifest(output, "mix", el, digests, states, spec)


# ---------------------------------------------------------------------------------------------
# manifest / verification
# ---------------------------------------------------------------------------------------------
def write_manifest(output, kind, ep_len, digests, states, spec=SPEC, file_hash=True):
    """Two content chains, because they answer different questions.

    content_sha256_state -- every column except pixels. A rebuild on any cluster MUST reproduce this;
                            it is the reproducibility claim.
    content_sha256_full  -- including pixels. Identifies this exact artifact. A rebuild will NOT
                            match it under MSAA (+/-1 LSB), which is why the file is distributed
                            rather than re-rendered; file_sha256 is what a downloader checks.
    """
    man = {
        "kind": kind,
        "spec_version": spec["version"],
        "spec_digest": spec_digest(spec),
        "spec": spec,
        "episodes": int(len(ep_len)),
        "frames": int(ep_len.sum()),
        "content_sha256_state": chain_digest(states),
        "content_sha256_full": chain_digest(digests),
        "file_sha256": sha256_file(output) if file_hash else None,
        "file_bytes": Path(output).stat().st_size,
        "episode_digests": digests,
        "episode_state_digests": states,
        "environment": env_fingerprint(),
    }
    p = Path(str(output) + ".manifest.json")
    p.write_text(json.dumps(man, indent=1))
    print(f"[manifest] {p}\n"
          f"[manifest] state  {man['content_sha256_state']}  (a rebuild must match this)\n"
          f"[manifest] full   {man['content_sha256_full']}\n"
          f"[manifest] file   {man['file_sha256']}\n"
          f"[manifest] episodes={man['episodes']} frames={man['frames']} "
          f"bytes={man['file_bytes'] / 1e9:.2f} GB", flush=True)
    return man


def recompute_digests(path, spec=SPEC, states_only=False):
    """Re-derive per-episode digests by reading the file back -- the deep check.

    `states_only` skips `pixels`, which is >99% of the bytes: the state chain then costs a few
    seconds on a 50 GB file instead of an hour, which is why the DEFAULT verify can afford it.
    """
    cols = state_columns(spec) if states_only else spec["columns"]
    full, states = [], []
    with h5py.File(path, "r") as f:
        off, el = np.asarray(f["ep_offset"]), np.asarray(f["ep_len"])
        for j in range(len(el)):
            a, n = int(off[j]), int(el[j])
            ep = {c: np.asarray(f[c][a:a + n]) for c in cols}
            if not states_only:
                full.append(episode_digest(ep, spec["columns"]))
            states.append(episode_digest(ep, state_columns(spec)))
    return full, states


def verify(path, deep=False, spec=SPEC):
    """Check a built or downloaded copy against its manifest.

    The STATE chain is the hard gate: it must match on any machine, and a mismatch means the rebuild
    genuinely diverged. The FULL chain additionally covers pixels -- it matches for a downloaded copy
    of this artifact, but a fresh re-render will differ by +/-1 LSB under MSAA, which is reported as
    a pixel-level note rather than a failure.
    """
    man_path = Path(str(path) + ".manifest.json")
    if not man_path.exists():
        raise SystemExit(f"no manifest beside {path}")
    man = json.loads(man_path.read_text())
    ok = True
    if man["spec_digest"] != spec_digest(spec):
        print(f"FAIL spec_digest: manifest {man['spec_digest'][:16]} != current {spec_digest(spec)[:16]}")
        ok = False

    if deep:
        full, states = recompute_digests(path, spec)
        src = "recomputed by re-reading the file"
    else:
        # THE STATE CHAIN IS ALWAYS RECOMPUTED FROM THE COLUMNS. Reading the stored
        # `episode_state_digest` instead compares the merge's own bookkeeping against the manifest
        # written from that same bookkeeping -- both are carried over from the block files, so the
        # check passed on a file whose columns were partly unwritten (see `_copy_into`). State is
        # cheap to re-read; only `pixels` is not, and that is what `--deep` adds.
        _, states = recompute_digests(path, spec, states_only=True)
        with h5py.File(path, "r") as f:
            full = [x.decode() for x in np.asarray(f["episode_digest"])]
        src = "state recomputed from the columns; full chain from the file's stored digests"

    if len(states) != len(man["episode_state_digests"]):
        print(f"FAIL episode count: {len(states)} vs manifest {len(man['episode_state_digests'])}")
        return False
    bad = [i for i, (a, b) in enumerate(zip(states, man["episode_state_digests"])) if a != b]
    if bad:
        print(f"FAIL state: {len(bad)}/{len(states)} episodes differ; first 10 -> {bad[:10]}")
        ok = False
    got_state, got_full = chain_digest(states), chain_digest(full)
    if got_state != man["content_sha256_state"]:
        print(f"FAIL content_sha256_state: {got_state}\n                        != {man['content_sha256_state']}")
        ok = False
    if got_full != man["content_sha256_full"]:
        npix = sum(1 for a, b in zip(full, man["episode_digests"]) if a != b)
        print(f"note  pixels differ in {npix}/{len(full)} episodes (state is unaffected). Expected for a "
              f"re-render under MSAA; NOT expected for a downloaded copy.")
    fp = env_fingerprint()
    drift = {k: (man["environment"].get(k), v) for k, v in fp.items() if man["environment"].get(k) != v}
    if drift:
        print(f"note  environment differs from the build machine: {drift}")
    print(f"{'PASS' if ok else 'FAIL'}  {path}  ({src})\n"
          f"      state {got_state}\n      full  {got_full}")
    return ok


# ---------------------------------------------------------------------------------------------
# preflight / selftest
# ---------------------------------------------------------------------------------------------
# Probe EPISODES, not single frames. Re-rendering one frozen state is NOT a valid determinism test:
# under MSAA it was stable for 250 consecutive renders while replaying an actual episode -- a
# sequence of differing states -- still flipped LSBs on a third of its frames.
GOLDEN_PROBES = [("play_ep0", "play", 0), ("play_ep500", "play", 500), ("noised_ep1", "noised", 1)]
PREFLIGHT_FRAMES = 30
PIXEL_TOL = 1              # MSAA resolve jitter is +/-1 LSB; anything larger is a different renderer
PIXEL_MEAN_TOL = 1e-3      # mean intensity over 30 frames -- LSB jitter moves it far less than this


def preflight(prep_path, gpu=0, emit=False, repeats=2, spec=SPEC):
    """Gate a build on whether THIS machine can produce the artifact, in a minute rather than a day.

    Two different bars, because the two halves of the data have different guarantees. STATE must be
    bit-identical to the reference -- a mismatch means a different mujoco/numpy or a broken SPEC, and
    the build is refused. PIXELS need only be within PIXEL_TOL of the reference: MSAA costs +/-1 LSB
    on a minority of frames, but a genuinely different renderer (other GPU vendor, other mujoco)
    would blow past that.
    """
    d = dict(np.load(prep_path))
    d["ep_len"] = np.minimum(d["ep_len"], PREFLIGHT_FRAMES)
    env = make_env(spec, gpu)
    got, distinct, pix = {}, {}, {}
    for name, kind, e in GOLDEN_PROBES:
        runs = [RENDERERS[kind](env, d, e, spec) for _ in range(repeats)]
        got[name] = {"state": episode_digest(runs[0], state_columns(spec)),
                     "pixel_mean": float(np.asarray(runs[0]["pixels"], np.float64).mean())}
        distinct[name] = len({episode_digest(r, state_columns(spec)) for r in runs})
        pix[name] = max(int(np.abs(r["pixels"].astype(np.int32)
                                   - runs[0]["pixels"].astype(np.int32)).max()) for r in runs)
    env.close()

    print(f"[preflight] spec {spec['version']} ({spec_digest(spec)[:16]})  {env_fingerprint()['gpu']}"
          f"  offsamples={spec['offsamples']} supersample={spec['supersample']}")
    ok = True
    for name, _, _ in GOLDEN_PROBES:
        if distinct[name] != 1:
            print(f"  {name}: STATE NONDETERMINISTIC -- {distinct[name]} distinct results in {repeats} "
                  f"replays of the same episode. The simulation is not reproducible on this machine.")
            ok = False
        if pix[name] > PIXEL_TOL:
            print(f"  {name}: pixels vary by {pix[name]} > PIXEL_TOL={PIXEL_TOL} across replays -- "
                  f"more than MSAA resolve jitter, something else is nondeterministic here.")
            ok = False
    if emit:
        print("\nGOLDEN = {")
        for name, _, _ in GOLDEN_PROBES:
            print(f'    "{name}": {{"state": "{got[name]["state"]}", '
                  f'"pixel_mean": {got[name]["pixel_mean"]:.9f}}},')
        print("}")
        return ok
    for name, _, _ in GOLDEN_PROBES:
        want = GOLDEN.get(name)
        if want is None:
            print(f"  {name}: no golden recorded (run `preflight --emit`)")
            ok = False
            continue
        if want["state"] != got[name]["state"]:
            print(f"  {name}: STATE MISMATCH -- this machine does not reproduce the reference "
                  f"simulation\n      expected {want['state']}\n      got      {got[name]['state']}")
            ok = False
        elif abs(want["pixel_mean"] - got[name]["pixel_mean"]) > PIXEL_MEAN_TOL:
            print(f"  {name}: PIXEL MISMATCH -- mean intensity {got[name]['pixel_mean']:.6f} vs "
                  f"reference {want['pixel_mean']:.6f}; this renderer differs from the reference.")
            ok = False
        else:
            print(f"  {name}: OK state {got[name]['state'][:16]}  "
                  f"pixels within tolerance ({repeats} replays, max delta {pix[name]})")
    print(f"[preflight] {'PASS -- this machine reproduces the reference renders' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    return ok


def selftest(prep_path, gpu=0, n_frames=200, spec=SPEC):
    """Prove the two properties the build relies on, cheaply, on THIS machine.

    (a) repeat determinism -- replaying an episode gives identical STATE columns
    (b) order independence  -- an episode's state is identical whether or not another ran before it

    Pixel jitter is measured and printed rather than asserted: under MSAA it is expected and does not
    affect any of the columns a rebuild is required to reproduce.
    """
    d = dict(np.load(prep_path))
    d["ep_len"] = np.minimum(d["ep_len"], n_frames)      # truncate: divergence shows up immediately
    env = make_env(spec, gpu)
    sc = state_columns(spec)
    ok = True
    for kind in KINDS:
        fn = RENDERERS[kind]
        r1 = fn(env, d, 1, spec)
        r2 = fn(env, d, 1, spec)
        fn(env, d, 0, spec)                              # a DIFFERENT episode in between
        r3 = fn(env, d, 1, spec)
        iso = make_env(spec, gpu)                        # and a process that has rendered NOTHING
        r4 = fn(iso, d, 1, spec)
        iso.close()
        a, b, c = (episode_digest(r, sc) for r in (r1, r2, r3))
        if episode_digest(r4, sc) != a:
            print(f"  {kind:7s} ISOLATION FAIL -- a fresh env gives a different episode than a reused one")
            ok = False
        jitter = max(int(np.abs(r["pixels"].astype(np.int32) - r1["pixels"].astype(np.int32)).max())
                     for r in (r2, r3))
        nfr = max(int((np.abs(r["pixels"].astype(np.int32) - r1["pixels"].astype(np.int32))
                       .reshape(len(r1["pixels"]), -1) > 0).any(1).sum()) for r in (r2, r3))
        print(f"  {kind:7s} state_repeat={'OK' if a == b else 'DIFFER'}  "
              f"state_order_independent={'OK' if a == c else 'DIFFER'}  digest={a[:16]}  "
              f"| pixel jitter {nfr}/{len(r1['pixels'])} frames, max {jitter} (MSAA, expected)")
        ok &= (a == b == c)
    # block partitioning must not move a single episode
    n = len(d["ep_len"])
    split_ok = True
    for w in (1, 3, 7, 36):
        flat = np.concatenate(blocks_for(n, w))
        if not np.array_equal(flat, np.arange(n)):
            print(f"  block split w={w}: NOT canonical order")
            split_ok = False
    print(f"  block split: canonical for w in (1,3,7,36) -> {'OK' if split_ok else 'FAIL'}")
    ok &= split_ok
    env.close()
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    return ok


# ---------------------------------------------------------------------------------------------
def main():
    from stable_worldmodel.data.utils import get_cache_dir

    root = get_cache_dir(sub_folder="datasets")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "render", "merge", "mix", "verify", "preflight", "selftest", "spec"])
    ap.add_argument("--kind", choices=[*KINDS, "mix"], default="noised")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--num-blocks", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--deep", action="store_true", help="verify: re-read pixels instead of trusting stored digests")
    ap.add_argument("--emit", action="store_true", help="preflight: print GOLDEN block to paste into this file")
    ap.add_argument("--repeats", type=int, default=2, help="preflight: replays per probe episode")
    ap.add_argument("--no-verify-source", action="store_true")
    ap.add_argument("--prep", type=Path, default=root / "_scene_build/prep.npz")
    ap.add_argument("--block-dir", type=Path, default=root / "_scene_build")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--play", type=Path, default=None, help="mix: expert input (default scene_play.h5)")
    ap.add_argument("--noised", type=Path, default=None, help="mix: noised input (default scene_play_noised.h5)")
    a = ap.parse_args()

    default_out = {"play": root / "scene_play.h5", "noised": root / "scene_play_noised.h5",
                   "mix": root / "scene_play_mix_noised.h5"}
    out = a.out or default_out[a.kind]

    if a.cmd == "spec":
        print(json.dumps({"spec": SPEC, "spec_digest": spec_digest(), "golden": GOLDEN,
                          "environment": env_fingerprint()}, indent=1))
    elif a.cmd == "prepare":
        prepare(a.prep, verify_source=not a.no_verify_source)
    elif a.cmd == "preflight":
        preflight(a.prep, a.gpu, emit=a.emit, repeats=a.repeats)
    elif a.cmd == "selftest":
        selftest(a.prep, a.gpu)
    elif a.cmd == "render":
        if a.block is None:
            raise SystemExit("--block is required")
        render_block(a.kind, a.block, a.num_blocks, a.gpu, a.prep,
                     a.block_dir / f"{a.kind}_block_{a.block}.h5")
    elif a.cmd == "merge":
        merge(a.kind, a.block_dir, a.num_blocks, out)
    elif a.cmd == "mix":
        mix(a.play or default_out["play"], a.noised or default_out["noised"],
            a.out or default_out["mix"])
    elif a.cmd == "verify":
        raise SystemExit(0 if verify(out, deep=a.deep) else 1)


if __name__ == "__main__":
    main()
