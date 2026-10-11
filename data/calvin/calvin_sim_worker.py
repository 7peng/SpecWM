"""One CALVIN PlayTableSimEnv in the simulator's own venv, driven over a pipe by src/calvin_sim_env.py.

WHY A WORKER. calvin_env needs pybullet and the legacy `gym`, neither of which the training venv
has, and installing them into a venv that four training ranks are importing from is not something
to do under a live run. The CALVIN builders already run the simulator under $CALVIN_VENV
(make_calvin_suite.py, build_calvin_noised.py render); this is that arrangement with a pipe on it.

PROTOCOL. Length-prefixed pickles over two dedicated pipe fds handed in by the parent (argv), NOT
stdin/stdout: pybullet and hydra print to fd 1 from C, which would corrupt a stdout protocol.
    ("build", scene_letter)                       -> ("ok", None)
    ("reset", robot_obs(15), scene_obs_raw(24))   -> ("ok", obs)
    ("step",  action(7) in [-1, 1], "cartesian_rel") -> ("ok", obs)
    ("close",)
obs = {"rgb_static": uint8 (200, 200, 3), "robot_obs": float64 (15,), "scene_obs_raw": float64 (24,)}
scene_obs_raw is in the SIMULATOR's per-scene block order (data/calvin/roster.py); the consumer
canonicalizes. The action is the pack's `action` column (rel_actions), applied exactly as the noised
builder applies it (build_calvin_noised.render_block).

EGL device: MUJOCO_EGL_DEVICE_ID (what scripts/eval_cem.py sets per GPU), else EGL_DEVICE, else 0.
"""
import os
import random
import pickle
import struct
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _send(f, obj):
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _recv(f):
    hdr = f.read(8)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(bytes(buf))


def _obs_of(o):
    return {"rgb_static": np.ascontiguousarray(o["rgb_obs"]["rgb_static"], dtype=np.uint8),
            "robot_obs": np.asarray(o["robot_obs"], np.float64).reshape(-1),
            "scene_obs_raw": np.asarray(o["scene_obs"], np.float64).reshape(-1)}


def main():
    rfd, wfd = int(sys.argv[1]), int(sys.argv[2])
    inp = os.fdopen(rfd, "rb", buffering=0)
    out = os.fdopen(wfd, "wb", buffering=0)
    gpu = os.environ.get("MUJOCO_EGL_DEVICE_ID") or os.environ.get("EGL_DEVICE") or "0"
    for v in ("CUDA_VISIBLE_DEVICES", "EGL_DEVICE", "EGL_VISIBLE_DEVICES"):
        os.environ[v] = str(gpu)
    os.environ.setdefault("CALVIN_USE_EGL", "1")
    from make_calvin_suite import build_env, CALVIN_SRC
    env = None
    build_random = build_numpy = None
    scene_letter = None
    resets = 0
    try:
        while True:
            msg = _recv(inp)
            if msg is None or msg[0] == "close":
                break
            print(f"[worker {os.getpid()}] {msg[0]} cid={getattr(env, 'cid', None)}", file=sys.stderr, flush=True)
            try:
                if msg[0] == "build":
                    if env is not None:
                        # PlayTableSimEnv holds the GLOBAL pybullet module and client id 0, and its
                        # __del__ disconnects that id again: a dropped old env would sever the new
                        # one's connection. Close it, then make its destructor a no-op.
                        env.close()
                        env.cid, env.ownsPhysicsClient = -1, False
                    scene_letter = str(msg[1])
                    build_random, build_numpy = random.getstate(), np.random.get_state()
                    env = build_env(scene_letter, str(CALVIN_SRC / "data"))
                    resets = 0
                    _send(out, ("ok", None))
                elif msg[0] == "reset":
                    if resets and os.environ.get('CALVIN_REBUILD_ON_RESET') == '1':
                        # Retain the expensive Python worker/imports, but rebuild the
                        # scene/controller exactly as a fresh process would. Pose reset
                        # and Bullet save/restore alone did not reproduce trajectories.
                        env.close()
                        env.cid, env.ownsPhysicsClient = -1, False
                        random.setstate(build_random)
                        np.random.set_state(build_numpy)
                        env = build_env(scene_letter, str(CALVIN_SRC / "data"))
                    resets += 1
                    o = env.reset(robot_obs=np.asarray(msg[1], np.float64).reshape(-1),
                                  scene_obs=np.asarray(msg[2], np.float64).reshape(-1))
                    _send(out, ("ok", _obs_of(o)))
                elif msg[0] == "step":
                    a = np.asarray(msg[1], np.float64).reshape(-1)
                    o, _, _, _ = env.step({"action": a, "type": "cartesian_rel"})
                    _send(out, ("ok", _obs_of(o)))
                else:
                    _send(out, ("err", f"unknown command {msg[0]!r}"))
            except Exception:
                _send(out, ("err", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
