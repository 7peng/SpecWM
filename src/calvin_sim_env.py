"""CALVIN goal-offset eval env for the swm CEM pipeline, registered as 'lewm/CalvinTask-v0'
(world.env_name: `calvin_sim_env:lewm/CalvinTask-v0`, so gym.make imports this module first).

THE SIMULATOR LIVES IN A WORKER SUBPROCESS under $CALVIN_VENV (data/calvin/calvin_sim_worker.py):
calvin_env needs pybullet and the legacy gym, which the training venv does not have. Each env owns
one worker; swm's process-parallel pool (eval.py) puts each env in its own process, so a 100-task
panel is 100 sims, one per process, exactly as the scene panel is 100 MuJoCo envs.

WHAT AN EPISODE IS. `config/eval/calvin.yaml`'s callables install the task after the reset:
`set_state(robot_obs, scene_obs, scene_id)` from the task set's first frame and
`set_goal(goal_scene_obs)` from its last. The pack's scene_obs is CANONICAL (red, blue, pink; see
data/calvin/roster.py) and the simulator wants its own per-scene order; the two are related by a
per-scene transposition, so the same permutation maps either way (asserted, not assumed).
`step` applies the 7-d rel_action exactly as the noised builder does, reads back the state, and
terminates on the goal predicate (data/calvin/goal_predicate.py), which is what world.terminateds
counts as success. `render` returns the last static-camera frame (200x200, the pack's own
resolution); MegaWrapper resizes it to the eval's 224 as the training loader resizes the pack.

info: success, factor errors under `err/<factor>`, and the moved-factor set of the task
(`task_factors`) -- what the task was about, for per-factor breakdowns.
"""
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs import registration

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "data" / "calvin"))
import roster as R              # noqa: E402
import goal_predicate as GP     # noqa: E402

LETTERS = "ABCD"
WORKER = _HERE.parent / "data" / "calvin" / "calvin_sim_worker.py"


def _send(f, obj):
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _recv(f):
    hdr = f.read(8)
    if len(hdr) < 8:
        raise RuntimeError("calvin sim worker closed the pipe (see its log)")
    (n,) = struct.unpack("<Q", hdr)
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise RuntimeError("calvin sim worker closed the pipe mid-message (see its log)")
        buf += chunk
    return pickle.loads(bytes(buf))


def _to_raw(scene_obs_canonical, scene_id):
    """Canonical -> simulator order. canonicalize() is a per-scene transposition of two block
    slots, hence its own inverse; the round trip is asserted rather than trusted."""
    c = np.asarray(scene_obs_canonical, np.float64).reshape(1, 24)
    sid = np.asarray([int(scene_id)])
    raw = R.canonicalize(c, sid)
    assert np.array_equal(R.canonicalize(raw, sid), c), "roster permutation is not an involution"
    return raw[0]


class CalvinSimWorker:
    def __init__(self, log_path):
        venv = os.environ.get("CALVIN_VENV")
        assert venv and Path(venv, "bin", "python").is_file(), (
            f"CALVIN_VENV={venv!r} has no bin/python; the simulator venv is required "
            f"(root .env.example names it)")
        r_in, w_in = os.pipe()      # parent -> worker
        r_out, w_out = os.pipe()    # worker -> parent
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env.setdefault("CALVIN_USE_EGL", "1")
        # PlayTableSimEnv records the calvin_env checkout's git state at construction, so the
        # checkout is declared safe for git and the worker runs from inside it
        src = env.get("CALVIN_ENV_SRC")
        assert src and Path(src).is_dir(), f"CALVIN_ENV_SRC={src!r} is not the calvin_env checkout (root .env.example names it)"
        env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
                    "GIT_CONFIG_VALUE_0": src})
        self.log = open(log_path, "ab")
        self.proc = subprocess.Popen(
            [str(Path(venv, "bin", "python")), str(WORKER), str(r_in), str(w_out)],
            pass_fds=(r_in, w_out), env=env, stdout=self.log, stderr=subprocess.STDOUT,
            cwd=src)
        os.close(r_in)
        os.close(w_out)
        self.w = os.fdopen(w_in, "wb", buffering=0)
        self.r = os.fdopen(r_out, "rb", buffering=0)
        self.scene = None

    def call(self, *msg):
        _send(self.w, msg)
        status, payload = _recv(self.r)
        if status != "ok":
            raise RuntimeError(f"calvin sim worker {msg[0]!r} failed:\n{payload}")
        return payload

    def build(self, letter):
        if self.scene != letter:
            self.call("build", letter)
            self.scene = letter

    def close(self):
        try:
            _send(self.w, ("close",))
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.log.close()


class CalvinTaskEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", tol=None, worker_log_dir=None):
        super().__init__()
        self.render_mode = render_mode
        self.tol = dict(GP.TOL, **(tol or {}))
        self.worker_log_dir = Path(worker_log_dir or os.environ.get("CALVIN_WORKER_LOG_DIR")
                                   or "/tmp/calvin_sim_workers")
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (15,), np.float32)})
        self._worker = None
        self._last = None           # last worker obs
        self._goal = None           # canonical (24,)
        self._start = None          # canonical (24,)
        self._scene_id = None
        self._task_factors = []
        self._t = 0

    # -- task installation (config/eval/calvin.yaml callables) ------------------------------
    def set_state(self, robot_obs, scene_obs, scene_id):
        sid = int(np.asarray(scene_id).reshape(-1)[0])
        letter = LETTERS[sid]
        if self._worker is None:
            self._worker = CalvinSimWorker(self.worker_log_dir / f"worker_{os.getpid()}.log")
        self._worker.build(letter)
        self._scene_id = sid
        self._start = np.asarray(scene_obs, np.float64).reshape(24).copy()
        self._last = self._worker.call("reset", np.asarray(robot_obs, np.float64).reshape(15),
                                       _to_raw(self._start, sid))
        self._t = 0
        # the simulator must hand back the state it was given: the reset is by state, so a
        # mismatch here is a roster or a units error, never physics
        back = self._canonical(self._last["scene_obs_raw"])
        err = np.abs(back - self._start)
        assert err[:6].max() < 1e-3 and np.abs(back[6:9] - self._start[6:9]).max() < 1e-3, (
            f"reset did not reproduce the requested scene_obs (max |delta| {err.max():.4f}); "
            f"canonical/raw order confused?")

    def set_goal(self, goal_scene_obs):
        self._goal = np.asarray(goal_scene_obs, np.float64).reshape(24).copy()
        if self._start is not None:
            self._task_factors = GP.moved_factors(self._start, self._goal, self.tol)

    # -- gym API -----------------------------------------------------------------------------
    def _canonical(self, raw):
        return R.canonicalize(np.asarray(raw, np.float64).reshape(1, 24),
                              np.asarray([self._scene_id]))[0]

    def _obs_info(self):
        if self._last is None:
            return {"proprio": np.zeros(15, np.float32)}, {"success": False, "task_factors": ""}
        cur = self._canonical(self._last["scene_obs_raw"])
        info = {"task_factors": ",".join(self._task_factors), "t": self._t}
        if self._goal is not None:
            ok = GP.factor_ok(cur, self._goal, self.tol)
            for k, v in GP.factor_errors(cur, self._goal).items():
                info[f"err/{k}"] = float(v)
            info["success"] = bool(all(ok.values()))
            info["task_ok"] = bool(all(ok[f] for f in self._task_factors)) if self._task_factors else True
        else:
            info["success"] = False
        return {"proprio": np.asarray(self._last["robot_obs"], np.float32)}, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # the task arrives through set_state/set_goal AFTER this reset (World._evaluate_from_dataset);
        # until then there is nothing to simulate and the frame is blank
        return self._obs_info()

    def step(self, action):
        assert self._worker is not None and self._last is not None, "step before set_state"
        a = np.asarray(action, np.float32).reshape(-1)
        assert a.shape == (7,), f"CALVIN action is 7-d (rel_actions), got {a.shape}"
        a = np.clip(a, -1.0, 1.0).astype(np.float64)
        # calvin_env's robot asserts gripper_action in (-1, 1): the recorded rel_actions gripper is
        # exactly +/-1 and a planned or random one is continuous, so it is thresholded at 0 here
        # (the only binarisation the sim accepts; the first floor run died on this assert)
        a[6] = 1.0 if a[6] >= 0.0 else -1.0
        self._last = self._worker.call("step", a)
        self._t += 1
        obs, info = self._obs_info()
        return obs, 0.0, bool(info["success"]), False, info

    def render(self):
        if self._last is None:
            return np.zeros((200, 200, 3), np.uint8)
        return self._last["rgb_static"]

    def close(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None


registration.register(id="lewm/CalvinTask-v0", entry_point=CalvinTaskEnv)
