"""The eval config's callables must reach the env, and the task set's goal must be winnable.

Two silent failures this pins:

  * A CALLABLE THAT DOES NOT EXIST IS SKIPPED WITHOUT A WORD. `world._apply_callables` guards with
    `hasattr(env, method)` -- but under the process-parallel pool `env` is `_RemoteUnwrapped`,
    whose `__getattr__` answers to ANY name, so that guard never fires and the worker's own
    `if hasattr(env, name)` drops the call. A scene config pointed at the cube env (or a renamed
    setter) would then evaluate against whatever target the env's own reset sampled, and report a
    success rate for it.
  * A GOAL THE PREDICATE CANNOT ACCEPT. The task builders write `block_pos` from a qpos slice and
    the env thresholds `object_joint_0.qpos[:3]` against a mocap position; the drawer/window/button
    factors are separate stores again. If any of those disagree the panel is unwinnable and every
    arm scores its floor.

Runs the real MuJoCo envs (construction and physics only -- no rendering, so it runs on a CPU box).

  PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python tests/adv_cem_env_protocol.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py                                                                     # noqa: E402
import hdf5plugin  # noqa: F401,E402                                            # Blosc codec
import gymnasium as gym                                                         # noqa: E402
import mujoco                                                                   # noqa: E402
import numpy as np                                                              # noqa: E402
from omegaconf import OmegaConf                                                 # noqa: E402

import stable_worldmodel as swm  # noqa: F401,E402   registers swm/OGB*-v0

CFG_DIR = os.path.join(REPO, "config", "eval")
N_EP = 12          # episodes to check per config; the whole set is 100


def _root():
    from stable_worldmodel.data.utils import get_cache_dir
    return get_cache_dir(sub_folder="datasets")


def _make_env(cfg):
    world = OmegaConf.to_container(cfg.world, resolve=False)
    world.pop("num_envs", None)
    world["max_episode_steps"] = 100
    name = world.pop("env_name")
    return gym.make(name, render_mode="rgb_array", **world).unwrapped


# scripts/eval_cem.py's task-set family per eval config (kept in sync by hand; the driver's own
# TASKSETS is the source of truth)
TASKSETS = {"scene": "scene_offset_tasks", "scene_explore": "scene_offset_tasks",
            "cube": "cube_offset_tasks"}


def check(config_name, offset=25):
    cfg = OmegaConf.load(os.path.join(CFG_DIR, f"{config_name}.yaml"))
    if "eval" not in cfg or "callables" not in cfg.eval:
        print(f"[{config_name}] no callables; skipped")
        return
    # the driver overrides eval.dataset_name/goal_offset_steps; the config's own defaults are the
    # 5-task built-in set, which is not what any recorded number used
    ds = f"{TASKSETS[config_name]}_O{offset}"
    path = _root() / f"{ds}.h5"
    if not path.exists():
        print(f"[{config_name}] {ds}.h5 absent under {_root()}; skipped")
        return
    env = _make_env(cfg)
    env.reset(seed=0)

    # --- every named method must exist on the env the config names
    for spec in cfg.eval.callables:
        m = str(spec["method"])
        assert hasattr(env, m), (
            f"[{config_name}] the env {type(env).__name__} has no `{m}`; under the parallel pool "
            f"this call is dropped silently and the episode is scored against the env's own "
            f"randomly sampled target")

    with h5py.File(path, "r") as f:
        cols = set(f.keys())
        # --- every in_dataset arg must name a column the task file carries
        for spec in cfg.eval.callables:
            for arg, d in (spec.get("args") or {}).items():
                if not d.get("in_dataset", True):
                    continue
                key = str(d["value"])
                col = key[len("goal_"):] if key.startswith("goal_") else key
                assert col in cols, (
                    f"[{config_name}] callable {spec['method']}({arg}=) reads '{key}', but "
                    f"{ds}.h5 has no column '{col}'; world._apply_callables drops the argument")

        el, eo = f["ep_len"][:], f["ep_offset"][:]
        assert int(el[0]) == offset + 1, (
            f"[{config_name}] {ds} episodes are {int(el[0])} frames; goal_offset_steps={offset} "
            f"needs {offset+1}")

        def install(row, targets_from):
            """Apply the config's callables for one env, exactly as world._apply_callables does."""
            for spec in cfg.eval.callables:
                kw = {}
                for arg, d in (spec.get("args") or {}).items():
                    if not d.get("in_dataset", True):
                        kw[arg] = d["value"]
                        continue
                    key = str(d["value"])
                    src = targets_from if key.startswith("goal_") else row
                    col = key[len("goal_"):] if key.startswith("goal_") else key
                    kw[arg] = f[col][src]
                getattr(env, spec["method"])(**kw)

        solved_at_start = solved_at_goal = 0
        for ep in range(min(N_EP, len(el))):
            s, g = int(eo[ep]), int(eo[ep]) + int(el[ep]) - 1
            install(s, g)                                    # start state + goal targets
            mujoco.mj_forward(env._model, env._data)
            solved_at_start += _all_ok(env)
            install(g, g)                                    # goal state + goal targets
            mujoco.mj_forward(env._model, env._data)
            solved_at_goal += _all_ok(env)

    n = min(N_EP, len(el))
    assert solved_at_goal == n, (
        f"[{config_name}] the success predicate accepts the recorded GOAL state in only "
        f"{solved_at_goal}/{n} episodes -- the panel is partly unwinnable and every arm reads its "
        f"floor on those tasks")
    assert solved_at_start == 0, (
        f"[{config_name}] {solved_at_start}/{n} episodes are ALREADY solved at the start frame; "
        f"the task builder's non-triviality filter and the env's tolerance disagree")
    print(f"[{config_name}] {len(cfg.eval.callables)} callables all present; goal winnable "
          f"{solved_at_goal}/{n}, start trivially solved {solved_at_start}/{n}: OK")


def _all_ok(env):
    r = env._compute_successes()
    if isinstance(r, tuple):
        cs, bs, ds, ws = r
        return bool(all(cs) and all(bs) and ds and ws)
    return bool(all(r))


def test_remote_unwrapped_defeats_the_hasattr_guard():
    """The world-side skip is inert under the parallel pool, so the worker must refuse loudly."""
    from parallel_env_pool import _RemoteUnwrapped, require_method

    class FakePool:
        def __init__(self):
            self.sent = []

        def _send_callable(self, idx, name, kw):
            self.sent.append((idx, name, kw))

    pool = FakePool()
    ru = _RemoteUnwrapped(pool, 0)
    assert hasattr(ru, "set_target_on_the_moon"), (
        "_RemoteUnwrapped no longer answers to arbitrary names -- update this test")
    ru.set_target_on_the_moon(x=1)
    assert pool.sent == [(0, "set_target_on_the_moon", {"x": 1})], pool.sent

    class Env:
        def set_state(self, **kw):
            pass

    require_method(Env(), "set_state")            # present: silent
    try:
        require_method(Env(), "set_target_on_the_moon")
    except AssertionError as e:
        assert "does not exist" in str(e), str(e)
    else:
        raise AssertionError(
            "the worker still SKIPS a callable the env does not have: the episode's targets would "
            "never be installed and the run would score the env's own randomly sampled goal")
    print("[pool ] world's hasattr guard is inert under _RemoteUnwrapped; the worker refuses a "
          "missing callable instead of skipping it: OK")


if __name__ == "__main__":
    test_remote_unwrapped_defeats_the_hasattr_guard()
    check("scene", 25)
    check("cube", 25)
    print("\nadv_cem_env_protocol: all checks passed")
