"""ParallelEnvPool: process-parallel, shared-memory drop-in for swm's ``EnvPool``, sharding envs
across spawned worker processes with their own EGL+CUDA contexts (eval.py installs it; one env per
process by default)."""

from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import Any

import multiprocessing as mp
import numpy as np
from gymnasium.vector.utils import batch_space

# swm's own stacking helpers, reused so assembly matches the serial pool
from stable_worldmodel.world.env_pool import (
    _broadcast_arg,
    _stack_fresh,
    _write_env_info,
)


def require_method(env, name):
    """Refuse a callable the env does not have, instead of skipping it: swm's `hasattr` guard runs on
    the main process against `_RemoteUnwrapped`, which answers to any name, so the worker is the only
    place a config/env mismatch is noticed (a skipped call installs the episode only partially)."""
    assert hasattr(env, name), (
        f"eval callable `{name}` does not exist on {type(env).__name__}; the episode would be "
        f"installed only partially and scored against the env's own randomly sampled target. "
        f"Check eval.callables against this env (config/eval/*.yaml).")


# gym spaces can carry unpicklable closures, so picklable specs cross the spawn pipe and the
# space is rebuilt on the parent
def _spec_of(space):
    import gymnasium.spaces as S
    if space is None:
        return None
    if isinstance(space, S.Box):
        return ("box", np.asarray(space.low), np.asarray(space.high), tuple(space.shape), str(space.dtype))
    if isinstance(space, S.Discrete):
        return ("discrete", int(space.n))
    if isinstance(space, S.Dict):
        return ("dict", {k: _spec_of(v) for k, v in space.spaces.items()})
    return ("raw", space)  # last resort; may still fail to pickle (surfaced as a worker error)


def _space_of(spec):
    import gymnasium.spaces as S
    if spec is None:
        return None
    kind = spec[0]
    if kind == "box":
        return S.Box(low=spec[1], high=spec[2], shape=spec[3], dtype=np.dtype(spec[4]))
    if kind == "discrete":
        return S.Discrete(spec[1])
    if kind == "dict":
        return S.Dict({k: _space_of(v) for k, v in spec[1].items()})
    return spec[1]


# --------------------------------------------------------------------------- worker process
def _worker_main(conn, shard, env_fns, primary):
    """Run one shard of envs in its own process (own EGL/CUDA context). ``shard`` lists the global
    env indices this worker owns (ascending); ``env_fns`` aligns with it; the primary worker
    reports the single-env spaces on startup."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        envs = [fn() for fn in env_fns]
    except Exception:  # pragma: no cover - surfaced to the parent
        # Full traceback, not repr(e): a construction failure in a spawned worker is otherwise a
        # bare "FileNotFoundError(2)" with nothing to say which file or which frame.
        import traceback
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return

    spaces = None
    if primary:
        spaces = (
            _spec_of(envs[0].action_space),
            _spec_of(envs[0].observation_space),
            _spec_of(getattr(envs[0].unwrapped, "variation_space", None)),
        )
    conn.send(("ready", spaces))

    # transport: key -> (np.ndarray view over SharedMemory, shm handle)
    transport: dict[str, tuple[np.ndarray, Any]] = {}
    # buffered set_state-style calls applied just before the next step (see _RemoteUnwrapped)
    pending: dict[int, list[tuple[str, dict]]] = {}

    def _local(gidx):
        return shard.index(gidx)

    def _apply_calls(gidx, calls):
        env = envs[_local(gidx)].unwrapped
        for name, kwargs in calls:
            require_method(env, name)
            getattr(env, name)(**kwargs)

    def _apply_pending():
        for gidx, calls in pending.items():
            _apply_calls(gidx, calls)
        pending.clear()

    def _sim_state(env):
        data = getattr(env.unwrapped, "data", None)
        if data is None or getattr(data, "qpos", None) is None:
            return None
        return np.concatenate([np.array(data.qpos, copy=True).ravel(), np.array(data.qvel, copy=True).ravel()])

    def _reobserve(gidx, calls):
        """Re-observe env `gidx` in place. Where `reset()` re-randomises the sim (the MuJoCo envs),
        the installed calls are re-applied and the observation re-taken from the restored state."""
        env = envs[_local(gidx)]
        before = _sim_state(env)
        _, info = env.reset()
        after = _sim_state(env)
        if before is None or after is None or np.array_equal(before, after):
            return info
        _apply_calls(gidx, calls)
        # every observation-derived key the env can recompute, not only pixels/qpos/qvel
        ob_info = getattr(env.unwrapped, "compute_ob_info", None)
        if ob_info is not None:
            for k, v in ob_info().items():
                if k in info:
                    info[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
        pix = None
        w = env
        while w is not None and pix is None:
            get_pixels = getattr(w, "_get_pixels", None)
            if get_pixels is not None:
                pix, info["render_time"] = get_pixels()
            w = getattr(w, "env", None)
        assert pix is not None, "no pixels wrapper to re-render from"
        info.update(pix)
        data = env.unwrapped.data
        for k, v in (("qpos", data.qpos), ("qvel", data.qvel)):
            if k in info:
                info[k] = np.array(v, copy=True)
        return info

    def _write_transport(gidx, info):
        nonarray = {}
        for k, v in info.items():
            if k in transport:
                transport[k][0][gidx, 0] = v  # transport indexed by global idx, slot [gidx, 0]
            else:
                nonarray[k] = v
        return nonarray

    try:
        while True:
            cmd = conn.recv()
            tag = cmd[0]

            if tag == "close":
                break

            elif tag == "attach":
                specs = cmd[1]                        # cmd[2] (array_keys) unused by workers
                for k, (name, shape, dtype) in specs.items():
                    shm = shared_memory.SharedMemory(name=name)
                    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                    transport[k] = (arr, shm)
                conn.send(("attached",))

            elif tag == "reset":
                seeds, options, mask, want_full = cmd[1], cmd[2], cmd[3], cmd[4]
                full = [None] * len(shard)
                nonarray = {}
                for li, gidx in enumerate(shard):
                    if mask is not None and not mask[li]:
                        continue
                    _, info = envs[li].reset(seed=seeds[li], options=options[li])
                    if want_full:
                        full[li] = info
                    else:
                        nonarray[gidx] = _write_transport(gidx, info)
                if want_full:
                    conn.send(("full", full))
                else:
                    conn.send(("reset_res", nonarray))

            elif tag == "step":
                actions, mask = cmd[1], cmd[2]
                _apply_pending()
                n = len(shard)
                rewards = np.zeros(n, dtype=np.float32)
                terminateds = np.zeros(n, dtype=bool)
                truncateds = np.zeros(n, dtype=bool)
                nonarray = {}
                for li, gidx in enumerate(shard):
                    if mask is not None and not mask[li]:
                        continue
                    _, rewards[li], terminateds[li], truncateds[li], info = envs[
                        li
                    ].step(actions[li])
                    nonarray[gidx] = _write_transport(gidx, info)
                conn.send(("step_res", rewards, terminateds, truncateds, nonarray))

            elif tag == "callable":
                # buffer a set_state-style call; applied at the next step or at a refresh
                gidx, name, kwargs = cmd[1], cmd[2], cmd[3]
                pending.setdefault(gidx, []).append((name, kwargs))

            elif tag == "refresh":
                # apply the buffered callables now and re-observe, without stepping
                gidxs = cmd[1]
                installed = {g: list(pending.get(g, [])) for g in gidxs}
                _apply_pending()
                nonarray = {}
                for gidx in gidxs:
                    info = _reobserve(gidx, installed[gidx])
                    nonarray[gidx] = _write_transport(gidx, info)
                conn.send(("refresh_res", nonarray))

            else:  # pragma: no cover
                conn.send(("error", f"unknown cmd {tag}"))
    except Exception:
        import traceback
        try:
            conn.send(("error", traceback.format_exc()))  # surface the real cause to the parent
        except Exception:
            pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        for _, shm in transport.values():
            try:
                shm.close()
            except Exception:
                pass
        conn.close()


# ------------------------------------------------------------------------- main-process proxies
class _RemoteUnwrapped:
    """Stands in for ``env.unwrapped`` on the main process: attribute access returns a call that is
    buffered and flushed to the owning worker at the next ``step``. The worker guards with the real
    env's ``hasattr``, so missing methods are skipped as in the serial pool."""

    def __init__(self, pool, idx):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_idx", idx)

    def pool_refresh(self):
        """Apply this env's buffered callables and pull its fresh observation into the pool's
        stacked infos. A real method, so `__getattr__` never turns it into a remote call."""
        pool = object.__getattribute__(self, "_pool")
        idx = object.__getattribute__(self, "_idx")
        pool.refresh([idx])

    def __getattr__(self, name):
        pool = object.__getattribute__(self, "_pool")
        idx = object.__getattribute__(self, "_idx")

        def _call(**kwargs):
            pool._send_callable(idx, name, kwargs)

        return _call


class _RemoteEnv:
    """Stands in for an env object so ``World._evaluate_from_dataset`` can reach
    ``pool.envs[i].unwrapped``."""

    def __init__(self, pool, idx):
        self._pool = pool
        self._idx = idx

    @property
    def unwrapped(self):
        return _RemoteUnwrapped(self._pool, self._idx)


# --------------------------------------------------------------------------------- the pool
class ParallelEnvPool:
    """Process-parallel, shared-memory drop-in for ``EnvPool``. ``env_fns`` is one zero-arg env
    factory per env, shipped to and called inside the workers; ``num_procs`` defaults to one process
    per env (MuJoCo EGL contexts sharing a process return each other's buffers above two envs)."""

    def __init__(self, env_fns: list, num_procs: int | None = None):
        self._n = len(env_fns)
        self._p = max(1, min(int(self._n if num_procs is None else num_procs), self._n))
        if self._p < self._n:
            print(f"[parallel-envs] WARNING: {self._n} envs over {self._p} processes puts up to "
                  f"{-(-self._n // self._p)} MuJoCo contexts in one process; renders were only "
                  f"verified clean at one env per process", flush=True)

        # contiguous shards: env i -> worker holding it; slot i is preserved on reassembly.
        bounds = np.linspace(0, self._n, self._p + 1).astype(int)
        self._shards = [list(range(bounds[w], bounds[w + 1])) for w in range(self._p)]
        self._shards = [s for s in self._shards if s]  # drop empties (n < p edge case)
        self._p = len(self._shards)

        self.seeds = np.zeros(self._n, dtype=np.int64)
        self._stacked_infos: dict[str, Any] | None = None
        self._transport: dict[str, tuple[np.ndarray, Any]] = {}
        self._array_keys: list[str] = []
        self._closed = True  # set False once workers are up (so a failed __init__ is safe)

        # spawn: a forked worker would inherit this process's CUDA and EGL state
        ctx = mp.get_context("spawn")
        self._procs = []
        self._conns = []
        for w, shard in enumerate(self._shards):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main,
                args=(child_conn, shard, [env_fns[i] for i in shard], w == 0),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._procs.append(proc)
            self._conns.append(parent_conn)
        self._closed = False

        primary_spaces = None
        for w, conn in enumerate(self._conns):
            tag, payload = conn.recv()
            if tag == "error":
                raise RuntimeError(f"env worker {w} failed to start: {payload}")
            if w == 0:
                primary_spaces = payload

        single_action, single_obs, single_var = (
            _space_of(primary_spaces[0]),
            _space_of(primary_spaces[1]),
            _space_of(primary_spaces[2]),
        )
        self._single_action_space = single_action
        self._single_observation_space = single_obs
        self._variation_space = single_var
        self._action_space = batch_space(single_action, self._n)
        self._observation_space = batch_space(single_obs, self._n)

        # proxy env objects so callers can reach ``.envs[i].unwrapped``
        self.envs = [_RemoteEnv(self, i) for i in range(self._n)]

    # ----------------------------------------------------------------- parity properties
    @property
    def num_envs(self) -> int:
        return self._n

    @property
    def action_space(self):
        return self._action_space

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def single_observation_space(self):
        return self._single_observation_space

    @property
    def variation_space(self):
        return self._variation_space

    @property
    def single_variation_space(self):
        return self._variation_space

    # ----------------------------------------------------------------- callable buffering
    def _send_callable(self, idx, name, kwargs):
        self._conns[self._worker_of(idx)].send(("callable", idx, name, kwargs))

    def _worker_of(self, gidx):
        for w, shard in enumerate(self._shards):
            if shard[0] <= gidx <= shard[-1]:
                return w
        raise IndexError(gidx)

    # ----------------------------------------------------------------- shared-mem transport
    def _setup_transport(self):
        """Allocate SharedMemory transport blocks for every np.ndarray key of the freshly
        stacked info dict and have workers attach. Called once, after the first reset."""
        specs = {}
        self._array_keys = []
        for k, v in self._stacked_infos.items():
            if isinstance(v, np.ndarray):
                shm = shared_memory.SharedMemory(create=True, size=max(int(v.nbytes), 1))
                arr = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
                self._transport[k] = (arr, shm)
                specs[k] = (shm.name, v.shape, v.dtype.str)
                self._array_keys.append(k)
        for conn in self._conns:
            conn.send(("attach", specs, self._array_keys))
        for conn in self._conns:
            assert self._recv(conn)[0] == "attached"

    def _apply_from_transport(self, gidx, nonarray_for_env):
        """Reconstruct env ``gidx``'s info from the shared transport (array keys) + the pickled
        tail (non-array keys) and write it into ``_stacked_infos`` via ``_write_env_info``."""
        info = {k: self._transport[k][0][gidx, 0] for k in self._array_keys}
        if nonarray_for_env:
            info.update(nonarray_for_env)
        _write_env_info(self._stacked_infos, gidx, info)

    @staticmethod
    def _recv(conn):
        """recv that raises the worker's traceback instead of a cryptic EOFError when it died."""
        msg = conn.recv()
        if msg and msg[0] == "error":
            raise RuntimeError(f"env worker crashed:\n{msg[1]}")
        return msg

    # ----------------------------------------------------------------------------- reset
    def reset(self, seed=None, options=None, mask=None):
        seeds = _broadcast_arg(seed, self._n, increment=True)
        opts = _broadcast_arg(options, self._n)

        # mask=None (or no stacked dict yet) => full rebuild via _stack_fresh, matching serial.
        want_full = mask is None or self._stacked_infos is None

        for w, shard in enumerate(self._shards):
            sub_seeds = [seeds[i] for i in shard]
            sub_opts = [opts[i] for i in shard]
            sub_mask = None if mask is None else [mask[i] for i in shard]
            self._conns[w].send(("reset", sub_seeds, sub_opts, sub_mask, want_full))

        if want_full:
            per_env = [None] * self._n
            for w in range(self._p):
                tag, full = self._recv(self._conns[w])
                assert tag == "full"
                for li, gidx in enumerate(self._shards[w]):
                    per_env[gidx] = full[li]
            self._stacked_infos = _stack_fresh(per_env)
            if not self._transport:
                self._setup_transport()
            else:
                # refresh transport contents to match the rebuilt dict
                for k in self._array_keys:
                    self._transport[k][0][...] = self._stacked_infos[k]
        else:
            for w in range(self._p):
                tag, nonarray = self._recv(self._conns[w])
                assert tag == "reset_res"
                for gidx in self._shards[w]:
                    if mask is None or mask[gidx]:
                        self._apply_from_transport(gidx, nonarray.get(gidx))

        # mirror serial: record provided seeds for the (unmasked) envs
        for i in range(self._n):
            if mask is not None and not mask[i]:
                continue
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]

        return None, self._stacked_infos

    # --------------------------------------------------------------------------- refresh
    def refresh(self, idxs=None):
        """Re-observe the given envs in place, without stepping: `World._evaluate_from_dataset`
        snapshots the infos of the reset before the callables install the episode and never
        re-observes, so the policy's first context frame would otherwise be the pre-episode state."""
        assert self._stacked_infos is not None, "refresh() before the first reset()"
        idxs = list(range(self._n)) if idxs is None else list(idxs)
        per_worker: dict[int, list[int]] = {}
        for g in idxs:
            per_worker.setdefault(self._worker_of(g), []).append(g)
        for w, gs in per_worker.items():
            self._conns[w].send(("refresh", gs))
        for w, gs in per_worker.items():
            tag, nonarray = self._recv(self._conns[w])
            assert tag == "refresh_res"
            for g in gs:
                self._apply_from_transport(g, nonarray.get(g))
        return self._stacked_infos

    # ------------------------------------------------------------------------------ step
    def step(self, actions, mask=None):
        # eval.py's callable wrapper defers the post-install refresh until the last env of a reset;
        # an installed but never re-observed episode would have the policy act on the pre-episode state
        assert not self.__dict__.get("_deferred_refresh"), (
            f"{len(self._deferred_refresh)} envs had their episode installed but were never "
            f"refreshed (deferred-refresh queue not flushed before the first step)")
        rewards = np.zeros(self._n, dtype=np.float32)
        terminateds = np.zeros(self._n, dtype=bool)
        truncateds = np.zeros(self._n, dtype=bool)

        for w, shard in enumerate(self._shards):
            sub_act = [actions[i] for i in shard]
            sub_mask = None if mask is None else [mask[i] for i in shard]
            self._conns[w].send(("step", sub_act, sub_mask))

        for w in range(self._p):
            tag, r, t, tr, nonarray = self._recv(self._conns[w])
            assert tag == "step_res"
            for li, gidx in enumerate(self._shards[w]):
                rewards[gidx] = r[li]
                terminateds[gidx] = t[li]
                truncateds[gidx] = tr[li]
                if mask is None or mask[gidx]:
                    self._apply_from_transport(gidx, nonarray.get(gidx))

        return None, rewards, terminateds, truncateds, self._stacked_infos

    # ----------------------------------------------------------------------------- close
    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        for _, shm in self._transport.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._transport.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
