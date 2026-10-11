"""The 25-d action block the planner scores must be the 25 numbers the env executes, in order.

Three paths have to agree on ONE layout and nothing checks it today:

  * TRAINING assembles a block by reading `frameskip` raw rows and letting
    `swm.data.Dataset.__getitem__` reshape (span, A) -> (num_steps, frameskip*A). The z-score is
    applied by the loader BEFORE that reshape (RowReadHDF5Dataset._load_slice returns the raw span),
    with per-CHANNEL stats -- so the block is time-major and each channel carries its own scale.
  * The SOLVER samples candidates of width `action_dim * action_block` and the planner cost feeds
    them to the action encoder whole.
  * The POLICY takes the first `receding_horizon` blocks, reshapes to (rh*action_block, A) and
    inverse-transforms each row with the per-channel scaler.

A channel-major block would pass every shape check in all three and mis-assign both the substep and
the scaler column. Labelled integers make any reordering visible.

  PYTHONPATH=src .venv/bin/python tests/adv_cem_action_layout.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np                                                              # noqa: E402
import torch                                                                    # noqa: E402
from sklearn import preprocessing                                               # noqa: E402

import stable_worldmodel as swm                                                 # noqa: E402
from utils import ZScoreNormalizer                                              # noqa: E402

A, FS, NSTEPS = 5, 5, 4          # action dim, frameskip, clip steps
HOR, RH = 5, 3                   # plan horizon (blocks), receding horizon (blocks)


def labelled_actions(n):
    """a[t, c] = 100*t + c -- every raw number names its own (frame, channel)."""
    return np.array([[100.0 * t + c for c in range(A)] for t in range(n)], dtype=np.float64)


def _train_block():
    """Exactly what swm.data.Dataset.__getitem__ does to _load_slice's (span, A) tensor."""
    return torch.from_numpy(labelled_actions(NSTEPS * FS)).reshape(NSTEPS, -1).numpy()


def test_train_block_is_time_major():
    raw = labelled_actions(NSTEPS * FS)
    block = _train_block()
    assert block.shape == (NSTEPS, FS * A)
    for t in range(NSTEPS):
        for s in range(FS):
            got = block[t, s * A:(s + 1) * A]
            want = raw[t * FS + s]
            assert np.array_equal(got, want), (
                f"train block[{t}] substep {s} = {got}, expected raw frame {t*FS+s} = {want}; "
                f"the block is not time-major")
    print("[train] block layout time-major, substep-major within the block: OK")


def test_eval_reshape_inverts_the_train_block():
    """swm.policy.WorldModelPolicy: plan[:, :RH].reshape(n, RH*action_block, -1)."""
    plan = torch.from_numpy(_train_block()[None, :RH])       # (1, RH, FS*A)
    flat = plan.reshape(1, RH * FS, -1).numpy()[0]           # (RH*FS, A)
    raw = labelled_actions(NSTEPS * FS)
    assert np.array_equal(flat, raw[: RH * FS]), (
        f"the executed prefix is not the first {RH*FS} raw actions of the plan:\n{flat[:3]}")
    print(f"[eval ] plan[:, :{RH}].reshape recovers raw frames 0..{RH*FS-1} in order: OK")


def test_zscore_roundtrip_train_vs_eval():
    """Train z-scores the (span, A) rows with per-channel stats and reshapes afterwards.
    Eval fits a per-channel StandardScaler on the same rows and inverse-transforms per row.
    Composition must be the identity on the raw numbers."""
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(4000, A)) * np.array([0.43, 0.46, 0.57, 0.57, 0.26]) + \
        np.array([0.016, 0.0, 0.05, 0.034, 0.08])

    sc = preprocessing.StandardScaler().fit(corpus)
    tmean = torch.from_numpy(corpus.mean(0, keepdims=True))
    tstd = torch.from_numpy(corpus.std(0, ddof=1, keepdims=True))   # torch.std default ddof=1
    norm = ZScoreNormalizer(tmean, tstd)

    raw = labelled_actions(NSTEPS * FS) * 0.001                     # keep the scale sane
    z_train = norm(torch.from_numpy(raw)).reshape(NSTEPS, -1).numpy()   # train: normalise, reshape
    # eval: the solver's candidate lives in this z-space; the policy reshapes and inverse-transforms
    back = sc.inverse_transform(torch.from_numpy(z_train)[:RH].reshape(RH * FS, A).numpy())
    err = np.abs(back - raw[: RH * FS]).max()
    # ddof 0 vs 1 on 4000 rows is a 1.25e-4 relative scale difference; the layout error it is
    # guarding against is order-1
    rel = err / max(np.abs(raw[: RH * FS]).max(), 1e-9)
    assert rel < 1e-3, (
        f"train-normalise -> block -> eval-inverse-transform is not the identity: max abs err "
        f"{err:.6g} (rel {rel:.3g}); a channel/substep transposition would land here")
    print(f"[both ] z-score round trip max |err| {err:.3e} (rel {rel:.2e}, ddof 0 vs 1 only): OK")


def test_policy_executes_the_scored_prefix():
    """End to end through the real swm.policy.WorldModelPolicy with a stub solver: the actions
    handed to the env must be the plan's first RH*FS raw rows, in order, once per step."""

    class StubSolver:
        def __init__(self, plan):
            self.plan = plan                      # (1, HOR, FS*A) in z-space
            self.calls = 0
            self.last_init = None

        def configure(self, *, action_space, n_envs, config):
            self.cfg = config
            self._n_envs = n_envs

        @property
        def action_dim(self):
            return A * FS

        @property
        def n_envs(self):
            return self._n_envs

        @property
        def horizon(self):
            return HOR

        def solve(self, info_dict, init_action=None):
            self.calls += 1
            self.last_init = init_action
            return {"actions": self.plan.clone()}

        def __call__(self, info, init_action=None):
            return self.solve(info, init_action)

    class StubEnv:
        num_envs = 1
        class _S:
            shape = (A,)
        single_action_space = _S()
        class _AS:
            shape = (1, A)
        action_space = _AS()

    raw = labelled_actions(HOR * FS) * 0.001
    rng = np.random.default_rng(1)
    corpus = rng.normal(size=(4000, A))
    sc = preprocessing.StandardScaler().fit(corpus)
    z = torch.from_numpy(sc.transform(raw)).reshape(1, HOR, FS * A).float()

    solver = StubSolver(z)
    cfg = swm.policy.PlanConfig(horizon=HOR, receding_horizon=RH, action_block=FS)
    pol = swm.policy.WorldModelPolicy(solver=solver, config=cfg,
                                      process={"action": sc}, transform={})
    # swm asserts the solver satisfies its Protocol; StubSolver has configure + __call__
    pol.set_env(StubEnv())

    got = []
    for _ in range(RH * FS):
        got.append(pol.get_action({"terminated": np.zeros((1, 1), dtype=bool)})[0].copy())
    got = np.stack(got)
    assert solver.calls == 1, f"solver replanned {solver.calls} times inside one receding horizon"
    err = np.abs(got - raw[: RH * FS]).max()
    assert err < 1e-4, (
        f"the env received actions that are not the plan's first {RH*FS} raw rows; max |err| "
        f"{err:.4g}\nfirst executed: {got[0]}\nfirst planned : {raw[0]}")
    print(f"[loop ] {RH*FS} executed actions == plan prefix, max |err| {err:.2e}: OK")

    # one more step forces a replan; the warm start must be the plan's UNEXECUTED tail
    pol.get_action({"terminated": np.zeros((1, 1), dtype=bool)})
    assert solver.calls == 2
    init = solver.last_init
    assert init is not None and init.shape == (1, HOR - RH, FS * A), (
        f"warm start shape {None if init is None else tuple(init.shape)} != "
        f"(1, {HOR-RH}, {FS*A})")
    assert torch.allclose(init, z[:, RH:], atol=1e-6), (
        "the warm start is not the plan's unexecuted tail (a strided or reversed slice)")
    print(f"[loop ] warm start == plan blocks {RH}..{HOR-1}: OK")


if __name__ == "__main__":
    test_train_block_is_time_major()
    test_eval_reshape_inverts_the_train_block()
    test_zscore_roundtrip_train_vs_eval()
    test_policy_executes_the_scored_prefix()
    print("test_cem_action_layout: OK")
