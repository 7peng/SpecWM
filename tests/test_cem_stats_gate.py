"""The planner's action scaler comes from the checkpoint's training file (`dataset.stats`) and
is never the identity by accident: a config without `dataset.stats` is refused, a constant action channel
is clamped as training clamps it and a mostly constant file is refused, a real file round-trips, the task sets'
own action column is identically zero, and every shipped eval config declares `dataset.stats`."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py                                                                     # noqa: E402
import hdf5plugin  # noqa: F401,E402
import numpy as np                                                              # noqa: E402
from omegaconf import OmegaConf                                                 # noqa: E402
from sklearn import preprocessing                                               # noqa: E402

import eval as E                                                                # noqa: E402


class _DS:
    def __init__(self, action):
        self._a = action
        self.column_names = ["action", "ep_idx", "step_idx"]

    def get_col_data(self, col):
        if col == "action":
            return self._a
        return np.zeros((len(self._a), 1))


def _ctx(stats_value, action):
    cfg = OmegaConf.create({
        "world": {}, "dataset": {"keys_to_cache": ["action"]},
        "eval": {"img_size": 224, "dataset_name": "scene_offset_tasks_O25"},
    })
    if stats_value is not None:
        cfg.dataset.stats = stats_value
    saved_world, saved_get = E.swm.World, E.get_dataset
    E.swm.World = lambda **kw: object()
    E.get_dataset = lambda c, name: _DS(action)
    try:
        return E.build_eval_context(cfg)
    finally:
        E.swm.World, E.get_dataset = saved_world, saved_get


def test_sklearn_makes_a_constant_column_the_identity():
    sc = preprocessing.StandardScaler().fit(np.zeros((64, 5), np.float32))
    assert list(sc.scale_) == [1.0] * 5, sc.scale_
    z = np.array([[1.5, -2.0, 0.3, 3.0, -1.0]])
    assert np.array_equal(sc.inverse_transform(z), z), (
        "premise broken: a constant-column scaler is no longer the identity")
    print("[stats] sklearn fits scale_=1 on a constant column, so inverse_transform is the "
          "identity and z-scores would be executed raw: mechanism confirmed")


def test_task_sets_carry_a_zero_action_column():
    from stable_worldmodel.data.utils import get_cache_dir
    root = get_cache_dir(sub_folder="datasets")
    seen = 0
    for name in ("scene_offset_tasks_O25", "cube_offset_tasks_O25"):
        p = root / f"{name}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            a = f["action"][:]
        assert np.abs(a).max() == 0.0, f"{name}: action column is no longer zero ({np.abs(a).max()})"
        seen += 1
        print(f"[stats] {name}.h5 action column is identically zero ({a.shape}), so it can never "
              f"serve as dataset.stats")
    if not seen:
        print(f"[stats] no task set under {root}; skipped the zero-column check")


def test_missing_stats_is_refused():
    try:
        _ctx(None, np.random.default_rng(0).normal(size=(64, 5)))
    except AssertionError as e:
        assert "dataset.stats" in str(e), str(e)
        print("[stats] a config without dataset.stats is refused instead of falling back to the "
              "task set: OK")
        return
    raise AssertionError("build_eval_context accepted a config with no dataset.stats")


def test_constant_action_channel_is_clamped_not_identity():
    a = np.random.default_rng(0).normal(size=(64, 5))
    a[:, 3] = 0.0
    sc = _ctx("some_file", a)["process"]["action"]
    assert abs(float(sc.scale_[3]) - 1e-6) < 1e-18, sc.scale_
    assert np.abs(np.asarray(sc.scale_)[[0, 1, 2, 4]] - a.std(0)[[0, 1, 2, 4]]).max() < 1e-9
    print("[stats] a constant action channel is clamped to std 1e-6, as training's normaliser: OK")


def test_wholly_constant_stats_file_is_refused():
    a = np.random.default_rng(0).normal(size=(64, 5))
    a[:, :3] = 0.0
    try:
        _ctx("some_file", a)
    except AssertionError as e:
        assert "constant" in str(e) and "[0, 1, 2]" in str(e), str(e)
        print("[stats] a stats file with mostly constant action channels is refused and named: OK")
        return
    raise AssertionError("build_eval_context accepted a stats file whose action column is constant")


def test_good_stats_pass_and_round_trip():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4096, 5)) * np.array([0.43, 0.46, 0.57, 0.57, 0.26])
    ctx = _ctx("scene_play", a)
    sc = ctx["process"]["action"]
    z = rng.normal(size=(7, 5))
    back = sc.inverse_transform(z)
    assert np.abs(sc.transform(back) - z).max() < 1e-9
    assert np.abs(np.asarray(sc.scale_) - a.std(0)).max() < 1e-9
    print(f"[stats] a real stats file fits std {np.round(sc.scale_, 3).tolist()} and round-trips: OK")


def test_every_shipped_eval_config_declares_stats():
    for p in sorted((REPO / "config" / "eval").glob("*.yaml")):
        cfg = OmegaConf.load(p)
        if "dataset" not in cfg or "keys_to_cache" not in (cfg.get("dataset") or {}):
            continue
        assert cfg.dataset.get("stats"), f"{p.name} caches columns but declares no dataset.stats"
    print("[stats] every shipped eval config that caches columns declares dataset.stats: OK")


if __name__ == "__main__":
    test_sklearn_makes_a_constant_column_the_identity()
    test_task_sets_carry_a_zero_action_column()
    test_missing_stats_is_refused()
    test_constant_action_channel_is_clamped_not_identity()
    test_wholly_constant_stats_file_is_refused()
    test_good_stats_pass_and_round_trip()
    test_every_shipped_eval_config_declares_stats()
    print("test_cem_stats_gate: OK")
