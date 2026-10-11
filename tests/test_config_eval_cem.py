"""scripts/eval_cem.py against config/eval/*.yaml: every override the driver emits names a key
the eval config declares, or carries the `+` prefix for one it does not; `action_block` is the
checkpoint's `action_encoder.input_dim` over the env's action width, read from config.json;
`train_meta` refuses a checkpoint without train_meta.json unless `--dataset` is given; missing
task sets are named before any GPU work."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_driver():
    spec = importlib.util.spec_from_file_location("eval_cem", ROOT / "scripts" / "eval_cem.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.n_tasks = lambda a, o: 100
    return m


@pytest.fixture(scope="module")
def drv():
    return load_driver()


class Args:
    def __init__(self, **kw):
        self.arm, self.policy, self.extra = "test", "lewm/x/w.pt", []
        self.meta = {"dataset": "x_play"}
        self.__dict__.update(kw)


def _eval_cfg(name):
    from hydra import compose, initialize_config_dir
    from utils import load_env
    load_env()
    with initialize_config_dir(config_dir=str(ROOT / "config" / "eval"), version_base=None):
        return compose(config_name=name, overrides=["policy=x"])


def test_overrides_name_declared_keys_or_add_with_plus(drv):
    from omegaconf import OmegaConf
    for name, ab in (("scene", 5), ("cube", 5), ("calvin", 10)):
        cfg = _eval_cfg(name)
        for ov in drv.overrides(Args(eval_config=name, action_block=ab), 25):
            key = ov.split("=", 1)[0]
            add = key.startswith("+")
            key = key.lstrip("+")
            parent, _, leaf = key.rpartition(".")
            node = OmegaConf.select(cfg, parent) if parent else cfg
            declared = node is not None and leaf in node
            assert declared != add, (name, ov, "declared" if declared else "undeclared")


def test_action_block_from_config_json(drv):
    with tempfile.TemporaryDirectory() as td:
        old = drv.STABLEWM_HOME
        drv.STABLEWM_HOME = Path(td)
        try:
            for env, in_dim, want in (("scene", 25, 5), ("cube", 15, 3), ("calvin", 70, 10)):
                run = Path(td) / "checkpoints" / "lewm" / env
                run.mkdir(parents=True)
                (run / "config.json").write_text(json.dumps({"action_encoder": {"input_dim": in_dim}}))
                assert drv.action_block(f"lewm/{env}/w.pt", env) == want
            with pytest.raises(SystemExit, match="config.json"):
                drv.action_block("lewm/absent/w.pt", "scene")
        finally:
            drv.STABLEWM_HOME = old


def test_train_meta_refuses_a_checkpoint_without_the_file(drv):
    with tempfile.TemporaryDirectory() as td:
        old = drv.STABLEWM_HOME
        drv.STABLEWM_HOME = Path(td)
        try:
            run = Path(td) / "checkpoints" / "lewm" / "r"
            run.mkdir(parents=True)
            with pytest.raises(SystemExit, match="train_meta.json"):
                drv.train_meta("lewm/r/w.pt")
            assert drv.train_meta("lewm/r/w.pt", dataset="scene_play")["dataset"] == "scene_play"
            (run / "train_meta.json").write_text(json.dumps({"dataset": "cube_play", "frameskip": 5}))
            m = drv.train_meta("lewm/r/w.pt")
            assert m["dataset"] == "cube_play" and m["frameskip"] == 5
            assert drv.train_meta("lewm/r/w.pt", dataset="other")["dataset"] == "other"
        finally:
            drv.STABLEWM_HOME = old


def test_missing_task_sets_are_named(drv):
    with tempfile.TemporaryDirectory() as td:
        old = drv.STABLEWM_HOME
        drv.STABLEWM_HOME = Path(td)
        try:
            (Path(td) / "datasets").mkdir()
            (Path(td) / "datasets" / "cube_offset_tasks_O25.h5").write_bytes(b"x")
            args = Args(eval_config="cube", action_block=5)
            assert drv.missing_tasksets(args, [25]) == []
            miss = [m.split("/")[-1] for m in drv.missing_tasksets(args, [25, 50, 75, 100])]
            assert miss == ["cube_offset_tasks_O50.h5", "cube_offset_tasks_O75.h5", "cube_offset_tasks_O100.h5"]
        finally:
            drv.STABLEWM_HOME = old


if __name__ == "__main__":
    d = load_driver()
    test_overrides_name_declared_keys_or_add_with_plus(d)
    test_action_block_from_config_json(d)
    test_train_meta_refuses_a_checkpoint_without_the_file(d)
    test_missing_task_sets_are_named(d)
    print("test_config_eval_cem: OK")
