"""scripts/eval_cem.py, the protocol driver: the block width eval.py asserts equals the one the
driver reads off config.json, horizon = ceil(offset / action_block) reaches the goal, the eval
config and task-set family follow train_meta's dataset, `dataset.stats` follows the training file,
the override list composes against every eval config, the random floor launches, an indivisible
input_dim is refused, and the run record identifies files rather than names."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import pytest                                                                   # noqa: E402

from eval import _action_encoder_in_dim                                         # noqa: E402
from networks import Embedder                                                   # noqa: E402


def load_driver():
    spec = importlib.util.spec_from_file_location("eval_cem", REPO / "scripts" / "eval_cem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.n_tasks = lambda a, o: 100          # the task-set h5 is not the subject here
    return mod


@pytest.fixture(scope="module")
def drv():
    return load_driver()


class Args:
    def __init__(self, **kw):
        self.extra, self.meta = [], {}
        self.__dict__.update(kw)


def _pick(ds):
    """main()'s eval-config rule, transcribed."""
    return "cube" if ds.startswith("cube_play") else "calvin" if ds.startswith("calvin") else "scene"


def test_action_block_detection_agrees_with_the_module_eval_reads():
    for kw in (dict(input_dim=25, emb_dim=8), dict(input_dim=25, smoothed_dim=25, emb_dim=8),
               dict(input_dim=70, emb_dim=8)):
        assert _action_encoder_in_dim(Embedder(**kw)) == kw["input_dim"]


def test_horizon_reaches_the_goal_frame(drv):
    for ab in (3, 5, 10):
        for offset in (25, 50, 75, 100):
            h = -(-offset // ab)
            assert h * ab >= offset and h * ab - offset < ab
            ov = drv.overrides(Args(action_block=ab, arm="a", policy="p", eval_config="scene"), offset)
            assert f"plan_config.horizon={h}" in ov


def test_eval_config_and_taskset_selection(drv):
    cases = {"scene_play": "scene", "scene_play_mix_lateonset": "scene", "cube_play": "cube",
             "cube_play_mix_lateonset": "cube", "calvin_abc_d": "calvin",
             "calvin_abc_d_mix_lateonset": "calvin", "": "scene"}
    for ds, want in cases.items():
        assert _pick(ds) == want and want in drv.TASKSETS
    fam = {c: drv.taskset(Args(eval_config=c), 25) for c in drv.TASKSETS}
    assert fam == {"scene": "scene_offset_tasks_O25", "cube": "cube_offset_tasks_O25",
                   "calvin": "calvin_offset_tasks_O25"}, fam
    for c in drv.TASKSETS:
        assert c in drv.ACTION_DIM and (REPO / "config" / "eval" / f"{c}.yaml").is_file()


def test_stats_override_follows_the_training_file(drv):
    args = Args(action_block=5, arm="arm", policy="lewm/x/w.pt", eval_config="scene",
                meta={"dataset": "scene_play_mix_lateonset"})
    ov = drv.overrides(args, 25)
    assert "dataset.stats=scene_play_mix_lateonset" in ov, ov
    assert "plan_config.horizon=5" in ov and "plan_config.action_block=5" in ov
    assert any(o.startswith("+results_dir=") for o in ov)
    args.meta = {}
    assert not any(o.startswith("dataset.stats=") for o in drv.overrides(args, 25))


def test_overrides_compose(drv):
    from hydra import compose, initialize_config_dir
    from utils import load_env
    load_env()
    for name, ab in (("scene", 5), ("cube", 5), ("calvin", 10)):
        args = Args(action_block=ab, arm="arm", policy="lewm/x/w.pt", eval_config=name,
                    meta={"dataset": "x_play"})
        ov = drv.overrides(args, 25)
        with initialize_config_dir(version_base=None, config_dir=str(REPO / "config" / "eval")):
            cfg = compose(config_name=name, overrides=ov)
        assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
        assert int(cfg.world.num_envs) == int(cfg.eval.num_eval)
        assert cfg.dataset.stats == "x_play" and cfg.policy == "lewm/x/w.pt"


def test_parse_sr(drv):
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "O25.log"
        log.write_text("noise\n{'success_rate': 45.0, 'episode_successes': array([True])}\nbye\n")
        assert drv.parse_sr(log) == 45.0
        log.write_text("{'success_rate': 12.0}\n{'success_rate': 45.0}\n")
        assert drv.parse_sr(log) == 45.0, "parse_sr must take the LAST success_rate"
        assert drv.parse_sr(Path(td) / "absent.log") is None


def test_random_floor_is_launchable(drv):
    for name in ("scene", "cube", "calvin"):
        ab = drv.action_block("random", name)
        assert ab == drv.config_action_block(name) > 0
        args = Args(action_block=ab, arm="floor", policy="random", eval_config=name)
        for offset in (25, 50, 75, 100):
            ov = drv.overrides(args, offset)
            h = int(next(o for o in ov if o.startswith("plan_config.horizon=")).split("=")[1])
            budget = int(next(o for o in ov if o.startswith("eval.eval_budget=")).split("=")[1])
            assert h * ab <= budget, f"{name} O{offset}: eval.py's launch assert would reject the floor"


def test_action_block_refuses_an_indivisible_input_dim(drv):
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "checkpoints" / "lewm" / "x"
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"action_encoder": {"input_dim": 23}}))
        (run / "w.pt").write_bytes(b"")
        old = drv.STABLEWM_HOME
        drv.STABLEWM_HOME = Path(td)
        try:
            drv.action_block("lewm/x/w.pt", "scene")
        except AssertionError as e:
            assert "not a multiple" in str(e), str(e)
        else:
            raise AssertionError("an indivisible input_dim was floor-divided")
        finally:
            drv.STABLEWM_HOME = old


def test_run_meta_records_what_was_read(drv):
    d = drv.data_id("scene_offset_tasks_O25")
    assert d["name"] == "scene_offset_tasks_O25" and "path" in d
    if not d.get("missing"):
        assert d["bytes"] > 0 and "mtime" in d
    assert drv.data_id("definitely_not_a_dataset_xyz").get("missing") is True


def test_protocol_change_on_an_existing_arm_is_refused(drv):
    prev = {"protocol": "ICLR-scene-cem-v2", "ckpt_sha256_16": "abc", "eval_config": "scene", "action_block": 5}
    new = {"protocol": drv.PROTOCOL, "ckpt_sha256_16": "abc", "eval_config": "scene", "action_block": 5}
    clashes = [k for k in ("protocol", "ckpt_sha256_16", "eval_config", "action_block") if prev[k] != new[k]]
    assert clashes == ["protocol"] and drv.PROTOCOL != "ICLR-scene-cem-v2"


if __name__ == "__main__":
    drv = load_driver()
    test_action_block_detection_agrees_with_the_module_eval_reads()
    test_horizon_reaches_the_goal_frame(drv)
    test_eval_config_and_taskset_selection(drv)
    test_stats_override_follows_the_training_file(drv)
    test_overrides_compose(drv)
    test_parse_sr(drv)
    test_random_floor_is_launchable(drv)
    test_action_block_refuses_an_indivisible_input_dim(drv)
    test_run_meta_records_what_was_read(drv)
    test_protocol_change_on_an_existing_arm_is_refused(drv)
    print("test_cem_driver: OK")
