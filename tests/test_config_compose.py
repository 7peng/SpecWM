"""Every recipe under config/train composes with `run_id=x` and resolves, and the resolved trees
hold the cross-recipe invariants: `action_encoder.input_dim` left to the runtime, `smoothed_dim`
equal to frameskip * 5 on scene/cube, factor teacher names declared under `loss.spectral.teachers`,
teacher specs present under config/teachers, `far_frames` on CALVIN recipes only, raw columns and
factor keys loaded, the clip geometry consistent with the predictor."""
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CFG_DIR = ROOT / "config" / "train"


@lru_cache(maxsize=None)
def composed():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from utils import load_env
    load_env()
    for v in ("SCENE_STAGE_DIR", "CUBE_STAGE_DIR", "CALVIN_STAGE_DIR", "OGBENCH_PACK_DIR",
              "CALVIN_PROBE_PACK_DIR", "STABLEWM_HOME"):
        os.environ.setdefault(v, "/tmp/nonexistent")
    ok, bad = {}, {}
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        for p in sorted(CFG_DIR.glob("*.yaml")):
            try:
                c = compose(config_name=p.stem, overrides=["run_id=x", "wandb.enabled=false"])
                ok[p.stem] = OmegaConf.to_container(c, resolve=True)
            except Exception as e:  # noqa: BLE001
                bad[p.stem] = f"{type(e).__name__}: {e}".replace("\n", " ")[:200]
    return ok, bad


def test_every_recipe_composes():
    ok, bad = composed()
    assert ok and not bad, "recipes that fail to compose:\n" + "\n".join(f"  {n}: {e}" for n, e in sorted(bad.items()))


def test_action_encoder_input_dim_is_left_to_the_runtime():
    ok, _ = composed()
    hard = {n: c["model"]["action_encoder"].get("input_dim") for n, c in ok.items()
            if c["model"]["action_encoder"].get("input_dim") not in ("???", None)}
    assert not hard, f"model.action_encoder.input_dim is set at runtime; literal in {hard}"


def test_smoothed_dim_matches_frameskip_times_action_dim():
    ok, _ = composed()
    bad = []
    for n, c in sorted(ok.items()):
        ds, sd = c["data"]["dataset"], c["model"]["action_encoder"].get("smoothed_dim")
        if sd is None or not str(ds.get("name", "")).startswith(("scene_play", "cube_play")):
            continue
        if int(sd) != int(ds["frameskip"]) * 5:
            bad.append(f"{n}: smoothed_dim={sd}, frameskip*5={int(ds['frameskip']) * 5}")
    assert not bad, "\n".join(bad)


def test_factor_teachers_and_specs_exist():
    ok, _ = composed()
    specs = {p.stem for p in (ROOT / "config" / "teachers").glob("*.yaml")}
    bad = []
    for n, c in sorted(ok.items()):
        sd = (c.get("loss") or {}).get("spectral") or {}
        teachers = {k: v for k, v in (sd.get("teachers") or {}).items() if v is not None}
        for f in sd.get("factors") or []:
            if f.get("teacher") is not None and f["teacher"] not in teachers:
                bad.append(f"{n}: factor teacher {f['teacher']!r} not in teachers {sorted(teachers)}")
        for k, v in teachers.items():
            if str(v.get("spec")) not in specs:
                bad.append(f"{n}: teacher {k} names spec {v.get('spec')!r}, no such file")
    assert not bad, "\n".join(bad)


def test_far_frames_only_on_calvin():
    ok, _ = composed()
    bad = [n for n, c in ok.items()
           if int(c["data"]["dataset"].get("far_frames", 0) or 0) and not n.startswith("calvin")]
    assert not bad, f"far_frames outside CALVIN: {bad}"
    for n, c in ok.items():
        if n.startswith("calvin") and (c.get("loss") or {}).get("spectral"):
            assert int(c["data"]["dataset"].get("far_frames", 0) or 0) >= 0


def test_loaded_columns_cover_raw_columns_and_factor_keys():
    ok, _ = composed()
    bad = []
    for n, c in sorted(ok.items()):
        ds = c["data"]["dataset"]
        loaded = set(ds.get("keys_to_load") or [])
        for col in ds.get("raw_columns") or []:
            if col not in loaded:
                bad.append(f"{n}: raw_columns names {col!r}, not in keys_to_load")
        sd = (c.get("loss") or {}).get("spectral") or {}
        for f in sd.get("factors") or []:
            if str(f.get("form")) in ("temporal", "quasimetric", "quasimetric_cached"):
                continue
            if f.get("key") not in loaded:
                bad.append(f"{n}: factor key {f.get('key')!r} not in keys_to_load")
            if f.get("key") not in set(ds.get("raw_columns") or []):
                bad.append(f"{n}: factor key {f.get('key')!r} not in raw_columns")
    assert not bad, "\n".join(bad)


def test_predictor_geometry_is_consistent():
    ok, _ = composed()
    bad = []
    for n, c in sorted(ok.items()):
        wm, pr, ds = c["wm"], c["model"]["predictor"], c["data"]["dataset"]
        if int(ds["num_steps"]) != int(wm["num_preds"]) + int(wm["history_size"]):
            bad.append(f"{n}: num_steps != num_preds + history_size")
        if pr.get("num_frames") != wm.get("history_size"):
            bad.append(f"{n}: predictor.num_frames != wm.history_size")
        if "ARPredictor" in str(pr.get("_target_")) and int(wm.get("max_pred_horizon", 1)) != 1:
            bad.append(f"{n}: ARPredictor with max_pred_horizon {wm.get('max_pred_horizon')}")
        if "HorizonPredictor" in str(pr.get("_target_")):
            if int(pr.get("max_horizon") or 0) < min(int(wm["max_pred_horizon"]), int(wm["num_preds"])):
                bad.append(f"{n}: predictor.max_horizon below the multistart's horizon")
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    test_every_recipe_composes()
    test_action_encoder_input_dim_is_left_to_the_runtime()
    test_smoothed_dim_matches_frameskip_times_action_dim()
    test_factor_teachers_and_specs_exist()
    test_far_frames_only_on_calvin()
    test_loaded_columns_cover_raw_columns_and_factor_keys()
    test_predictor_geometry_is_consistent()
    print(f"test_config_compose: {len(composed()[0])} recipes OK")
