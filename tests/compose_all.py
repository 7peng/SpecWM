"""Compose every train config in one process and report which ones break.

One process, not one per config: the entry points each pay a fresh torch import, which turns a
two-minute check into an hours-long one. `utils.load_env()` runs once here for the same reason --
it is what makes the `${oc.env:...}` interpolations resolvable.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils import load_env

load_env()
cfg_dir = ROOT / "config" / "train"
names = sorted(p.stem for p in cfg_dir.glob("*.yaml") if p.stem != "common")
ok, bad = [], []
with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
    for n in names:
        try:
            OmegaConf.resolve(compose(config_name=n))
            ok.append(n)
        except Exception as e:
            bad.append((n, f"{type(e).__name__}: {e}".replace("\n", " ")[:160]))
for n, e in bad:
    print(f"FAIL {n} :: {e}")
print(f"composed {len(ok)} ok, {len(bad)} failed, of {len(names)}")
sys.exit(1 if bad else 0)
