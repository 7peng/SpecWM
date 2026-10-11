"""Write the eval-side config.json (the resolved `model` tree) for a run from its hydra config.

train_common writes config.json at epoch end, so a run scored from step snapshots has none until
then and scripts/eval_cem.py cannot read its action block. This writes the same tree from
outputs/<date>/<time>/.hydra/config.yaml with action_encoder.input_dim filled the way train_common
fills it at runtime (frameskip x the 7-d CALVIN / 5-d OGBench action, by data name).

    python scripts/write_config_json.py <hydra_dir> <ckpt_dir>
"""
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
h, out = Path(sys.argv[1]), Path(sys.argv[2]) / "config.json"
if out.exists():
    sys.exit(0)
cfg = OmegaConf.load(h / "config.yaml")
OmegaConf.set_struct(cfg, False)
m = OmegaConf.to_container(cfg.model, resolve=True)
name = str(cfg.data.dataset.name)
adim = 7 if name.startswith("calvin") else 5
if m["action_encoder"].get("input_dim") in ("???", None):
    m["action_encoder"]["input_dim"] = int(cfg.data.dataset.frameskip) * adim
tmp = out.with_suffix(".json.tmp")
json.dump(m, open(tmp, "w"), indent=2)
tmp.replace(out)
print(f"wrote {out}: {m['predictor']['_target_']}, action input_dim {m['action_encoder']['input_dim']}")
