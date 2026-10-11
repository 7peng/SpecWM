#!/usr/bin/env python
"""Export planner-loadable weights from a run's step snapshots.

    python scripts/export_step_weights.py RUN_ID COMPOSED_CFG_YAML STEP [STEP ...]

Writes $STABLEWM_HOME/checkpoints/lewm/RUN_ID/config.json (the resolved `model` block, with
action_encoder.input_dim = frameskip * action_dim) and weights_step_<STEP>.pt (the `model` state
dict of step_snaps/step_<STEP>.pt), the layout scripts/eval_cem.py and swm's load_pretrained read.
"""
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

ACTION_DIM = 5

run, cfg_path, steps = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
cfg = OmegaConf.load(cfg_path)
model = OmegaConf.to_container(cfg.model, resolve=True)
d = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / "lewm" / run
if model["action_encoder"].get("input_dim") in (None, "???"):
    # the width the checkpoint was trained with, read off its own action-encoder weight; the
    # scene constant is the fallback for a snapshot without one
    first = torch.load(d / "step_snaps" / f"step_{steps[0]}.pt", map_location="cpu", weights_only=False)["model"]
    w = next((v for k, v in first.items() if k.startswith("action_encoder.") and v.ndim >= 2), None)
    model["action_encoder"]["input_dim"] = int(w.shape[-2] if w.ndim == 3 else w.shape[-1]) if w is not None \
        else int(cfg.data.dataset.frameskip) * ACTION_DIM
(d / "config.json").write_text(json.dumps(model, indent=2))
for s in steps:
    blob = torch.load(d / "step_snaps" / f"step_{s}.pt", map_location="cpu", weights_only=False)
    torch.save(blob["model"], d / f"weights_step_{s}.pt")
    print(f"weights_step_{s}.pt ({len(blob['model'])} tensors, blob step {blob.get('step')})")
print(d / "config.json")
