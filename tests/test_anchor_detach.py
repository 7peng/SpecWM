"""`loss.spectral.anchor.stop_gradient`: the prediction anchor stops reaching the encoder through
the target row while the predictor still trains on it and the kernel term is untouched; the
override composes on every temporal recipe and changes that one key only."""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
for v in ("SCENE_STAGE_DIR", "CUBE_STAGE_DIR", "CALVIN_STAGE_DIR", "OGBENCH_PACK_DIR",
          "CALVIN_PROBE_PACK_DIR", "STABLEWM_HOME"):
    os.environ.setdefault(v, "/tmp/nonexistent")

import train_spec as T  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from tdwm_harness import StubModule, labelled_emb, make_batch  # noqa: E402

OVERRIDE = "loss.spectral.anchor.stop_gradient=true"
B, D = 3, 12


def _compose(name, *ov):
    with initialize_config_dir(config_dir=str(ROOT / "config" / "train"), version_base=None):
        return compose(config_name=name, overrides=["run_id=test", *ov])


RAW = _compose("scene_temporal")
N = int(RAW.data.dataset.num_steps)


class FreePredictor(nn.Module):
    """Independent of the encoder: the only door the anchor has into the encoder is the target row."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.w = nn.Parameter(torch.zeros(()))
        self.register_buffer("noise", torch.randn(B * N, D, generator=torch.Generator().manual_seed(3)))

    def forward(self, ce, be, hor, amask):
        return self.lin(self.noise[: ce.size(0)]) + self.w


def build(seed=0, **over):
    torch.manual_seed(seed)
    emb = labelled_emb(B, N, D, scale=0.4).clone().requires_grad_(True)
    act = torch.randn(B, N, D, requires_grad=True)
    c = T._PlainCfg.wrap(OmegaConf.to_container(RAW, resolve=True))
    for k, v in over.items():
        (c["loss"]["spectral"]["anchor"] if k == "stop_gradient" else c["loss"]["spectral"])[k] = v
    mod = StubModule(c, emb, act, predictor=FreePredictor())
    return mod, emb


def run(**over):
    mod, emb = build(**over)
    out = mod(make_batch(B, N))
    out["loss"].backward()
    return mod, emb, out


def test_recipe_has_a_live_anchor():
    sd = RAW.loss.spectral
    assert float(sd.anchor.weight) > 0 and not bool(sd.anchor.stop_gradient)


def test_stop_gradient_cuts_the_encoder_and_keeps_the_predictor():
    _, emb_on, out_on = run(weight=0.0, stop_gradient=True)          # anchor alone
    assert float(out_on["pred_anchor_loss"]) > 0
    assert emb_on.grad is None or float(emb_on.grad.abs().max()) == 0.0
    mod_off, emb_off, out_off = run(weight=0.0)
    assert float(emb_off.grad.abs().max()) > 0, "the attached anchor moves the encoder"
    mod_on, _, _ = run(weight=0.0, stop_gradient=True)
    g_on, g_off = mod_on.model.predictor.w.grad, mod_off.model.predictor.w.grad
    assert g_on is not None and float(g_on.abs()) > 0
    assert torch.allclose(g_on, g_off, atol=1e-6), (float(g_on), float(g_off))


def test_stop_gradient_leaves_the_values_and_the_kernel_gradient_alone():
    _, emb_on, out_on = run(stop_gradient=True)
    _, emb_off, out_off = run()
    for k in ("spec_loss", "pred_anchor_loss", "loss"):
        assert torch.allclose(out_on[k], out_off[k]), k
    _, emb_kernel, _ = run(anchor={"weight": 0.0})                    # kernel term alone
    assert torch.allclose(emb_on.grad, emb_kernel.grad, atol=1e-7, rtol=0), (
        f"max diff {float((emb_on.grad - emb_kernel.grad).abs().max())}")


def test_override_changes_one_key_on_every_temporal_recipe():
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
        return o
    for env in ("scene", "cube", "calvin"):
        a = flat(OmegaConf.to_container(_compose(f"{env}_temporal"), resolve=False))
        b = flat(OmegaConf.to_container(_compose(f"{env}_temporal", OVERRIDE), resolve=False))
        diff = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
        assert diff == ["loss.spectral.anchor.stop_gradient"], (env, diff)
        assert b["loss.spectral.anchor.stop_gradient"] is True


if __name__ == "__main__":
    test_recipe_has_a_live_anchor()
    test_stop_gradient_cuts_the_encoder_and_keeps_the_predictor()
    test_stop_gradient_leaves_the_values_and_the_kernel_gradient_alone()
    test_override_changes_one_key_on_every_temporal_recipe()
    print("test_anchor_detach: OK")
