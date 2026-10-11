"""One step of `temporal_forward` under the live scene recipes (resolved by hydra) on the stub
harness with a stub frozen teacher: the loss is weight * spec_loss + anchor.weight *
pred_anchor_loss, both encoder and predictor receive gradient, the teacher is asked once per step,
spec_loss and pred_anchor_loss match an fp64 reference, and the per-voice monitors reach the
logger."""
import os
import sys
from pathlib import Path

import numpy as np
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

B, D = 3, 12


def _compose(name):
    with initialize_config_dir(config_dir=str(ROOT / "config" / "train"), version_base=None):
        raw = compose(config_name=name, overrides=["run_id=test"])
    return T._PlainCfg.wrap(OmegaConf.to_container(raw, resolve=True))


class StubTeacher(nn.Module):
    """d(i, j) = |t_i - t_j| + 40 [clip_i != clip_j] over the B*N rows."""

    def __init__(self, B, N):
        super().__init__()
        t = torch.arange(N).float().repeat(B)
        c = torch.arange(B).repeat_interleave(N).float()
        self.d = (t[:, None] - t[None, :]).abs() + 40.0 * (c[:, None] != c[None, :]).float()
        self.calls = 0

    def all_pairs_dist(self, pixels):
        self.calls += 1
        return self.d.clone()


def build(name, **sd_over):
    cfg = _compose(name)
    N = int(cfg["data"]["dataset"]["num_steps"])
    emb = labelled_emb(B, N, D, scale=0.4).clone().requires_grad_(True)
    act = torch.randn(B, N, D, requires_grad=True)
    cfg["loss"]["spectral"].update(sd_over)
    mod = StubModule(cfg, emb, act)
    mod.model.predictor.emb_true = emb.detach().roll(1, dims=1)   # not the oracle: a live anchor
    mod.teacher_linear = StubTeacher(B, N)
    return mod, emb, cfg, N


def _bctr(M, m):
    rs = (M * m).sum(-1, keepdims=True) / m.sum(-1, keepdims=True)
    cs = (M * m).sum(-2, keepdims=True) / m.sum(-2, keepdims=True)
    return M - rs - cs + (M * m).sum() / m.sum()


def test_quasimetric_recipe_step():
    mod, emb, cfg, N = build("scene_quasimetric")
    sd = cfg["loss"]["spectral"]
    out = mod(make_batch(B, N))
    w_sd, w_pa = float(sd["weight"]), float(sd["anchor"]["weight"])
    expect = w_sd * float(out["spec_loss"]) + w_pa * float(out["pred_anchor_loss"])
    assert abs(float(out["loss"]) - expect) < 1e-5, (float(out["loss"]), expect)
    assert mod.teacher_linear.calls == 1

    out["loss"].backward()
    assert emb.grad is not None and float(emb.grad.abs().sum()) > 0
    pw = mod.model.predictor.w
    assert pw.grad is not None and float(pw.grad.abs()) > 0

    # fp64 reference of spec_loss (score_diagonal: every pair of the clip) and of the anchor
    H = int(cfg["wm"]["history_size"])
    sigma = float(sd["factors"][0]["sigma"])
    m = np.ones((N, N)) if bool(sd["score_diagonal"]) else 1.0 - np.eye(N)
    E = emb.detach().double().numpy()
    d = mod.teacher_linear.d.double().numpy()
    es = []
    for c in range(B):
        G = np.exp(-d[c * N:(c + 1) * N, c * N:(c + 1) * N] / sigma)
        h = E[c] / np.linalg.norm(E[c], axis=-1, keepdims=True)
        Sc, Kc = _bctr(h @ h.T, m), _bctr(G, m)
        varK = (Kc ** 2 * m).sum() / m.sum()
        es.append(((Sc - Kc) ** 2 * m).sum() / m.sum() / varK)
    ref = float(np.mean(es))
    got = float(out["spec_loss"])
    assert abs(got - ref) / ref < 2e-5, (got, ref)

    pf = np.arange(H - 1, N - 1) + 1
    pred_rows = np.stack([np.roll(E[b], 1, axis=0)[pf] for b in range(B)])
    tgt_rows = np.stack([E[b][pf] for b in range(B)])
    cn = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)  # noqa: E731
    ref_anchor = float(1.0 - (cn(pred_rows) * cn(tgt_rows)).sum(-1).mean())
    assert abs(float(out["pred_anchor_loss"]) - ref_anchor) < 2e-6

    logged = {k.split("/", 1)[1] for k in mod.logged}
    for k in ("spec_loss", "pred_anchor_loss", "gram_corr_within", "gram_corr_linear", "loss"):
        assert k in logged, (k, sorted(logged))


def test_temporal_recipe_step():
    mod, emb, cfg, N = build("scene_temporal")
    out = mod(make_batch(B, N))
    sd = cfg["loss"]["spectral"]
    expect = float(sd["weight"]) * float(out["spec_loss"]) + float(sd["anchor"]["weight"]) * float(out["pred_anchor_loss"])
    assert abs(float(out["loss"]) - expect) < 1e-5
    assert mod.teacher_linear.calls == 0, "the temporal kernel needs no teacher"
    out["loss"].backward()
    assert float(emb.grad.abs().sum()) > 0
    assert "train/gram_corr_lag" in mod.logged, sorted(mod.logged)


def test_predictions_query_recipe_step():
    mod, emb, cfg, N = build("scene_clip_temporal")
    assert cfg["loss"]["spectral"]["queries"] == "predictions"
    out = mod(make_batch(B, N))
    out["loss"].backward()
    assert "train/gram_corr_cross_pred" in mod.logged
    assert float(emb.grad.abs().sum()) > 0 and float(mod.model.predictor.w.grad.abs()) > 0


if __name__ == "__main__":
    test_quasimetric_recipe_step()
    test_temporal_recipe_step()
    test_predictions_query_recipe_step()
    print("test_tdwm_live_forward: OK")
