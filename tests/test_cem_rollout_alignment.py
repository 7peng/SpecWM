"""Which action block conditions which predicted transition in `DeltaCosineCostModel`: on the
ARPredictor branch step i's window ends on block i with the executed history blocks before it and
the cost equals an explicit rollout loop; on the HorizonPredictor branch step i is conditioned on
block i alone at horizon 1. A plan of T blocks takes exactly T predictor steps."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import torch                                                                    # noqa: E402
from torch import nn                                                            # noqa: E402

from eval import DeltaCosineCostModel                                           # noqa: E402
from jepa import JEPA                                                           # noqa: E402

D, A, FS = 8, 3, 4          # embed dim, raw action dim, frameskip (block = FS*A)
B, S, T = 2, 3, 5           # envs, candidates, plan steps
HW = 3                      # predictor context frames


class TinyEncoder(nn.Module):
    class Cfg:
        hidden_size = D

    def __init__(self):
        super().__init__()
        self.config = self.Cfg()

    def forward(self, pixels, interpolate_pos_encoding=False, **kw):
        n = pixels.shape[0]
        v = pixels.reshape(n, -1).mean(1, keepdim=True)
        tok = torch.arange(D, dtype=torch.float32).view(1, D) * 0.1 + v
        return type("O", (), {"last_hidden_state": torch.stack([tok, tok], dim=1)})()


class LabelActionEncoder(nn.Module):
    """(B,T,FS*A) -> (B,T,D): the block's first raw channel broadcast, so a label survives."""

    def forward(self, x):
        return x[..., :1].expand(*x.shape[:-1], D).clone()


class RecordingHorizon(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_pos = nn.Parameter(torch.zeros(1, HW, D))
        self.calls = []

    def forward(self, ctx_emb, bridge, horizons, attn_mask=None):
        self.calls.append((ctx_emb.detach().clone(), bridge.detach().clone(), horizons.detach().clone()))
        return bridge[:, -1] * 0 + bridge.mean(1)


class ARPredictor(nn.Module):
    """Named for eval.py's dispatch; returns x + c so both context and action shape the output."""

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, HW, D))
        self.calls = []

    def forward(self, x, c):
        self.calls.append((x.detach().clone(), c.detach().clone()))
        return x + c


def build(predictor):
    return JEPA(encoder=TinyEncoder(), predictor=predictor, action_encoder=LabelActionEncoder(),
                projector=nn.Identity(), pred_proj=nn.Identity(), embed_dim=D,
                emb_norm=False, pred_norm=False).eval()


def labelled_candidates():
    c = torch.zeros(B, S, T, FS * A)
    for i in range(T):
        c[:, :, i, :] = float(i)
    return c


def labelled_history():
    """action_hist[b, s, k] = block executed from history frame k, oldest first: -(HW-1-k)."""
    h = torch.zeros(B, S, HW - 1, FS * A)
    for k in range(HW - 1):
        h[:, :, k, :] = float(k - (HW - 1))
    return h


def _info(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"pixels": torch.rand(B, S, HW, 3, 8, 8, generator=g),
            "goal": torch.rand(B, S, 1, 3, 8, 8, generator=g),
            "action_hist": labelled_history()}


def test_ar_branch_window_and_cost():
    pred = ARPredictor()
    model = build(pred)
    cm = DeltaCosineCostModel(model)
    assert cm.ar
    info = _info()
    with torch.inference_mode():
        cost = cm.get_cost(dict(info), labelled_candidates())
    assert cost.shape == (B, S) and len(pred.calls) == T
    for i, (x, c) in enumerate(pred.calls):
        assert x.shape[1] == HW and c.shape[1] == HW, (x.shape, c.shape)
        blocks = c[0, :, 0].tolist()
        assert blocks == [float(j) for j in range(i - HW + 1, i + 1)], (i, blocks)
    assert not torch.equal(pred.calls[-1][0], pred.calls[0][0]), "the context never advanced"

    # the explicit loop: H frames of context, window slides one frame per predicted block
    with torch.inference_mode():
        ctx = model.encode({"pixels": info["pixels"][:, 0]})["emb"]                    # (B, HW, D)
        goal = model.encode({"pixels": info["goal"][:, 0]})["emb"][:, -1]              # (B, D)
        ce = ctx.unsqueeze(1).expand(B, S, HW, D).reshape(B * S, HW, D)
        acts = torch.cat([labelled_history().reshape(B * S, HW - 1, -1),
                          labelled_candidates().reshape(B * S, T, -1)], 1)
        for i in range(T):
            a = model.action_encoder(acts[:, i:i + HW])
            ce = torch.cat([ce, model.predict(ce[:, -HW:], a)[:, -1:]], 1)
        pn = torch.nn.functional.normalize(ce[:, -1], dim=-1).reshape(B, S, D)
        gn = torch.nn.functional.normalize(goal, dim=-1)
        ref = 1.0 - (pn * gn.unsqueeze(1)).sum(-1)
    assert torch.allclose(cost, ref, atol=1e-6), float((cost - ref).abs().max())


def test_ar_branch_needs_the_history_and_a_window_that_fits():
    cm = DeltaCosineCostModel(build(ARPredictor()))
    info = _info()
    info.pop("action_hist")
    try:
        with torch.inference_mode():
            cm.get_cost(info, labelled_candidates())
    except AssertionError as e:
        assert "action_hist" in str(e)
    else:
        raise AssertionError("the AR path ran without action_hist")
    info = _info()
    info["pixels"] = torch.rand(B, S, HW + 1, 3, 8, 8)
    info["action_hist"] = torch.zeros(B, S, HW, FS * A)
    try:
        with torch.inference_mode():
            cm.get_cost(info, labelled_candidates())
    except AssertionError as e:
        assert "window" in str(e)
        return
    raise AssertionError("a history longer than the predictor's window was accepted")


def test_horizon_branch_block_i_drives_step_i():
    pred = RecordingHorizon()
    cm = DeltaCosineCostModel(build(pred))
    assert not cm.ar
    with torch.inference_mode():
        cost = cm.get_cost(_info(), labelled_candidates())
    assert cost.shape == (B, S) and len(pred.calls) == T
    for i, (ctx, bridge, hor) in enumerate(pred.calls):
        seen = sorted({round(float(v), 3) for v in bridge.reshape(-1)})
        assert seen == [float(i)], (i, seen)
        assert int(hor[0]) == 1 and ctx.shape[1] == HW
    assert not torch.allclose(pred.calls[-1][0], pred.calls[0][0])


if __name__ == "__main__":
    test_ar_branch_window_and_cost()
    test_ar_branch_needs_the_history_and_a_window_that_fits()
    test_horizon_branch_block_i_drives_step_i()
    print("test_cem_rollout_alignment: OK")
