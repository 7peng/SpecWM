"""Shared CPU harness for the train_spec tests: `temporal_forward` against a stub model whose
`encode` returns LABELLED embeddings (row (b, t) carries b and t in its first two coordinates), so
any misrouting of frames, clips, starts or horizons is visible in the output."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_spec as T  # noqa: E402


def labelled_emb(B, N, D, scale=1.0):
    """emb[b, t] = [b, t, deterministic noise...]."""
    e = torch.zeros(B, N, D)
    e[..., 0] = torch.arange(B).float()[:, None]
    e[..., 1] = torch.arange(N).float()[None, :]
    g = torch.Generator().manual_seed(1234)
    e[..., 2:] = torch.randn(B, N, D - 2, generator=g) * scale
    return e


class StubPredictor(nn.Module):
    """HorizonPredictor interface. Records every call and returns the ORACLE row
    emb_true[b, start + horizon] decoded from the context's last frame, so `pred_loss` is exactly
    0 iff the multistart index is aligned."""

    def __init__(self, emb_true):
        super().__init__()
        self.emb_true = emb_true
        self.calls = []
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, ce, be, hor, amask):
        self.calls.append((ce.detach().clone(), be.detach().clone(),
                           hor.detach().clone(), None if amask is None else amask.detach().clone()))
        b = ce[:, -1, 0].round().long()
        s = ce[:, -1, 1].round().long()
        out = self.emb_true[b, (s + hor).clamp(max=self.emb_true.size(1) - 1)]
        return out + 0.0 * ce[:, -1] + self.w


class ARPredictor(nn.Module):
    """ARPredictor interface (the name is what train_spec dispatches on): `pos_embedding` of
    `num_frames` rows, forward(x, c) -> (B, T, D) with position t the oracle row emb_true[b, t+1]."""

    def __init__(self, emb_true, num_frames):
        super().__init__()
        self.emb_true = emb_true
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, emb_true.size(-1)))
        self.calls = []
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, x, c):
        self.calls.append((x.detach().clone(), c.detach().clone()))
        b = x[..., 0].round().long()
        s = x[..., 1].round().long()
        out = self.emb_true[b, (s + 1).clamp(max=self.emb_true.size(1) - 1)]
        return out + 0.0 * x + self.w


class StubModel(nn.Module):
    def __init__(self, emb, act_emb, predictor=None):
        super().__init__()
        self._emb, self._act = emb, act_emb
        self.predictor = predictor if predictor is not None else StubPredictor(emb)

    def encode(self, batch):
        return {"emb": self._emb, "act_emb": self._act}

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def predict_horizon(self, ce, be, hor, amask=None):
        return self.predictor(ce, be, hor, amask)


class StubSigreg(nn.Module):
    def forward(self, x):
        return x.float().pow(2).mean() * 0.0


class StubModule(nn.Module):
    """The Lightning-module stand-in `temporal_forward` is bound to. `cfg` is a `_PlainCfg`
    (dict) or a DictConfig; a dict is used as the step config directly."""

    def __init__(self, cfg, emb, act_emb, predictor=None):
        super().__init__()
        self.model = StubModel(emb, act_emb, predictor)
        self.sigreg = StubSigreg()
        self.gpu_aug = lambda px: (px, None)
        self.global_step = 0
        self.trainer = None
        self.logged = {}
        self._cfg = cfg
        if isinstance(cfg, dict):
            self._plain_cfg = cfg

    def log_dict(self, d, **kw):
        self.logged.update(d)

    def forward(self, batch, stage="train"):
        return T.temporal_forward(self, batch, stage, self._cfg)


def make_cfg(N, H, max_ph, spectral=None, sigreg_weight=0.0, frameskip=5):
    """A minimal step config: wm geometry, sigreg weight, an optional loss.spectral block."""
    return T._PlainCfg.wrap({
        "data": {"dataset": {"far_frames": 0, "frameskip": frameskip}},
        "wm": {"history_size": H, "num_preds": N - H, "max_pred_horizon": max_ph},
        "loss": {"sigreg": {"weight": sigreg_weight}, "spectral": spectral},
        "quasimetric_head": None,
    })


def make_batch(B, N, act_dim=4, px_c=3, px_h=2, px_w=2):
    g = torch.Generator().manual_seed(7)
    return {
        "pixels": torch.randint(0, 255, (B, N, px_c, px_h, px_w), generator=g,
                                dtype=torch.uint8).float(),
        "action": torch.randn(B, N, act_dim, generator=g),
    }
