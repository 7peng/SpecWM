"""JEPA world model: ViT encoder -> projector -> emb; predictor over emb conditioned on embedded
actions (eval.DeltaCosineCostModel rolls it forward for planning)."""

import torch
from einops import rearrange
from torch import nn

from networks import RMSNorm


class JEPA(nn.Module):
    """One embedding per frame: the encoder's CLS token through `projector` and the optional
    parameter-free RMSNorm (`emb_norm`); predictions through `pred_proj` and `pred_norm`.
    `window_size`, `window_causal`, `window_inert` and `bottleneck_dim` are accepted from saved
    configs and must hold their per-frame values."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        pool="cls",
        emb_norm=True,
        pred_norm=False,
        embed_dim=None,
        window_size=1,
        window_causal=False,
        window_inert=False,
        bottleneck_dim=None,
    ):
        super().__init__()
        assert pool == "cls", f"pool must be cls, got {pool!r}"
        assert int(window_size) == 1 and not window_causal and not window_inert, \
            "the encoder emits one embedding per frame (window_size 1, no causal window)"
        assert isinstance(emb_norm, bool), f"emb_norm is on or off, got {emb_norm!r}"
        self.pool = pool
        self.window_size = 1
        self.window_causal = False
        self.n_views = 1
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.emb_norm = RMSNorm(eps=1e-6) if emb_norm else nn.Identity()
        self.pred_norm = RMSNorm(eps=1e-6) if pred_norm else nn.Identity()
        self.embed_dim = embed_dim

    def encode(self, info, window_size=None):
        """info: dict with 'pixels' (B,S,C,H,W) and optionally 'action'. Adds 'trunk' (B,S,D), the
        pre-projector CLS features, 'emb' (B,S,D) and 'act_emb'. `window_size` is accepted for
        call-site compatibility and must be None or 1."""
        assert window_size is None or int(window_size) == 1, "one frame per embedding"
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        # input is always the ViT's native size, so pos-encoding interpolation is a no-op
        output = self.encoder(pixels, interpolate_pos_encoding=False)
        pixels_emb = output.last_hidden_state[:, 0]                 # CLS token, one per frame
        info["trunk"] = rearrange(pixels_emb, "(b t) e -> b t e", b=b)
        emb = self.emb_norm(self.projector(pixels_emb))
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        """Next-step prediction. emb: (B,T,D); act_emb: (B,T,A) AdaLN conditioning."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_norm(self.pred_proj(rearrange(preds, "b t d -> (b t) d")))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def predict_horizon(self, ctx_emb, bridge_act_emb, horizons, attn_mask=None):
        """Direct t-step-ahead prediction (HorizonPredictor). ctx_emb (B,F,D), bridge action
        tokens (B,K,D), horizons (B,) long, attn_mask (B,1,1,K) bool. Returns (B,D)."""
        preds = self.predictor(ctx_emb, bridge_act_emb, horizons, attn_mask)
        return self.pred_norm(self.pred_proj(preds))
