"""Model components: norms, attention blocks, the action embedder, the projector MLP and the two
predictors (next-step ARPredictor, variable-horizon HorizonPredictor)."""
import logging

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Parameter-free RMS normalization: y = target_scale * x / sqrt(mean(x^2, -1) + eps).

    With target_scale=1.0 each row has RMS 1 (||y|| = sqrt(d)); use 1/sqrt(d) for unit-norm rows.
    Deliberately no learnable affine; anchors magnitude only.
    """

    def __init__(self, eps=1e-6, target_scale=1.0):
        super().__init__()
        self.eps = eps
        self.target_scale = target_scale

    def forward(self, x):
        # fp32 for a stable rsqrt under bf16 autocast, then restore the dtype
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.target_scale).type_as(x)

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads)
                   for t in qkv)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (identity at init)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, context=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * \
            self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * \
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=ConditionalBlock,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, context=None, attn_mask=None):

        if hasattr(self, "input_proj") and not isinstance(self.input_proj, nn.Identity):
            x = self.input_proj(x)
            if context is not None:   # cross-attn action tokens share input_dim with states
                context = self.input_proj(context)

        if c is not None and hasattr(self, "cond_proj") and not isinstance(self.cond_proj, nn.Identity):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, c, context=context, attn_mask=attn_mask)
        x = self.norm(x)

        if hasattr(self, "output_proj") and not isinstance(self.output_proj, nn.Identity):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """Flatten-and-embed action encoder: (B, T, input_dim) action blocks -> (B, T, emb_dim).

    ``smoothed_dim`` is the width of the kernel-1 Conv1d and defaults to input_dim (full rank);
    a smaller value discards input_dim - smoothed_dim dimensions of every action block and is only
    for reloading checkpoints trained that way (warns).
    """

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=None,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        if smoothed_dim is None:
            smoothed_dim = input_dim
        elif smoothed_dim < input_dim:
            logging.warning(
                "Embedder: smoothed_dim=%d < input_dim=%d -- patch_embed is a rank-%d map, so "
                "%d of every action block's %d dimensions are discarded before the predictor. "
                "Set smoothed_dim=null (=input_dim) unless reloading a checkpoint trained this way.",
                smoothed_dim, input_dim, smoothed_dim, input_dim - smoothed_dim, input_dim)
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x

class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x, c):
        """x: (B, T, d) state context tokens; c: (B, T, d) AdaLN conditioning (action embedding)."""
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x

class CrossAttnHead(nn.Module):
    """Multi-head attention with separate query / key-value streams and no internal norm; the caller
    applies the (AdaLN-modulated) norm. `attn_mask` is a boolean mask (True=attend) for
    cross-attention; pass `causal=False` for both self- and cross-attention here."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dim = dim
        self.dropout = dropout
        # Off by default (SDPA path, bit-identical to existing checkpoints); set via `enable_fused_attention`.
        self.fused = False
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _compute_dtype(self):
        """The dtype a plain `nn.Linear` would produce here; derived (not hardcoded to bf16) so
        non-autocast eval keeps full precision."""
        if torch.is_autocast_enabled("cuda") and self.to_q.weight.is_cuda:
            return torch.get_autocast_dtype("cuda")
        return self.to_q.weight.dtype

    def _attend(self, scores, attn_mask, drop, dt):
        """Masked softmax in fp32 regardless of `dt`, matching SDPA's internal accumulate."""
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        p = scores.float().softmax(-1).to(dt)
        return F.dropout(p, drop, True) if drop > 0 else p

    def _manual(self, x_q, x_kv, attn_mask, drop, dt):
        """Explicit bmm -> softmax -> bmm; identical arithmetic to SDPA on a different kernel."""
        h = self.heads
        q = rearrange(self.to_q(x_q), "b n (h d) -> b h n d", h=h)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=h)
                for t in self.to_kv(x_kv).chunk(2, dim=-1))
        p = self._attend(torch.matmul(q, k.transpose(-1, -2)) * self.dim_head ** -0.5,
                         attn_mask, drop, q.dtype)
        return self.to_out(rearrange(torch.matmul(p, v), "b h n d -> b n (h d)"))

    def _folded(self, x_q, x_kv, attn_mask, drop, dt):
        """Weight-folded cross-attention: K and V are never materialised.
        scores_h = x_q (Wq_h^T Wk_h) x_kv^T; out = sum_h (P_h x_kv)(Wv_h^T Wo_h^T).
        Costs O(q*D^2) per head, so only worthwhile when keys far outnumber queries (`forward` gates)."""
        h, dh, D = self.heads, self.dim_head, self.dim
        s, qn, kn = x_q.size(0), x_q.size(1), x_kv.size(1)
        # Fold the weights in fp32, then cast once.
        wq = self.to_q.weight.view(h, dh, D).float()
        wk = self.to_kv.weight[:h * dh].view(h, dh, D).float()
        wv = self.to_kv.weight[h * dh:].view(h, dh, D).float()
        wo = self.to_out[0].weight.view(D, h, dh).float()
        m = torch.einsum("hdi,hdj->hij", wq, wk).permute(1, 0, 2).reshape(D, h * D).to(dt)
        n = torch.einsum("hdi,ehd->hie", wv, wo).reshape(h * D, D).to(dt)
        xq, xkv = x_q.to(dt), x_kv.to(dt)
        # Contract against the (small) query side first; the key side would cost (kn/qn)x more.
        a = (xq.reshape(s * qn, D) @ m).view(s, qn, h, D).permute(0, 2, 1, 3).reshape(s, h * qn, D)
        sc = (torch.bmm(a, xkv.transpose(1, 2)) * dh ** -0.5).view(s, h, qn, kn)
        p = self._attend(sc, attn_mask, drop, dt)
        z = torch.bmm(p.view(s, h * qn, kn), xkv).view(s, h, qn, D)
        z = z.permute(0, 2, 1, 3).reshape(s * qn, h * D)
        o = z @ n
        b = self.to_out[0].bias
        if b is not None:
            o = o + b.to(dt)
        return self.to_out[1](o.view(s, qn, D))

    def forward(self, x_q, x_kv=None, attn_mask=None, causal=False):
        self_attn = x_kv is None
        x_kv = x_q if self_attn else x_kv
        drop = self.dropout if self.training else 0.0
        # `causal` is unsupported by both fused paths, so it falls through to SDPA.
        if self.fused and not causal:
            dt = self._compute_dtype()
            # Fold only when keys outnumber queries enough to pay for the O(q*D^2) weight products;
            # self-attention (kn == qn) stays on the manual path.
            if not self_attn and x_kv.size(1) > 2 * x_q.size(1):
                return self._folded(x_q, x_kv, attn_mask, drop, dt)
            return self._manual(x_q, x_kv, attn_mask, drop, dt)
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """DiT-style block conditioned on a single vector `c` via AdaLN (shift+scale+gate on each of
    three sublayers): self-attn(states) -> cross-attn(states -> action tokens) -> MLP. `c` is
    (B, 1, dim) and broadcasts over the query tokens; `attn_mask` restricts the cross-attention."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.self_attn = CrossAttnHead(dim, heads, dim_head, dropout)
        self.cross_attn = CrossAttnHead(dim, heads, dim_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ctx_norm = nn.LayerNorm(dim)   # plain norm for the action (key/value) tokens
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
        # Not AdaLN-zero: weight=0 but gate biases=1 (shift/scale biases=0), so at init each block is
        # a standard residual x + sublayer(LN(x)) and horizons are differentiated from step 1.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        with torch.no_grad():
            for gi in (2, 5, 8):  # gate slices in [sh1,sc1,g1, sh2,sc2,g2, sh3,sc3,g3]
                self.adaLN_modulation[-1].bias[gi * dim:(gi + 1) * dim].fill_(1.0)

    def forward(self, x, context, c, attn_mask=None):
        (sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3) = self.adaLN_modulation(c).chunk(9, dim=-1)
        ctx = self.ctx_norm(context)
        x = x + g1 * self.self_attn(modulate(self.norm1(x), sh1, sc1), causal=False)
        x = x + g2 * self.cross_attn(modulate(self.norm2(x), sh2, sc2), ctx, attn_mask=attn_mask)
        x = x + g3 * self.mlp(modulate(self.norm3(x), sh3, sc3))
        return x

class HorizonPredictor(nn.Module):
    """Direct variable-horizon predictor: predicts the state embedding `t` steps ahead from a
    `num_frames` history window by cross-attending the `t` bridge action tokens, with the horizon `t`
    injected via AdaLN. Bridge actions arrive pre-embedded (by JEPA.action_encoder); the caller masks
    them per-horizon. Readout is the last (most-recent) context state token.

    use_horizon_cond=False ablates the explicit `t`: a learned `null_c` replaces the horizon
    embedding. Exactly one of {horizon_emb, null_c} is allocated (no DDP-unused parameter)."""

    def __init__(self, *, num_frames, max_horizon, depth, heads, mlp_dim, input_dim, hidden_dim,
                 output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0, use_horizon_cond=True,
                 action_substep=1, window_size=1):
        super().__init__()
        self.use_horizon_cond = bool(use_horizon_cond)
        # one action token per frame step: both layout factors are fixed at 1
        self.action_substep = int(action_substep)
        self.window_size = int(window_size)
        assert self.action_substep == 1 and self.window_size == 1, (
            f"HorizonPredictor takes one action token per frame step; got action_substep="
            f"{action_substep}, window_size={window_size}")
        self.state_in = (nn.Linear(input_dim, hidden_dim)
                         if input_dim != hidden_dim else nn.Identity())
        self.action_in = (nn.Linear(input_dim, hidden_dim)
                          if input_dim != hidden_dim else nn.Identity())
        self.state_pos = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)
        self.action_pos = nn.Parameter(
            torch.randn(1, max_horizon * self.action_substep * self.window_size, hidden_dim) * 0.02)
        if self.use_horizon_cond:
            self.horizon_emb = nn.Embedding(max_horizon + 1, hidden_dim)  # index by t in 1..max_horizon
            nn.init.normal_(self.horizon_emb.weight, std=0.02)
        else:
            # Non-zero init: c must be != 0 so SiLU(c) != 0 and the zero-init AdaLN can train.
            self.null_c = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        out_dim = output_dim or input_dim
        self.out = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(self, ctx_emb, bridge_act_emb, horizons, attn_mask=None):
        """ctx_emb (B,F,Din) state context; bridge_act_emb (B,K,Din) pre-embedded bridge actions
        (K=max_horizon); horizons (B,) long in 1..max_horizon; attn_mask (B,1,1,K) bool (True=attend)
        masking each row to its first `t` action tokens. Returns (B, out_dim)."""
        x = self.dropout(self.state_in(ctx_emb) + self.state_pos)         # (B,F,H)
        # K = bridge_act_emb.size(1) may be < max_horizon; slice action_pos to the actual bridge length.
        nb = bridge_act_emb.size(1)
        assert nb <= self.action_pos.size(1), (
            f"bridge has {nb} action tokens but action_pos covers {self.action_pos.size(1)} (max_horizon)")
        ctx = self.action_in(bridge_act_emb) + self.action_pos[:, :nb]    # (B,K,H)
        if self.use_horizon_cond:
            c = self.horizon_emb(horizons).unsqueeze(1)                   # (B,1,H)
        else:
            c = self.null_c.expand(x.size(0), -1, -1)                     # (B,1,H)
        for blk in self.blocks:
            x = blk(x, ctx, c, attn_mask=attn_mask)
        x = self.norm(x)[:, -1]                                           # last/most-recent state token
        return self.out(x)                                               # (B, out_dim)

