"""Encoder throughput fast paths: region-reassociated ViT forward and fused
cross-attention toggling.
"""
import types

import torch
import torch.nn.functional as F

from networks import CrossAttnHead


def _region_ln(x, w, b, eps, dt):
    """LayerNorm in the derived compute dtype `dt` (ATen still accumulates statistics in fp32 for
    bf16 input). Outside autocast dt stays fp32, so eval's CEM rollout keeps full precision."""
    with torch.autocast("cuda", enabled=False):
        return F.layer_norm(x.to(dt), (x.shape[-1],), w.to(dt), b.to(dt), eps)


def _region(r, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, W1, B1, W2, B2,
            lnB_w, lnB_b, lnA_w, lnA_b, h, hd, scaling, eps, dt):
    """One compiled unit spanning (norm) -> attn -> (add+norm) -> mlp -> add. A plain function of
    tensors (not a Module), so every layer shares one compiled graph. The residual stream is the
    only tensor crossing the boundary."""
    Bn, S, D = r.shape
    sh = (Bn, S, h, hd)
    y = _region_ln(r, lnB_w, lnB_b, eps, dt)
    q = F.linear(y, Wq, Bq).view(sh).transpose(1, 2)
    k = F.linear(y, Wk, Bk).view(sh).transpose(1, 2)
    v = F.linear(y, Wv, Bv).view(sh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, scale=scaling)
    r = r + F.linear(o.transpose(1, 2).reshape(Bn, S, D), Wo, Bo)
    y = _region_ln(r, lnA_w, lnA_b, eps, dt)
    return r + F.linear(F.gelu(F.linear(y, W1, B1)), W2, B2)


def _region_args(Lyr):
    """The weight arguments `_region` takes out of one ViTLayer, in its parameter order."""
    a, m = Lyr.attention, Lyr.mlp
    return (a.q_proj.weight, a.q_proj.bias, a.k_proj.weight, a.k_proj.bias,
            a.v_proj.weight, a.v_proj.bias, a.o_proj.weight, a.o_proj.bias,
            m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias,
            Lyr.layernorm_before.weight, Lyr.layernorm_before.bias,
            Lyr.layernorm_after.weight, Lyr.layernorm_after.bias)


def _region_forward(self, pixels, interpolate_pos_encoding=False, **kw):
    """Drop-in replacement for `ViTModel.forward`, bound onto the same module object so the
    module tree (and state dict) is untouched. `interpolate_pos_encoding` passes through to the
    stock HF `self.embeddings`; the region path only re-associates the block loop."""
    # other HF forward kwargs (bool_masked_pos, head_mask, output_*) are not implemented;
    # None/False passes through, anything truthy is refused
    unsupported = {k: v for k, v in kw.items() if v}
    assert not unsupported, (
        f"region encoder does not implement {sorted(unsupported)}; it re-associates the block loop "
        f"only. Set encoder_opt=none to use the stock HF forward.")
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    fn = self._region_fn
    dt = (torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda")
          else self.layernorm.weight.dtype)
    r = self.embeddings(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
    L0 = self.layers[0]
    eps, h = L0.layernorm_before.eps, L0.attention.num_attention_heads
    hd, scaling = L0.attention.head_dim, L0.attention.scaling
    for Lyr in self.layers:
        r = fn(r, *_region_args(Lyr), h, hd, scaling, eps, dt)
    y = _region_ln(r, self.layernorm.weight, self.layernorm.bias, eps, dt)
    return BaseModelOutputWithPooling(last_hidden_state=y)


def enable_region_encoder(hf, compile_mode="max-autotune-no-cudagraphs"):
    """Re-associate an HF ViT's block loop into one compiled region; patches `forward` in place.
    Parameter- and state-dict-identical, so checkpoints are interchangeable across the flag."""
    # `_region` does not implement HF's per-block or attention dropout; refuse non-zero dropout
    for i, L in enumerate(hf.layers):
        assert float(L.dropout.p) == 0.0, (
            f"layers[{i}].dropout.p={L.dropout.p}; the region encoder does not implement dropout")
    assert float(hf.config.attention_probs_dropout_prob) == 0.0, \
        "attention_probs_dropout_prob != 0 is not implemented by the region encoder"
    assert float(hf.config.hidden_dropout_prob) == 0.0, \
        "hidden_dropout_prob != 0 is not implemented by the region encoder"
    torch._dynamo.config.recompile_limit = 64
    hf._region_fn = torch.compile(_region, mode=compile_mode, dynamic=False)
    hf.forward = types.MethodType(_region_forward, hf)
    return len(hf.layers)


def enable_fused_attention(module, enabled=True):
    """Flip `CrossAttnHead.fused` on every head under `module` in place; returns the count.
    A kernel change only (no parameters added or reshaped): state dicts are interchangeable
    across the flag. Reassociates GEMMs, so numerics move within bf16 precision noise; do not
    switch it mid-run."""
    n = 0
    for m in module.modules():
        if isinstance(m, CrossAttnHead):
            m.fused = bool(enabled)
            n += 1
    return n
