"""Hypothesis 9: `HorizonPredictor` -- horizon table, mask polarity, state_pos broadcast.

  - `horizon_emb` is indexed by t in 1..max_horizon and row 0 is never read;
  - a horizon above max_horizon raises rather than wrapping;
  - `attn_mask` True = attend: masked-out bridge tokens cannot change the output;
  - `state_pos` is added unsliced, so a context of F != num_frames BROADCASTS rather than raising
    (audit B19). This test records which F values are silently accepted.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from networks import HorizonPredictor  # noqa: E402

torch.manual_seed(0)
F_, MH, Dh = 3, 4, 16
P = HorizonPredictor(num_frames=F_, max_horizon=MH, depth=2, heads=2, mlp_dim=32,
                     input_dim=Dh, hidden_dim=Dh, dropout=0.0, emb_dropout=0.0).eval()

B, Kt = 5, MH
ctx = torch.randn(B, F_, Dh)
bri = torch.randn(B, Kt, Dh)
hor = torch.tensor([1, 2, 3, 4, 1])
mask = (torch.arange(Kt)[None, :] < hor[:, None])[:, None, None, :]

with torch.no_grad():
    y0 = P(ctx, bri, hor, mask)
    # (a) masked-out bridge tokens must not move the output
    bri2 = bri.clone()
    for b in range(B):
        bri2[b, hor[b]:] = torch.randn(Kt - int(hor[b]), Dh) * 100.0
    y1 = P(ctx, bri2, hor, mask)
assert float((y0 - y1).abs().max()) < 1e-5, (
    f"masked bridge tokens changed the prediction by {float((y0 - y1).abs().max())}: "
    "attn_mask polarity is inverted")
# and UNmasked tokens do move it, so the mask is not simply ignoring everything
with torch.no_grad():
    y2 = P(ctx, bri2, hor, None)
assert float((y0 - y2).abs().max()) > 1e-4
print("attn_mask polarity: True = attend, masked tokens inert")

# (b) horizon_emb row 0 is never used, rows 1..max are.
# AT INIT NOTHING IS: `CrossAttentionBlock.adaLN_modulation[-1].weight` is zero-initialised, so
# the modulation does not depend on `c` and `horizon_emb` receives EXACTLY zero gradient on the
# first step -- contrary to the block's own comment ("horizons are differentiated from step 1").
# It bootstraps through the adaLN linear, which does receive gradient, so the conditioning is live
# from step 2 on. Both facts are recorded here.
P.zero_grad()
P(ctx, bri, hor, mask).sum().backward()
gh = P.horizon_emb.weight.grad
g_ada = P.blocks[0].adaLN_modulation[-1].weight.grad
print(f"at init: |grad horizon_emb| = {float(gh.abs().sum()):.3e}, "
      f"|grad adaLN weight| = {float(g_ada.abs().sum()):.3e}")
assert float(gh.abs().sum()) == 0.0, (
    "horizon_emb now receives gradient at init -- the adaLN zero-init changed; update this record")
assert float(g_ada.abs().sum()) > 0, "the conditioning path is dead, not merely asleep"
# one optimizer step on the adaLN linear alone, and the horizon table is live
with torch.no_grad():
    for blk in P.blocks:
        blk.adaLN_modulation[-1].weight -= 1e-2 * blk.adaLN_modulation[-1].weight.grad
P.zero_grad()
P(ctx, bri, hor, mask).sum().backward()
gh = P.horizon_emb.weight.grad
assert float(gh[0].abs().sum()) == 0.0, "horizon row 0 received gradient -- horizons are 0-indexed"
assert float(gh[1:].abs().sum()) > 0, "horizon_emb still dead after the adaLN moved"
print("horizon_emb indexed 1..max_horizon; row 0 unused; live after one adaLN step")

# (c) a horizon above max_horizon must raise
try:
    P(ctx, bri, torch.full((B,), MH + 1), mask)
    raise SystemExit("horizon > max_horizon must raise, not wrap")
except (IndexError, RuntimeError):
    pass
print("horizon > max_horizon raises")

# (d) state_pos broadcast (audit B19, unfixed): which context lengths are silently accepted
accepted = []
for F2 in (1, 2, 3, 4):
    try:
        with torch.no_grad():
            out = P(torch.randn(B, F2, Dh), bri, hor, mask)
        accepted.append((F2, tuple(out.shape)))
    except Exception as e:
        accepted.append((F2, type(e).__name__))
print("state_pos: context length -> result:", accepted)
assert accepted[0][1] == (B, Dh), (
    "a 1-frame context now raises -- B19 was fixed elsewhere; update this record")
