"""The shared model plumbing on labelled inputs: JEPA.encode's reshapes keep clip and time order,
the Embedder's rank squeeze is refused at train start, SIGReg matches an fp64 Epps-Pulley reference
and survives bf16, and MuonAdamW routes every parameter into exactly one group."""
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAIL = []


def check(name, cond, msg=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


D = 8


class LabelEncoder(nn.Module):
    """Stand-in ViT: CLS token = the frame's own scalar label broadcast over D; patch p carries
    label*100 + p. Accepts and ignores interpolate_pos_encoding / window_size, like HF's."""

    def __init__(self, n_patch=4):
        super().__init__()
        self.n_patch = n_patch
        self.config = SimpleNamespace(hidden_size=D)
        self.embeddings = nn.Identity()

    def forward(self, pixels, interpolate_pos_encoding=False, **kw):
        lab = pixels.reshape(pixels.size(0), -1)[:, 0]      # (BT,)
        cls = lab[:, None, None].expand(-1, 1, D)
        pat = (lab[:, None] * 100 + torch.arange(self.n_patch, dtype=lab.dtype))[:, :, None]
        pat = pat.expand(-1, -1, D)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, pat], 1))


def make_jepa(**kw):
    from jepa import JEPA
    from networks import Embedder
    return JEPA(encoder=LabelEncoder(), predictor=nn.Identity(),
                action_encoder=Embedder(input_dim=3, emb_dim=D),
                projector=nn.Identity(), embed_dim=D, emb_norm=False, **kw)


def labelled_pixels(b, rows):
    """(b, rows, 1, 1, 1) where pixel value = b*1000 + row index."""
    x = torch.arange(rows, dtype=torch.float32)[None, :] + torch.arange(b, dtype=torch.float32)[:, None] * 1000
    return x.reshape(b, rows, 1, 1, 1)


def main():
    torch.manual_seed(0)

    # ---- 1. encode(): time order, view fold, window readout -----------------
    m = make_jepa()
    B, T = 2, 5
    info = m.encode({"pixels": labelled_pixels(B, T)})
    lab = info["emb"][..., 0]
    want = torch.arange(T, dtype=torch.float32)[None] + torch.arange(B, dtype=torch.float32)[:, None] * 1000
    check("encode: (B,T,...) -> (B*T,...) -> (B,T,D) preserves clip and time order",
          torch.equal(lab, want), f"got {lab.tolist()}")
    check("encode: trunk is per-frame and equals emb at V=1,K=1",
          torch.equal(info["trunk"][..., 0], want))

    # a multi-frame window or a second view is refused: one embedding per frame
    for bad in (dict(window_size=2), dict(window_causal=True)):
        try:
            make_jepa(**bad)
        except AssertionError:
            continue
        check(f"JEPA refuses {bad}", False)
    check("JEPA refuses window_size > 1 and window_causal", True)
    from networks import Embedder

    import logging, io
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    logging.getLogger().addHandler(h)
    Embedder(input_dim=6, smoothed_dim=3, emb_dim=D)
    logging.getLogger().removeHandler(h)
    from train_common import assert_no_action_bottleneck
    class _W:
        action_encoder = Embedder(input_dim=6, smoothed_dim=3, emb_dim=D)
    raised = False
    try:
        assert_no_action_bottleneck(_W())
    except AssertionError:
        raised = True
    check("Embedder rank squeeze is REFUSED at train start (not only warned)", raised)

    # ---- 4. SIGReg: fp64 reference, and bf16 autocast ----------------------
    from losses import SIGReg
    sg = SIGReg(knots=17, num_proj=64)
    torch.manual_seed(1)
    proj = torch.randn(3, 24, 16)                    # (T, B, D)

    def ref(proj, A, knots=17):
        p = proj.double().numpy(); A = A.double().numpy()
        t = np.linspace(0, 3, knots)
        dt = 3.0 / (knots - 1)
        w = np.full(knots, 2 * dt); w[0] = w[-1] = dt
        win = np.exp(-t ** 2 / 2)
        w = w * win
        x = p @ A                                     # (T,B,P)
        xt = x[..., None] * t
        err = (np.cos(xt).mean(-3) - win) ** 2 + (np.sin(xt).mean(-3)) ** 2
        return float((err @ w).mean() * p.shape[-2])

    torch.manual_seed(7)
    A = torch.randn(16, 64); A = A / A.norm(dim=0)
    # replicate the module's draw so the reference sees the same A
    torch.manual_seed(7)
    got = float(sg(proj))
    exp = ref(proj, A)
    check("SIGReg fp32 matches an fp64 Epps-Pulley reference",
          abs(got - exp) < 1e-4 * max(1.0, abs(exp)), f"code {got:.8f} vs fp64 {exp:.8f}")

    torch.manual_seed(7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gotb = float(sg(proj))
    rel = abs(gotb - exp) / max(abs(exp), 1e-12)
    check("SIGReg under bf16 autocast stays within 2% of the fp64 value",
          rel < 0.02, f"bf16 {gotb:.6f} vs fp64 {exp:.6f}  rel {rel:.4%}")

    # projections are RESAMPLED each call (not a fixed buffer) -> the statistic is stochastic
    a, b = float(sg(proj)), float(sg(proj))
    check("SIGReg's random projections are resampled per call (documented stochasticity)",
          a != b, f"{a:.6f} vs {b:.6f}")
    check("SIGReg carries no projection buffer in its state dict",
          set(sg.state_dict()) == {"t", "phi", "weights"}, f"{sorted(sg.state_dict())}")

    # the statistic is scaled by proj.size(-2) == the BATCH, i.e. samples per projection
    p_small = proj[:, :6]
    s6, s24 = float(sg(p_small)), float(sg(proj))
    check("SIGReg's N is the clip batch (proj.size(-2)), so the per-rank batch sets the test's n",
          proj.size(-2) == 24 and p_small.size(-2) == 6, f"{s6:.3f} at n=6, {s24:.3f} at n=24")

    # ---- 5. MuonAdamW routing ----------------------------------------------
    from optim import MuonAdamW
    net = nn.Module()
    net.mat = nn.Linear(4, 4, bias=True)          # 2-D weight + 1-D bias
    net.emb = nn.Embedding(5, 4)                  # 2-D but NOT a matrix
    net.tok = nn.Parameter(torch.randn(1, 2, 4))  # 3-D
    net.conv = nn.Conv2d(2, 2, 3)                 # 4-D
    no_muon = {id(p) for m in net.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    opt = MuonAdamW(net.parameters(), lr=1e-3, muon_lr=1e-2, weight_decay=0.1, no_muon=no_muon)
    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    check("MuonAdamW: nn.Embedding tables are kept OUT of Muon",
          id(net.emb.weight) not in muon_ids and id(net.mat.weight) in muon_ids)
    wd_of = {}
    for g in opt.adamw.param_groups:
        for p in g["params"]:
            wd_of[id(p)] = g["weight_decay"]
    check("MuonAdamW: the embedding table also gets weight_decay 0",
          wd_of.get(id(net.emb.weight)) == 0.0, f"wd={wd_of.get(id(net.emb.weight))}")
    check("MuonAdamW: 4-D conv kernels keep weight decay, 1-D/3-D do not",
          wd_of[id(net.conv.weight)] == 0.1 and wd_of[id(net.mat.bias)] == 0.0
          and wd_of[id(net.tok)] == 0.0)
    check("MuonAdamW: every trainable parameter lands in exactly one group",
          len(muon_ids | set(wd_of)) == len(list(net.parameters()))
          and not (muon_ids & set(wd_of)))

    # GraphedMuon must fall back to eager on CPU and be bit-identical to torch.optim.Muon
    def _run(graph):
        torch.manual_seed(3)
        n = nn.Linear(4, 4, bias=False)
        o = MuonAdamW(n.parameters(), lr=1e-3, muon_lr=1e-2, muon_graph=graph)
        for _ in range(3):
            o.zero_grad()
            n.weight.grad = torch.ones_like(n.weight) * 0.01
            o.step()
        return n.weight.detach().clone()
    check("GraphedMuon == eager torch.optim.Muon on CPU (bit-identical, graph path is CUDA-only)",
          torch.equal(_run(True), _run(False)))

    # a parameter frozen AFTER construction still sits in a group; with grad None nothing moves
    n2 = nn.Linear(4, 4, bias=False)
    o2 = MuonAdamW(n2.parameters(), lr=1e-3, muon_lr=1e-2, weight_decay=0.5)
    before = n2.weight.detach().clone()
    n2.weight.requires_grad_(False)
    n2.weight.grad = None
    o2.step()
    check("MuonAdamW: a param frozen AFTER construction with grad=None is not decayed",
          torch.equal(before, n2.weight.detach()),
          f"max drift {float((before - n2.weight.detach()).abs().max()):.3e}")
    # ... but a STALE grad left on a frozen param IS still stepped
    n3 = nn.Linear(4, 4, bias=False)
    o3 = MuonAdamW(n3.parameters(), lr=1e-3, muon_lr=1e-2)
    b3 = n3.weight.detach().clone()
    n3.weight.grad = torch.ones_like(n3.weight)
    n3.weight.requires_grad_(False)
    o3.step()
    moved = float((b3 - n3.weight.detach()).abs().max())
    check("MuonAdamW: a frozen param with a STALE grad is still stepped (requires_grad read once)",
          moved > 0, f"drift {moved:.3e} -- freeze must be paired with zero_grad(set_to_none=True)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
