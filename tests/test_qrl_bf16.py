"""bf16 autocast on the quasimetric head: `encode_z` ends in an autocast-listed Linear, so the
IQE interval test runs on bf16 latents; measured on a real teacher head at training scale (skipped
when the checkpoint is not readable). The IQE sweep itself is exact given the same z."""
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from losses import QRLHead  # noqa: E402

CKPT = Path(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))) / ("checkpoints/lewm/scene-qrl-linear-1/"
            "step_snaps/step_2000.pt")
D = 192


def _real_head():
    blob = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    h = QRLHead(embed_dim=D, hidden=384, z_dim=384, group_size=16, push_linear=True)
    h.load_state_dict(blob["heads"]["qrl_head"], strict=True)
    return h.eval()


def _walk(n, step, seed=0):
    """A clip of RMS-1 embedding rows: a random walk, renormalised each step (what emb_norm
    hands the head)."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(D, generator=g)
    rows = []
    for _ in range(n):
        e = e + step * torch.randn(D, generator=g)
        rows.append(e / e.pow(2).mean().sqrt())
    return torch.stack(rows)[None]                     # (1,n,D)


@pytest.mark.skipif(not CKPT.exists(), reason="real teacher checkpoint not readable here")
def test_bf16_error_on_the_real_head_at_training_scale():
    h = _real_head()
    # calibrate the walk so the fp32 adjacent distance sits at step_cost
    lo, hi = 1e-4, 3.0
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        emb = _walk(24, mid)
        z = h.encode_z(emb)
        d_adj = h._dist(z[:, :-1], z[:, 1:]).mean().item()
        lo, hi = (mid, hi) if d_adj < h.step_cost else (lo, mid)
    emb = _walk(24, (lo * hi) ** 0.5)

    z32 = h.encode_z(emb)
    D32 = h.pairwise_dist(z32)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        zb = h.encode_z(emb)
        Db = h.pairwise_dist(zb)[0].float()
    N = D32.shape[0]
    ar = torch.arange(N)
    gaps = ar[None, :] - ar[:, None]
    rows = []
    for g in (1, 2, 4, 8, 16, 23):
        m = gaps == g
        a, b = D32[m].mean().item(), Db[m].mean().item()
        rows.append((g, a, b, (b - a) / max(a, 1e-9)))
    print(f"[bf16] z dtype fp32/{zb.dtype}   step_cost={h.step_cost}")
    print(f"[bf16] {'gap':>4}{'d fp32':>10}{'d bf16':>10}{'rel':>9}")
    for g, a, b, r in rows:
        print(f"[bf16] {g:>4}{a:>10.4f}{b:>10.4f}{r:>+9.3%}")
    off = ~torch.eye(N, dtype=torch.bool)
    allrel = ((Db - D32).abs()[off] / D32[off].clamp_min(1e-9))
    print(f"[bf16] all off-diagonal pairs: mean |rel| {allrel.mean():.3%}, "
          f"p90 {allrel.quantile(0.9):.3%}, max {allrel.max():.3%}")
    # the adjacent bond is the one L_step prices; report it as the headline
    adj_rel = rows[0][3]
    assert abs(adj_rel) < 0.5, f"adjacent-bond bf16 error {adj_rel:.1%}"


@pytest.mark.skipif(not CKPT.exists(), reason="real teacher checkpoint not readable here")
def test_iqe_itself_is_exact_under_autocast_the_error_is_encode_z():
    """Localise it: given the SAME z, the IQE sweep is autocast-safe (its GEMM sums integers in
    {-1,0,1}); the loss is entirely in the Linear that produces z."""
    h = _real_head()
    z = h.encode_z(_walk(8, 0.05))
    d32 = h.pairwise_dist(z)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        db = h.pairwise_dist(z)
    err = (d32 - db.float()).abs().max().item()
    print(f"[bf16] same z, autocast on/off: max|dd| = {err:.3e}")
    assert err == 0.0


if __name__ == "__main__":
    if not CKPT.exists():
        print(f"test_qrl_bf16: skipped, no teacher checkpoint at {CKPT}")
    else:
        test_bf16_error_on_the_real_head_at_training_scale()
        test_iqe_itself_is_exact_under_autocast_the_error_is_encode_z()
        print("test_qrl_bf16: OK")
