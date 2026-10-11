"""The IQE quasimetric primitive against an fp64 reference: `_iqe_components` on random and tied
coordinates, zero self-distance, asymmetry, the triangle inequality, exact chunking, the grad path,
and what bf16 autocast does to it."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from losses import QRLHead  # noqa: E402


def iqe_ref(u, v, G):
    """fp64 union-of-intervals reference. u, v: (..., Z). Returns (..., Z//G)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lead = u.shape[:-1]
    Z = u.shape[-1]
    C = Z // G
    out = np.zeros(lead + (C,), dtype=np.float64)
    for idx in np.ndindex(*lead):
        for c in range(C):
            sl = slice(c * G, (c + 1) * G)
            iv = [(a, b) for a, b in zip(u[idx][sl], v[idx][sl]) if a < b]
            iv.sort()
            tot, cur_s, cur_e = 0.0, None, None
            for a, b in iv:
                if cur_s is None:
                    cur_s, cur_e = a, b
                elif a <= cur_e:
                    cur_e = max(cur_e, b)
                else:
                    tot += cur_e - cur_s
                    cur_s, cur_e = a, b
            if cur_s is not None:
                tot += cur_e - cur_s
            out[idx + (c,)] = tot
    return out


def _head(**kw):
    torch.manual_seed(0)
    return QRLHead(embed_dim=8, hidden=16, z_dim=16, group_size=4, push_linear=True, **kw)


def test_iqe_components_match_fp64_reference():
    torch.manual_seed(1)
    h = _head()
    u = torch.randn(5, 16)
    v = torch.randn(5, 16)
    got = h._iqe_components(u, v, 4).numpy()
    ref = iqe_ref(u.numpy(), v.numpy(), 4)
    err = np.abs(got - ref).max()
    print(f"[iqe] max|got-ref| fp32 = {err:.3e}  (ref mean {ref.mean():.4f})")
    assert err < 1e-5, err


def test_iqe_components_with_ties_and_degenerate_intervals():
    """Repeated coordinates and exact ties are what a sort-based sweep gets wrong."""
    h = _head()
    u = torch.tensor([[0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0]])
    v = torch.tensor([[0.0, 1.0, 1.0, 2.0, 3.0, -1.0, 2.0, 2.5]])
    got = h._iqe_components(u, v, 4).numpy()
    ref = iqe_ref(u.numpy(), v.numpy(), 4)
    print(f"[iqe-ties] got={got}  ref={ref}")
    assert np.abs(got - ref).max() < 1e-6


def test_dist_zero_on_self_and_asymmetric():
    torch.manual_seed(2)
    h = _head()
    z = torch.randn(4, 16)
    d_self = h._dist(z, z)
    print(f"[iqe] max d(x,x) = {d_self.abs().max().item():.3e}")
    assert d_self.abs().max().item() < 1e-6
    dij = h._dist(z[:, None, :], z[None, :, :])
    asym = (dij - dij.T).abs().max().item()
    print(f"[iqe] max|d(i,j)-d(j,i)| = {asym:.4f}  (mean d {dij.mean().item():.4f})")
    assert asym > 1e-3, "IQE must be asymmetric"


def test_triangle_inequality():
    torch.manual_seed(3)
    h = _head()
    z = torch.randn(12, 16)
    D = h._dist(z[:, None, :], z[None, :, :])
    viol = (D[:, None, :] - (D[:, :, None] + D[None].transpose(0, 1))).amax()
    # d(i,k) <= d(i,j) + d(j,k)
    D3 = D[:, :, None] + D[None, :, :]           # (i,j,k) = d(i,j)+d(j,k)
    worst = (D[:, None, :] - D3).max().item()
    print(f"[iqe] worst triangle violation = {worst:.3e}")
    assert worst < 1e-5, worst


@pytest.mark.parametrize("chunk", [1, 2, 3, 8, 16])
def test_pairwise_dist_chunking_is_exact(chunk):
    torch.manual_seed(4)
    h = _head()
    z = torch.randn(2, 7, 16)          # N=7, not divisible by 2/3/8
    ref = h.pairwise_dist(z, chunk=7)
    got = h.pairwise_dist(z, chunk=chunk)
    err = (ref - got).abs().max().item()
    print(f"[chunk={chunk}] max|d-d| = {err:.3e}")
    assert err == 0.0


def test_pairwise_dist_grad_matches_pairwise_dist():
    torch.manual_seed(5)
    h = _head()
    z = torch.randn(2, 6, 16)
    a = h.pairwise_dist(z)
    b = h.pairwise_dist_grad(z)
    err = (a - b).abs().max().item()
    print(f"[grad-path] max|pairwise_dist - pairwise_dist_grad| = {err:.3e}")
    assert err == 0.0


def test_iqe_under_bf16_autocast_matches_fp32():
    """The trainer runs under bf16 autocast. `_iqe_components` upcasts with .float(), but the
    prefix-sum GEMM is an autocast-listed op: if it degrades, `covered` flips and the union
    length is wrong by whole intervals."""
    torch.manual_seed(6)
    h = _head()
    u = torch.randn(6, 16)
    v = torch.randn(6, 16)
    ref = iqe_ref(u.numpy(), v.numpy(), 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = h._iqe_components(u, v, 4).float().numpy()
    err = np.abs(got - ref).max()
    print(f"[iqe-autocast] max|got-ref| = {err:.3e}  dtype-in={u.dtype}")
    assert err < 1e-5, err


def test_encode_z_precision_under_autocast():
    """encode_z's Linear IS autocast-listed, so z arrives in bf16 and the interval test
    `x < y` collapses coordinate pairs closer than a bf16 ulp. Quantifies the effect on the
    ADJACENT-frame distance -- the quantity L_step constrains."""
    torch.manual_seed(7)
    h = QRLHead(embed_dim=32, hidden=64, z_dim=64, group_size=16, push_linear=True)
    emb = torch.randn(3, 8, 32)
    emb_next = emb + 0.01 * torch.randn_like(emb)          # a small step, as adjacent frames are
    z32 = h.encode_z(emb)
    zn32 = h.encode_z(emb_next)
    d32 = h._dist(z32, zn32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        zb = h.encode_z(emb)
        znb = h.encode_z(emb_next)
        db = h._dist(zb, znb)
    print(f"[encode_z] z dtype fp32={z32.dtype} autocast={zb.dtype}")
    rel = ((db.float() - d32).abs() / d32.abs().clamp_min(1e-9)).mean().item()
    print(f"[encode_z] adjacent d: fp32 mean {d32.mean().item():.5f}  "
          f"bf16 mean {db.float().mean().item():.5f}  mean rel err {rel:.4f}")
    assert zb.dtype == torch.bfloat16, "documenting the dtype the head actually sees"


if __name__ == "__main__":
    test_iqe_components_match_fp64_reference()
    test_iqe_components_with_ties_and_degenerate_intervals()
    test_dist_zero_on_self_and_asymmetric()
    test_triangle_inequality()
    for c in (1, 2, 3, 8, 16):
        test_pairwise_dist_chunking_is_exact(c)
    test_pairwise_dist_grad_matches_pairwise_dist()
    test_iqe_under_bf16_autocast_matches_fp32()
    test_encode_z_precision_under_autocast()
    print("test_qrl_iqe: OK")
