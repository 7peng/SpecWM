"""The within-clip Gram term on a planted kernel: `_block_ctr` against an fp64 reference,
`_direct_gram_within_term` zero at an exact realisation, invariant to clip permutation, not to a
within-clip frame shuffle, 1 on a collapsed embedding, the `live` mask polarity, and the
degenerate-clip handling of `_factor_block_kernel` / `_cmse_blocks`."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_spec as T  # noqa: E402

torch.manual_seed(0)
C, N, D = 5, 9, 16
SD = {"weight": 1.0, "score_diagonal": False}
SDD = {"weight": 1.0, "score_diagonal": True}
M = 1.0 - torch.eye(N)


def term(sd, out, emb, Kb, live, Gabs):
    return T._direct_gram_within_term(sd, out, emb, Kb, live, emb[:, 3:], torch.arange(3, N), Gabs, {})


def ref_block_ctr(M_, m):
    M_, m = np.asarray(M_, np.float64), np.asarray(m, np.float64)
    rs = (M_ * m).sum(-1, keepdims=True) / np.maximum(m.sum(-1, keepdims=True), 1e-12)
    cs = (M_ * m).sum(-2, keepdims=True) / np.maximum(m.sum(-2, keepdims=True), 1e-12)
    return M_ - rs - cs + (M_ * m).sum((-2, -1), keepdims=True) / m.sum()


def test_block_ctr():
    X = torch.randn(C, N, N).double()
    got = T._block_ctr(X, M.double()).numpy()
    assert np.abs(got - ref_block_ctr(X.numpy(), M.numpy())).max() < 1e-12
    off = torch.arange(C).double().view(C, 1, 1) * 3.7
    g2 = T._block_ctr(X + off, M.double()).numpy()
    assert np.abs((got - g2)[:, ~np.eye(N, dtype=bool)]).max() < 1e-11, "not centring per clip"


def _planted():
    h = torch.nn.functional.normalize(torch.randn(C, N, D), dim=-1)
    Gabs = (h @ h.mT).detach()
    Kb = T._block_ctr(Gabs, M)
    Kb = Kb / Kb.diagonal(dim1=1, dim2=2).sum(-1).abs().clamp_min(1e-12)[:, None, None] * N
    return h, Gabs, Kb


def test_exact_realisation_and_invariances():
    h, Gabs, Kb = _planted()
    out = {}
    l0 = term(SD, out, h.clone().requires_grad_(True), Kb, None, Gabs)
    assert float(l0) < 1e-10, float(l0)
    assert float(out["gram_corr_within"]) > 0.99      # masked centring is not idempotent
    assert float(term(SDD, {}, h, Kb, None, Gabs)) < 1e-10, "score_diagonal breaks the exact case"
    assert abs(float(term(SD, {}, h * 7.0, Kb, None, Gabs))) < 1e-9, "row scale must not matter"

    perm = torch.randperm(C)
    assert abs(float(term(SD, {}, h[perm], Kb[perm], None, Gabs[perm])) - float(l0)) < 1e-9
    bad = torch.roll(torch.arange(C), 1)
    assert float(term(SD, {}, h[bad], Kb, None, Gabs)) > 0.5, "the term is not per clip"

    fperm = torch.randperm(N)
    while bool((fperm == torch.arange(N)).all()):
        fperm = torch.randperm(N)
    assert float(term(SD, {}, h[:, fperm], Kb, None, Gabs)) > 0.1, "a frame shuffle is invisible"
    lf2 = term(SD, {}, h[:, fperm], Kb[:, fperm][:, :, fperm], None, Gabs[:, fperm][:, :, fperm])
    assert float(lf2) < 1e-9

    lz = term(SD, {}, torch.ones(C, N, D), Kb, None, Gabs)
    assert abs(float(lz) - 1.0) < 1e-5, f"a collapsed embedding must read 1, got {float(lz)}"


def test_block_corr_limits_and_live_polarity():
    h, Gabs, Kb = _planted()
    S = h @ h.mT
    one = torch.zeros(N, N)
    one[0, 1] = 1.0
    c1 = T._block_corr(S, T._block_ctr(Kb, one), one)
    assert not torch.isfinite(c1) or abs(float(c1)) <= 1.0
    allm = torch.zeros(N, N)
    assert float(T._block_corr(S, T._block_ctr(Kb, allm), allm)) == 0.0

    live = torch.zeros(C, dtype=torch.bool)
    live[0] = True
    lL = term(SD, {}, h, Kb, live, Gabs)
    h2 = h.clone()
    h2[0] = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
    lL2 = term(SD, {}, h2, Kb, live, Gabs)
    assert float(lL) < 1e-9 < float(lL2), "`live` True must mean COUNTED"


def test_degenerate_clips():
    Cd, Nd, Dd = 4, 12, 16
    tt = torch.arange(Nd).float()
    base = (tt[:, None] - tt[None, :]).abs()
    dfull = torch.zeros(Cd * Nd, Cd * Nd)
    for c in range(Cd):
        dfull[c * Nd:(c + 1) * Nd, c * Nd:(c + 1) * Nd] = base * (1.0 if c < Cd - 1 else 1e-4)
    cid = torch.arange(Cd * Nd) // Nd
    dfull = dfull + 40.0 * (cid[:, None] != cid[None, :]).float()
    fq = {"form": "quasimetric", "teacher": "tea", "sigma": 1.0, "weight": 1.0, "orient": True}
    Kbd, lived = T._factor_block_kernel([fq], {}, Cd, Nd, {"tea": dfull}, device="cpu")
    Gd = T._abs_block_gram([fq], Cd, Nd, {"tea": dfull}, "cpu")
    assert bool(lived.all()), "a scaled-down clip passes `live` (trace-normalised)"
    embd = torch.nn.functional.normalize(torch.randn(Cd, Nd, Dd), dim=-1)
    od = {}
    ld = T._direct_gram_within_term(SD, od, embd, Kbd, lived, embd[:, 3:], torch.arange(3, Nd), Gd, {})
    assert torch.isfinite(ld)
    assert float(od["kern_std_min_frac"]) < 1e-2, "the monitor does not see the degenerate clip"
    dz = dfull.clone()
    dz[3 * Nd:4 * Nd, 3 * Nd:4 * Nd] = 0.0
    Kbz, livez = T._factor_block_kernel([fq], {}, Cd, Nd, {"tea": dz}, device="cpu")
    Gz = T._abs_block_gram([fq], Cd, Nd, {"tea": dz}, "cpu")
    assert not bool(livez[3]), "an exactly constant clip must be dropped from `live`"
    lz = T._direct_gram_within_term(SD, {}, embd, Kbz, livez, embd[:, 3:], torch.arange(3, Nd), Gz, {})
    assert torch.isfinite(lz) and float(lz) < 1e3


if __name__ == "__main__":
    test_block_ctr()
    test_exact_realisation_and_invariances()
    test_block_corr_limits_and_live_polarity()
    test_degenerate_clips()
    print("test_tdwm_within_gram: OK")
