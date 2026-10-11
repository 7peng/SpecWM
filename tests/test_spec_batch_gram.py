"""The batch-scope objective: the tempered target's trace and rank, zero at the spectral
embedding of the target (alpha 1), an fp64 reference of the loss at a random table (alpha 0.5),
and the `normalise: relative` form."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_spec as T  # noqa: E402


def _sd(alpha, modes, **kw):
    return {"temper": {"alpha": alpha, "modes": modes}, "weight": 1.0, "anchor": {"weight": 0.0}, **kw}


def test_batch_gram():
    torch.manual_seed(0)
    n, d, m = 60, 16, 128
    H0 = torch.nn.functional.normalize(torch.randn(n, d, dtype=torch.float64), dim=-1)
    G = H0 @ H0.T

    T1 = T._spec_target(G.float(), 1.0, m)
    assert abs(float(T1.diagonal().sum()) - n) < 1e-3
    Cm = np.eye(n) - 1.0 / n
    Kc = Cm @ G.numpy() @ Cm
    ref = n * Kc / np.trace(Kc)
    assert np.abs(T1.numpy() - ref).max() < 1e-3
    assert int((torch.linalg.eigvalsh(T1) > 1e-4).sum()) <= d

    w, U = np.linalg.eigh(ref)
    keep = w > 1e-8
    Hs = U[:, keep] * np.sqrt(w[keep])
    emb = torch.tensor(np.concatenate([Hs, np.zeros((n, d - Hs.shape[1]))], 1), dtype=torch.float32)[None]
    pred, pos = emb[:, 3:], torch.arange(3, n)
    out = {}
    l = T._direct_gram_batch_term(_sd(1.0, m), out, emb, pred, pos, G.float())
    assert float(l) < 1e-6, f"spectral embedding scores {float(l)}"
    assert float(out["gram_corr_batch"]) > 0.9999

    Hr = torch.randn(1, n, d)
    l = T._direct_gram_batch_term(_sd(0.5, m), {}, Hr, Hr[:, 3:], pos, G.float())
    wk = np.clip(w, 0, None) ** 0.5
    Tt = n * (U * wk) @ U.T / wk.sum()
    hn = Hr[0].double().numpy()
    hn = hn / np.linalg.norm(hn, axis=1, keepdims=True)
    hb = hn * np.sqrt(np.clip(np.diag(Tt), 0, None))[:, None]
    Sc = Cm @ (hb @ hb.T) @ Cm
    e_ref = ((Sc - Tt) ** 2).mean()
    assert abs(float(l) - e_ref) < 1e-4 * max(1.0, e_ref)

    lc = T._direct_gram_batch_term(_sd(1.0, m, normalise="relative"), {}, torch.ones(1, n, d),
                                   torch.ones(1, n - 3, d), pos, G.float())
    wk1 = np.clip(w, 0, None)
    T1n = n * (U * wk1) @ U.T / wk1.sum()
    s = np.sqrt(np.clip(np.diag(T1n), 0, None))[:, None]
    Sc1 = Cm @ (s @ s.T) @ Cm
    assert abs(float(lc) - ((Sc1 - T1n) ** 2).mean() / (T1n ** 2).mean()) < 1e-4, float(lc)
    lz = T._direct_gram_batch_term(_sd(1.0, m, normalise="relative"), {}, emb, pred, pos, G.float())
    assert float(lz) < 1e-5, float(lz)


if __name__ == "__main__":
    test_batch_gram()
    print("test_spec_batch_gram: OK")
