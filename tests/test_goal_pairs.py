"""OGBench GCDataset goal sampling (sample_goal_pairs) and the contrastive term on those pairs."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from losses import CRLHeads  # noqa: E402
from train_spec import _crl_term  # noqa: E402
from utils import sample_goal_pairs  # noqa: E402


def test_goals_follow_the_geometric_law_inside_the_episode():
    g = torch.Generator().manual_seed(0)
    lengths = [5000, 3000, 40, 4000]
    ep, t, goal = sample_goal_pairs(lengths, [0, 1, 3], 200000, 0.99, g)
    L = torch.tensor(lengths)[ep]
    assert set(ep.tolist()) <= {0, 1, 3}, "anchors only from the episodes given"
    assert (t >= 0).all() and (t <= L - 2).all() and (goal > t).all() and (goal <= L - 1).all()
    k = (goal - t)[t <= L - 1 - 2000].double()                # anchors with room: the cap has probability 0.99^2000
    assert abs(k.mean().item() - 100.0) < 1.5, k.mean()       # 1 / (1 - 0.99)
    assert abs((k == 1).double().mean().item() - 0.01) < 0.002
    share = torch.bincount(ep, minlength=4).double() / len(ep)
    w = torch.tensor([4999.0, 2999.0, 0.0, 3999.0]); w /= w.sum()
    assert (share - w).abs().max() < 0.01, "anchors uniform over frames, so episodes weighted by length"


def test_pair_crl_matches_numpy_binary_nce():
    torch.manual_seed(0)
    B, N, D, n = 3, 6, 8, 4
    emb, pred, pos = torch.randn(B, N, D), torch.randn(B, 3, D), torch.tensor([1, 2, 3])
    pairs = torch.randn(B, 2 * n, D)
    heads = CRLHeads(D, hidden_dims=(16,), latent_dim=12)
    out = {}
    loss = float(_crl_term({"weight": 1.0, "form": "binary", "reduction": "anchor_sum", "anchor": {"weight": 0.0}},
                           out, emb, pred, pos, heads, pairs))
    with torch.no_grad():
        a, g = heads.phi(pairs[:, :n].reshape(-1, D)), heads.psi(pairs[:, n:].reshape(-1, D))
        z = (a @ g.T).double().numpy() / np.sqrt(12)
    lab = np.eye(B * n)
    bce = np.maximum(z, 0) - z * lab + np.log1p(np.exp(-np.abs(z)))      # optax.sigmoid_binary_cross_entropy
    assert abs(loss - bce.sum(1).mean()) < 1e-4, (loss, bce.sum(1).mean())
    K = B * n; p = 1 / K
    assert abs(float(out["crl_loss_trivial"]) - K * -(p * np.log(p) + (1 - p) * np.log(1 - p))) < 1e-5
