"""The contrastive RL term equals OGBench CRL's binary NCE (through its phi/psi heads) and the InfoNCE form on the
same anchor-goal pairs, with goals from an anchor's own clip left out; binary NCE without heads is refused."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest  # noqa: E402
from losses import CRLHeads  # noqa: E402
from train_spec import _crl_term  # noqa: E402


def _pairs(B, N, D, seed):
    torch.manual_seed(seed)
    t = torch.arange(N - 1).expand(B, N - 1)
    off = torch.distributions.Geometric(probs=torch.tensor(0.2)).sample(t.shape).long() + 1
    gt = torch.minimum(t + off, torch.full_like(t, N - 1))
    clip = np.repeat(np.arange(B), N - 1); lab = np.eye(B * (N - 1))
    keep = (clip[:, None] != clip[None, :]) | lab.astype(bool)
    return t, gt, lab, keep


def _rows(emb, gt):
    B, N, D = emb.shape
    return emb[:, :N - 1].reshape(-1, D), emb[torch.arange(B)[:, None], gt].reshape(-1, D)


def test_crl_binary_with_heads_matches_numpy():
    torch.manual_seed(0)
    B, N, D = 4, 6, 8
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    heads = CRLHeads(D, hidden_dims=(16, 16), latent_dim=12)
    cfg = {"discount": 0.8, "weight": 1.0, "form": "binary", "anchor": {"weight": 0.0}}
    torch.manual_seed(5); out = {}
    loss = float(_crl_term(cfg, out, emb, pred, pos, heads))
    t, gt, lab, keep = _pairs(B, N, D, 5)
    anc, goal = _rows(emb, gt)
    with torch.no_grad():
        logits = (heads.phi(anc) @ heads.psi(goal).T).double().numpy() / np.sqrt(12)
    bce = np.maximum(logits, 0) - logits * lab + np.log1p(np.exp(-np.abs(logits)))   # optax.sigmoid_binary_cross_entropy
    ref = (bce * keep).sum() / keep.sum()
    assert abs(loss - ref) < 1e-5, (loss, ref)
    assert (gt > t).all() and (gt <= N - 1).all(), "positives must lie later in the clip"


def test_crl_infonce_matches_numpy():
    torch.manual_seed(0)
    B, N, D = 4, 6, 8
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    cfg = {"discount": 0.8, "weight": 1.0, "form": "infonce", "anchor": {"weight": 0.0}}
    torch.manual_seed(5); out = {}
    loss = float(_crl_term(cfg, out, emb, pred, pos))
    t, gt, lab, keep = _pairs(B, N, D, 5)
    anc, goal = _rows(emb, gt)
    logits = (anc @ goal.T).double().numpy() / np.sqrt(D)
    z = np.where(keep, logits, -np.inf)
    lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
    ref = (lse - logits.diagonal()).mean()
    assert abs(loss - ref) < 1e-5, (loss, ref)
    assert abs(float(out["crl_loss_trivial"]) - np.log(keep[0].sum())) < 1e-6


def test_crl_binary_without_heads_is_refused():
    emb = torch.randn(2, 4, 8)
    with pytest.raises(AssertionError, match="log 2 floor"):
        _crl_term({"discount": 0.8, "form": "binary", "anchor": {"weight": 0.0}}, {}, emb, torch.randn(2, 1, 8), torch.tensor([1]))


def test_crl_positive_offsets_follow_the_discount():
    torch.manual_seed(0); g = 0.8669
    off = torch.distributions.Geometric(probs=torch.tensor(1 - g)).sample((200000,)).long() + 1
    assert abs(off.float().mean().item() - 1 / (1 - g)) < 0.05


def test_crl_anchor_sum_is_pair_mean_times_goals():
    torch.manual_seed(0)
    B, N, D = 4, 6, 8
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    heads = CRLHeads(D, hidden_dims=(16,), latent_dim=12)
    base = {"discount": 0.8, "weight": 1.0, "form": "binary", "anchor": {"weight": 0.0}}
    torch.manual_seed(5); o1 = {}; l_mean = float(_crl_term({**base, "reduction": "pair_mean"}, o1, emb, pred, pos, heads))
    torch.manual_seed(5); o2 = {}; l_sum = float(_crl_term({**base, "reduction": "anchor_sum"}, o2, emb, pred, pos, heads))
    K = 1 + (B - 1) * (N - 1)
    assert abs(l_sum - K * l_mean) < 1e-4 * K, (l_sum, K * l_mean)
    p = 1.0 / K; h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert abs(float(o1["crl_loss_trivial"]) - h) < 1e-6 and abs(float(o2["crl_loss_trivial"]) - K * h) < 1e-5


def test_truncated_positives_follow_the_clip_truncated_law():
    torch.manual_seed(0)
    B, N, D, g = 64, 8, 4, 0.8
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    cfg = {"discount": g, "form": "infonce", "positives": "truncated", "negatives": "all", "anchor": {"weight": 0.0}}
    hits = np.zeros((N - 1, N))
    for i in range(300):
        torch.manual_seed(i)
        out = {}; _crl_term(cfg, out, emb, pred, pos)
        torch.manual_seed(i)
        t = torch.arange(N - 1).expand(B, N - 1); room = (N - 1 - t).double()
        u = torch.rand(t.shape, dtype=torch.float64)
        gt = t + torch.minimum(torch.floor(torch.log1p(-u * (1 - g ** room)) / np.log(g)).long() + 1, room.long())
        for a, b in zip(t.reshape(-1).tolist(), gt.reshape(-1).tolist()):
            hits[a, b] += 1
    for a in (0, 3, 5):
        R = N - 1 - a; ref = g ** np.arange(R); ref /= ref.sum()
        emp = hits[a, a + 1:a + 1 + R] / hits[a].sum()
        assert hits[a, :a + 1].sum() == 0 and np.abs(emp - ref).max() < 0.02, (a, emp, ref)


def test_within_clip_negatives_mask_only_duplicates():
    torch.manual_seed(1)
    B, N, D = 3, 6, 8
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    cfg = {"discount": 0.8, "form": "infonce", "positives": "truncated", "negatives": "all", "anchor": {"weight": 0.0}}
    torch.manual_seed(7); out = {}; loss = float(_crl_term(cfg, out, emb, pred, pos))
    torch.manual_seed(7)
    t = torch.arange(N - 1).expand(B, N - 1); room = (N - 1 - t).double()
    u = torch.rand(t.shape, dtype=torch.float64)
    gt = (t + torch.minimum(torch.floor(torch.log1p(-u * (1 - 0.8 ** room)) / np.log(0.8)).long() + 1, room.long())).numpy()
    e = emb.numpy().astype(np.float64)
    anc = e[:, :N - 1].reshape(-1, D); goal = e[np.arange(B)[:, None], gt].reshape(-1, D)
    z = anc @ goal.T / np.sqrt(D)
    clip = np.repeat(np.arange(B), N - 1); ga, ta = gt.reshape(-1), t.numpy().reshape(-1)
    same = clip[:, None] == clip[None, :]
    keep = ~(same & ((ga[None, :] == ga[:, None]) | (ga[None, :] == ta[:, None]))) | np.eye(len(z), dtype=bool)
    zm = np.where(keep, z, -np.inf); m = zm.max(1, keepdims=True)
    ref = (np.log(np.exp(zm - m).sum(1)) + m[:, 0] - z.diagonal()).mean()
    assert abs(loss - ref) < 1e-5, (loss, ref)
    assert same[keep & ~np.eye(len(z), dtype=bool)].any(), "own-clip goals must remain as negatives"


def test_far_rows_join_infonce_as_negatives_only():
    torch.manual_seed(2)
    B, N, D, nf = 3, 6, 8, 2
    emb = torch.randn(B, N, D); pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3]); far = torch.randn(B, nf, D)
    cfg = {"discount": 0.8, "form": "infonce", "positives": "truncated", "negatives": "all", "anchor": {"weight": 0.0}}
    torch.manual_seed(9); out = {}; loss = float(_crl_term(cfg, out, emb, pred, pos, None, None, far))
    torch.manual_seed(9)
    t = torch.arange(N - 1).expand(B, N - 1); room = (N - 1 - t).double()
    u = torch.rand(t.shape, dtype=torch.float64)
    gt = (t + torch.minimum(torch.floor(torch.log1p(-u * (1 - 0.8 ** room)) / np.log(0.8)).long() + 1, room.long())).numpy()
    e = emb.numpy().astype(np.float64)
    anc = e[:, :N - 1].reshape(-1, D)
    goal = np.concatenate([e[np.arange(B)[:, None], gt].reshape(-1, D), far.numpy().astype(np.float64).reshape(-1, D)])
    z = anc @ goal.T / np.sqrt(D)
    clip = np.repeat(np.arange(B), N - 1); ga, ta = gt.reshape(-1), t.numpy().reshape(-1)
    same = clip[:, None] == clip[None, :]
    keep = ~(same & ((ga[None, :] == ga[:, None]) | (ga[None, :] == ta[:, None]))) | np.eye(len(anc), dtype=bool)
    keep = np.concatenate([keep, np.ones((len(anc), B * nf), bool)], 1)
    zm = np.where(keep, z, -np.inf); m = zm.max(1, keepdims=True)
    ref = (np.log(np.exp(zm - m).sum(1)) + m[:, 0] - z[np.arange(len(anc)), np.arange(len(anc))]).mean()
    assert abs(loss - ref) < 1e-5, (loss, ref)
