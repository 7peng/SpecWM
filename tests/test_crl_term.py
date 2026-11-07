"""The contrastive RL term equals OGBench CRL's binary NCE on the same anchor-goal pairs, with goals from an
anchor's own clip left out."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_spec import _crl_term  # noqa: E402


def test_crl_matches_numpy_binary_nce():
    torch.manual_seed(0)
    B, N, D = 4, 6, 8
    emb = torch.randn(B, N, D)
    pred = torch.randn(B, 3, D); pos = torch.tensor([1, 2, 3])
    cfg = {"discount": 0.8, "weight": 1.0, "anchor": {"weight": 0.0}}
    torch.manual_seed(5); out = {}
    loss = float(_crl_term(cfg, out, emb, pred, pos))
    torch.manual_seed(5)
    t = torch.arange(N - 1).expand(B, N - 1)
    off = torch.distributions.Geometric(probs=torch.tensor(0.2)).sample(t.shape).long() + 1
    gt = torch.minimum(t + off, torch.full_like(t, N - 1)).numpy()
    e = emb.numpy().astype(np.float64)
    anc = e[:, :N - 1].reshape(-1, D); goal = e[np.arange(B)[:, None], gt].reshape(-1, D)
    logits = anc @ goal.T / np.sqrt(D)
    lab = np.eye(len(anc)); clip = np.repeat(np.arange(B), N - 1)
    keep = (clip[:, None] != clip[None, :]) | lab.astype(bool)
    bce = np.maximum(logits, 0) - logits * lab + np.log1p(np.exp(-np.abs(logits)))   # optax.sigmoid_binary_cross_entropy
    ref = (bce * keep).sum() / keep.sum()
    assert abs(loss - ref) < 1e-5, (loss, ref)
    assert (gt > t.numpy()).all() and (gt <= N - 1).all(), "positives must lie later in the clip"


def test_crl_positive_offsets_follow_the_discount():
    torch.manual_seed(0); g = 0.8669
    off = torch.distributions.Geometric(probs=torch.tensor(1 - g)).sample((200000,)).long() + 1
    assert abs(off.float().mean().item() - 1 / (1 - g)) < 0.05
