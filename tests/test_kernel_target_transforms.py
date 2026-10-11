"""`distance_affinity`: exp(-d/sigma), the "mean" symmetrisation averages the two directed
affinities (not the distances); and the quasimetric factor's consumers in train_spec build a
finite, differentiable term from a directed distance matrix."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quasimetric.kernel_targets import distance_affinity  # noqa: E402


def test_directed_affinity_symmetrization():
    d = torch.tensor([[0., 1.], [9., 0.]])
    k = distance_affinity(d, 2., "mean")
    expected = (np.exp(-0.5) + np.exp(-4.5)) / 2
    assert abs(k[0, 1].item() - expected) < 1e-7
    assert torch.equal(k, k.T)
    assert torch.equal(k.diag(), torch.ones(2))
    assert k[0, 1] > torch.exp(-((d + d.T) / 4))[0, 1]
    symmetric = (d + d.T) / 2
    assert torch.equal(distance_affinity(symmetric, 2.), torch.exp(-symmetric / 2))
    assert torch.equal(distance_affinity(symmetric, 2., "mean"), distance_affinity(symmetric, 2.))
    for bad in ({"sigma": 0.0}, {"symmetrize": "roundtrip"}):
        try:
            distance_affinity(d, bad.get("sigma", 1.0), bad.get("symmetrize", "none"))
        except AssertionError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_training_consumers():
    import train_spec as training
    n, c = 6, 2
    torch.manual_seed(0)
    d = torch.rand(c * n, c * n) * 8
    d.fill_diagonal_(0)
    factor = dict(form="quasimetric", teacher="test", sigma=2., weight=1.)
    raw = distance_affinity(d, 2., "none")
    expected = torch.stack([raw[i * n:(i + 1) * n, i * n:(i + 1) * n] for i in range(c)])
    absolute = training._abs_block_gram([factor], c, n, {"test": d}, "cpu")
    torch.testing.assert_close(absolute, expected)
    blocks, live = training._factor_block_kernel([factor], {}, c, n, {"test": d}, "cpu")
    assert blocks.shape == (c, n, n) and bool(live.all())
    emb = torch.randn(c, n, 12, requires_grad=True)
    pred = torch.randn(c, n - 3, 12, requires_grad=True)
    cfg = dict(weight=1., anchor=dict(weight=2., stop_gradient=False))
    loss = training._direct_cross_gram_term(cfg, {}, emb, pred, torch.arange(3, n), blocks, live, {}, absolute)
    loss.backward()
    for grad in (emb.grad, pred.grad):
        assert torch.isfinite(grad).all() and grad.norm() > 0


if __name__ == "__main__":
    test_directed_affinity_symmetrization()
    test_training_consumers()
    print("test_kernel_target_transforms: OK")
