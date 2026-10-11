"""`FrozenQRLTeacher`: loads a step snapshot, a Lightning checkpoint or a weights_epoch file with
its heads sidecar to the same weights, stays in eval mode under `.train(True)`, carries no trainable
parameter, replaces its unused halves with Identity, and hands out one distance under every
accessor (per-clip, flat, from pixels) with the configured symmetrisation."""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quasimetric import FrozenQRLTeacher  # noqa: E402
from losses import QRLHead                # noqa: E402

HKW = dict(embed_dim=16, hidden=32, z_dim=32, group_size=8, push_linear=True)


class _TinyModel(nn.Module):
    """Stands in for jepa.JEPA: encode({'pixels'}) -> {'emb'}, plus the halves a teacher drops."""

    def __init__(self, embed_dim=16):
        super().__init__()
        self.proj = nn.Linear(12, embed_dim)
        self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim))
        self.predictor = nn.Linear(2, 2)
        self.action_encoder = nn.Linear(2, 2)
        self.pred_proj = nn.Linear(2, 2)
        self.pred_norm = nn.Identity()

    def encode(self, batch):
        px = batch["pixels"]
        B, N = px.shape[:2]
        return {"emb": self.projector(self.proj(px.reshape(B, N, -1).float()))}


def _blobs(tmp):
    torch.manual_seed(0)
    m, h = _TinyModel(), QRLHead(**HKW)
    snap = tmp / "step_7.pt"
    torch.save({"model": m.state_dict(), "heads": {"qrl_head": h.state_dict()}, "step": 7}, snap)
    lightning = tmp / "last.ckpt"
    sd = {f"model.{k}": v for k, v in m.state_dict().items()}
    sd.update({f"qrl_head.{k}": v for k, v in h.state_dict().items()})
    torch.save({"state_dict": sd}, lightning)
    epoch = tmp / "weights_epoch_3.pt"
    torch.save(m.state_dict(), epoch)
    torch.save({"qrl_head": h.state_dict()}, tmp / "heads_epoch_3.pt")
    return snap, lightning, epoch


def _teacher(ckpt, symmetrize="mean"):
    return FrozenQRLTeacher(str(ckpt), {"_target_": f"{__name__}._TinyModel", "embed_dim": 16},
                            HKW, symmetrize=symmetrize)


def test_every_checkpoint_format_loads_the_same_teacher(tmp_path):
    snap, lightning, epoch = _blobs(tmp_path)
    ts = [_teacher(p) for p in (snap, lightning, epoch)]
    px = torch.randn(2, 4, 12)
    ref = ts[0].all_pairs_dist(px)
    for t in ts[1:]:
        assert torch.equal(t.all_pairs_dist(px), ref)
    os.remove(tmp_path / "heads_epoch_3.pt")
    with pytest.raises(AssertionError, match="sidecar"):
        _teacher(epoch)


def test_frozen_eval_and_pruned(tmp_path):
    t = _teacher(_blobs(tmp_path)[0])
    assert not t.training
    t.train(True)
    assert not t.training and not t.model.training
    assert all(not p.requires_grad for p in t.parameters())
    for name in ("predictor", "action_encoder", "pred_proj", "pred_norm"):
        assert isinstance(getattr(t.model, name), nn.Identity), name
    with pytest.raises(AssertionError):
        _teacher(_blobs(tmp_path)[0], symmetrize="roundtrip")


def test_accessors_agree_and_symmetrise(tmp_path):
    snap = _blobs(tmp_path)[0]
    t = _teacher(snap, symmetrize="mean")
    px = torch.randn(1, 6, 12)
    z = t.z_of(px)
    per_clip = t.key_dist_from_z(z)[0]
    flat = t.all_pairs_dist_from_z(z.reshape(1, -1, z.shape[-1]))
    assert torch.equal(per_clip, flat) and torch.equal(t.all_pairs_dist(px), flat)
    assert torch.equal(flat, flat.T) and float(flat.diagonal().abs().max()) < 1e-6
    raw = t.head.pairwise_dist(z)[0]
    assert torch.equal(flat, 0.5 * (raw + raw.T))
    tn = _teacher(snap, symmetrize="none")
    directed = tn.all_pairs_dist(px)
    assert torch.equal(directed, raw) and float((directed - directed.T).abs().max()) > 1e-4
    with pytest.raises(AssertionError, match="one flat block"):
        t.all_pairs_dist_from_z(t.z_of(torch.randn(3, 4, 12)))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        for i, fn in enumerate((test_every_checkpoint_format_loads_the_same_teacher,
                                test_frozen_eval_and_pruned, test_accessors_agree_and_symmetrise)):
            d = Path(td) / str(i)
            d.mkdir()
            fn(d)
    print("test_qrl_frozen_teacher: OK")
