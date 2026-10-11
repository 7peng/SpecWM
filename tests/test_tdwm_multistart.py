"""The multistart index on labelled frames: every (start, horizon) the config allows appears
exactly once, the context is the H frames ending at `start`, posframe names the frame each
prediction targets, the action bridge carries the actions of frames start .. start+h-1 of the same
clip and never the zero-padded final block. Checked on the HorizonPredictor path and on the
ARPredictor path (one step ahead, one action token per context frame)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm_harness import ARPredictor, StubModule, labelled_emb, make_batch, make_cfg  # noqa: E402
import train_spec as T  # noqa: E402


def _expected(N, H, max_ph):
    s, h = [], []
    for i in range(H - 1, N - 1):
        nt = min(max_ph, N - 1 - i)
        s += [i] * nt
        h += list(range(1, nt + 1))
    return s, h


def _labelled_act(B, N, D):
    act = torch.zeros(B, N, D)
    act[..., 0] = torch.arange(B).float()[:, None]
    act[..., 1] = torch.arange(N).float()[None, :]
    return act


def _run(N, H, max_ph, B=3, D=8, ar=False):
    emb = labelled_emb(B, N, D)
    act = _labelled_act(B, N, D)
    pred = ARPredictor(emb, num_frames=H) if ar else None
    mod = StubModule(make_cfg(N, H, max_ph), emb, act, predictor=pred)
    out = mod(make_batch(B, N))
    return mod, out


def check_horizon_geometry(N, H, max_ph, B=3):
    mod, out = _run(N, H, max_ph, B=B)
    s_ref, h_ref = _expected(N, H, max_ph)
    P = len(s_ref)
    pairs = sorted(zip(s_ref, h_ref))
    assert len(set(pairs)) == len(pairs) and all(h >= 1 for h in h_ref)
    assert all(H - 1 <= s <= N - 2 for s in s_ref) and max(s + h for s, h in pairs) <= N - 1

    ce, be, hor, amask = mod.model.predictor.calls[0]
    assert ce.shape == (B * P, H, 8), ce.shape
    ce_ = ce.view(B, P, H, -1)
    for b in range(B):
        for p in range(P):
            got = ce_[b, p, :, 1].round().long().tolist()
            assert got == list(range(s_ref[p] - H + 1, s_ref[p] + 1)), (N, H, max_ph, b, p, got)
            assert int(ce_[b, p, 0, 0].round()) == b, "context row came from another clip"
    assert hor.view(B, P)[0].tolist() == h_ref
    assert float(out["pred_loss"]) < 1e-10, "an oracle predictor must score pred_loss 0"
    assert float(out["pred_cos"]) > 1 - 1e-6

    Tm = max(h_ref)
    be_ = be.view(B, P, Tm, -1)
    am = amask.view(B, P, Tm)
    for b in range(B):
        for p in range(P):
            live = am[b, p].nonzero().flatten().tolist()
            assert live == list(range(h_ref[p])), f"amask {live} for horizon {h_ref[p]}"
            got = be_[b, p, : h_ref[p], 1].round().long().tolist()
            assert got == list(range(s_ref[p], s_ref[p] + h_ref[p])), (N, H, max_ph, got)
            assert be_[b, p, : h_ref[p], 0].round().long().eq(b).all(), "bridge crossed clips"
            assert N - 1 not in got, "the padded final action block was consumed"


def test_horizon_predictor_geometries():
    for (N, H, max_ph) in [(35, 3, 1), (35, 3, 4), (12, 3, 32), (10, 3, 4), (6, 1, 2)]:
        check_horizon_geometry(N, H, max_ph)


def test_ar_predictor_geometry():
    N, H, B = 23, 3, 3
    mod, out = _run(N, H, 1, B=B, ar=True)
    s_ref, _ = _expected(N, H, 1)
    P = len(s_ref)
    x, c = mod.model.predictor.calls[0]
    assert x.shape == (B * P, H, 8) and c.shape == (B * P, H, 8)
    x_, c_ = x.view(B, P, H, -1), c.view(B, P, H, -1)
    for b in range(B):
        for p in range(P):
            want = list(range(s_ref[p] - H + 1, s_ref[p] + 1))
            assert x_[b, p, :, 1].round().long().tolist() == want
            assert c_[b, p, :, 1].round().long().tolist() == want, "action tokens are not the context frames'"
            assert c_[b, p, :, 0].round().long().eq(b).all()
    assert float(out["pred_loss"]) < 1e-10


def test_ar_predictor_refuses_multi_step_horizons():
    emb = labelled_emb(2, 12, 8)
    mod = StubModule(make_cfg(12, 3, 4), emb, _labelled_act(2, 12, 8), predictor=ARPredictor(emb, 3))
    try:
        mod(make_batch(2, 12))
    except AssertionError as e:
        assert "max_pred_horizon" in str(e)
        return
    raise AssertionError("an ARPredictor accepted max_pred_horizon 4")


def test_multistart_cache_key():
    T._MULTISTART_CACHE.clear()
    a = T._multistart_index(*_expected(35, 3, 1), 3, 35, 1, "cpu")
    b = T._multistart_index(*_expected(35, 3, 4), 3, 35, 4, "cpu")
    assert a[0].numel() != b[0].numel(), "the multistart cache collided across max_pred_horizon"


if __name__ == "__main__":
    test_horizon_predictor_geometries()
    test_ar_predictor_geometry()
    test_ar_predictor_refuses_multi_step_horizons()
    test_multistart_cache_key()
    print("test_tdwm_multistart: OK")
