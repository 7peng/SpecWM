"""Training entry point of the spectral world model: the encoder's within-clip (or whole-batch) Gram is
regressed onto a target kernel and the predictor onto the frames it predicts. The same entry point trains
the quasimetric teachers (quasimetric_head set, no kernel term).

    python src/train_spec.py --config-name=<recipe> run_id=<id>

TEACHER_DUMP=<path.npz> writes one batch of teacher distances and exits (the sigma solve reads it).
"""
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import train_common
from losses import SIGReg, QRLHead
from quasimetric import FrozenQRLTeacher, distance_affinity
from utils import GPUPixelNormalize


class _PlainCfg(dict):
    """A resolved plain-dict copy of the DictConfig for the training step: attribute access and `.get` like
    OmegaConf, without the reference cycle a DictConfig `.get` on a missing key leaks per call."""
    __slots__ = ()

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    @classmethod
    def wrap(cls, x):
        if isinstance(x, dict):
            return cls({k: cls.wrap(v) for k, v in x.items()})
        if isinstance(x, list):
            return [cls.wrap(v) for v in x]
        return x


def _step_cfg(mod, cfg):
    """The step's view of `cfg`, converted once per process and cached on the module."""
    pc = getattr(mod, "_plain_cfg", None)
    if pc is None:
        pc = _PlainCfg.wrap(OmegaConf.to_container(cfg, resolve=True))
        mod._plain_cfg = pc
    return pc


def _quasimetric_monitors(head, output, device):
    """The live quasimetric head's per-step readings (set by `QRLHead.qrl_loss`)."""
    l_push, l_step, lam = head._parts
    output["quasimetric_push_loss"] = torch.as_tensor(l_push, device=device)
    output["quasimetric_step_loss"] = torch.as_tensor(l_step, device=device)
    output["quasimetric_lambda"] = torch.as_tensor(lam, device=device)
    d_mean, d_std = head._d_stats
    output["quasimetric_d_mean"] = torch.as_tensor(d_mean, device=device)
    output["quasimetric_d_std"] = torch.as_tensor(d_std, device=device)
    output["quasimetric_near_frac"] = torch.as_tensor(head._near_frac, device=device)
    output["quasimetric_z_group_share"] = torch.as_tensor(head._z_group_share, device=device)
    output["quasimetric_z_group_unif"] = torch.as_tensor(head._z_group_unif, device=device)
    output["quasimetric_z_rms"] = torch.as_tensor(head._z_rms, device=device)
    output["quasimetric_alpha"] = head.alpha.detach().squeeze()
    output["quasimetric_scale"] = F.softplus(head.raw_scale).detach().squeeze()
    output["quasimetric_n_pairs"] = torch.as_tensor(head._n_pairs, device=device)


# ------------------------------------------------------------------------------------ target kernels

def _cos_kernel_fp32(X, sigma):
    """exp(-(1 - cos)/sigma) on the rows of X, in fp32 with TF32 off (1 - cos is a cancellation)."""
    with torch.autocast("cuda", enabled=False):
        _tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            Xn = F.normalize(X.float(), dim=-1)
            return torch.exp(-(1.0 - Xn @ Xn.t()).clamp_min(0.0) / float(sigma))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _tf32


def _factor_kernel(form, X, sigma, f):
    """(n, n) similarity kernel of one state factor on its raw coordinates X (n, d): laplace exp(-d/sigma) on
    the coordinates or their (cos, sin) image (`embed: cos_sin`), cos on the row angle, indicator on equality."""
    if form == "indicator":
        return (X[:, None, :] == X[None, :, :]).all(-1).float()
    if form == "cos":
        return _cos_kernel_fp32(X, sigma)
    assert form == "laplace", f"factor form {form!r}; expected laplace | cos | indicator"
    if str(f.get("embed", "none")) == "cos_sin":
        X = torch.cat([X.cos(), X.sin()], -1)
    with torch.autocast(X.device.type, enabled=False):
        d = torch.cdist(X.float(), X.float(), compute_mode="donot_use_mm_for_euclid_dist")
    return torch.exp(-d / float(sigma))


def _flatten_spectrum(Kc, modes, power):
    """The centred kernel restricted to its `modes` leading eigenvectors with eigenvalues^power."""
    w, U = torch.linalg.eigh((Kc + Kc.t()) * 0.5)
    w, U = w.flip(0)[:modes], U.flip(1)[:, :modes]
    w = (w.clamp_min(0) ** power) * (w > 0)                   # power 0 must not revive null modes
    return (U * w) @ U.t()


def _factor_rows(f, rows):
    """The factor's own coordinates: its column, restricted to `dims` or `slice`."""
    Y = rows[str(f["key"])]
    if f.get("dims", None) is not None:
        Y = Y[..., [int(d) for d in f["dims"]]]
    elif f.get("slice", None) is not None:
        Y = Y[..., int(f["slice"][0]):int(f["slice"][1])]
    return Y


def _lag_sep(N, C, device, times=None):
    """(C, N, N) |t_i - t_j|: the row index, or `times` (C, N) when the clip's rows are not contiguous."""
    if times is None:
        idx = torch.arange(N, device=device, dtype=torch.float32)
        return (idx[:, None] - idx[None, :]).abs().unsqueeze(0).expand(C, N, N)
    t = times.to(device=device, dtype=torch.float32)
    return (t[:, :, None] - t[:, None, :]).abs()


def _blockify_center_orient(Kfull, C, N, f):
    """(C*N, C*N) -> (C, N, N) diagonal blocks, each centred (and sign-oriented under `orient`) in its own clip."""
    Kf = Kfull.reshape(C, N, C, N).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
    Kf = Kf - Kf.mean(-1, keepdim=True) - Kf.mean(-2, keepdim=True) + Kf.mean((-2, -1), keepdim=True)
    if f.get("orient", False):
        tr = Kf.diagonal(dim1=1, dim2=2).sum(-1)
        Kf = torch.where(tr[:, None, None] < 0, -Kf, Kf)
    return Kf


def _accum_block(Kb, wsum, live, Kf, f):
    """Per-clip trace-normalise one voice's blocks and add them in at the voice's weight. A clip whose block
    is degenerate is zeroed and dropped from `live`; a voice degenerate on more than 1 - min_live of the
    clips is refused."""
    tr = Kf.diagonal(dim1=1, dim2=2).sum(-1).abs()                      # (C,)
    ok = tr > 1e-9 * max(float(tr.max()), 1e-30)
    assert float(ok.float().mean()) >= float(f.get("min_live", 0.5)), (
        f"factor {f.get('key', f.get('teacher'))}/{f['form']} has a degenerate within-clip block on "
        f"{100 * (1 - float(ok.float().mean())):.0f}% of clips: it barely varies inside a clip")
    live = ok if live is None else (live & ok)
    Kf = torch.where(ok[:, None, None], Kf / tr.clamp_min(1e-12)[:, None, None], torch.zeros_like(Kf))
    w = float(f.get("weight", 1.0))
    return Kb + Kf * w if Kb is not None else Kf * w, wsum + w, live


def _factor_block_kernel(factors, rows, C, N, tdist=None, device=None, times=None):
    """K* on within-clip pairs: the (C, N, N) diagonal blocks of the factor sum, each voice centred and
    trace-normalised per clip. `tdist` holds the teachers' distance matrices for the quasimetric voices,
    `times` the row times when the clip carries far frames. Returns (Kb, live)."""
    Kb, wsum, live = None, 0.0, None
    for f in factors:
        form = str(f["form"])
        if form == "temporal":
            Kf = torch.exp(-_lag_sep(N, C, device, times) / float(f["sigma"]))
            Kf = Kf - Kf.mean(-1, keepdim=True) - Kf.mean(-2, keepdim=True) + Kf.mean((-2, -1), keepdim=True)
        elif form in ("quasimetric", "quasimetric_cached"):
            Kf = _blockify_center_orient(distance_affinity(tdist[str(f["teacher"])], f["sigma"], "none"), C, N, f)
        else:
            Kf = _blockify_center_orient(_factor_kernel(form, _factor_rows(f, rows), f.get("sigma", 1.0), f), C, N, f)
        Kb, wsum, live = _accum_block(Kb, wsum, live, Kf, f)
    return Kb * (N / wsum), live


def _abs_block_gram(factors, C, N, tdist, device, rows=None, times=None):
    """The absolute within-clip target: per clip, the weighted mean over the voices of their similarity
    kernels, diagonal blocks only, uncentred (diagonal 1, entries in (0, 1])."""
    G, wsum = None, 0.0
    for f in factors:
        form = str(f["form"])
        w = float(f.get("weight", 1.0))
        if form == "temporal":
            Kf = torch.exp(-_lag_sep(N, C, device, times) / float(f["sigma"]))
        elif form in ("quasimetric", "quasimetric_cached"):
            Kf = distance_affinity(tdist[str(f["teacher"])], f["sigma"], "none")
            Kf = Kf.reshape(C, N, C, N).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        else:
            Kf = _factor_kernel(form, _factor_rows(f, rows), f.get("sigma", 1.0), f)
            Kf = Kf.reshape(C, N, C, N).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        G = Kf * w if G is None else G + Kf * w
        wsum += w
    return (G / wsum).float()


def _abs_batch_gram(factors, rows):
    """The absolute kernel over every frame of the batch: the weighted mean of the voices' similarity
    kernels, uncentred."""
    G, wsum = None, 0.0
    for f in factors:
        Kf = _factor_kernel(str(f["form"]), _factor_rows(f, rows), f.get("sigma", 1.0), f).float()
        w = float(f.get("weight", 1.0))
        G = Kf * w if G is None else G + Kf * w
        wsum += w
    return G / wsum


# ------------------------------------------------------------------------------------ block statistics

def _block_ctr(M, m):
    """Masked double-centering inside each (N, N) block of M."""
    rs = (M * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1e-12)
    cs = (M * m).sum(-2, keepdim=True) / m.sum(-2, keepdim=True).clamp_min(1e-12)
    return M - rs - cs + (M * m).sum((-2, -1), keepdim=True) / m.sum().clamp_min(1e-12)


def _block_corr(S, Kc, m, live=None):
    """Correlation of S and Kc on each clip's own masked pairs, averaged uniformly over the live clips."""
    Sc, Kk = _block_ctr(S, m), _block_ctr(Kc, m)
    num = (Sc * Kk * m).sum((-2, -1))
    den = ((Sc ** 2 * m).sum((-2, -1)) * (Kk ** 2 * m).sum((-2, -1))).sqrt().clamp_min(1e-12)
    c = num / den
    if live is None:
        return c.mean()
    w = live.to(c.dtype)
    return (c * w).sum() / w.sum().clamp_min(1.0)


def _cmse_blocks(Sc, Kc, m, live, output, std_key, kstd_key, min_live=0.5):
    """The centred block MSE on double-centred blocks Sc (the Gram) and Kc (the absolute kernel) over the
    pairs m: per clip ((Sc - Kc)^2).mean / varK, clips without target variation dropped, live-weighted mean.
    Writes the Gram's and the kernel's centred RMS under `std_key` / `kstd_key`."""
    ms = m.sum().clamp_min(1.0)
    varG = (Sc ** 2 * m).sum((-2, -1)) / ms
    varK = (Kc ** 2 * m).sum((-2, -1)) / ms
    e = ((Sc - Kc) ** 2 * m).sum((-2, -1)) / ms
    e = e / varK.clamp_min(1e-12)
    w = torch.ones_like(e) if live is None else live.to(e.dtype)
    ok_k = varK > 1e-9 * float(varK.max().clamp_min(1e-30))
    assert float(ok_k.float().mean()) >= float(min_live), (
        f"{100 * (1 - float(ok_k.float().mean())):.0f}% of clips have no within-clip variation in the "
        f"absolute kernel; check the teacher and loss.spectral.factors[].sigma")
    w = w * ok_k.to(w.dtype)
    l = (e * w).sum() / w.sum().clamp_min(1.0)
    output[std_key] = ((varG * w).sum() / w.sum().clamp_min(1.0)).sqrt().detach()
    _ksw = ((varK * w).sum() / w.sum().clamp_min(1.0)).sqrt()
    output[kstd_key] = _ksw.detach()
    _sk = varK.clamp_min(0).sqrt()
    _skw = torch.where(w > 0, _sk, _sk.max().expand_as(_sk))
    output["kern_std_min_frac"] = (_skw.min() / _ksw.clamp_min(1e-30)).detach()
    return l


# ------------------------------------------------------------------------------------ loss terms

def _pred_anchor_term(sd_cfg, output, emb, pred_emb, posframe):
    """The prediction anchor: 1 - cos(pred_i, h_{posframe_i}) averaged over the predictions, at
    `anchor.weight`. `anchor.stop_gradient` stops the gradient through the encoder row."""
    pa = sd_cfg.get("anchor", None) or {}
    w = float(pa.get("weight", 0.0))
    B, N, D = emb.shape
    P = pred_emb.size(1)
    with torch.autocast(emb.device.type, enabled=False):
        rows = emb.float()
        if bool(pa.get("stop_gradient", False)):
            rows = rows.detach()
        hp = F.normalize(pred_emb.float(), dim=-1)
        he = F.normalize(rows, dim=-1)
        pf = posframe.long()
        bi = torch.arange(B, device=emb.device)[:, None]
        tgt = he[bi, pf[None, :].expand(B, P) % N]
        d0_cos = (hp * tgt).sum(-1)
        if w <= 0:
            return emb.sum() * 0.0
        output["pred_anchor_cos"] = d0_cos.detach().mean()
        l = (1.0 - d0_cos).mean()
        output["pred_anchor_loss"] = l.detach()
        return w * l


def _direct_gram_within_term(sd_cfg, output, emb, Kb, live, pred_emb, posframe, Gabs, mon):
    """The within-clip kernel term on the encoder rows: per clip, the centred block MSE of the unit rows'
    cosine Gram against the absolute kernel (`score_diagonal` scores the self pairs too), plus
    the prediction anchor. Monitors the per-clip correlation with K* and with each named voice."""
    B, N, D = emb.shape
    with torch.autocast(emb.device.type, enabled=False):
        m = 1.0 - torch.eye(N, device=emb.device, dtype=torch.float32)
        h = F.normalize(emb.float(), dim=-1)
        K = _block_ctr(Kb.detach().float(), m)
        S = h @ h.mT
        c = _block_corr(S, K, m, live)
        output["gram_corr_within"] = c.detach()
        with torch.no_grad():
            for _nm, _Km in mon.items():
                output[f"gram_corr_{_nm}"] = _block_corr(S.detach(), _block_ctr(_Km.detach().float(), m), m, live)
        cm = m + torch.eye(N, device=emb.device, dtype=m.dtype) if bool(sd_cfg.get("score_diagonal", False)) else m
        l = _cmse_blocks(_block_ctr(S, cm), _block_ctr(Gabs.detach().float(), cm), cm, live,
                         output, "gram_std_within", "kern_std_within", min_live=float(sd_cfg.get("min_live", 0.5)))
        output["spec_loss"] = l.detach()
        loss = float(sd_cfg.get("weight", 1.0)) * l
    return loss + _pred_anchor_term(sd_cfg, output, emb, pred_emb, posframe)


def _direct_cross_gram_term(sd_cfg, output, emb, pred_emb, posframe, Kb, live, mon, Gabs):
    """The within-clip kernel term on the prediction-vs-row block: per clip, the centred block MSE of
    cos(pred_i, h_j) against the absolute kernel read at the predicted frames, plus the prediction anchor."""
    B, N, D = emb.shape
    P = pred_emb.size(1)
    assert Kb.size(0) == B, f"the kernel has {Kb.size(0)} blocks for {B} clips"
    with torch.autocast(emb.device.type, enabled=False):
        pf = posframe.long()
        m = torch.ones(P, N, device=emb.device, dtype=torch.float32)
        K = _block_ctr(Kb.detach().float()[:, pf, :], m)                    # (B, P, N)
        hp = F.normalize(pred_emb.float(), dim=-1)
        he = F.normalize(emb.float(), dim=-1)
        S = hp @ he.mT                                                      # (B, P, N)
        c = _block_corr(S, K, m, live)
        output["gram_corr_cross_pred"] = c.detach()
        l = _cmse_blocks(_block_ctr(S, m), _block_ctr(Gabs.detach().float()[:, pf, :], m), m, live,
                         output, "gram_std_cross_pred", "kern_std_cross_pred", min_live=float(sd_cfg.get("min_live", 0.5)))
        output["spec_loss"] = l.detach()
        with torch.no_grad():
            for _nm, _Km in mon.items():
                output[f"gram_corr_{_nm}"] = _block_corr(S.detach(), _block_ctr(_Km.detach().float()[:, pf, :], m), m, live)
        loss = float(sd_cfg.get("weight", 1.0)) * l
        return loss + _pred_anchor_term(sd_cfg, output, emb, pred_emb, posframe)


def _spec_target(G, alpha, modes, center=True):
    """The tempered target of the fully specified objective: the (double-centred) kernel's spectrum raised
    to `alpha` on its top `modes` positive directions (all of them when `modes` is None), rescaled to trace n."""
    n = G.size(0)
    Cm = torch.eye(n, device=G.device) - 1.0 / n
    Kc = Cm @ G.float() @ Cm if center else G.float()
    T = _flatten_spectrum(Kc, n if modes is None else modes, alpha)
    return n * T / T.diagonal().sum().clamp_min(1e-12)


def _direct_gram_batch_term(sd_cfg, output, emb, pred_emb, posframe, G):
    """The fully specified objective on the whole batch: every row at the length the tempered target
    prescribes, the squared error over all n^2 entries of the doubly centred Gram against the target, plus
    the prediction anchor."""
    n = G.size(0)
    tp = sd_cfg.get("temper", None) or {}
    center = bool(tp.get("center", True))
    _tm = tp.get("modes", 128)
    with torch.autocast(emb.device.type, enabled=False):
        T = _spec_target(G.detach(), float(tp.get("alpha", 0.5)), None if _tm is None else int(_tm), center)
        Cm = torch.eye(n, device=G.device) - 1.0 / n
        hn = F.normalize(emb.reshape(n, -1).float(), dim=-1)
        hb = hn * T.diagonal().clamp_min(0).sqrt()[:, None]
        Sc = Cm @ (hb @ hb.mT) @ Cm if center else hb @ hb.mT
        e = ((Sc - T) ** 2).mean()
        output["spec_mse"] = e.detach()
        if str(sd_cfg.get("normalise", "entry")) == "relative":
            e = e / (T ** 2).mean().clamp_min(1e-12)
        output["spec_loss"] = e.detach()
        with torch.no_grad():
            sc, tc = Sc - Sc.mean(), T - T.mean()
            output["gram_corr_batch"] = (sc * tc).sum() / (sc.norm() * tc.norm()).clamp_min(1e-12)
            output["spec_target_rank"] = (torch.linalg.eigvalsh(T) > 1e-6 * n).sum().float()
    return float(sd_cfg.get("weight", 1.0)) * e + _pred_anchor_term(sd_cfg, output, emb, pred_emb, posframe)


def _spectral_term(mod, sd_cfg, output, batch, emb, pred_emb, posframe, far=None):
    """The spectral loss. scope=clip builds K* per clip and scores it on the encoder rows (queries=rows) or
    on the prediction-vs-row block (queries=predictions); scope=batch builds the tempered whole-batch target.
    `far` carries the far frames' rows and times (clip scope only)."""
    B, N = emb.size(0), emb.size(1)
    factors = sd_cfg["factors"]
    frows = {}
    for f in factors:
        if str(f["form"]) in ("quasimetric", "quasimetric_cached", "temporal"):
            continue
        key = str(f["key"])
        assert key in batch, f"factor {f.get('name', key)!r} reads batch[{key!r}]; add it to data.dataset.keys_to_load"
        frows[key] = batch[key].reshape(-1, batch[key].shape[-1])
    scope = str(sd_cfg["scope"])
    assert scope in ("clip", "batch"), f"loss.spectral.scope={scope!r}; expected clip | batch"
    if scope == "clip":
        embw, Nw, times = emb, N, None
        if far is not None:
            embw = torch.cat([emb, far["emb"]], 1)
            Nw, times = embw.size(1), far["times"]
            assert times.shape == (B, Nw), f"far times {tuple(times.shape)} vs rows {(B, Nw)}"
        with torch.no_grad():
            td = {}
            for f in factors:
                form = str(f["form"])
                if form not in ("quasimetric", "quasimetric_cached"):
                    continue
                name = str(f["teacher"])
                if name in td:
                    continue
                teacher = getattr(mod, f"teacher_{name}", None)
                assert teacher is not None, f"factor names teacher {name!r}; add it under loss.spectral.teachers"
                if form == "quasimetric_cached":
                    zkey = str(sd_cfg.get("cached_key", "teacher_z"))
                    assert zkey in batch, f"form=quasimetric_cached reads batch[{zkey!r}]; point data.dataset.depth_dir at the teacher-z store"
                    sl = f["slice"]
                    zr = batch[zkey].float().reshape(-1, batch[zkey].shape[-1])[:, int(sl[0]):int(sl[1])]
                    assert zr.shape[0] == B * Nw, f"cached z has {zr.shape[0]} rows against {B * Nw} frames"
                    td[name] = teacher.all_pairs_dist_from_z(zr).float()
                else:
                    td[name] = teacher.all_pairs_dist(batch["pixels"]).float()
            if os.environ.get("TEACHER_DUMP") and td:
                path = os.environ["TEACHER_DUMP"]
                np.savez(path, N=N, C=B, **{f"d_{k}": v.cpu().numpy() for k, v in td.items()})
                raise SystemExit(f"[teacher-dump] wrote {path} { {k: tuple(v.shape) for k, v in td.items()} }")
            Kg, live = _factor_block_kernel(factors, frows, B, Nw, td, device=emb.device, times=times)
            Gabs = _abs_block_gram(factors, B, Nw, td, emb.device, rows=frows, times=times)
            Kmon = {str(f["monitor"]): _factor_block_kernel([f], frows, B, Nw, td, device=emb.device, times=times)[0]
                    for f in factors if f.get("monitor", None)}
        assert Kg.shape == (B, Nw, Nw), f"clip targets need one (N, N) block per clip, got {tuple(Kg.shape)}"
        queries = str(sd_cfg.get("queries", "rows"))
        assert queries in ("rows", "predictions"), f"loss.spectral.queries={queries!r}; expected rows | predictions"
        if queries == "predictions":
            assert far is None, "queries=predictions takes no far frames"
            return _direct_cross_gram_term(sd_cfg, output, emb, pred_emb, posframe, Kg, live, Kmon, Gabs)
        return _direct_gram_within_term(sd_cfg, output, embw, Kg, live, pred_emb, posframe, Gabs, Kmon)
    with torch.no_grad():
        G = _abs_batch_gram(factors, frows)
    return _direct_gram_batch_term(sd_cfg, output, emb, pred_emb, posframe, G)


_MULTISTART_CACHE = {}


def _multistart_index(starts, hors, H, N, max_ph, device):
    """The multistart (start, horizon) index lists as device tensors, built once per geometry."""
    key = (H, N, max_ph, str(device))
    hit = _MULTISTART_CACHE.get(key)
    if hit is None:
        hit = _MULTISTART_CACHE[key] = (torch.tensor(starts, device=device), torch.tensor(hors, device=device))
        assert hit[0].tolist() == starts and hit[1].tolist() == hors
    return hit


def temporal_forward(self, batch, stage, cfg):
    """Encode the clip, predict every reachable (start, horizon) frame from its history, and sum the
    objective: SIGReg at its weight (a monitor at 0), the live quasimetric head of a teacher run, and the
    kernel term with its prediction anchor."""
    cfg = _step_cfg(self, cfg)
    H, Tm = cfg.wm.history_size, cfg.wm.num_preds
    sw = cfg.loss.sigreg.weight
    sd_cfg = cfg.loss.get("spectral", None)
    qk_cfg = cfg.get("quasimetric_head", None)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    clean_px, _ = self.gpu_aug(batch["pixels"])            # ImageNet normalisation on the GPU
    batch["pixels"] = clean_px
    output = self.model.encode(batch)
    emb = output["emb"]                                        # (B, T, D)
    # far frames (data.dataset.far_frames): rows of the same episode appended after the contiguous clip;
    # they enter the kernel term only
    n_far = int(cfg.data.dataset.get("far_frames", 0) or 0)
    far = None
    if n_far:
        fs = float(cfg.data.dataset.frameskip)
        far = {"emb": emb[:, -n_far:], "times": batch["step_idx"].reshape(emb.size(0), -1).float() / fs}
        emb = emb[:, :-n_far]
        output["emb"] = emb
    act_emb = output["act_emb"]
    N, B = emb.size(1), emb.size(0)
    assert N == batch["pixels"].size(1) - n_far, f"encoder returned {N} rows for {batch['pixels'].size(1)} frames"
    assert act_emb.size(1) == N, f"action bridge has {act_emb.size(1)} tokens for {N} frames"

    # multistart: every (start, horizon) with start >= H-1 and start + horizon <= N-1, horizon <= max_pred_horizon
    max_ph = int(cfg.wm.get("max_pred_horizon", Tm))
    starts, hors = [], []
    for i in range(H - 1, N - 1):
        nt = min(max_ph, N - 1 - i)
        starts += [i] * nt
        hors += list(range(1, nt + 1))
    max_h = max(hors)
    starts, hors = _multistart_index(starts, hors, H, N, max_ph, emb.device)
    P = starts.numel()
    ar_H = torch.arange(H, device=emb.device)
    ce = emb[:, starts[:, None] - (H - 1) + ar_H].reshape(B * P, H, -1)
    hor = hors[None, :].expand(B, P).reshape(B * P)
    if type(getattr(self.model, "predictor", None)).__name__ == "ARPredictor":
        # the causal predictor reads the H-frame context with its H action tokens; its last position is the
        # prediction of the next frame
        assert max_h == 1, f"ARPredictor predicts one step ahead; wm.max_pred_horizon must be 1, got {max_ph}"
        assert self.model.predictor.pos_embedding.size(1) >= H, "model.predictor.num_frames must cover wm.history_size"
        ae = act_emb[:, starts[:, None] - (H - 1) + ar_H].reshape(B * P, H, -1)
        pred_emb = self.model.predict(ce, ae)[:, -1].reshape(B, P, -1)
    else:
        ar_T = torch.arange(max_h, device=emb.device)
        amask = (ar_T[None, :] < hors[:, None])[None].expand(B, P, max_h).reshape(B * P, 1, 1, max_h)
        bri = starts[:, None] + ar_T
        ae_pad = torch.cat([act_emb, act_emb.new_zeros(B, max_h, act_emb.size(-1))], dim=1)
        be = ae_pad[:, bri].reshape(B * P, max_h, -1)
        pred_emb = self.model.predict_horizon(ce, be, hor, amask).reshape(B, P, -1)
    posframe = starts + hors
    ms_keys = emb.reshape(B * N, -1)
    ms_pos_idx = (torch.arange(B, device=emb.device)[:, None] * N + posframe[None, :]).reshape(B * P)
    tgt_emb = ms_keys[ms_pos_idx].reshape(B, P, -1)
    with torch.no_grad():
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["pred_cos"] = F.cosine_similarity(
            pred_emb.flatten(0, 1).float(), tgt_emb.flatten(0, 1).float(), dim=-1).mean()

    if sw > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    else:
        with torch.no_grad():                                  # logged only
            output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = sw * output["sigreg_loss"]
    if qk_cfg is not None:
        z = self.quasimetric_head.encode_z(emb)
        output["quasimetric_loss"] = self.quasimetric_head.qrl_loss(z)
        _quasimetric_monitors(self.quasimetric_head, output, emb.device)
        output["loss"] = output["loss"] + float(qk_cfg.get("weight", 1.0)) * output["quasimetric_loss"]
    if sd_cfg is not None:
        output["loss"] = output["loss"] + _spectral_term(self, sd_cfg, output, batch, emb, pred_emb, posframe, far)

    train_common.log_step_metrics(self, output, emb, stage)
    return output


def build_loss_modules(cfg, gpu_norm):
    """The loss modules attached to the Lightning module: SIGReg, the pixel normaliser, the frozen quasimetric
    teachers the factors name, and the live quasimetric head of a teacher run."""
    sw = float(cfg.loss.sigreg.get("weight", 0.0))
    if sw > 0 and bool((cfg.get("wm", None) or {}).get("emb_norm", True)):
        print(f"==> loss.sigreg.weight={sw} with emb_norm on: SIGReg acts as an isotropy term on the sphere", flush=True)
    mods = {"sigreg": SIGReg(**cfg.loss.sigreg.kwargs),
            "gpu_aug": GPUPixelNormalize()}
    sd = cfg.loss.get("spectral", None)
    if sd is not None:
        # the factor sigmas are in the columns' own units: every column read must be raw
        raw = set(cfg.data.dataset.get("raw_columns", []) or [])
        if cfg.data.dataset.get("depth_dir", None):
            raw.add(str(cfg.data.dataset.get("depth_key", "depth_raw")))
        for f in sd.factors:
            if str(f["form"]) in ("quasimetric", "quasimetric_cached", "temporal"):
                continue
            assert str(f["key"]) in raw, (
                f"loss.spectral factor {f.get('name', f['key'])!r} reads {f['key']!r}, which is not in "
                "data.dataset.raw_columns; its sigma is calibrated in raw units")
        # a null teacher entry is one an inherited recipe names and this arm does not use
        for name, tc in (sd.get("teachers", None) or {}).items():
            if tc is None:
                continue
            ty = OmegaConf.load(str(Path(__file__).resolve().parents[1] / "config" / "teachers" / f"{tc['spec']}.yaml"))
            if OmegaConf.is_missing(ty.model.action_encoder, "input_dim"):
                # set at runtime from frameskip x action width, so absent from every stored config; the teacher
                # never runs its action encoder but its checkpoint carries weights of the teacher's own width
                sfs, sin = int(cfg.data.dataset.frameskip), int(cfg.model.action_encoder.input_dim)
                assert sin % sfs == 0, f"student action_encoder.input_dim {sin} is not a multiple of its frameskip {sfs}"
                ty.model.action_encoder.input_dim = int(ty.model.action_encoder.frameskip) * (sin // sfs)
            mods[f"teacher_{name}"] = FrozenQRLTeacher(
                tc["ckpt"], ty.model, OmegaConf.to_container(ty.head, resolve=True),
                symmetrize=tc.get("symmetrize", "mean"))
            print(f"==> frozen teacher {name}: {tc['spec']} symmetrize={tc.get('symmetrize', 'mean')}", flush=True)
    qk = cfg.get("quasimetric_head", None)
    if qk is not None:
        mods["quasimetric_head"] = QRLHead(**OmegaConf.to_container(qk.get("head", {}), resolve=True))
    return mods


@hydra.main(version_base=None, config_path="../config/train", config_name="scene_temporal")
def main(cfg):
    train_common.run(cfg, temporal_forward, build_loss_modules, gpu_norm=True)


if __name__ == "__main__":
    main()
