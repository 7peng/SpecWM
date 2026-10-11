"""Batched GPU ridge+MLP probe head: splits and alphas are batch dimensions of one
bmm pass. Scalers fit on train rows only; alpha selected on an inner val split of
the train rows, then refit on full train; penalty is sklearn's mse + (alpha/n)||W||^2.
"""
import numpy as np
import torch

LR = 3e-3
# whitening floor as a fraction of the top eigenvalue (scale-free)
WHITEN_EPS = 1e-3
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def r2_splits_ridge_mlp(X, Y, n_splits=3, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0),
                        ridge_alphas=RIDGE_ALPHAS, device=None, seed0=0, max_rows=5000,
                        h=256, steps=3000, groups=None):
    """Ridge and MLP on the same rows, splits, and scoring convention -> (ridge, mlp).
    One-block alias of r2_ridge_mlp_multi."""
    blk = ("_", X, Y) if groups is None else ("_", X, Y, groups)
    out = r2_ridge_mlp_multi([blk], n_splits=n_splits, alphas=alphas, ridge_alphas=ridge_alphas,
                             device=device, seed0=seed0, max_rows=max_rows, h=h, steps=steps)
    return out["_"]


def _fit_batch_masked(Xtr, Ytr, Xte, alphas, ymask, steps, h, seed, rmask, block_id, n_buckets=4):
    """Xtr (B,n,d), Ytr (B,n,k) zero-padded, Xte (B,m,d), alphas (B,) -> preds (B,m,k).
    `ymask` (B,1,k) marks the real target columns and `rmask` (B,n,1) the real rows; each
    element's loss normaliser and (alpha/n) penalty use its own real counts. `block_id[b]` seeds
    element b's init, so a block's init is independent of the batch it rides in. Elements are
    bucketed by real-row count and each bucket is fitted padded to its own max."""
    B, n, d = Xtr.shape
    k = Ytr.shape[-1]
    assert ymask.shape == (B, 1, k), f"ymask must be (B,1,k)=({B},1,{k}), got {tuple(ymask.shape)}"
    assert rmask.shape == (B, n, 1), f"rmask must be (B,n,1)=({B},{n},1), got {tuple(rmask.shape)}"
    g = torch.Generator(device=Xtr.device).manual_seed(seed)
    dt = Xtr.dtype
    W1 = torch.empty(B, d, h, device=Xtr.device, dtype=dt)
    W2 = torch.empty(B, h, k, device=Xtr.device, dtype=dt)
    for _b in range(B):
        g.manual_seed(seed * 100003 + int(block_id[_b]))
        W1[_b] = torch.randn(d, h, generator=g, device=Xtr.device, dtype=dt) * (2.0 / d) ** 0.5
        W2[_b] = torch.randn(h, k, generator=g, device=Xtr.device, dtype=dt) * (2.0 / h) ** 0.5
    a = alphas.view(B)
    # count in float32: bf16 cannot represent integer sums past 256
    nvalid = ymask.float().sum(dim=(1, 2)).clamp_min(1.0)               # (B,) real columns per block
    nrow = rmask.float().sum(dim=(1, 2)).clamp_min(1.0)                 # (B,) real rows per block
    use_graph = Xtr.is_cuda and steps >= 64
    n_buckets = max(1, min(n_buckets, B))
    order = torch.argsort(nrow)                                         # ascending real-row count
    bounds = [int(round(i * B / n_buckets)) for i in range(n_buckets + 1)]
    W1f, b1f = W1.detach().clone(), torch.zeros(B, 1, h, device=Xtr.device, dtype=dt)
    W2f, b2f = W2.detach().clone(), torch.zeros(B, 1, k, device=Xtr.device, dtype=dt)
    W3f = torch.zeros(B, d, k, device=Xtr.device, dtype=dt)

    for _bi in range(n_buckets):
        idx = order[bounds[_bi]:bounds[_bi + 1]]
        if idx.numel() == 0:
            continue
        bn = int(nrow[idx].max().item())                               # bucket's max real rows
        Bb = idx.numel()
        W1_b = W1[idx].detach().clone().requires_grad_()
        b1_b = torch.zeros(Bb, 1, h, device=Xtr.device, dtype=dt, requires_grad=True)
        W2_b = W2[idx].detach().clone().requires_grad_()
        b2_b = torch.zeros(Bb, 1, k, device=Xtr.device, dtype=dt, requires_grad=True)
        W3_b = torch.zeros(Bb, d, k, device=Xtr.device, dtype=dt, requires_grad=True)
        Xtr_b, Ytr_b, rmask_b = Xtr[idx, :bn], Ytr[idx, :bn], rmask[idx, :bn]
        ymask_b, a_b, nvalid_b, nrow_b = ymask[idx], a[idx], nvalid[idx], nrow[idx]
        # capturable=True keeps Adam's step counters on-device (a CUDA-graph requirement)
        opt = torch.optim.Adam([W1_b, b1_b, W2_b, b2_b, W3_b], lr=LR, capturable=use_graph)

        def _step(zero_none):
            # set_to_none=False inside a captured region: grads must stay at static addresses
            opt.zero_grad(set_to_none=zero_none)
            pred = torch.relu(Xtr_b @ W1_b + b1_b) @ W2_b + Xtr_b @ W3_b + b2_b
            mse = (((pred - Ytr_b) ** 2) * ymask_b * rmask_b).sum(dim=(1, 2)) / (nrow_b * nvalid_b)
            pen = ((W1_b ** 2).sum(dim=(1, 2)) + (W2_b ** 2).sum(dim=(1, 2))
                   + (W3_b ** 2).sum(dim=(1, 2))) * (a_b / nrow_b)
            (mse + pen).sum().backward()
            opt.step()

        # enable_grad: the caller may be inside torch.no_grad() (Lightning's validation loop)
        with torch.enable_grad():
            captured = False
            if use_graph:
                # CUDA-graph the step loop: warmup on a side stream, capture one step, replay
                try:
                    warm = 5
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(warm):
                            _step(zero_none=False)
                    torch.cuda.current_stream().wait_stream(s)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        _step(zero_none=False)
                    for _ in range(steps - warm - 1):
                        graph.replay()
                    captured = True
                except Exception as e:
                    print(f"[probe-mlp] CUDA-graph capture failed ({type(e).__name__}: {e}); eager fallback",
                          flush=True)
            if not captured:
                for _ in range(steps):
                    _step(zero_none=True)
        with torch.no_grad():                                          # scatter params back to full B
            W1f[idx], b1f[idx], W2f[idx], b2f[idx], W3f[idx] = (
                W1_b.detach(), b1_b.detach(), W2_b.detach(), b2_b.detach(), W3_b.detach())
    with torch.no_grad():
        return torch.relu(Xte @ W1f + b1f) @ W2f + Xte @ W3f + b2f


def r2_ridge_mlp_multi(blocks, n_splits=3, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0),
                       ridge_alphas=RIDGE_ALPHAS, device=None, seed0=0, max_rows=5000,
                       h=256, steps=3000, fit_dtype=None, select_steps=None):
    """Many (key, X, Y[, groups]) probe blocks in one batched GPU pass ->
    {key: (ridge_r2, mlp_r2)}. Blocks may differ in target width and row count
    (zero-padded and masked; exact) but must share the X width; `groups` makes that
    block's splits episode-grouped. The MLP is anchored on the ridge prediction and
    fits only the residual, so with no nonlinear structure it returns ridge exactly.
    fit_dtype sets the MLP-fit dtype only; select_steps caps the alpha-selection pass.
    R^2 is sklearn's r2_score default (per output dim, uniform average).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # save + restore tf32 so training matmuls keep their numerics
    _tf32_matmul0 = torch.backends.cuda.matmul.allow_tf32
    _tf32_cudnn0 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # subsample rows once per block
    prep, grps = [], []
    for blk in blocks:
        key, X, Y = blk[:3]
        g = np.asarray(blk[3]) if len(blk) > 3 and blk[3] is not None else None
        X = np.asarray(X, np.float32)
        Y = np.asarray(Y, np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        if len(X) > max_rows:
            sel = np.random.default_rng(0).choice(len(X), max_rows, replace=False)
            X, Y = X[sel], Y[sel]
            g = g[sel] if g is not None else None
        prep.append((key, torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)))
        grps.append(g)

    Bn = len(prep)
    import zlib
    bkey = [zlib.crc32(str(k).encode()) % 1000003 for k, _, _ in prep]   # stable per-block seed id
    widths = {int(X.shape[1]) for _, X, _ in prep}
    assert len(widths) == 1, f"blocks must share X width, got {sorted(widths)}; split the call per width"
    d = widths.pop()
    kmax = max(int(Y.shape[1]) for _, _, Y in prep)
    Ns = [int(X.shape[0]) for _, X, _ in prep]

    def _idxset(bi, s):
        """(tr1, va, trf, te) row indices for block bi, split s. With groups, test and
        inner-val hold out whole episodes; without, permutation slicing."""
        n = Ns[bi]
        if grps[bi] is not None:
            from diagnostics.probe_heads import group_split
            trf, te = group_split(grps[bi], seed0 + s)
            itr, iva = group_split(grps[bi][trf], seed0 + s + 7919, test_frac=0.25)
            tr1, va = trf[itr], trf[iva]
        else:
            perm = np.random.default_rng(s).permutation(n)
            ntr = int(round(n * 0.8))
            nin = int(round(ntr * 0.75))
            tr1, va, trf, te = perm[:nin], perm[nin:ntr], perm[:ntr], perm[ntr:]
        return tuple(torch.as_tensor(np.ascontiguousarray(a), device=dev)
                     for a in (tr1, va, trf, te))

    idxsets = [[_idxset(bi, s) for s in range(n_splits)] for bi in range(Bn)]

    Nmax = max(Ns)
    Xpad = torch.zeros(Bn, Nmax, d, device=dev)
    Ypad = torch.zeros(Bn, Nmax, kmax, device=dev)
    kmask_r = torch.zeros(Bn, kmax, device=dev, dtype=torch.float64)
    for bi, (_k, X, Y) in enumerate(prep):
        Xpad[bi, :Ns[bi]] = X
        Ypad[bi, :Ns[bi], :Y.shape[1]] = Y
        kmask_r[bi, :Y.shape[1]] = 1.0
    Xpad64, Ypad64 = Xpad.double(), Ypad.double()

    def _rows(rows_list, m=None):
        """Per-block row sets of unequal length -> (B, m) padded gather index + (B, m, 1)
        row mask. Pad index 0 gathers a real row, so consumers must mask after centering."""
        if m is None:
            m = max(len(r) for r in rows_list)
        idx = torch.zeros(Bn, m, dtype=torch.long, device=dev)
        rm = torch.zeros(Bn, m, 1, device=dev)
        for bi, r in enumerate(rows_list):
            idx[bi, :len(r)] = r
            rm[bi, :len(r), 0] = 1.0
        return idx, rm

    def _take(P, idx):
        return torch.gather(P, 1, idx.unsqueeze(-1).expand(Bn, idx.shape[1], P.shape[-1]))

    # pad every split to the global max per stage kind, so per-split tensors stack
    mmax = [max(int(idxsets[bi][s][j].numel()) for bi in range(Bn) for s in range(n_splits))
            for j in range(4)]
    stages = []
    for s in range(n_splits):
        stages.append({
            "tr1": _rows([idxsets[bi][s][0] for bi in range(Bn)], mmax[0]),
            "va": _rows([idxsets[bi][s][1] for bi in range(Bn)], mmax[1]),
            "trf": _rows([idxsets[bi][s][2] for bi in range(Bn)], mmax[2]),
            "te": _rows([idxsets[bi][s][3] for bi in range(Bn)], mmax[3]),
        })

    def _whiten_batched(idx, rm):
        """(mu, Wm) with Wm = (C + eps I)^{-1/2} of the masked rows, by Newton-Schulz coupled
        iteration; the Frobenius norm stands in for lambda_max in the floor. TF32 off inside."""
        Xg = _take(Xpad, idx)
        nb = rm.sum(1, keepdim=True)
        mu = (Xg * rm).sum(1, keepdim=True) / nb
        A = (Xg - mu) * rm
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            C = A.transpose(1, 2) @ A / (nb - 1.0).clamp_min(1.0)
            eye = torch.eye(d, device=dev, dtype=C.dtype)
            M = C + WHITEN_EPS * C.norm(dim=(1, 2), keepdim=True).clamp_min(1e-30) * eye
            s = M.norm(dim=(1, 2), keepdim=True)          # Frobenius >= lambda_max -> spectrum in (0,1]
            Y = M / s
            Z = eye.expand_as(M).contiguous()
            for _ in range(30):
                T = 1.5 * eye - 0.5 * (Z @ Y)
                Y = Y @ T
                Z = T @ Z
            Wm = Z / s.sqrt()                              # -> (C + eps I)^(-1/2)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return mu, Wm

    white = []
    for s in range(n_splits):
        st = stages[s]
        mu1, Wm1 = _whiten_batched(*st["tr1"])
        muf, Wmf = _whiten_batched(*st["trf"])
        white.append({
            "tr1": ((_take(Xpad, st["tr1"][0]) - mu1) @ Wm1) * st["tr1"][1],
            "va": ((_take(Xpad, st["va"][0]) - mu1) @ Wm1) * st["va"][1],
            "trf": ((_take(Xpad, st["trf"][0]) - muf) @ Wmf) * st["trf"][1],
            "te": ((_take(Xpad, st["te"][0]) - muf) @ Wmf) * st["te"][1],
        })

    def _r2(Yt, pred):
        """R^2 per output dim, uniform average (sklearn's r2_score default)."""
        ss_res = ((Yt - pred) ** 2).sum(0)                                   # (k,) per output dim
        ss_tot = ((Yt - Yt.mean(0, keepdim=True)) ** 2).sum(0)               # (k,) test variance, as sklearn
        r = torch.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot.clamp_min(1e-12),
                        torch.zeros_like(ss_res))
        return float(r.mean())

    # ridge first: the reported baseline and the MLP's frozen anchor. Alpha selected on the
    # inner val rows; batched float64 Gram solves; padded dims/rows never fit or scored.
    ridge_r2, anch_in, anch_full = {}, {}, {}
    eye_d = torch.eye(d, device=dev, dtype=torch.float64)
    alphas_r = torch.as_tensor(list(ridge_alphas), device=dev, dtype=torch.float64)
    Ar = len(ridge_alphas)

    def _std_fit(idx, rm):
        rm64 = rm.double()
        Xf, Yf = _take(Xpad64, idx), _take(Ypad64, idx)
        nb = rm64.sum(1, keepdim=True)
        xm = (Xf * rm64).sum(1, keepdim=True) / nb
        xs = ((((Xf - xm) * rm64) ** 2).sum(1, keepdim=True)
              / (nb - 1.0).clamp_min(1.0)).sqrt().clamp_min(1e-8)
        ym_ = (Yf * rm64).sum(1, keepdim=True) / nb
        ys_ = ((((Yf - ym_) * rm64) ** 2).sum(1, keepdim=True)
               / (nb - 1.0).clamp_min(1.0)).sqrt().clamp_min(1e-8)
        # mask after centering: padded rows would otherwise carry -mu/sd into the Gram
        return ((Xf - xm) / xs) * rm64, ((Yf - ym_) / ys_) * rm64, xm, xs, ym_, ys_

    def _predict_b(W, idx, xm, xs, ym_, ys_):
        return ((_take(Xpad64, idx) - xm) / xs) @ W * ys_ + ym_              # original units, padded k

    def _r2_masked(idx, rm, P):
        rm64 = rm.double()
        Yt = _take(Ypad64, idx)
        nb = rm64.sum(-2, keepdim=True).clamp_min(1.0)
        mu = (Yt * rm64).sum(-2, keepdim=True) / nb
        ssr = (((Yt - P) * rm64) ** 2).sum(-2)
        sst_raw = (((Yt - mu) * rm64) ** 2).sum(-2)
        sst = sst_raw.clamp_min(1e-12)
        # constant-on-the-scored-rows dims score 0 (sklearn's convention)
        r = torch.where(sst_raw > 1e-12, 1.0 - ssr / sst, torch.zeros_like(ssr))
        return (r * kmask_r).sum(-1) / kmask_r.sum(-1)       # (..., B) uniform over real dims

    for s in range(n_splits):
        st = stages[s]
        A_, Bm, xm, xs, ym_, ys_ = _std_fit(*st["tr1"])
        G = A_.transpose(1, 2) @ A_
        XY = A_.transpose(1, 2) @ Bm
        Gs = G.unsqueeze(0) + alphas_r.view(Ar, 1, 1, 1) * eye_d
        W = torch.linalg.solve(Gs, XY.unsqueeze(0).expand(Ar, *XY.shape))    # (A, B, d, k)
        r2va = _r2_masked(st["va"][0], st["va"][1],
                          _predict_b(W, st["va"][0], xm.unsqueeze(0), xs.unsqueeze(0),
                                     ym_.unsqueeze(0), ys_.unsqueeze(0)))    # (A, B)
        bidx = r2va.argmax(0)                                                # first-max tie-break
        Wb = W[bidx, torch.arange(Bn, device=dev)]
        p_tr = _predict_b(Wb, st["tr1"][0], xm, xs, ym_, ys_)
        p_va = _predict_b(Wb, st["va"][0], xm, xs, ym_, ys_)
        A2, B2, xm2, xs2, ym2, ys2 = _std_fit(*st["trf"])
        G2 = A2.transpose(1, 2) @ A2 + alphas_r[bidx].view(Bn, 1, 1) * eye_d
        W2r = torch.linalg.solve(G2, A2.transpose(1, 2) @ B2)
        p_trf = _predict_b(W2r, st["trf"][0], xm2, xs2, ym2, ys2)
        p_te = _predict_b(W2r, st["te"][0], xm2, xs2, ym2, ys2)
        r2te = _r2_masked(st["te"][0], st["te"][1], p_te)
        for bi in range(Bn):
            kk = int(prep[bi][2].shape[1])
            n1, nv, nf, nt = (int(idxsets[bi][s][j].numel()) for j in range(4))
            anch_in[(bi, s)] = (p_tr[bi, :n1, :kk].float(), p_va[bi, :nv, :kk].float())
            anch_full[(bi, s)] = (p_trf[bi, :nf, :kk].float(), p_te[bi, :nt, :kk].float())
            ridge_r2[(bi, s)] = float(r2te[bi])

    # pass 1: alpha selection on the inner val rows; the MLP's target is the ridge residual
    m1 = white[0]["tr1"].shape[1]
    xi, yi, xv, al, ym, rmk, meta = [], [], [], [], [], [], []
    for bi, (key, X, Y) in enumerate(prep):
        kk = Y.shape[1]
        for s in range(n_splits):
            tr = idxsets[bi][s][0]
            R = Y[tr] - anch_in[(bi, s)][0]                                   # residual, original units
            rmu, rsd = R.mean(0, keepdim=True), R.std(0, keepdim=True).clamp_min(1e-8)
            yt = torch.zeros(m1, kmax, device=dev)
            yt[:len(R), :kk] = (R - rmu) / rsd
            m = torch.zeros(1, kmax, device=dev)
            m[0, :kk] = 1.0
            for a in alphas:
                xi.append(white[s]["tr1"][bi]); yi.append(yt); xv.append(white[s]["va"][bi])
                al.append(a); ym.append(m); rmk.append(stages[s]["tr1"][1][bi])
                meta.append((bi, s, a, rmu, rsd))
    fdt = fit_dtype or torch.float32
    s1 = int(select_steps) if select_steps else steps
    P = _fit_batch_masked(torch.stack(xi).to(fdt), torch.stack(yi).to(fdt), torch.stack(xv).to(fdt),
                          torch.as_tensor(al, device=dev, dtype=fdt),
                          torch.stack(ym).to(fdt), steps=s1, h=h, seed=seed0,
                          rmask=torch.stack(rmk).to(fdt),
                          block_id=[bkey[m[0]] for m in meta]).float()
    best = {}
    for i, (bi, s, a, rmu, rsd) in enumerate(meta):
        Y = prep[bi][2]
        va = idxsets[bi][s][1]
        kk = Y.shape[1]
        pred = anch_in[(bi, s)][1] + P[i, :len(va), :kk] * rsd + rmu  # anchor + residual, original units
        r2 = _r2(Y[va], pred)
        if (bi, s) not in best or r2 > best[(bi, s)][1]:
            best[(bi, s)] = (a, r2)

    # pass 2: refit on the full train rows at the selected alpha
    mf = white[0]["trf"].shape[1]
    xt2, yt2, xe2, al2, ym2, rmk2, meta2 = [], [], [], [], [], [], []
    for bi, (key, X, Y) in enumerate(prep):
        kk = Y.shape[1]
        for s in range(n_splits):
            tr = idxsets[bi][s][2]
            R = Y[tr] - anch_full[(bi, s)][0]
            rmu, rsd = R.mean(0, keepdim=True), R.std(0, keepdim=True).clamp_min(1e-8)
            yt = torch.zeros(mf, kmax, device=dev)
            yt[:len(R), :kk] = (R - rmu) / rsd
            m = torch.zeros(1, kmax, device=dev)
            m[0, :kk] = 1.0
            xt2.append(white[s]["trf"][bi]); yt2.append(yt); xe2.append(white[s]["te"][bi])
            al2.append(best[(bi, s)][0]); ym2.append(m); rmk2.append(stages[s]["trf"][1][bi])
            meta2.append((bi, s, rmu, rsd))
    P2 = _fit_batch_masked(torch.stack(xt2).to(fdt), torch.stack(yt2).to(fdt),
                           torch.stack(xe2).to(fdt),
                           torch.as_tensor(al2, device=dev, dtype=fdt),
                           torch.stack(ym2).to(fdt), steps=steps, h=h, seed=seed0,
                           rmask=torch.stack(rmk2).to(fdt),
                           block_id=[bkey[m[0]] for m in meta2]).float()

    # score: ridge and (anchor + residual) on the same rows / splits / units
    acc = {}
    for i, (bi, s, rmu, rsd) in enumerate(meta2):
        key, X, Y = prep[bi]
        te = idxsets[bi][s][3]
        kk = Y.shape[1]
        pred = anch_full[(bi, s)][1] + P2[i, :len(te), :kk] * rsd + rmu
        acc.setdefault(key, ([], []))
        acc[key][0].append(ridge_r2[(bi, s)])
        acc[key][1].append(_r2(Y[te], pred))
    torch.backends.cuda.matmul.allow_tf32 = _tf32_matmul0
    torch.backends.cudnn.allow_tf32 = _tf32_cudnn0
    return {key: (float(np.mean(r)), float(np.mean(m))) for key, (r, m) in acc.items()}
