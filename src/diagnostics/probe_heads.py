"""Probe readout heads (ridge, MLP) and the shared episode-grouped fit protocol."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CV_ALPHAS = np.logspace(-2, 4, 13)


MIN_GROUPS = 3          # below this a grouped split cannot hold out an episode and train on the rest


def group_split(groups, seed, test_frac=0.2):
    """Deterministic (train_idx, test_idx) holding out whole episodes; None when there are
    fewer than MIN_GROUPS episodes to split."""
    from sklearn.model_selection import GroupShuffleSplit
    if len(np.unique(groups)) < MIN_GROUPS:
        return None
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def grouped_fit(X, Y, groups, ridge_alpha="cv", n_splits=5, seed0=0, min_rows=60):
    """Held-out ridge R^2 averaged over n_splits episode-grouped splits.

    Per split: standardize X and Y on the train rows, fit, score on the held-out
    episodes. ridge_alpha="cv" selects alpha per split (grouped inner
    splits of the train rows) and also fits a fixed alpha-10 twin. Split seeds are seed0..seed0+n_splits-1, so
    fits sharing groups and seeds share splits.

    Returns {"r2", "alpha", "r2_a10"} ("alpha"/"r2_a10" only under cv); r2 is nan
    when no split yields a valid fit."""
    from sklearn.metrics import r2_score
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    g = np.asarray(groups)
    if len(X) < min_rows:
        return {"r2": float("nan")}
    cv = isinstance(ridge_alpha, str)
    if cv:
        assert ridge_alpha == "cv", f"ridge_alpha must be a number or 'cv', got {ridge_alpha!r}"
    r2s, r2s_a10, alphas = [], [], []
    for s in range(n_splits):
        sp = group_split(g, seed0 + s)
        if sp is None:
            break
        tr, te = sp
        Ytr, Yte = Y[tr], Y[te]
        if min(float(Ytr.std(axis=0).max()), float(Yte.std(axis=0).max())) < 1e-8:
            continue
        ysc = StandardScaler().fit(Ytr)
        Ytr, Yte = ysc.transform(Ytr), ysc.transform(Yte)
        xsc = StandardScaler().fit(X[tr])
        Xtr, Xte = xsc.transform(X[tr]), xsc.transform(X[te])
        if cv:
            a_best = select_alpha_grouped(X[tr], Y[tr], g[tr], seed0)
            alphas.append(a_best)
            m = Ridge(alpha=a_best).fit(Xtr, Ytr)
            m10 = Ridge(alpha=10.0).fit(Xtr, Ytr)
            r2s_a10.append(float(r2_score(Yte, m10.predict(Xte).reshape(Yte.shape))))
        else:
            m = Ridge(alpha=float(ridge_alpha)).fit(Xtr, Ytr)
        r2s.append(float(r2_score(Yte, m.predict(Xte).reshape(Yte.shape))))
    if not r2s:
        return {"r2": float("nan")}
    out = {"r2": float(np.mean(r2s))}
    if cv:
        out["alpha"] = float(np.median(alphas))
        out["r2_a10"] = float(np.mean(r2s_a10))
    return out


MLP_SPLITS = 3
MLP_MAX_ROWS = 1200
MLP_ALPHAS = (10.0, 100.0, 1000.0)
MLP_KW = dict(hidden_layer_sizes=(64,), solver="lbfgs", max_iter=3000)


def n_splits_for(head, n_splits):
    return min(n_splits, MLP_SPLITS) if head == "mlp" else n_splits


def select_alpha_grouped(X, Y, g, seed=0, n_inner=2):
    """Ridge alpha from CV_ALPHAS chosen on `n_inner` grouped inner splits of `X` (seeds
    seed+1000+i), scoring per-output-dim R^2 uniformly averaged; 10.0 when no split is
    valid."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    g = np.asarray(g)
    tot = np.zeros(len(CV_ALPHAS))
    seen = 0
    for s_in in range(n_inner):
        isp = group_split(g, seed + 1000 + s_in)
        if isp is None:
            continue
        itr, ite = isp
        Yte = Y[ite]
        if Yte.std(axis=0).max() < 1e-8:
            continue
        mu = X[itr].mean(0)
        sd = X[itr].std(0)                      # population std, as StandardScaler uses
        sd = np.where(sd < 1e-12, 1.0, sd)
        A = (X[itr] - mu) / sd
        ymu = Y[itr].mean(0)
        G = A.T @ A
        c = A.T @ (Y[itr] - ymu)
        B = (X[ite] - mu) / sd
        eye = np.eye(G.shape[0])
        sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
        ok = sst > 1e-12
        if not ok.any():
            continue
        for ai, a in enumerate(CV_ALPHAS):
            P = B @ np.linalg.solve(G + a * eye, c) + ymu
            sse = ((Yte - P) ** 2).sum(0)
            # uniform average over output dims, matching sklearn's r2_score default
            tot[ai] += float(np.mean(1.0 - sse[ok] / sst[ok]))
        seen += 1
    return float(CV_ALPHAS[int(np.argmax(tot))]) if seen else 10.0


def _fit_predict(Xtr, Ytr, Xte, head, seed, ridge_alpha):
    sx = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sx.transform(Xtr), sx.transform(Xte)
    if head == "ridge":
        return Ridge(alpha=ridge_alpha).fit(Xtr_s, Ytr).predict(Xte_s)
    sy = StandardScaler().fit(Ytr)
    Yt = sy.transform(Ytr)
    yfit = Yt.ravel() if Yt.shape[1] == 1 else Yt
    # alpha selected on an inner validation split of the train rows only
    itr, iva = train_test_split(np.arange(len(Xtr_s)), test_size=0.25, random_state=seed)
    best, bmod = -np.inf, None
    for a in MLP_ALPHAS:
        m = MLPRegressor(alpha=a, random_state=seed, **MLP_KW).fit(Xtr_s[itr], yfit[itr])
        p = np.asarray(m.predict(Xtr_s[iva])).reshape(len(iva), -1)
        sst = ((Yt[iva] - Yt[iva].mean(0)) ** 2).sum()
        r2 = 1.0 - ((Yt[iva] - p) ** 2).sum() / sst if sst > 1e-12 else -np.inf
        if r2 > best:
            best, bmod = r2, a
    m = MLPRegressor(alpha=bmod, random_state=seed, **MLP_KW).fit(Xtr_s, yfit)
    P = np.asarray(m.predict(Xte_s)).reshape(-1, Ytr.shape[1])
    return sy.inverse_transform(P)


def r2_splits(X, Y, head="ridge", n_splits=20, ridge_alpha=10.0, min_rows=60, groups=None):
    """Held-out R^2 averaged over repeated splits (episode-grouped when `groups` is
    given), variance-weighted across output dims. Fixed seeds range(n_splits), so
    fits sharing head/n_splits/groups share splits; the row cap applies to both heads
    with the same rng."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) < min_rows:
        return np.nan
    if len(X) > MLP_MAX_ROWS:
        sel = np.random.default_rng(0).choice(len(X), MLP_MAX_ROWS, replace=False)
        X, Y = X[sel], Y[sel]
        if groups is not None:
            groups = np.asarray(groups)[sel]
    if isinstance(ridge_alpha, str) and groups is not None:
        ridge_alpha = select_alpha_grouped(X, Y, groups)
    elif isinstance(ridge_alpha, str):
        ridge_alpha = 10.0
    scores = []
    for seed in range(n_splits_for(head, n_splits)):
        if groups is not None:
            sp = group_split(groups, seed)
            if sp is None:
                break
            tr, te = sp
            Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
        else:
            Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=seed)
        try:
            P = _fit_predict(Xtr, Ytr, Xte, head, seed, ridge_alpha)
        except Exception:
            continue
        P = np.asarray(P).reshape(Yte.shape)   # sklearn ravels single-column targets
        sst = ((Yte - Yte.mean(0)) ** 2).sum()
        if sst > 1e-12:
            scores.append(1.0 - ((Yte - P) ** 2).sum() / sst)
    return float(np.mean(scores)) if scores else np.nan
