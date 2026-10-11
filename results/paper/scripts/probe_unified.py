"""One probe instrument for both protocols: features through diagnostics.probe_encoders.build_encoder
(fp32, explicit ImageNet normalisation, the planner's CLS -> projector embedding) and the fit through
diagnostics.probe_heads.grouped_fit (grouped 5-fold CV ridge, alpha by inner CV, R^2 averaged over
the target's coordinates).

  suite    : a pack from build_probe_pack.py; per cell the rows are grouped into 20 contiguous chunks
             (placements are i.i.d., so chunks are exchangeable) and grouped_fit runs over them; the
             env number is the median over in-distribution cells (`obj`), ood cells separately.
  heldout  : 5,000 frames from the seeded validation episodes of the run's own corpus, grouped by
             episode, every raw state target.

    python results/paper/scripts/probe_unified.py --protocol suite --env cube --pack probe_pack_ogbench_v2/cube --final --random --out data/x.csv
    python results/paper/scripts/probe_unified.py --protocol heldout --env calvin --final --random --out data/y.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from utils import load_env  # noqa: E402

load_env()
from diagnostics.probe_heads import grouped_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_val_3env import features  # noqa: E402
from probe_val_episodes import val_episodes  # noqa: E402

HOME = Path(os.environ["STABLEWM_HOME"])
D = HOME / "datasets"
ENVS = {
    "scene": dict(mix=D / "scene_play_mix_lateonset.h5",
                  runs=["scene-lewm-mixlate-1", "scene-mixlate-1", "scene-mixlate-det-1", "scene-mixlate-lag-1"],
                  targets={"cube_xyz": ("qpos", slice(14, 17)), "drawer": ("qpos", slice(23, 24)), "window": ("qpos", slice(24, 25)),
                           "buttons": ("button_states", slice(0, 2)), "arm": ("qpos", slice(0, 7))}),
    "cube": dict(mix=D / "cube_play_mix_lateonset.h5",
                 runs=["cube-lewm-mixlate-1", "cube-mixlate-t2-1", "cube-mixlate-t2-det-1", "cube-mixlate-lag-1"],
                 targets={"cube_xyz": ("qpos", slice(14, 17)), "arm": ("qpos", slice(0, 7))}),
    "calvin": dict(mix=D / "calvin_abc_d_mix_lateonset.h5",
                   runs=["calvin-lewm-mixlate-1", "calvin-mixlate-1", "calvin-mixlate-det-1", "calvin-mixlate-lag-1"],
                   targets={"red_xyz": ("scene_obs", slice(6, 9)), "blue_xyz": ("scene_obs", slice(12, 15)), "pink_xyz": ("scene_obs", slice(18, 21)),
                            "slider": ("scene_obs", slice(0, 1)), "drawer": ("scene_obs", slice(1, 2)), "switch": ("scene_obs", slice(3, 4)),
                            "arm": ("robot_obs", slice(0, 15))}),
}
CHUNKS = 20


def final_ckpt(run):
    return str(max((HOME / "checkpoints" / "lewm" / run).glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1])))


def jobs(a, spec):
    out = [("random-init", 0, "random-init")] if a.random else []
    for run in (a.runs or spec["runs"]):
        if a.final:
            out.append((run, -1, final_ckpt(run)))
        for s in a.steps:
            out.append((run, s, str(HOME / "checkpoints" / "lewm" / run / "step_snaps" / f"step_{s}.pt")))
    return out


def suite(a, spec):
    pack = HOME / a.pack; ix = json.load(open(pack / "index.json")); z = np.load(pack / "cells.npz")
    rows, Y = z["obj_in_cam.rows"], z["obj_in_cam.Y"]; names = ix["groups"]["obj_in_cam"]["names"]; split = ix["groups"]["obj_in_cam"]["split"]
    frames = np.load(pack / "frames.npy", mmap_mode="r")
    need = np.unique(rows.reshape(-1)); pos = {r: i for i, r in enumerate(need)}
    px = np.asarray(frames[need])
    print(f"[suite] {a.env} pack {a.pack}: {rows.shape[0]} cells x {rows.shape[1]} rows, {len(need)} frames", flush=True)
    recs = []
    for run, step, ck in jobs(a, spec):
        X = features(ck, px)
        groups = np.arange(rows.shape[1]) * CHUNKS // rows.shape[1]
        fits = Parallel(n_jobs=min(8, rows.shape[0]))(delayed(grouped_fit)(X[[pos[r] for r in rows[c]]], Y[c].astype(np.float64), groups, ridge_alpha="cv", n_splits=5, seed0=0) for c in range(rows.shape[0]))
        r2 = np.array([f["r2"] for f in fits])
        idm = np.array([s != "OOD" for s in split])
        rec = dict(env=a.env, run=run, step=step, pack=a.pack, obj=float(np.median(r2[idm])), obj_ood=float(np.median(r2[~idm])) if (~idm).any() else np.nan, n_cells=int(idm.sum()))
        rec.update({f"cell:{n}": float(v) for n, v in zip(names, r2)})
        recs.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith("cell:")}, flush=True)
    return recs


def mlp_fit(X, y, ep, split):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from diagnostics.probe_heads import group_split, MLP_KW
    tr, te = group_split(ep, split); xs = StandardScaler().fit(X[tr]); ys = StandardScaler().fit(y[tr])
    m = MLPRegressor(alpha=10.0, random_state=0, **MLP_KW).fit(xs.transform(X[tr]), ys.transform(y[tr]).ravel() if y.shape[1] == 1 else ys.transform(y[tr]))
    return r2_score(ys.transform(y[te]), m.predict(xs.transform(X[te])).reshape(-1, y.shape[1]))


def heldout(a, spec):
    with h5py.File(spec["mix"], "r") as f:
        ep = f["ep_idx"][:].astype(np.int64); val, n_ep = val_episodes(ep)
        keep = np.array(sorted(val))
        if a.subset != 'all':
            b = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / f'branch_episodes_{a.env}.csv').set_index('episode')
            twin = b.loc[keep, 'family_has_train'].values | (b.loc[keep, 'placement_twin_in_train'].values if a.env == 'calvin' else False)
            clean = ~twin      # OGBench play episodes all start from one reset spot, so placement identity means nothing there
            keep = keep[clean] if a.subset == 'clean' else keep[~clean]
        pool = np.flatnonzero(np.isin(ep, keep))
        sel = np.sort(np.random.default_rng(a.seed).choice(pool, size=min(a.n, len(pool)), replace=False))
        ep_rows = ep[sel]; px = f["pixels"][sel]
        if a.env == "calvin":
            assert f["scene_obs"].attrs.get("roster") == "red,blue,pink", "mixture scene_obs is not roster-canonical"
        Yd = {k: f[col][sel][:, sl].astype(np.float64) for k, (col, sl) in spec["targets"].items()}
    print(f"[heldout] {a.env} {spec['mix'].name} subset={a.subset}: {len(sel)} frames from {len(np.unique(ep_rows))} of {len(keep)} episodes ({len(val)} validation, {n_ep} total), pixels {px.shape}", flush=True)
    recs = []
    def centre(M):
        u, inv = np.unique(ep_rows, return_inverse=True); mu = np.zeros((len(u), M.shape[1])); cnt = np.bincount(inv)
        for j in range(M.shape[1]): mu[:, j] = np.bincount(inv, M[:, j]) / cnt
        return M - mu[inv]
    if a.centre_episodes:
        Yd = {k: centre(y) for k, y in Yd.items()}
    for run, step, ck in jobs(a, spec):
        X = features(ck, px)
        if a.centre_episodes: X = centre(X.astype(np.float64))
        ev = np.linalg.eigvalsh(np.cov(X.astype(np.float64).T)).clip(min=0); eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum())
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True); G = Xn[:2000] @ Xn[:2000].T; mean_cos = float((G.sum() - np.trace(G)) / (G.shape[0] * (G.shape[0] - 1)))
        if a.head == "mlp":
            fits = Parallel(n_jobs=min(24, 3 * len(Yd)))(delayed(mlp_fit)(X, y, ep_rows, k) for y in Yd.values() for k in range(3))
            fits = [{"r2": float(np.mean(fits[3 * i:3 * i + 3]))} for i in range(len(Yd))]
        else:
            fits = Parallel(n_jobs=min(8, len(Yd)))(delayed(grouped_fit)(X, y, ep_rows, ridge_alpha="cv", n_splits=5, seed0=0) for y in Yd.values())
        rec = dict(env=a.env, run=run, step=step, subset=a.subset, centred=a.centre_episodes, head=a.head, n_frames=len(sel), n_episodes=len(np.unique(ep_rows)), eff_rank=eff_rank, mean_cos=mean_cos)
        rec.update({k: f["r2"] for k, f in zip(Yd, fits)})
        rec["obj"] = rec["cube_xyz"] if a.env != "calvin" else float(np.median([rec["red_xyz"], rec["blue_xyz"], rec["pink_xyz"]]))
        recs.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["suite", "heldout"], required=True); ap.add_argument("--env", choices=list(ENVS), required=True)
    ap.add_argument("--pack", default=None, help="suite: pack directory under $STABLEWM_HOME")
    ap.add_argument("--runs", nargs="*", default=None); ap.add_argument("--steps", type=int, nargs="*", default=[])
    ap.add_argument("--final", action="store_true"); ap.add_argument("--random", action="store_true")
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear", help="heldout: ridge (grouped_fit) or a 64-hidden MLP (lbfgs, alpha 10) on 3 grouped splits")
    ap.add_argument("--centre-episodes", action="store_true", help="heldout: subtract each episode's mean from features and targets before the fit (within-episode R^2: object motion, not placement)")
    ap.add_argument("--subset", choices=["all", "clean", "twin"], default="all",
                    help="heldout: all validation episodes; clean = no branch-family member in training (CALVIN: nor a placement twin); twin = the rest (data/branch_episodes_<env>.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); spec = ENVS[a.env]
    recs = suite(a, spec) if a.protocol == "suite" else heldout(a, spec)
    d = pd.DataFrame(recs); p = Path(a.out)
    if p.exists():
        d = pd.concat([pd.read_csv(p), d], ignore_index=True).drop_duplicates(subset=["env", "run", "step"] + [c for c in ("pack", "subset", "centred", "head") if c in d], keep="last")
    d.to_csv(p, index=False); print(f"[out] {p}")


if __name__ == "__main__":
    main()
