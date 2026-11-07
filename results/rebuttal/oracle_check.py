"""External check of the reachability graph: expert hitting time from OGBench's oracles (--oracle plan: the
noise-free keyframe plans, fixed segment durations; markov: the closed-loop controllers without noise). For
held-out Scene frame pairs (s, g) reachable on the graph whose object configuration differs in exactly one
factor (cube moved, one button toggled, drawer or window moved) and agrees within the env's success tolerance
elsewhere, the simulator is set to s, that factor's target to g's value, and the factor's plan oracle runs
until the env reports success. Compares the hitting time with graph distance, recorded time and each model's
embedding distance.
    python results/rebuttal/oracle_check.py DATASET.h5 GRAPH.npz EMBDIR OUT.csv [--per 40] [--workers 8]"""
import argparse, multiprocessing as mp, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, hdf5plugin  # noqa: F401
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from correlate import model_dist, spearman  # noqa: E402

TOL, MAX_STEPS = 0.04, 600
FACTORS = ("cube", "button", "drawer", "window")


def object_diff(qa, ba, qb, bb):
    """The factors whose state differs between two frames beyond the env's success tolerance."""
    d = []
    if np.linalg.norm(qa[14:17] - qb[14:17]) > TOL: d.append("cube")
    if (ba != bb).any(): d.append("button")
    if abs(qa[23] - qb[23]) > TOL: d.append("drawer")
    if abs(qa[24] - qb[24]) > TOL: d.append("window")
    return d


_ENV = _OR = None


def _init():
    global _ENV, _OR
    from ogbench.manipspace.envs.scene_env import SceneEnv
    from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
    from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
    from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
    _ENV = SceneEnv(env_type="scene", mode="data_collection", ob_type="states", terminate_at_goal=False)
    _ENV.reset(seed=0)
    if os.environ.get("ORACLE_KIND", "plan") == "markov":
        from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
        from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
        _OR = {"cube": CubeMarkovOracle(env=_ENV), "button": ButtonMarkovOracle(env=_ENV),
               "drawer": DrawerMarkovOracle(env=_ENV), "window": WindowMarkovOracle(env=_ENV)}
    else:
        _OR = {"cube": CubePlanOracle(env=_ENV, noise=0.0), "button": ButtonPlanOracle(env=_ENV, noise=0.0),
               "drawer": DrawerPlanOracle(env=_ENV, noise=0.0), "window": WindowPlanOracle(env=_ENV, noise=0.0)}


def _rollout(job):
    """Steps until the factor's success check passes from s, or -1 within MAX_STEPS."""
    import mujoco
    k, factor, qs, vs, bs, qg, bg = job
    u = _ENV
    u.reset(seed=int(k))
    u.set_state(qs, vs, bs.astype(np.int64))
    u._target_task = factor
    if factor == "cube":
        u._target_block = 0
        u._data.mocap_pos[u._cube_target_mocap_ids[0]] = qg[14:17]
        u._data.mocap_quat[u._cube_target_mocap_ids[0]] = [1.0, 0.0, 0.0, 0.0]
    elif factor == "button":
        u._target_button = int(np.flatnonzero(bs != bg)[0]); u._target_button_states = bg.astype(np.int64).copy()
    elif factor == "drawer":
        u._target_drawer_pos = float(qg[23]); u._model.site("drawer_handle_center_target").pos[1] = float(qg[23])
    else:
        u._target_window_pos = float(qg[24]); u._model.site("window_handle_center_target").pos[0] = float(qg[24])
    mujoco.mj_kinematics(u._model, u._data)
    ob, info = u.compute_observation(), u.get_reset_info()
    orc = _OR[factor]; orc.reset(ob, info)
    for step in range(1, MAX_STEPS + 1):
        ob, _, _, _, info = u.step(np.clip(np.asarray(orc.select_action(ob, info)), -1, 1))
        if info["success"]:
            return k, step
    return k, -1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("graph"); ap.add_argument("emb"); ap.add_argument("out")
    ap.add_argument("--per", type=int, default=40); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--oracle", choices=["plan", "markov"], default="plan"); a = ap.parse_args(); os.environ["ORACLE_KIND"] = a.oracle
    g = np.load(a.graph); A, D, ep, t, V = g["anchors"], g["D"], g["ep"], g["t"], g["rows"]
    with h5py.File(a.dataset, "r") as f:
        q = f["qpos"][:][V].astype(np.float64); v = f["qvel"][:][V].astype(np.float64); b = f["button_states"][:][V]
    in_drawer = lambda x: (0.19 <= x[14] <= 0.47) and (-0.45 <= x[15] <= 0.0) and (x[16] <= 0.15)  # noqa: E731  around the drawer box at any opening
    rng = np.random.default_rng(0)
    same = ep[A][:, None] == ep[None, :]; indep = ~same & ((ep[A] % 1000)[:, None] != (ep % 1000)[None, :])
    fwd = same & (t[None, :] > t[A][:, None])
    pairs = []
    for scope, mask in (("same episode", fwd), ("independent", indep)):
        ai, ti = np.nonzero(mask & np.isfinite(D)); order = rng.permutation(len(ai))
        got = {f_: 0 for f_ in FACTORS}
        for k in order:
            i, j = A[ai[k]], ti[k]
            dif = object_diff(q[i], b[i], q[j], b[j])
            if len(dif) != 1 or got[dif[0]] >= a.per or in_drawer(q[i]) or in_drawer(q[j]):
                continue
            got[dif[0]] += 1; pairs.append((scope, dif[0], ai[k], j))
            if all(c >= a.per for c in got.values()):
                break
        print(f"[pairs] {scope}: " + ", ".join(f"{f_} {c}" for f_, c in got.items()), flush=True)
    jobs = [(k, fac, q[A[ai]], v[A[ai]], b[A[ai]], q[j], b[j]) for k, (_, fac, ai, j) in enumerate(pairs)]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        steps = dict(pool.imap_unordered(_rollout, jobs))
    rec = [dict(scope=s_, factor=fac, anchor=int(ai), target=int(j), graph=float(D[ai, j]), recorded=float(t[j] - t[A[ai]]) if s_ == "same episode" else np.nan,
                oracle=steps[k]) for k, (s_, fac, ai, j) in enumerate(pairs)]
    emb = Path(a.emb)
    for nm, metric in [("scene-mixlate-lag-1", "cos"), ("scene-mixlate-1", "cos"), ("scene-lewm-mixlate-1", "l2"), ("raw-state", "l2"), ("raw-pixels", "l2")]:
        Dm = model_dist(np.load(emb / f"{nm}.npy"), A, metric)
        for r in rec:
            r[nm] = float(Dm[r["anchor"], r["target"]])
    import pandas as pd
    d = pd.DataFrame(rec); d.to_csv(a.out, index=False)
    ok = d[d.oracle > 0]
    print(f"\n[oracle] success {len(ok)}/{len(d)}; by factor: " + ", ".join(f"{f_} {(d[d.factor == f_].oracle > 0).mean():.0%}" for f_ in FACTORS))
    print(f"[oracle] median steps: oracle {ok.oracle.median():.0f}, graph {ok.graph.median():.0f}; oracle <= graph for {(ok.oracle <= ok.graph).mean():.0%}")
    for scope in ("same episode", "independent"):
        s = ok[ok.scope == scope]
        if len(s) < 10:
            continue
        line = f"[{scope}] n {len(s)}: Spearman oracle vs graph {spearman(s.oracle, s.graph):+.2f}"
        if scope == "same episode":
            line += f", oracle vs recorded {spearman(s.oracle, s.recorded):+.2f}, graph vs recorded {spearman(s.graph, s.recorded):+.2f}"
        print(line)
        print(f"[{scope}] model distance vs oracle steps: " + ", ".join(f"{nm} {spearman(s[nm], s.oracle):+.2f}" for nm in ("scene-mixlate-lag-1", "scene-mixlate-1", "scene-lewm-mixlate-1", "raw-state", "raw-pixels")))


if __name__ == "__main__":
    main()
