"""Held-out-episode probe for the nine Fig. 3 runs, the Fig. 4 companion: 5,000 frames from the seeded
validation episodes (the 10% train_common never trains on, torch.randperm seed 3072), the planner's
embedding (CLS -> projector), raw state targets, grouped ridge with cv alpha over 5 episode-held-out
splits (probe_heads.grouped_fit), at snapshots 12k..28k every 4k. Same frames for every checkpoint of
an environment.

CALVIN's pixel mixture is not on this node; its two source files are, and the mixture is the play
file's episodes followed by the noised file's, so a mixture (episode, step) maps onto a source row.
CALVIN frames are 200x200 and are resized the way training does (torchvision bilinear, antialias).

    python results/paper/scripts/probe_val_3env.py --env scene --out data/probe_val_3env.csv
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
from torchvision.transforms.v2 import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from utils import load_env  # noqa: E402

load_env()
from diagnostics.probe_encoders import build_encoder, random_encoder, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from diagnostics.probe_heads import grouped_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_val_episodes import val_episodes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data" / "calvin"))
from roster import canonicalize  # noqa: E402

HOME = Path(os.environ["STABLEWM_HOME"])
ENVS = {
    "scene": dict(mix=HOME / "datasets" / "scene_play_mix_noised.h5", sources=None,
                  runs=["scene-lewm-mixm-1", "scene-mixdiag-1", "scene-mixdiag-det-1", "scene-lewm-mixlate-1", "scene-mixlate-1", "scene-mixlate-det-1", "scene-mixlate-lag-1"],
                  targets={"cube_xyz": ("qpos", slice(14, 17)), "drawer": ("qpos", slice(23, 24)), "window": ("qpos", slice(24, 25)),
                           "buttons": ("button_states", slice(0, 2)), "arm": ("qpos", slice(0, 7))}),
    "cube": dict(mix=HOME / "datasets" / "cube_play_mix_lateonset.h5", sources=None,
                 runs=["cube-lewm-mixlate-1", "cube-mixlate-t2-1", "cube-mixlate-t2-det-1", "cube-mixlate-lag-1"],
                 targets={"cube_xyz": ("qpos", slice(14, 17)), "arm": ("qpos", slice(0, 7))}),
    "calvin": dict(mix=HOME / "datasets" / "calvin_abc_d_mix_noised.h5", sources=[HOME / "calvin_abc_d.h5", HOME / "calvin_abc_d_noised.h5"],
                   runs=["calvin-lewm-mixm-1", "calvin-mixdiag-1", "calvin-mixdiag-det-1", "calvin-lewm-mixlate-1", "calvin-mixlate-1", "calvin-mixlate-det-1", "calvin-mixlate-lag-1"],
                   targets={"red_xyz": ("scene_obs", slice(6, 9)), "blue_xyz": ("scene_obs", slice(12, 15)), "pink_xyz": ("scene_obs", slice(18, 21)),
                            "slider": ("scene_obs", slice(0, 1)), "drawer": ("scene_obs", slice(1, 2)), "switch": ("scene_obs", slice(3, 4)),
                            "arm": ("robot_obs", slice(0, 15))}),
}
STEPS = [12000, 16000, 20000, 24000, 28000]


def sample_rows(mix, n, seed):
    with h5py.File(mix, "r") as f:
        ep = f["ep_idx"][:].astype(np.int64)
        val, n_ep = val_episodes(ep)
        pool = np.flatnonzero(np.isin(ep, list(val)))
        rows = np.sort(np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False))
        return rows, ep[rows], len(val), n_ep, f["step_idx"][:][rows] if "step_idx" in f else None


def read_columns(spec, rows, ep_rows, step_rows, targets):
    """pixels (N,H,W,3) uint8 and the target arrays, read from the mixture or from its source files."""
    if spec["sources"] is None:
        with h5py.File(spec["mix"], "r") as f:
            px = f["pixels"][rows]
            Y = {k: f[col][rows][:, sl] for k, (col, sl) in targets.items()}
        return px, Y
    play, noised = spec["sources"]
    with h5py.File(play, "r") as fp, h5py.File(noised, "r") as fn:
        n_play = len(fp["ep_len"]); off_p = fp["ep_offset"][:]; off_n = fn["ep_offset"][:]
        src = np.where(ep_rows < n_play, off_p[np.minimum(ep_rows, n_play - 1)] + step_rows,
                       off_n[np.maximum(ep_rows - n_play, 0)] + step_rows)
        is_play = ep_rows < n_play
        px = np.empty((len(rows),) + fp["pixels"].shape[1:], np.uint8)
        Y = {k: np.empty((len(rows), sl.stop - sl.start), np.float64) for k, (_, sl) in targets.items()}
        for f, m in ((fp, is_play), (fn, ~is_play)):
            if not m.any():
                continue
            order = np.argsort(src[m]); idx = src[m][order]; dest = np.flatnonzero(m)[order]
            px[dest] = f["pixels"][idx]
            cols = {}
            for col in {c for c, _ in targets.values()}:
                v = f[col][idx]
                if col == "scene_obs" and not f[col].attrs.get("roster"):
                    v = canonicalize(v, f["scene_id"][idx].reshape(-1))    # the play file is in simulator order
                cols[col] = v
            for k, (col, sl) in targets.items():
                Y[k][dest] = cols[col][:, sl]
        return px, Y


@torch.no_grad()
def features(ckpt, px, dev="cuda", bs=256):
    enc, proj, _ = random_encoder(device=dev, seed=0) if ckpt == "random-init" else build_encoder(ckpt, device=dev)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    X = []
    for c in range(0, len(px), bs):
        x = torch.from_numpy(px[c:c + bs]).permute(0, 3, 1, 2)
        if x.shape[-1] != 224:
            x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x = (x.to(dev).float() / 255.0 - mean) / std
        X.append(proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu().numpy())
    return np.concatenate(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS)
    ap.add_argument("--out", required=True); ap.add_argument("--runs", nargs="*", default=None, help="subset of the env's runs")
    ap.add_argument("--random", action="store_true", help="probe a random-init encoder (width 192) instead of checkpoints; one row per env")
    ap.add_argument("--final", action="store_true", help="probe each run's final checkpoint (its highest weights_epoch_*.pt) instead of --steps; step is recorded as -1")
    a = ap.parse_args()
    spec = ENVS[a.env]
    rows, ep_rows, n_val, n_ep, step_rows = sample_rows(spec["mix"], a.n, a.seed)
    print(f"[{a.env}] {len(rows)} frames from {len(np.unique(ep_rows))} of the {n_val} validation episodes ({n_ep} total)", flush=True)
    px, Y = read_columns(spec, rows, ep_rows, step_rows, spec["targets"])
    print(f"[{a.env}] pixels {px.shape}", flush=True)
    out = []
    def final_ckpt(run):
        return str(max((HOME / "checkpoints" / "lewm" / run).glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1])))

    jobs = [("random-init", 0)] if a.random else [(run, -1) for run in (a.runs or spec["runs"])] if a.final else [(run, s) for run in (a.runs or spec["runs"]) for s in a.steps]
    for run, s in jobs:
        if True:
            ck = "random-init" if a.random else final_ckpt(run) if a.final else str(HOME / "checkpoints" / "lewm" / run / "step_snaps" / f"step_{s}.pt")
            X = features(ck, px)
            rec = dict(env=a.env, run=run, step=s, n_frames=len(rows), n_episodes=len(np.unique(ep_rows)))
            fits = Parallel(n_jobs=min(8, len(Y)))(delayed(grouped_fit)(X, y.astype(np.float64), ep_rows, ridge_alpha="cv", n_splits=5, seed0=0) for y in Y.values())
            for k, r in zip(Y, fits):
                rec[k] = r["r2"]
            out.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    d = pd.DataFrame(out); p = Path(a.out)
    if p.exists():
        d = pd.concat([pd.read_csv(p), d], ignore_index=True)
    d.to_csv(p, index=False); print(f"[out] {p}")


if __name__ == "__main__":
    main()
