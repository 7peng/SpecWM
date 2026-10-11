"""The live probe protocol (grouped ridge, cv alpha, 5 episode-held-out splits) run on the episodes
the encoder NEVER TRAINED ON: train_common holds out round((1 - train_split) * n_episodes) episodes by
a torch.randperm seeded with cfg.seed over the sorted episode ids, and this script re-derives that set
(--episodes val), or samples the training episodes instead (--episodes train) for the paired reading.

    python results/paper/scripts/probe_val_episodes.py --ckpt lewm/scene-lin-state-1/weights_epoch_1.pt \
        --dataset $SCENE_STAGE_DIR/scene_play.h5 --episodes val --n 10000 --out results/paper/data/probe_val.csv
"""
import argparse
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch

ICLR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ICLR / "src"))
from utils import load_env  # noqa: E402
load_env()
from diagnostics.probe_encoders import build_encoder, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from diagnostics.probe_heads import grouped_fit  # noqa: E402

TARGETS = {"cube_xyz": ("qpos", slice(14, 17)), "drawer": ("qpos", slice(23, 24)), "window": ("qpos", slice(24, 25)), "buttons": ("button_states", slice(0, 2)), "arm": ("qpos", slice(0, 7))}


def val_episodes(ep_col, seed=3072, train_split=0.9):
    uniq = np.unique(ep_col)
    perm = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(seed)).numpy()
    n_val = max(1, int(round(len(uniq) * (1 - train_split))))
    return set(uniq[perm[:n_val]].tolist()), len(uniq)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--episodes", choices=["val", "train"], default="val")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
    f = h5py.File(a.dataset, "r"); ep = f["ep_idx"][:].astype(np.int64); val, n_ep = val_episodes(ep)
    pool = np.flatnonzero(np.isin(ep, list(val)) if a.episodes == "val" else ~np.isin(ep, list(val)))
    rows = np.sort(np.random.default_rng(a.seed).choice(pool, size=min(a.n, len(pool)), replace=False))
    ck = a.ckpt if Path(a.ckpt).is_absolute() else str(Path(__import__("os").environ["STABLEWM_HOME"]) / "checkpoints" / a.ckpt)
    enc, proj, _ = build_encoder(ck); mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    X = []
    with torch.no_grad():
        for c in range(0, len(rows), 256):
            px = torch.from_numpy(f["pixels"][rows[c:c + 256]]).permute(0, 3, 1, 2)
            x = ((px.float() / 255.0 - mean) / std).cuda()
            X.append(proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu().numpy())
    X = np.concatenate(X); g = ep[rows]
    out = []
    for name, (col, sl) in TARGETS.items():
        Y = f[col][rows][:, sl].astype(np.float64)
        r = grouped_fit(X, Y, g, ridge_alpha="cv", n_splits=5, seed0=0)
        out.append(dict(ckpt=a.ckpt, episodes=a.episodes, n_frames=len(rows), n_episodes=len(np.unique(g)), target=name, r2=r["r2"], r2_a10=r.get("r2_a10", np.nan)))
    d = pd.DataFrame(out); print(f"{a.ckpt}  {a.episodes} episodes ({len(np.unique(g))} of {n_ep}), {len(rows)} frames"); print(d[["target", "r2", "r2_a10"]].round(3).to_string(index=False))
    p = Path(a.out); d.to_csv(p, mode="a", header=not p.exists(), index=False)


if __name__ == "__main__":
    main()
