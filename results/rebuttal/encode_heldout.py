"""Embed every frame of a corpus's held-out validation episodes with several encoders in one pass over the
pixels (the probe's encode path: /255, ImageNet normalisation, ViT CLS token, projector). Episodes are read
by parallel worker processes. Writes one float16 (n_frames, d) array per model plus the frame rows, and
refuses a model whose adjacent frames are not more similar than random frame pairs.
    python results/rebuttal/encode_heldout.py DATASET.h5 OUTDIR name=ckpt ... [--random] [--workers 8]"""
import argparse, sys
from pathlib import Path
import h5py, hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "results" / "paper" / "scripts"))
from diagnostics.probe_encoders import build_encoder, random_encoder, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from probe_val_episodes import val_episodes  # noqa: E402


class Episodes(Dataset):
    """One held-out episode's pixels per item, read from the worker's own file handle."""

    def __init__(self, path, spans):
        self.path, self.spans, self.f = path, spans, None

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, k):
        if self.f is None:
            self.f = h5py.File(self.path, "r")
        a, b = self.spans[k]
        return k, torch.from_numpy(self.f["pixels"][a:b])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("outdir"); ap.add_argument("models", nargs="*")
    ap.add_argument("--random", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=500); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.dataset, "r") as f:
        ep = f["ep_idx"][:].astype(np.int64)
    val, _ = val_episodes(ep)
    rows = np.flatnonzero(np.isin(ep, list(val)))
    cut = np.r_[0, np.flatnonzero(np.diff(ep[rows]) != 0) + 1, len(rows)]
    spans = [(int(rows[cut[i]]), int(rows[cut[i + 1] - 1]) + 1) for i in range(len(cut) - 1)]
    assert sum(b - a for a, b in spans) == len(rows), "held-out episodes are not contiguous row ranges"
    nets = {}
    for spec in a.models:
        name, ck = spec.split("=", 1)
        nets[name] = build_encoder(ck)[:2]
    if a.random:
        nets["random-init"] = random_encoder()[:2]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).cuda(); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).cuda()
    E = {k: np.zeros((len(rows), 192), np.float16) for k in nets}
    offs = np.r_[0, np.cumsum([b - a for a, b in spans])]
    dl = DataLoader(Episodes(a.dataset, spans), batch_size=None, num_workers=a.workers, prefetch_factor=2)
    with torch.no_grad():
        for n_done, (k, px) in enumerate(dl):
            x_all = px.cuda(non_blocking=True).permute(0, 3, 1, 2)
            for s in range(0, len(x_all), a.batch):
                x = (x_all[s:s + a.batch].float() / 255.0 - mean) / std
                for name, (enc, proj) in nets.items():
                    z = proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])
                    E[name][offs[k] + s: offs[k] + s + len(x)] = z.float().cpu().numpy()
            if n_done % 20 == 0:
                print(f"[encode] {n_done + 1}/{len(spans)} episodes", flush=True)
    rng = np.random.default_rng(0); i = rng.choice(len(rows) - 1, 4000, replace=False)
    i = i[ep[rows[i]] == ep[rows[i + 1]]]; j = rng.choice(len(rows), len(i), replace=False)
    for name, Z in E.items():
        Zn = Z.astype(np.float32); Zn /= np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12
        adj, rnd = float((Zn[i] * Zn[i + 1]).sum(1).mean()), float((Zn[i] * Zn[j]).sum(1).mean())
        print(f"[gate] {name}: adjacent cos {adj:.3f}, random-pair cos {rnd:.3f}", flush=True)
        assert name == "random-init" or adj > rnd, f"{name}: adjacent frames no more similar than random ones; encode path is wrong"
        np.save(out / f"{name}.npy", Z)
    np.save(out / "rows.npy", rows)
    print(f"[done] {len(rows):,} frames x {len(E)} models -> {out}")


if __name__ == "__main__":
    main()
