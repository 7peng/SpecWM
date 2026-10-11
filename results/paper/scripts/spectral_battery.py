"""Spectral battery, no Nystrom: on one frame sample of scene play, each kernel is built over every
pair, centred and eigendecomposed; each encoder's embedding is ridge-fit to every eigenvector on the
fit episodes and scored by R^2 on the held-out episodes. Also the kernel's own eigenvectors as a
readout of the state factors (the ceiling) and the untrained encoder (the floor).

    python results/paper/scripts/spectral_battery.py cache     # embed the sample with every encoder
    python results/paper/scripts/spectral_battery.py battery   # kernels, eigenvectors, readouts
    python results/paper/scripts/spectral_battery.py plot
    python results/paper/scripts/spectral_battery.py cache-snaps <shard> <n_shards>   # every step snapshot
    python results/paper/scripts/spectral_battery.py battery-snaps
    python results/paper/scripts/spectral_battery.py plot-conversion
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ICLR = HERE.parents[1]
sys.path.insert(0, str(ICLR / "src"))
from utils import load_env  # noqa: E402

load_env()
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import torch  # noqa: E402

STAGE = Path(os.environ["SCENE_STAGE_DIR"])
CK = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / "lewm"
OUT = HERE / "battery"
N_EP, PER_EP, FIT_EP, EP_LEN, SEED = 500, 40, 350, 1000, 0
MODES = 128
ENCODERS = {  # label: checkpoint
    "untrained": "random:0",
    "LeWM": CK / "scene-lewm-fs5-2" / "weights_epoch_5.pt",
    "temporal (under)": CK / "scene-spec-lag-1" / "step_snaps" / "step_9000.pt",
    "quasimetric (under)": CK / "scene-spec-qm-1" / "weights_epoch_1.pt",
    "state (under)": CK / "scene-spec-state-2" / "weights_epoch_1.pt",
    "proprio (under)": CK / "scene-spec-proprio-1" / "weights_epoch_1.pt",
    "depth (under)": CK / "scene-spec-depth-1" / "weights_epoch_1.pt",
    "state (full)": CK / "scene-full-state-2" / "weights_epoch_1.pt",
    "proprio (full)": CK / "scene-full-proprio-2" / "weights_epoch_1.pt",
    "depth (full)": CK / "scene-full-depth-2" / "weights_epoch_1.pt",
    "lin proprio": CK / "scene-lin-proprio-1" / "weights_epoch_1.pt",           # the kernel-dominance pair: linear label kernels
    "lin state": CK / "scene-lin-state-1" / "weights_epoch_1.pt",
}
KERNELS = {  # name: (column, form, sigma of the fully specified arm)
    "state": ("observation", "laplace", 5.6042),
    "proprio": ("qpos_arm_cossin", "laplace", 1.169),
    "depth": ("depth_z", "cos", 1.0258),
}
RUNS = {  # label: (run dir, objective)
    "temporal": ("scene-spec-lag-1", "under"), "quasimetric": ("scene-spec-qm-1", "under"), "state": ("scene-spec-state-2", "under"),
    "proprio": ("scene-spec-proprio-1", "under"), "depth": ("scene-spec-depth-1", "under"),
    "state_full": ("scene-full-state-2", "full"), "proprio_full": ("scene-full-proprio-2", "full"), "depth_full": ("scene-full-depth-2", "full"),
    "LeWM": ("scene-lewm-fs5-2", "control"),
}
TARGETS = {"cube": ("qpos", [14, 15, 16]), "arm": ("qpos", list(range(7))), "drawer": ("qpos", [23]), "window": ("qpos", [24]), "buttons": ("button_states", [0, 1])}


def sample():
    rng = np.random.default_rng(SEED)
    eps = np.sort(rng.choice(EP_LEN, size=N_EP, replace=False))
    offs = np.linspace(0, EP_LEN - 1, PER_EP).round().astype(int)
    frames = (eps[:, None] * EP_LEN + offs[None, :]).ravel()
    ep_of = np.repeat(eps, PER_EP)
    fit_eps = set(eps[:FIT_EP].tolist())
    fit = np.array([e in fit_eps for e in ep_of])
    return frames, ep_of, fit


def cache(only=None):
    """Embed the sample with every encoder, or only the labels named on the command line."""
    from diagnostics.probe_encoders import build_encoder, random_encoder, IMAGENET_MEAN, IMAGENET_STD
    OUT.mkdir(exist_ok=True)
    frames, ep_of, fit = sample()
    with h5py.File(STAGE / "scene_play.h5", "r") as f:
        px = np.stack([f["pixels"][int(i)] for i in frames])                       # (n, 224, 224, 3) uint8
        if not (only and (OUT / "sample.npz").exists()):                          # a filtered cache adds encoders to an existing sample
            cols = {"qpos": f["qpos"][:][frames], "observation": f["observation"][:][frames], "button_states": f["button_states"][:][frames]}
    if not (only and (OUT / "sample.npz").exists()):
        with h5py.File(STAGE / "scene_play_depth.h5", "r") as f:
            cols["depth_z"] = np.stack([f["depth_z"][int(i)] for i in frames]).astype(np.float16)
        q = cols["qpos"][:, :6]; cols["qpos_arm_cossin"] = np.concatenate([np.cos(q), np.sin(q)], 1)
        np.savez(OUT / "sample.npz", frames=frames, ep_of=ep_of, fit=fit, **cols)
    else:
        assert np.array_equal(np.load(OUT / "sample.npz")["frames"], frames), "the existing sample is not this sample"
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x_all = torch.from_numpy(px).permute(0, 3, 1, 2)
    for label, ck in ENCODERS.items():
        if only and label not in only:
            continue
        enc, proj, _ = random_encoder(seed=0) if str(ck).startswith("random") else build_encoder(str(ck))
        out = []
        with torch.no_grad():
            for c in range(0, len(frames), 256):
                x = ((x_all[c:c + 256].float() / 255.0 - mean) / std).cuda()
                out.append(proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu().numpy())
        emb = np.concatenate(out)
        d_adj = np.linalg.norm(emb[1:] - emb[:-1], axis=1)[np.diff(ep_of) == 0].mean()
        d_rand = np.linalg.norm(emb - emb[np.random.default_rng(1).permutation(len(emb))], axis=1).mean()
        assert d_adj < d_rand, f"{label}: adjacent frames not closer than random ({d_adj:.3f} vs {d_rand:.3f})"
        np.save(OUT / f"emb_{label.replace(' ', '_').replace('(', '').replace(')', '')}.npy", emb)
        print(f"{label:22s} emb {emb.shape} adjacent/random {d_adj / d_rand:.2f}", flush=True)


def snapshots():
    """(label, step, path) for every step snapshot of every run, LeWM by epoch (6,423 steps each)."""
    out = []
    for label, (run, _) in RUNS.items():
        d = CK / run
        for f in sorted((d / "step_snaps").glob("step_*.pt"), key=lambda q: int(q.stem.split("_")[1])):
            out.append((label, int(f.stem.split("_")[1]), f))
        if label == "LeWM":
            for f in sorted(d.glob("weights_epoch_*.pt"), key=lambda q: int(q.stem.split("_")[-1])):
                out.append((label, 6423 * int(f.stem.split("_")[-1]), f))
    return out


def embed(ck, px_all, mean, std):
    from diagnostics.probe_encoders import build_encoder, random_encoder
    enc, proj, _ = random_encoder(seed=0) if str(ck).startswith("random") else build_encoder(str(ck))
    out = []
    with torch.no_grad():
        for c in range(0, len(px_all), 256):
            x = ((px_all[c:c + 256].float() / 255.0 - mean) / std).cuda()
            out.append(proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu().numpy())
    return np.concatenate(out)


def cache_snaps(shard, n_shards):
    from diagnostics.probe_encoders import IMAGENET_MEAN, IMAGENET_STD
    (OUT / "snaps").mkdir(exist_ok=True)
    S = np.load(OUT / "sample.npz"); frames = S["frames"]
    with h5py.File(STAGE / "scene_play.h5", "r") as f:
        px = torch.from_numpy(np.stack([f["pixels"][int(i)] for i in frames])).permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    todo = [(l, st, pth) for i, (l, st, pth) in enumerate(snapshots()) if i % n_shards == shard]
    for label, step, pth in todo:
        dst = OUT / "snaps" / f"{label}_{step}.npy"
        if dst.exists():
            continue
        np.save(dst, embed(pth, px, mean, std)); print(f"[shard {shard}] {label} {step}", flush=True)


def battery_snaps():
    import train_spec as T
    S = np.load(OUT / "sample.npz"); fit = S["fit"]
    U = {}
    for kname, (col, form, sigma) in KERNELS.items():
        X = torch.as_tensor(np.asarray(S[col]), device="cuda").float()
        with torch.no_grad():
            K = T._factor_kernel(form, X, sigma, {"form": form}); K = K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()
            w, V = torch.linalg.eigh(0.5 * (K + K.T))
        U[kname] = V.flip(1).cpu().numpy()[:, :MODES]
        del K, X; torch.cuda.empty_cache()
    rows = [dict(run="untrained", step=0, kernel=k, **{f"m{i}": float(v) for i, v in enumerate(ridge_r2(np.load(OUT / "emb_untrained.npy"), U[k], fit, S["ep_of"]))}) for k in KERNELS]
    for pth in sorted((OUT / "snaps").glob("*.npy")):
        label, step = pth.stem.rsplit("_", 1); E = np.load(pth)
        for k in KERNELS:
            r2 = ridge_r2(E, U[k], fit, S["ep_of"])
            rows.append(dict(run=label, step=int(step), kernel=k, **{f"m{i}": float(v) for i, v in enumerate(r2)}))
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["kernel", "run", "step"]); df.to_csv(OUT / "conversion.csv", index=False)
    print(df.groupby(["kernel", "run"]).step.max().to_string())


def plot_conversion():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(OUT / "conversion.csv")
    bands = {"modes 1-8": range(0, 8), "modes 9-32": range(8, 32), "modes 33-128": range(32, 128)}
    for name, idx in bands.items():
        df[name] = df[[f"m{i}" for i in idx]].mean(1)
    HUE = {"modes 1-8": "#4C72B0", "modes 9-32": "#DD8452", "modes 33-128": "#55A868"}
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 12, "axes.facecolor": "#EBEBEB", "axes.edgecolor": "#EBEBEB"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.14, wspace=0.1, hspace=0.42)
    for c, kname in enumerate(KERNELS):
        for r, (obj, suffix) in enumerate((("Fully specified", "_full"), ("Underspecified", ""))):
            ax = axes[r][c]
            g = df[(df.kernel == kname) & (df.run == kname + suffix)].sort_values("step")
            u = df[(df.kernel == kname) & (df.run == "untrained")]
            for name, col in HUE.items():
                ax.plot(g.step, g[name], color=col, lw=2.2, label=name)
                ax.axhline(float(u[name].iloc[0]), color=col, lw=1.0, ls=(0, (3, 3)), alpha=0.8)
            ax.set_title(f"{obj}, {kname} kernel", fontsize=13, weight="bold", loc="left", pad=8)
            ax.set_ylim(0, 1); ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
        axes[1][c].set_xlabel("Training step", fontsize=12)
    for r in range(2):
        axes[r][0].set_ylabel("R² of eigenvectors\nfrom embedding", fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#444", lw=1.0, ls=(0, (3, 3)))); l.append("untrained encoder")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005))
    plt.savefig(HERE / "fig2_conversion.png", dpi=200, facecolor="white"); print(HERE / "fig2_conversion.png")


CV_ALPHAS = np.logspace(-2, 4, 13)


def _r2_cols(P, Yt):
    return 1.0 - ((P - Yt) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0).clamp_min(1e-12)


def _ridge_all(Xtr, Ytr, Xte, alphas):
    """Held-out predictions for every alpha at once: features standardised on the train rows, targets centred."""
    mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False); sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd; ym = Ytr.mean(0)
    G, c, eye = A.T @ A, A.T @ (Ytr - ym), torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    return {float(a): B @ torch.linalg.solve(G + float(a) * eye, c) + ym for a in alphas}


def ridge_r2(X, Y, fit, groups, n_inner=2, seed=0):
    """Held-out R^2 of every column of Y from a ridge readout of X, fitted on the `fit` rows and scored on the rest.
    The penalty is chosen per column from CV_ALPHAS on `n_inner` episode-grouped inner splits of the fit rows (the
    paper probe protocol's grid and selection), so a column no readout predicts scores near zero, not below it.
    Runs in float64 on the GPU when one is available."""
    from sklearn.model_selection import GroupShuffleSplit
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev); Y = torch.as_tensor(np.asarray(Y), dtype=torch.float64, device=dev)
    g = np.asarray(groups); fit = np.asarray(fit, dtype=bool)
    fi = np.flatnonzero(fit); Xf, Yf, gf = X[fi], Y[fi], g[fi]
    tot = torch.zeros((len(CV_ALPHAS), Y.shape[1]), dtype=torch.float64, device=dev)
    for s in range(n_inner):
        itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1000 + s).split(np.zeros(len(gf)), groups=gf))
        P = _ridge_all(Xf[itr], Yf[itr], Xf[ite], CV_ALPHAS)
        for ai, a in enumerate(CV_ALPHAS):
            tot[ai] += _r2_cols(P[float(a)], Yf[ite])
    best = CV_ALPHAS[tot.argmax(0).cpu().numpy()]
    te = torch.as_tensor(np.flatnonzero(~fit), device=dev)
    P = _ridge_all(Xf, Yf, X[te], np.unique(best)); Yt = Y[te]
    pred = torch.stack([P[float(a)][:, j] for j, a in enumerate(best)], 1)
    return _r2_cols(pred, Yt).cpu().numpy()


def battery():
    import train_spec as T
    S = np.load(OUT / "sample.npz"); fit = S["fit"]
    embs = {l: np.load(OUT / f"emb_{l.replace(' ', '_').replace('(', '').replace(')', '')}.npy") for l in ENCODERS}
    res = {"n": int(len(fit)), "n_fit": int(fit.sum()), "modes": MODES, "kernels": {}}
    for kname, (col, form, sigma) in KERNELS.items():
        X = torch.as_tensor(np.asarray(S[col]), device="cuda").float()
        with torch.no_grad():
            K = T._factor_kernel(form, X, sigma, {"form": form})
            n = K.shape[0]; K = K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()
            w, U = torch.linalg.eigh(0.5 * (K + K.T))
        w = w.flip(0).cpu().numpy(); U = U.flip(1).cpu().numpy()[:, :MODES]
        share = (np.clip(w, 0, None) / np.clip(w, 0, None).sum())[:MODES]
        entry = {"eig_share": share.tolist(), "encoders": {}, "ceiling": {}}
        for label, E in embs.items():
            r2 = ridge_r2(E, U, fit, S["ep_of"])
            entry["encoders"][label] = {"per_mode": r2.tolist(), "band0-7": float(r2[:8].mean()), "band8-31": float(r2[8:32].mean()), "band32-127": float(r2[32:128].mean())}
        for tname, (tc, dims) in TARGETS.items():
            Y = np.asarray(S[tc])[:, dims].astype(np.float64)
            entry["ceiling"][tname] = float(ridge_r2(U, Y, fit, S["ep_of"]).mean())
        res["kernels"][kname] = entry
        print(kname, "| ceiling:", {k: round(v, 2) for k, v in entry["ceiling"].items()}, "| band0-7:",
              {l: round(e["band0-7"], 2) for l, e in entry["encoders"].items()}, flush=True)
    json.dump(res, open(OUT / "battery.json", "w"))


ABLATION = {  # label: (checkpoint, kernel it pairs with, objective)
    "state (kernel, play+explore)": (CK / "scene-fullex-state-1" / "weights_epoch_1.pt", "state", "kernel"),
    "state (regression, play+explore)": (CK / "scene-gtex-state-1" / "weights_epoch_1.pt", "state", "regression"),
    "proprio (kernel, play+explore)": (CK / "scene-fullex-proprio-1" / "weights_epoch_1.pt", "proprio", "kernel"),
    "proprio (regression, play+explore)": (CK / "scene-gtex-proprio-1" / "weights_epoch_1.pt", "proprio", "regression"),
    "depth (kernel)": (CK / "scene-full-depth-3" / "weights_epoch_1.pt", "depth", "kernel"),
    "depth (regression)": (CK / "scene-gt-depth-2" / "weights_epoch_1.pt", "depth", "regression"),
}


def battery_ablation():
    """The kernel-vs-regression pairs read against their own kernel's eigenvectors: per-mode R^2,
    band means and the eigenvalue-weighted conversion, written to battery_ablation.json."""
    import train_spec as T
    from diagnostics.probe_encoders import IMAGENET_MEAN, IMAGENET_STD
    S = np.load(OUT / "sample.npz"); frames, fit = S["frames"], S["fit"]
    with h5py.File(STAGE / "scene_play.h5", "r") as f:
        px = torch.from_numpy(np.stack([f["pixels"][int(i)] for i in frames])).permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    embs = {}
    for label, (ck, _, _) in ABLATION.items():
        dst = OUT / f"emb_{label.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('+', '_')}.npy"
        if not dst.exists():
            np.save(dst, embed(ck, px, mean, std))
        embs[label] = np.load(dst); print(label, embs[label].shape, flush=True)
    res = {"n": int(len(fit)), "n_fit": int(fit.sum()), "modes": MODES, "encoders": {}}
    for kname, (col, form, sigma) in KERNELS.items():
        X = torch.as_tensor(np.asarray(S[col]), device="cuda").float()
        with torch.no_grad():
            K = T._factor_kernel(form, X, sigma, {"form": form})
            K = K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()
            w, U = torch.linalg.eigh(0.5 * (K + K.T))
        w = w.flip(0).cpu().numpy(); U = U.flip(1).cpu().numpy()[:, :MODES]
        share = (np.clip(w, 0, None) / np.clip(w, 0, None).sum())[:MODES]
        for label, (_, k, obj) in ABLATION.items():
            if k != kname:
                continue
            r2 = ridge_r2(embs[label], U, fit, S["ep_of"])
            res["encoders"][label] = {"kernel": k, "objective": obj, "per_mode": r2.tolist(), "band0-7": float(r2[:8].mean()),
                                      "band8-31": float(r2[8:32].mean()), "band32-127": float(r2[32:128].mean()),
                                      "weighted": float((r2 * share).sum() / share.sum())}
            print(kname, label, {b: round(res["encoders"][label][b], 3) for b in ("band0-7", "band8-31", "band32-127", "weighted")}, flush=True)
        del K, X; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "battery_ablation.json", "w"))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(OUT / "battery.json"))
    HUE = {"state": "#55A868", "proprio": "#8172B3", "depth": "#C44E52"}
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 12, "axes.facecolor": "#EBEBEB", "axes.edgecolor": "#EBEBEB"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.3, wspace=0.12)
    x = np.arange(1, MODES + 1)
    for ax, (kname, entry) in zip(axes, res["kernels"].items()):
        enc = entry["encoders"]
        for label, ls, col, lw in ((f"{kname} (full)", "-", HUE[kname], 2.4), (f"{kname} (under)", "--", HUE[kname], 2.0), ("LeWM", "-", "#8E8E8E", 2.0), ("untrained", ":", "#444444", 1.8)):
            y = np.convolve(np.asarray(enc[label]["per_mode"]), np.ones(5) / 5, mode="same")
            ax.plot(x, y, ls=ls, color=col, lw=lw, label=label)
        ax.set_title(f"{kname} kernel", fontsize=15, weight="bold", loc="left", pad=10)
        ax.set_xscale("log"); ax.set_xlim(1, MODES); ax.set_ylim(-0.05, 1.0)
        ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Eigenvector index", fontsize=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=10.5)
    axes[0].set_ylabel("R² of eigenvector from embedding\n(5-mode running mean)", fontsize=12)
    plt.savefig(HERE / "fig2_spectral.png", dpi=200, facecolor="white"); print(HERE / "fig2_spectral.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache-snaps":
        cache_snaps(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "cache" and len(sys.argv) > 2:
        cache(only=set(sys.argv[2:]))
    else:
        {"cache": cache, "battery": battery, "plot": plot, "battery-snaps": battery_snaps, "plot-conversion": plot_conversion,
         "battery-ablation": battery_ablation}[cmd]()
