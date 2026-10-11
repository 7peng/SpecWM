"""Spectral conversion of the centering / tempering ablation arms, on OGBench Scene and CALVIN: on a frame sample
of the training corpus (500 episodes x 40 frames, 350 episodes fit / 150 held out), the fully specified state kernel
of the recipe (the factor sum of `loss.spectral.factors`, centred, no tempering) is built over every pair and
eigendecomposed; every 2k-step snapshot of each arm is embedded and ridge-fit to each of the top 128 eigenvectors,
scored by held-out R^2 (ridge penalty per eigenvector by inner episode-grouped cross-validation). The heatmap (step x eigenvector, one panel per arm and env) shows how fast each form
converts the kernel's modes.

    python results/paper/scripts/kab_conversion.py compute --env scene     # -> battery/kab_conversion.csv
    python results/paper/scripts/kab_conversion.py compute --env calvin    # -> battery/kab_conversion_calvin.csv
    python results/paper/scripts/kab_conversion.py plot-fade --out results/paper/figures/ablations/figB_centering_tempering_conversion.png
    python results/paper/scripts/kab_conversion.py plot --out results/paper/figures/ablations/figB_centering_tempering_heatmap.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ICLR = HERE.parents[1]
sys.path.insert(0, str(HERE / "scripts")); sys.path.insert(0, str(ICLR / "src"))
from spectral_battery import OUT, CK, MODES, embed, ridge_r2  # noqa: E402

N_EP, PER_EP, FIT_EP, SEED = 500, 40, 350, 0
STEPS = list(range(2000, 33000, 2000))
LABELS = ["neither", "no tempering", "no centering", "centred, tempered"]
SCENE_STEM = "scene-kab-full-state-buttons-pa05-rel-"
ENVS = {
    "scene": dict(h5=("SCENE_STAGE_DIR", "scene_play.h5"), cols=["qpos", "button_states"], csv="kab_conversion.csv",
                  sample="sample.npz", share="kab_state_eigshare.npy", final="weights_epoch_1.pt",
                  runs=[SCENE_STEM + s for s in ("raw-s2", "notemper-s2", "nocenter-s2", "s2")]),
    "calvin": dict(h5=("CALVIN_STAGE_DIR", "calvin_abc_d_mix_lateonset.h5"), cols=["robot_obs", "scene_obs", "scene_id"],
                   csv="kab_conversion_calvin.csv", sample="sample_calvin.npz", share="kab_state_eigshare_calvin.npy",
                   final="weights_epoch_1.pt", runs=[f"calvin-kab-full-state-{s}-1" for s in ("raw", "center", "temper", "both")]),
}


def factors(env):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ICLR / "config" / "train" / f"{env}_state_ablation_centered_tempered.yaml")
    return [OmegaConf.to_container(f) for f in cfg.loss.spectral.factors]


def sample(env, spec):
    """Frames of `N_EP` episodes at `PER_EP` evenly spaced offsets, the first `FIT_EP` episodes fit; cached
    with the factor columns beside the pixels' indices (scene reuses the spectral battery's sample)."""
    import h5py, hdf5plugin  # noqa: F401
    pth = OUT / spec["sample"]
    if pth.exists():
        return np.load(pth)
    with h5py.File(Path(os.environ[spec["h5"][0]]) / spec["h5"][1], "r") as f:
        ep = f["ep_idx"][:]
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]); lens = np.diff(np.r_[starts, len(ep)])
        rng = np.random.default_rng(SEED)
        eps = np.sort(rng.choice(len(starts), size=N_EP, replace=False))
        frames = np.concatenate([starts[e] + np.linspace(0, lens[e] - 1, PER_EP).round().astype(int) for e in eps])
        ep_of = np.repeat(eps, PER_EP); fit = np.isin(ep_of, eps[:FIT_EP])
        cols = {c: f[c][:][frames] for c in spec["cols"]}
    np.savez(pth, frames=frames, ep_of=ep_of, fit=fit, **cols)
    return np.load(pth)


def kernel_eigvecs(env, S):
    import torch
    import train_spec as T
    rows = {c: torch.as_tensor(np.asarray(S[c]), device="cuda").float() for c in ENVS[env]["cols"]}
    K = None
    with torch.no_grad():
        for f in factors(env):
            Kf = float(f.get("weight", 1.0)) * T._factor_kernel(f["form"], T._factor_rows(f, rows), f.get("sigma"), f)
            K = Kf if K is None else K + Kf
            del Kf
        K = K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()
        w, U = torch.linalg.eigh(0.5 * (K + K.T))
    w = w.flip(0).cpu().numpy(); U = U.flip(1).cpu().numpy()[:, :MODES]
    share = np.clip(w, 0, None); share = (share / share.sum())[:MODES]
    return U, share


def pixels(spec, frames):
    import h5py, hdf5plugin  # noqa: F401
    import torch
    import torchvision.transforms.functional as TF
    with h5py.File(Path(os.environ[spec["h5"][0]]) / spec["h5"][1], "r") as f:
        px = torch.from_numpy(f["pixels"][np.sort(frames)]).permute(0, 3, 1, 2)
    if px.shape[-1] != 224:
        px = torch.cat([TF.resize(px[c:c + 512], [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                        for c in range(0, len(px), 512)])
    return px


def compute(env):
    import pandas as pd
    import torch
    from diagnostics.probe_encoders import IMAGENET_MEAN, IMAGENET_STD
    spec = ENVS[env]
    S = sample(env, spec); frames, fit = S["frames"], S["fit"]
    assert np.all(np.diff(frames) > 0), "frames must be sorted so the pixel rows align with the factor rows"
    U, share = kernel_eigvecs(env, S); np.save(OUT / spec["share"], share)
    px = pixels(spec, frames)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rows = []
    for label, run in [("untrained", "random:0"), *zip(LABELS, spec["runs"])]:
        for step in ([0] if run == "random:0" else STEPS + ["final"]):
            pth = run if run == "random:0" else CK / run / (spec["final"] if step == "final" else f"step_snaps/step_{step}.pt")
            if run != "random:0" and not pth.exists():
                continue
            r2 = ridge_r2(embed(pth, px, mean, std), U, fit, S["ep_of"])
            rows.append(dict(arm=label, step=33375 if step == "final" else step, **{f"m{i}": float(v) for i, v in enumerate(r2)}))
            print(f"{label:18s} {step!s:>6}: bands {r2[:8].mean():.3f} {r2[8:32].mean():.3f} {r2[32:128].mean():.3f} "
                  f"weighted {(r2 * share).sum() / share.sum():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / spec["csv"], index=False)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd
    cmap = LinearSegmentedColormap.from_list("house", ["#EEEEF3", "#B4C0DB", "#4C72B0", "#1F3A63"])
    titles = ["raw kernel", "+ centering", "+ tempering", "+ centering + tempering"]
    envs = [(e, "OGBench Scene" if e == "scene" else "CALVIN") for e in ENVS if (OUT / ENVS[e]["csv"]).exists()]
    cols = [f"m{i}" for i in range(MODES)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.facecolor": "#EEEEF3", "axes.edgecolor": "#EEEEF3"})
    fig, axes = plt.subplots(len(envs), 4, figsize=(12.5, 3.6 * len(envs)), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.14 / len(envs), hspace=0.42, wspace=0.08)
    for row, (env, name) in zip(axes, envs):
        df = pd.read_csv(OUT / ENVS[env]["csv"])
        steps = sorted(df[df.arm != "untrained"].step.unique())
        for ax, label, title in zip(row, LABELS, titles):
            g = df[df.arm == label].set_index("step").reindex(steps)
            im = ax.imshow(g[cols].values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap=cmap, interpolation="nearest",
                           extent=(0.5, MODES + 0.5, -0.5, len(steps) - 0.5))
            ax.set_yticks(range(0, len(steps), 2)); ax.set_yticklabels([f"{s / 1000:g}k" if s != 33375 else "final" for s in steps[::2]])
            ax.set_xticks([1, 32, 64, 96, 128]); ax.set_title(title, fontsize=12, pad=6, color="#222")
            ax.tick_params(length=0, colors="#555")
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax is not row[0]:
                ax.tick_params(labelleft=False)
        row[0].set_ylabel(f"{name}\nTraining step")
    for ax in axes[-1]:
        ax.set_xlabel("Eigenvector index")
    cax = fig.add_axes([0.915, 0.14 / len(envs), 0.015, 0.93 - 0.14 / len(envs)])
    cb = fig.colorbar(im, cax=cax, label="Held-out R² of eigenvector"); cb.outline.set_visible(False); cb.ax.tick_params(length=0, colors="#555")
    fig.savefig(out, dpi=170, facecolor="white"); print(out)


def plot_curves(out):
    """Conversion against training step: mean held-out R^2 of the kernel's leading (1-8), mid (9-32) and tail (33-128)
    eigenvectors, one line per arm, one row per environment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    bands = [("modes 1-8", 0, 8), ("modes 9-32", 8, 32), ("modes 33-128", 32, 128)]
    styles = [("raw kernel", "-."), ("+ centering", ":"), ("+ tempering", "--"), ("+ centering + tempering", "-")]
    envs = [(e, "OGBench Scene" if e == "scene" else "CALVIN") for e in ENVS if (OUT / ENVS[e]["csv"]).exists()]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.facecolor": "#EEEEF3", "axes.edgecolor": "#EEEEF3"})
    fig, axes = plt.subplots(len(envs), 3, figsize=(12.0, 3.4 * len(envs)), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.34 / len(envs), hspace=0.5, wspace=0.16)
    for row, (env, name) in zip(axes, envs):
        df = pd.read_csv(OUT / ENVS[env]["csv"])
        for ax, (title, lo, hi) in zip(row, bands):
            cols = [f"m{i}" for i in range(lo, hi)]
            for (label, key), (_, ls) in zip(zip(styles, LABELS), styles):
                g = df[df.arm == key].sort_values("step"); g = g[g.step != 33375]
                ax.plot(g.step / 1000, g[cols].mean(1), ls, color="#4C72B0", lw=2.0, marker="o", ms=3.5, label=label[0], zorder=3)
            u = df[df.arm == "untrained"]
            if len(u):
                ax.axhline(float(u.iloc[0][cols].mean()), color="#7F7F7F", lw=1.2, ls=(0, (2, 2)), label="untrained encoder", zorder=2)
            ax.set_xlim(0, 33); ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32]); ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title(f"{name}, {title}", fontsize=12, pad=6, color="#222")
            ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors="#555")
            for sp in ax.spines.values():
                sp.set_visible(False)
        row[0].set_ylabel("Held-out R² of eigenvectors")
    for ax in axes[-1]:
        ax.set_xlabel("Training step (k)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01), handlelength=3.2, columnspacing=1.8)
    fig.savefig(out, dpi=170, facecolor="white"); print(out)


def plot_fade(out, steps=(2000, 4000, 8000, 16000, 32000)):
    """The spectral-recovery figure through training: held-out R^2 of each kernel eigenvector (9-mode running mean)
    at the given steps, earlier steps faded, one panel per arm and one row per environment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    titles = ["raw kernel", "+ centering", "+ tempering", "+ centering + tempering"]
    bands = ((0, 8, "#DCDCDC"), (8, 32, "#E4E4E4"), (32, 128, "#EBEBEB"))
    alphas = np.linspace(0.18, 1.0, len(steps))
    envs = [(e, "OGBench Scene" if e == "scene" else "CALVIN") for e in ENVS if (OUT / ENVS[e]["csv"]).exists()]
    cols = [f"m{i}" for i in range(MODES)]; x = np.arange(1, MODES + 1)
    smooth = lambda r: pd.Series(np.asarray(r, dtype=float)).rolling(9, center=True, min_periods=1).mean().to_numpy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.facecolor": "#EEEEF3", "axes.edgecolor": "#EEEEF3"})
    fig, axes = plt.subplots(len(envs), 4, figsize=(13, 3.6 * len(envs)), sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.26 / len(envs), hspace=0.45, wspace=0.1)
    for row, (env, name) in zip(axes, envs):
        df = pd.read_csv(OUT / ENVS[env]["csv"])
        mn = min(smooth(r).min() for _, r in df[cols].iterrows())          # held-out R² can be negative
        floor = 0.0 if mn > -0.05 else float(np.floor(10 * mn) / 10)
        for ax, label, title in zip(row, LABELS, titles):
            g = df[df.arm == label].set_index("step")
            for st, al in zip(steps, alphas):
                if st not in g.index:
                    continue
                ax.plot(x, smooth(g.loc[st, cols].values), color="#4C72B0", lw=2.2, alpha=al, label=f"{st // 1000}k steps", zorder=3)
            u = df[df.arm == "untrained"]
            if len(u):
                ax.plot(x, smooth(u.iloc[0][cols].values), color="#444444", lw=1.0, ls=(0, (2, 2)), label="untrained encoder", zorder=2)
            for lo, hi, shade in bands:
                ax.axvspan(lo + 0.5, hi + 0.5, color=shade, zorder=0, lw=0)
            ax.set_xlim(1, MODES); ax.set_ylim(floor, 1); ax.set_xticks([1, 32, 64, 96, 128])
            if floor < 0:
                ax.axhline(0, color="#999999", lw=0.8, zorder=1)
            ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors="#555")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=12, loc="left", pad=7, color="#222")
        row[0].set_ylabel(f"{name}\nR² of eigenvector from embedding")
    for ax in axes[-1]:
        ax.set_xlabel("Eigenvector index")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(out, dpi=200, facecolor="white"); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["compute", "plot", "plot-curves", "plot-fade"]); ap.add_argument("--env", default="scene", choices=list(ENVS))
    ap.add_argument("--out", default=str(HERE / "figures" / "ablations" / "figB_centering_tempering_heatmap.png")); a = ap.parse_args()
    {"compute": lambda: compute(a.env), "plot": lambda: plot(a.out), "plot-curves": lambda: plot_curves(a.out), "plot-fade": lambda: plot_fade(a.out)}[a.cmd]()
