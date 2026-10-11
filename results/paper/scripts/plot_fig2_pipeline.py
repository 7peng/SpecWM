"""Figure 2 (remake): one scene clip -> its target kernel as a matrix -> the encoder's Gram of the same clip
-> the embedding rows against the kernel's spectral embedding, both coloured by time. Real data: a 23-row clip
at frameskip 5 from scene_play_mix_lateonset (a play episode), the clip clock kernel exp(-|i-j|/7), and the
encoder of a trained within-clip run (CLS -> projector, unit rows).

    python results/paper/scripts/plot_fig2_pipeline.py --ckpt lewm/scene-mixdiag-lag-1/step_snaps/step_16000.pt --out results/paper/fig2_pipeline.png
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from utils import load_env  # noqa: E402

load_env()
from diagnostics.probe_encoders import build_encoder, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

HOME = Path(os.environ["STABLEWM_HOME"])
BLUE, LIGHT = "#4C72B0", "#B4C0DB"


def clip(path, episode, start, T, fs):
    with h5py.File(path, "r") as f:
        off = int(f["ep_offset"][episode]); idx = off + start + np.arange(T) * fs
        return f["pixels"][idx]


@torch.no_grad()
def embed(ckpt, px):
    enc, proj, _ = build_encoder(str(HOME / "checkpoints" / ckpt), device="cpu")
    x = torch.from_numpy(px).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    h = proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float()
    return torch.nn.functional.normalize(h, dim=-1).numpy()


def centred(K):
    n = len(K); J = np.eye(n) - 1.0 / n
    return J @ K @ J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lewm/scene-mixdiag-lag-1/step_snaps/step_16000.pt")
    ap.add_argument("--data", default=str(HOME / "datasets" / "scene_play_mix_lateonset.h5"))
    ap.add_argument("--episode", type=int, default=3); ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--T", type=int, default=23); ap.add_argument("--fs", type=int, default=5); ap.add_argument("--sigma", type=float, default=7.0)
    ap.add_argument("--centred", action="store_true", help="show the double-centred kernel and Gram (what the loss compares) on a diverging scale")
    ap.add_argument("--out", required=True); ap.add_argument("--elev", type=float, default=23); ap.add_argument("--azim", type=float, default=-71); ap.add_argument("--sheet", action="store_true", help="render a contact sheet of viewpoints of panel d instead")
    ap.add_argument("--dump", default=None, help="write the two 3-d point clouds (psi, Z) and frame indices to this JSON and exit")
    a = ap.parse_args()
    px = clip(a.data, a.episode, a.start, a.T, a.fs)
    t = np.arange(a.T)
    K = np.exp(-np.abs(t[:, None] - t[None, :]) / a.sigma)
    H = embed(a.ckpt, px); G = H @ H.T
    Kc, Gc = centred(K), centred(G)
    w, V = np.linalg.eigh(Kc); order = np.argsort(w)[::-1]; psi = V[:, order[:3]] * np.sqrt(w[order[:3]])
    Hc = H - H.mean(0); u, s, vt = np.linalg.svd(Hc, full_matrices=False); Z = u[:, :3] * s[:3]
    for k in range(3):
        if np.corrcoef(psi[:, k], Z[:, k])[0, 1] < 0: Z[:, k] *= -1
    corr = float(np.corrcoef(Kc[np.triu_indices(a.T, 1)], Gc[np.triu_indices(a.T, 1)])[0, 1])

    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("house", ["#FFFFFF", "#B4C0DB", "#4C72B0", "#1F3A63"])
    tramp = LinearSegmentedColormap.from_list("house_t", ["#B4C0DB", "#4C72B0", "#1F3A63"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
    grey, dark = "#7F7F7F", "#333333"
    FW, FH = 14.0, 6.0
    if a.dump:
        import json
        json.dump({"T": int(a.T), "fs": int(a.fs), "sigma": a.sigma, "corr": corr, "t": t.tolist(), "psi": psi.tolist(), "Z": Z.tolist()}, open(a.dump, "w"))
        print(a.dump); return
    if a.sheet:
        views = [(e, z) for e in (15, 30, 45) for z in (-60, -30, 0, 30)]
        fs = plt.figure(figsize=(16, 12))
        for n, (e, z) in enumerate(views):
            ax = fs.add_subplot(3, 4, n + 1, projection="3d"); r = max(np.ptp(psi[:, j]) for j in range(3)) * 0.62
            ax.plot(psi[:, 0], psi[:, 1], psi[:, 2], color="#B9BFCC", lw=1.0); ax.scatter(psi[:, 0], psi[:, 1], psi[:, 2], c=t, cmap=tramp, s=30, depthshade=False)
            for j in range(3):
                v = np.zeros(3); v[j] = r; ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color="#9AA1AE", lw=1.0); ax.text(*(v * 1.12), ["ψ1", "ψ2", "ψ3"][j], fontsize=9, color="#7F7F7F")
            ax.view_init(elev=e, azim=z); ax.set_axis_off(); ax.set_title(f"elev {e}, azim {z}", fontsize=10)
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim): setter(-r * 1.1, r * 1.1)
        fs.savefig(a.out, dpi=110, facecolor="white"); print(a.out); return
    fig = plt.figure(figsize=(FW, FH))
    S = 2.55 / FW; SH = 2.55 / FH                      # one square panel, in figure fractions
    GUT = (1 - 2 * 0.04 - 4 * S) / 3
    X0 = [0.04 + k * (S + GUT) for k in range(4)]
    Y0 = 0.165
    def panel(k): return fig.add_axes([X0[k], Y0, S, SH])
    TY = Y0 + SH + 0.012
    def title(ax, main, sub=None, k=None):
        x = X0[k] if k is not None else ax.get_position().x0
        fig.text(x, TY, main, fontsize=15, color="#222222", va="bottom", ha="left")
    def clean(ax, bg=True):
        ax.set_facecolor("#EEEEF3" if bg else "white"); ax.tick_params(length=0, labelsize=10.5, colors=grey)
        for sp in ax.spines.values(): sp.set_visible(False)

    # (a) the clip, spanning the width
    show = np.linspace(0, a.T - 1, 8).round().astype(int); gap = 6; W = px.shape[2]
    strip = np.full((px.shape[1], len(show) * W + (len(show) - 1) * gap, 3), 255, np.uint8)
    for k, i in enumerate(show): strip[:, k * (W + gap):k * (W + gap) + W] = px[i]
    sw = X0[3] + S - X0[0]; sh = sw * FW / FH * strip.shape[0] / strip.shape[1]
    axc = fig.add_axes([X0[0], 0.985 - sh, sw, sh]); axc.imshow(strip); axc.set_xticks([]); axc.set_yticks([]); clean(axc, bg=False)
    for k, i in enumerate(show): axc.text(k * (W + gap) + W / 2, px.shape[1] + 12, f"{i}", ha="center", va="top", fontsize=10.5, color=grey, clip_on=False)

    ticks = [0, a.T // 2, a.T - 1]
    if a.centred:
        Kshow, Gshow = Kc, Gc; vlim = float(max(abs(Kc).max(), abs(Gc).max())); vmin, vmax = -vlim, vlim
        ramp = LinearSegmentedColormap.from_list("house_div", ["#B4503A", "#E9C4B8", "#FFFFFF", "#B4C0DB", "#1F3A63"]); cticks = [vmin, 0, vmax]; clabels = [f"{vmin:.1f}", "0", f"{vmax:.1f}"]
    else:
        Kshow, Gshow = K, G; vmin, vmax = 0, 1; cticks = [0, 1]; clabels = ["0", "1"]
    ax1 = panel(0); ax1.imshow(Kshow, cmap=ramp, vmin=vmin, vmax=vmax); clean(ax1, bg=False)
    title(ax1, "Centred target kernel" if a.centred else "Target kernel", k=0)
    ax1.set_xticks(ticks); ax1.set_yticks(ticks); ax1.set_xlabel("input j", color=grey, fontsize=11, labelpad=4); ax1.set_ylabel("input i", color=grey, fontsize=11, labelpad=4)
    ax2 = panel(1); im = ax2.imshow(Gshow, cmap=ramp, vmin=vmin, vmax=vmax); clean(ax2, bg=False)
    title(ax2, "Centred encoder Gram" if a.centred else "Encoder Gram", k=1)
    ax2.set_xticks(ticks); ax2.set_yticks(ticks); ax2.set_xlabel("input j", color=grey, fontsize=11, labelpad=4)
    cax = fig.add_axes([X0[1], Y0 - 0.09, S, 0.013]); cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=cticks)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=10.5, colors=grey); cb.ax.set_xticklabels(clabels)
    cax.text(0.5, -1.6, "similarity", transform=cax.transAxes, ha="center", va="top", fontsize=10.5, color=grey)

    def scatter3(k, P, main, labels, elev=None, azim=None):
        elev = a.elev if elev is None else elev; azim = a.azim if azim is None else azim
        ax = fig.add_axes([X0[k] - 0.13, Y0 - 0.13, S + 0.26, SH + 0.2], projection="3d"); ax.patch.set_alpha(0)
        ax.view_init(elev=elev, azim=azim); ax.set_proj_type("ortho"); ax.set_axis_off(); ax.computed_zorder = False
        r = max(np.ptp(P[:, j]) for j in range(3)) * 0.55; z0 = P[:, 2].min() - 0.4 * r
        # the floor and the curve's shadow on it are the only depth cues
        gx, gy = np.meshgrid(np.linspace(-1.25 * r, 1.25 * r, 2), np.linspace(-1.25 * r, 1.25 * r, 2))
        ax.plot_surface(gx, gy, np.full_like(gx, z0), color="#E6E7EC", alpha=1.0, shade=False, zorder=0)
        # the two floor axes, labelled along the plane's front edges
        fr = 1.25 * r; ax.text(0, -fr * 1.22, z0, labels[0], ha="center", va="top", fontsize=11, color=grey, zorder=6)
        ax.text(fr * 1.22, 0, z0, labels[1], ha="center", va="top", fontsize=11, color=grey, zorder=6)
        ax.plot(P[:, 0], P[:, 1], np.full(len(P), z0), color="#C4C7CF", lw=1.3, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], np.full(len(P), z0), color="#C4C7CF", s=10, depthshade=False, zorder=1)
        for i in range(len(P)):
            ax.plot([P[i, 0], P[i, 0]], [P[i, 1], P[i, 1]], [z0, P[i, 2]], color="#D3D6DE", lw=0.6, zorder=2)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#8C93A3", lw=1.2, zorder=3)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=t, cmap=tramp, s=34, edgecolor="white", lw=0.5, depthshade=False, zorder=4)
        ax.scatter(P[[0, -1], 0], P[[0, -1], 1], P[[0, -1], 2], c=[t[0], t[-1]], cmap=tramp, vmin=t[0], vmax=t[-1], s=90, marker="D", edgecolor="white", lw=0.8, depthshade=False, zorder=5)
        for i, lab, dx in ((0, "input 0", -14), (a.T - 1, f"input {a.T - 1}", 14)):
            ax.annotate(lab, ax.transData.transform(proj3d.proj_transform(*P[i], ax.get_proj())[:2]) if False else (0, 0), xycoords="figure pixels", visible=False)
        ax._end_labels = [(0, "input 0", "right"), (a.T - 1, f"input {a.T - 1}", "left")]
        for setter in (ax.set_xlim, ax.set_ylim): setter(-1.45 * r, 1.45 * r)
        zt = P[:, 2].max() + 0.15 * r; ax.set_zlim(z0, zt)
        ax.set_box_aspect((2.9 * r, 2.9 * r, zt - z0), zoom=1.1)      # one scale on all three axes, as in the viewer
        title(ax, main, k=k)
        fig.canvas.draw()
        from mpl_toolkits.mplot3d import proj3d
        for i, lab, ha in ax._end_labels:
            x2, y2, _ = proj3d.proj_transform(P[i, 0], P[i, 1], P[i, 2], ax.get_proj())
            ax.annotate(lab, (x2, y2), xytext=(-12 if ha == "right" else 12, 4), textcoords="offset points", ha=ha, va="center", fontsize=10.5, color=grey, annotation_clip=False, zorder=7)
        return ax
    ax3 = scatter3(2, psi, "Spectral embedding", ("$\\psi_1$", "$\\psi_2$", "$\\psi_3$"))
    ax4 = scatter3(3, Z, "Encoder rows", ("PC 1", "PC 2", "PC 3"))
    fig.savefig(a.out, dpi=170, facecolor="white"); print(a.out, "corr", round(corr, 3))


if __name__ == "__main__":
    main()
