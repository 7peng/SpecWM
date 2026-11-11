"""Rebuttal figure: CEM success on OGBench Scene and Cube (goal offset 25) and CALVIN (offset 50), 100 tasks each, for the
contrastive baselines against LeWM and SpecWM (temporal). Every run is the mean of its 20k/24k/28k cells; a bar is the
mean over seeds, whisker one standard deviation, dots the individual seeds. Data: data/contrastive_cem.csv (baselines),
data/finals_seeds.csv (LeWM, SpecWM), data/cem_3env_floor.csv (Random).
    python results/rebuttal/plot_contrastive_cem.py OUT.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "paper" / "scripts"))
import style as S  # noqa: E402

D = ROOT / "results" / "paper" / "data"
F = pd.read_csv(D / "finals_seeds.csv")
C = pd.read_csv(D / "contrastive_cem.csv")
FLOOR = pd.read_csv(D / "cem_3env_floor.csv").set_index("env")["random_policy"]
ENVS = [("scene", "OGBench Scene"), ("cube", "OGBench Cube"), ("calvin", "CALVIN")]


def seeds(env, arm):
    return C[(C.env == env) & (C.arm == arm)].sort_values("seed").mean_20k_28k.to_numpy()


def arms(env):
    f = F[F.env == env]
    return [("Random", np.array([float(FLOOR[env])]), S.RANDOM, None),
            ("LeWM", f[f.arm == "sigreg"].mean_20k_30k.to_numpy(), S.LEWM, None),
            ("CRL", seeds(env, "crl"), S.tint(S.QUASIMETRIC, 0.35), "////"),
            ("Temporal\nInfoNCE", seeds(env, "infonce"), S.QUASIMETRIC, "////"),
            ("SpecWM\n(temporal)", f[f.arm == "temporal"].mean_20k_30k.to_numpy(), S.TEMPORAL, None)]


S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 4.6), sharey=True)
fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.21, wspace=0.05)
W, G = 0.94, 0.48
for ax, (env, title) in zip(axes, ENVS):
    A = arms(env)
    xs = [i * (W + G) for i in range(len(A))]
    labels = []
    for x, (lab, v, c, hatch) in zip(xs, A):
        val = float(v.mean())
        ax.bar(x, val, W - S.PAIR_GAP, color=c, edgecolor=S.tint(c) if hatch else "none", linewidth=0, hatch=hatch, zorder=3)
        top = val
        if len(v) > 1:
            sd = float(v.std(ddof=1)); S.whisker(ax, x, val - sd, val + sd); top = val + sd
        if lab != "Random":
            ax.scatter(x + np.linspace(-0.22, 0.22, len(v)) if len(v) > 1 else [x], v, s=12, color="#333333", zorder=6)
            top = max(top, float(v.max()))
        S.value(ax, x, top + 1.6, f"{val:.0f}")
        labels.append(lab + (f"\n({len(v)} seed{'s' if len(v) > 1 else ''})" if lab != "Random" and len(v) < 3 else ""))
        print(env, lab.replace("\n", " "), np.round(v, 1).tolist(), round(val, 1))
    ax.set_xticks(xs, labels, fontsize=S.VALUE_SMALL); ax.set_xlim(-0.72, xs[-1] + 0.72)
    S.clean(ax, yticks=range(0, 101, 20), ylim=(0, 100)); S.title(ax, title)
S.ylabel(axes[0], "Success (%)")
S.save(fig, sys.argv[1]); fig.savefig(Path(sys.argv[1]).with_suffix(".pdf"), facecolor="white")
