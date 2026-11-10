"""Rebuttal figure: CEM success on OGBench Scene (100 tasks, goal offset 25) for the temporal kernel with the
Equation 6 row rescaling (the form Theorem 3 assumes) against the reported objective. Reported bars are the
Fig. 3 numbers; every bar is the mean over three seeds of the 20k/24k/28k snapshot mean, whisker one standard
deviation.
    python results/rebuttal/plot_rescale_cem.py OUT.png"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "paper" / "scripts"))
import style as S  # noqa: E402

D = ROOT / "results" / "paper" / "data"
f = pd.read_csv(D / "finals_seeds.csv"); f = f[f.env == "scene"]
floor = float(pd.read_csv(D / "cem_3env_floor.csv").set_index("env").loc["scene", "random_policy"])


def cells(run):
    out = []
    for s in (20000, 24000, 28000):
        m = re.search(r"success_rate'?:\s*([0-9.]+)", (ROOT / "results" / f"{run}-s{s}" / "O25.txt").read_text())
        out.append(float(m.group(1)))
    return np.array(out)


resc = np.array([cells(r).mean() for r in ("scene-temporal-rescaled-2", "scene-temporal-rescaled-s2", "scene-temporal-rescaled-s3")])
lewm, temp = f[f.arm == "sigreg"].mean_20k_30k, f[f.arm == "temporal"].mean_20k_30k
BARS = [("Random", floor, None, None, S.RANDOM, None),
        ("LeWM", lewm.mean(), lewm.std(ddof=1), None, S.LEWM, None),
        ("SpecWM\n(reported)", temp.mean(), temp.std(ddof=1), None, S.TEMPORAL, None),
        ("SpecWM\n(rescaled)", resc.mean(), resc.std(ddof=1), None, S.TEMPORAL, "////")]

S.apply(); plt.rcParams["hatch.linewidth"] = 0.9
PANEL_W = 0.935 / (3 + 2 * 0.07)                       # one Fig. 3 panel, as the Fig. 8 / 9 ablations
pw = PANEL_W * S.FULL; fig = plt.figure(figsize=(pw + 1.0, 4.6))
ax = fig.add_axes([0.8 / (pw + 1.0), 0.19, pw / (pw + 1.0), 0.71])
W, G = 0.94, 0.48                                      # Fig. 3 bar width and gap
xs = [i * (W + G) for i in range(len(BARS))]
for x, (lab, val, sd, pts, c, hatch) in zip(xs, BARS):
    ax.bar(x, val, W - S.PAIR_GAP, color=c, edgecolor=S.tint(c) if hatch else "none", linewidth=0, hatch=hatch, zorder=3)
    top = val
    if sd is not None:
        S.whisker(ax, x, val - sd, val + sd); top = val + sd
    if pts is not None:
        ax.scatter(x + np.array([-0.22, 0, 0.22]), pts, s=12, color="#333333", zorder=6); top = max(top, pts.max())
    S.value(ax, x, top + (2.6 if pts is not None else 1.2), f"{val:.0f}")
ax.set_xticks(xs, [b[0] for b in BARS], fontsize=S.VALUE_SMALL); ax.set_xlim(-0.72, xs[-1] + 0.72)
S.clean(ax, yticks=range(0, 101, 20), ylim=(0, 100)); S.title(ax, "OGBench Scene"); S.ylabel(ax, "Success (%)")
S.save(fig, sys.argv[1]); fig.savefig(Path(sys.argv[1]).with_suffix(".pdf"), facecolor="white")
print("reported", temp.round(1).tolist(), "rescaled", resc.round(1).tolist())
