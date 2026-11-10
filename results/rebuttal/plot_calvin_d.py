"""Rebuttal figure: CEM success on CALVIN (100 tasks, goal offset 50) in the training scenes A-C (the Fig. 3 numbers)
and in the held-out scene D (task_ABC_D validation split, never seen in training). Each bar is the mean over three
seeds of the 20k/24k/28k snapshot mean, whisker one standard deviation.
    python results/rebuttal/plot_calvin_d.py OUT.png"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "paper" / "scripts"))
import style as S  # noqa: E402

f = pd.read_csv(ROOT / "results" / "paper" / "data" / "finals_seeds.csv"); f = f[f.env == "calvin"]
GROUPS = [("LeWM", "sigreg", S.LEWM), ("SpecWM\n(quasimetric)", "attached", S.QUASIMETRIC), ("SpecWM\n(temporal)", "temporal", S.TEMPORAL)]


def d_mean(run):
    v = []
    for s in (20000, 24000, 28000):
        m = re.search(r"success_rate'?:\s*([0-9.]+)", (ROOT / "results" / f"{run}-D-s{s}" / "O50.txt").read_text())
        v.append(float(m.group(1)))
    return np.mean(v)


S.apply()
PANEL_W = 0.935 / (3 + 2 * 0.07)
pw = PANEL_W * S.FULL; fig = plt.figure(figsize=(pw + 1.0, 4.9))
ax = fig.add_axes([0.8 / (pw + 1.0), 0.25, pw / (pw + 1.0), 0.65])
W, G = 0.53, 0.5
x, ticks = 0.0, []
for label, arm, c in GROUPS:
    rows = f[f.arm == arm]
    seen, unseen = rows.mean_20k_30k.to_numpy(), np.array([d_mean(r) for r in rows.run])
    for k, (vals, hatch) in enumerate(((seen, None), (unseen, S.HATCH))):
        xc = x + k * W; val, sd = vals.mean(), vals.std(ddof=1)
        ax.bar(xc, val, W - S.PAIR_GAP, color=c, edgecolor=S.tint(c) if hatch else "none", linewidth=0, hatch=hatch, zorder=3)
        S.whisker(ax, xc, val - sd, val + sd)
        S.value(ax, xc, val + sd + 1.2, f"{val:.0f}", fs=S.VALUE_SMALL)
    print(label.replace("\n", " "), "A-C", seen.round(1), "D", unseen.round(1))
    ticks.append(x + W / 2); x += 2 * W + G
ax.set_xticks(ticks, [g[0] for g in GROUPS], fontsize=S.VALUE_SMALL); ax.set_xlim(-0.5, x - G - W + 0.5)
S.clean(ax, yticks=range(0, 81, 20), ylim=(0, 80)); S.title(ax, "CALVIN"); S.ylabel(ax, "Success (%)")
hs = [mp.Patch(facecolor="#6E6E74", edgecolor="none"), mp.Patch(facecolor="#6E6E74", edgecolor=S.tint("#6E6E74"), hatch=S.HATCH, linewidth=0)]
S.legend(fig, hs, ["Scenes A–C (training)", "Scene D (held out)"], ncol=1, columnspacing=1.2)
S.save(fig, sys.argv[1]); fig.savefig(Path(sys.argv[1]).with_suffix(".pdf"), facecolor="white")
