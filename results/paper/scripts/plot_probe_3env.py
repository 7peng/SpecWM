"""Figure 4: linear probe R^2 of the object (median block on CALVIN) per environment at each run's
FINAL checkpoint, for the random-init encoder, LeWM (SIGReg), the quasimetric and the temporal arm.
Both sources come from scripts/probe_unified.py (one feature path, one grouped-CV ridge):
`--source suite` reads data/probe_suite_unified_<env>.csv (the placement suite, final weights + random
encoder), `--source val` reads data/probe_heldout_unified_<env>.csv (held-out validation episodes of
the run's own corpus, final weights + random encoder).

    python results/paper/scripts/plot_probe_3env.py --source val --out results/paper/figures/fig4_probe.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
ENVS = [("cube", "OGBench Cube"), ("scene", "OGBench Scene"), ("calvin", "CALVIN")]
# groups: single bars (random, LeWM) then one group per kernel with an attached (dark) and a detached
# (light) bar; a kernel without a detached run shows its attached bar alone
SINGLES = [("Random", "#D9D9D9", "Random"), ("LeWM", "#7F7F7F", "LeWM")]
KERNELS = [("Temporal", "#1F3A63"), ("Quasimetric", "#4C72B0")]
NAMES = {"Temporal": "SpecWM (temporal)", "Quasimetric": "SpecWM (quasimetric)"}
CALVIN_RUNS = r"random-init|calvin-lewm-mixlate-\d|calvin-mixlate-far-(det-)?\d|calvin-mixlate-lag-far-(det-)?\d"
HATCH = "////"                   # detached = the kernel's colour with a faint hatch (a tint of the same colour); attached = solid


def tint(hex_colour, f=0.55):
    """hex colour blended toward white by f; the hatch line colour for a bar of that colour."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))
GREY_ENVS = set()


def arm_of(run):
    """(kernel, variant) of a run id: variant is None for singles, attached/detached for kernels."""
    if run == "random-init": return ("Random", None)
    if "lewm" in run: return ("LeWM", None)
    kernel = "Temporal" if "-lag-" in run else "Quasimetric"
    return (kernel, "detached" if "-det-" in run else "attached")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--col", default="id_p50")
    ap.add_argument("--source", choices=["suite", "val"], default="val")
    ap.add_argument("--ablation", action="store_true", help="attached and detached anchor side by side (appendix); default attached only")
    a = ap.parse_args()
    stem = "probe_suite_unified_" if a.source == "suite" else "probe_heldout_unified_"
    files = sorted((HERE / "data").glob(f"{stem}*.csv")) + ([HERE / "data" / "probe_heldout_paper_calvin.csv"] if a.source == "val" else [])
    v = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    v = v[(v.env != "calvin") | v.run.str.match(CALVIN_RUNS)]                      # CALVIN: the far-frame forms only
    if "subset" in v: v = v[v.subset.fillna("all") == "all"]
    if "centred" in v: v = v[v.centred.fillna(False) != True]
    if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
    v = v[~v.run.str.contains(r"-n\d+w-|-qrl-|fs5|grouped|play|kab")].drop_duplicates(subset=["env", "run"], keep="last")   # the fig8 arms only (all seeds)
    d = v[["env", "run", "obj"]].rename(columns={"obj": a.col})
    d["kernel"], d["variant"] = zip(*d.run.map(arm_of))
    d = d.groupby(["env", "kernel", "variant"], dropna=False)[a.col].agg(["mean", "std", "count"]).reset_index()   # mean over seeds, +/- one s.d.
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.07)
    W, G = 0.53, 0.5
    if not a.ablation: W, G = 0.94, 0.48                                      # attached-only: fat bars, labels clear
    plt.rcParams["hatch.linewidth"] = 0.9
    for ax, (env, title) in zip(axes, ENVS):
        g = d[d.env == env]; x = 0.0; ticks, labels = [], []; grey = env in GREY_ENVS
        def bar(xc, row, colour, width, fs=style.VALUE, hatch=None):
            val = float(row["mean"]); c = "#B8B8BE" if grey and colour not in ("#D9D9D9",) else colour
            ax.bar(xc, val, width, color=c, zorder=3, edgecolor=tint(c) if hatch else "none", linewidth=0, hatch=hatch)
            top = max(val, 0)
            if int(row["count"]) > 1:                                                  # +/- one sample standard deviation over seeds
                lo, hi = val - float(row["std"]), val + float(row["std"]); wc = "#8A8A8A" if grey else "#333333"
                ax.plot([xc, xc], [lo, hi], color=wc, lw=1.0, zorder=5); ax.plot([xc - 0.06, xc + 0.06], [lo, lo], color=wc, lw=1.0, zorder=5); ax.plot([xc - 0.06, xc + 0.06], [hi, hi], color=wc, lw=1.0, zorder=5)
                top = max(hi, 0)
            ax.text(xc, top + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=fs, color="#9A9A9A" if grey else "#555")
        for kernel, colour, lab in SINGLES:
            row = g[g.kernel == kernel]
            if len(row): bar(x, row.iloc[0], colour, W - style.PAIR_GAP)
            ticks.append(x); labels.append(lab); x += W + G
        for kernel, colour in KERNELS:
            have = {v: g[(g.kernel == kernel) & (g.variant == v)].iloc[0] for v in ("attached", "detached") if len(g[(g.kernel == kernel) & (g.variant == v)])}
            if not have: continue
            if not a.ablation:
                if "attached" in have: bar(x, have["attached"], colour, W - style.PAIR_GAP)
                ticks.append(x); labels.append(kernel); x += W + G
                continue
            for k, v in enumerate(("attached", "detached")):
                if v in have:
                    bar(x + k * W, have[v], colour, W - style.PAIR_GAP, fs=style.VALUE_SMALL, hatch=HATCH if v == "detached" else None)
                else:                                                    # placeholder: a dashed outline at the attached height, no value
                    c = "#B8B8BE" if grey else colour
                    ref = float(have["attached"]["mean"])
                    ax.bar(x + k * W, ref, W - style.PAIR_GAP, facecolor="none", edgecolor=c, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
                    ax.text(x + k * W, ref + 0.012, "tbd", ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#9A9A9A")
            ticks.append(x + W / 2); labels.append(kernel); x += W * 2 + G
        ax.set_xticks([]); ax.set_xlim(-0.5, x - G - W + 0.5) if a.ablation else ax.set_xlim(-0.72, x - W - G + 0.72)
        ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors="#555")
        ax.axhline(0, color="#C8C8CE", lw=0.8, zorder=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        pass
        ax.set_title(title + (" (preliminary)" if grey else ""), fontsize=style.TITLE, pad=10, color="#8A8A8A" if grey else "#222")
    axes[0].set_ylabel("R²", fontsize=style.LABEL, color="#333")
    import matplotlib.patches as mp
    hs = [mp.Patch(facecolor=c, edgecolor="none") for _, c, _ in SINGLES] + [mp.Patch(facecolor=c, edgecolor="none") for _, c in KERNELS]
    ls = [lab for _, _, lab in SINGLES] + [NAMES[k] for k, _ in KERNELS]
    if a.ablation:
        hs.append(mp.Patch(facecolor="#6E6E74", edgecolor=tint("#6E6E74"), hatch=HATCH, linewidth=0)); ls.append("prediction anchor detached")
    fig.legend(hs, ls, loc="lower center", ncol=len(hs), frameon=False, fontsize=style.LEGEND, bbox_to_anchor=(0.5, 0.01), handlelength=1.4, columnspacing=2.2)
    fig.subplots_adjust(bottom=0.17, top=0.88)
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
