"""Centering / tempering ablation of the fully specified state kernel on OGBench Scene and CALVIN: held-out
linear-probe R^2 per target at the final checkpoint for the raw kernel, with centering, with tempering, and with both
(the paper form). Reads data/probe_heldout_unified_scene.csv and data/probe_heldout_paper_calvin.csv; prints the
markdown tables and writes figures/ablations/tableB_centering_tempering.md.

    python results/paper/scripts/kab_table.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
LABELS = ["raw kernel", "+ centering", "+ tempering", "+ centering + tempering (paper)"]
SCENE_STEM = "scene-kab-full-state-buttons-pa05-rel-"
ENVS = [
    ("OGBench Scene", "probe_heldout_unified_scene.csv", ["cube_xyz", "arm", "drawer", "window", "buttons"],
     [SCENE_STEM + s for s in ("raw-s2", "notemper-s2", "nocenter-s2", "s2")]),
    ("CALVIN", "probe_heldout_paper_calvin.csv", ["red_xyz", "blue_xyz", "pink_xyz", "slider", "drawer", "switch", "arm"],
     [f"calvin-kab-full-state-{s}-1" for s in ("raw", "center", "temper", "both")]),
]


def table(name, csv, targets, runs):
    v = pd.read_csv(HERE / "data" / csv)
    if "subset" in v: v = v[v.subset.fillna("all") == "all"]
    if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
    v = v[v.step == -1].drop_duplicates("run", keep="last").set_index("run")
    cols = [t for t in targets if t in v.columns]
    rows = [f"**{name}**", "", "| arm | " + " | ".join(cols) + " | mean |", "|---|" + "---|" * (len(cols) + 1)]
    for label, run in zip(LABELS, runs):
        if run not in v.index:
            rows.append(f"| {label} | " + " | ".join("tbd" for _ in cols) + " | tbd |"); continue
        vals = [float(v.loc[run, c]) for c in cols]
        rows.append(f"| {label} | " + " | ".join(f"{x:.2f}" for x in vals) + f" | {sum(vals) / len(vals):.2f} |")
    return "\n".join(rows)


def main():
    out = "\n\n".join(table(*e) for e in ENVS); print(out)
    (HERE / "figures" / "ablations" / "tableB_centering_tempering.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
