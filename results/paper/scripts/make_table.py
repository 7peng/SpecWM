"""Table 1: the exact saved OGBench Scene probe output used for the submission.

Reads data/table1_probe_exact.json; writes figures/table1_probes.{md,tex}.

    python results/paper/scripts/make_table.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TARGETS = [("cube", "Cube"), ("arm", "Arm"), ("drawer", "Drawer"), ("window", "Window"), ("buttons", "Buttons")]


def main():
    artifact = json.loads((HERE / "data" / "table1_probe_exact.json").read_text())
    rows = [(row["objective"], row["kernel"], [float(row["r2"][key]) for key, _ in TARGETS])
            for row in artifact["rows"]]
    best = [max(row[2][j] for row in rows if row[0] != "Reference") for j in range(len(TARGETS))]
    out = HERE / "figures"
    md = ["| objective | kernel | " + " | ".join(n for _, n in TARGETS) + " |", "|---|---|" + "---|" * len(TARGETS)]
    for gname, label, v in rows:
        md.append(f"| {gname} | {label} | " + " | ".join(("**%.2f**" % x) if best[j] - x <= 0.0100001 else "%.2f" % x for j, x in enumerate(v)) + " |")
    (out / "table1_probes.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{table}[t]", "\\centering", "\\caption{Linear probe $R^2$ on episode-grouped held-out readout splits, OGBench Scene. Bold marks the best result per column and results within 0.01 of it.}",
           "\\label{tab:kernel_probes}", "\\begin{tabular}{llccccc}", "\\toprule",
           "Objective & Kernel & " + " & ".join(n for _, n in TARGETS) + " \\\\", "\\midrule"]
    groups = []
    for gname, label, values in rows:
        if not groups or groups[-1][0] != gname:
            groups.append((gname, []))
        groups[-1][1].append((label, values))
    for gi, (gname, arms) in enumerate(groups):
        if gi:
            tex.append("\\midrule")
        tex.append(f"\\multirow{{{len(arms)}}}{{*}}{{{gname}}}")
        for label, vals in arms:
            cells = " & ".join(("\\textbf{%.2f}" % x) if best[j] - x <= 0.0100001 else "%.2f" % x for j, x in enumerate(vals))
            tex.append(f" & {label} & {cells} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "table1_probes.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
