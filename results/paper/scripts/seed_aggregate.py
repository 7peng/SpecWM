"""Per-seed CEM cells for the paper arms: seed 1 from this checkout's results/, seeds 2 and 3 from the collaborator's
replicate results ($SEED_RESULTS_DIR, or this checkout when a seed was trained here), one row per (env, arm, seed) with the
final-checkpoint success and the mean over the 20k/24k/28k snapshots. Writes data/finals_seeds.csv and prints
the per-arm mean and seed range.

    python results/paper/scripts/seed_aggregate.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import cem_cell  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OURS = HERE.parent          # results/
THEIRS = Path(os.environ.get("SEED_RESULTS_DIR") or OURS)
ARMS = {  # env -> arm -> (run stem, offset); seed s appends -s
    "cube": {"sigreg": ("cube-lewm-mixlate", 25), "attached": ("cube-mixlate-t2", 25), "detached": ("cube-mixlate-t2-det", 25), "temporal": ("cube-mixlate-lag", 25), "temporal-detached": ("cube-mixlate-lag-det", 25)},
    "scene": {"sigreg": ("scene-lewm-mixlate", 25), "attached": ("scene-mixlate", 25), "detached": ("scene-mixlate-det", 25), "temporal": ("scene-mixlate-lag", 25), "temporal-detached": ("scene-mixlate-lag-det", 25)},
    "calvin": {"sigreg": ("calvin-lewm-mixlate", 50),
               "attached": ({1: "calvin-mixlate-far-1", 2: "calvin-mixlate-far-2", 3: "calvin-mixlate-far-3"}, 50),
               "detached": ({1: "calvin-mixlate-far-det-1", 2: "calvin-mixlate-far-det-2", 3: "calvin-mixlate-far-det-3"}, 50),
               "temporal": ({1: "calvin-mixlate-lag-far-4", 2: "calvin-mixlate-lag-far-5", 3: "calvin-mixlate-lag-far-6"}, 50),
               "temporal-detached": ({1: "calvin-mixlate-lag-far-det-1", 2: "calvin-mixlate-lag-far-det-2", 3: "calvin-mixlate-lag-far-det-3"}, 50)},
}


def cell(root, run, tag, off):
    return cem_cell(run, tag, off, roots=(root, OURS, THEIRS))


def main():
    rows = []
    for env, arms in ARMS.items():
        for arm, (stem, off) in arms.items():
            for seed, root in ((1, OURS), (2, THEIRS), (3, THEIRS)):
                run = stem[seed] if isinstance(stem, dict) else f"{stem}-{seed}"     # explicit per-seed run names, or stem-seed
                root = next((r for r in (root, OURS, THEIRS) if any(r.glob(f"{run}-*"))), root)   # whichever tree holds the run's cells
                cells = {s: cell(root, run, f"s{s}", off) for s in range(4000, 33000, 4000)}
                fin = cell(root, run, "final", off)
                if np.isnan(fin) and all(np.isnan(v) for v in cells.values()):
                    continue
                m20 = [cells[s] for s in (20000, 24000, 28000) if not np.isnan(cells[s])]
                rows.append(dict(env=env, arm=arm, seed=seed, run=run, offset=off, final=fin, mean_20k_30k=np.mean(m20) if m20 else np.nan, n_20k_30k=len(m20), **{f"s{s}": v for s, v in cells.items()}))
    d = pd.DataFrame(rows); d.to_csv(HERE / "data" / "finals_seeds.csv", index=False)
    g = d.groupby(["env", "arm"])
    summ = g.agg(n_seeds=("seed", "count"), final_mean=("final", "mean"), final_min=("final", "min"), final_max=("final", "max"), m20_mean=("mean_20k_30k", "mean"), m20_min=("mean_20k_30k", "min"), m20_max=("mean_20k_30k", "max")).round(1)
    print(summ.to_string()); print(f"[out] {HERE / 'data' / 'finals_seeds.csv'}")


if __name__ == "__main__":
    main()
