# Results

Every number in the paper is produced by a script in `paper/scripts/` from the files in this folder
(run names are the training run ids; `paper/RUNS.tsv` maps each to its recipe);
`paper/MANIFEST.md` maps each figure and table to its script, data and runs.

## Runs

`<env>-<arm>-<seed>`. Seed 1 of every arm was trained in this checkout; seeds 2 and 3 on a second machine
and read from `$SEED_RESULTS_DIR` by `seed_aggregate.py`.

| env | LeWM | SpecWM temporal | SpecWM quasimetric | stop-gradient variants |
|---|---|---|---|---|
| cube | `cube-lewm-mixlate-{1,2,3}` | `cube-mixlate-lag-{1,2,3}` | `cube-mixlate-t2-{1,2,3}` | `cube-mixlate-lag-det-*`, `cube-mixlate-t2-det-*` |
| scene | `scene-lewm-mixlate-{1,2,3}` | `scene-mixlate-lag-{1,2,3}` | `scene-mixlate-{1,2,3}` | `scene-mixlate-lag-det-*`, `scene-mixlate-det-*` |
| CALVIN | `calvin-lewm-mixlate-{1,2,3}` | `calvin-mixlate-lag-far-{4,5,6}` | `calvin-mixlate-far-{1,2,3}` | `calvin-mixlate-lag-far-det-*`, `calvin-mixlate-far-det-*` |

CALVIN's SpecWM arms use the far-frame form of the objective (eight extra same-episode frames per clip at
300–3,000 raw frames, the kernel a mean of two Laplacians; `calvin_temporal`, `calvin_quasimetric`). Play-data
ablation: `{scene,cube}-play-lag-1`, `{scene,cube}-lewm-play-1`, `calvin-play-lag-far-1`,
`calvin-play-lag-far-sigreg-1`, `calvin-lewm-play-1`. Random-policy floors: `scene-floor-random-v3`,
`cube-floor-random`, `calvin-floor-random`. Centering / tempering ablation of the fully specified state
kernel: `scene-kab-full-state-buttons-pa05-rel-{,nocenter-,notemper-,raw-}s2`.

## Planning (CEM)

`scripts/eval_cem.py <cell> lewm/<run>/<weights>.pt --offsets <k>` writes
`results/<run>-s<step>/O<k>.txt` (100 fixed start–goal tasks; scene and cube at goal offset 25, CALVIN at 50;
solver settings in `config/eval/solver/cem.yaml`). The figure value is the mean over the 20k, 24k and 28k
step snapshots, then over seeds; whiskers are one sample standard deviation over seeds.
`paper/data/finals_seeds.csv` holds every cell (`seed_aggregate.py`), `cem_3env_floor.csv` the floors.

| env | Random | LeWM | SpecWM (temporal) | SpecWM (quasimetric) |
|---|---|---|---|---|
| cube (O25) | 13 | 52 ± 5 | 68 ± 3 | 71 ± 3 |
| scene (O25) | 21 | 49 ± 4 | 77 ± 3 | 74 ± 4 |
| CALVIN (O50) | 17 | 29 ± 3 | 51 ± 1 | 47 ± 2 |

## Probing

`paper/PROBING_PROTOCOL.md` is the protocol: 5,000 frames from 200 held-out validation episodes of the
run's own corpus, the final checkpoint, grouped ridge with cross-validated alpha, R² on held-out episodes,
a random-init encoder as the floor. `paper/scripts/probe_unified.py --protocol heldout` writes
`data/probe_heldout_unified_<env>.csv` (`probe_heldout_paper_calvin.csv` for the CALVIN far-frame arms).

## Figures

`paper/figures/` holds the main-text figures and Table 1 under the paper's numbers (`fig2_pipeline`,
`fig3_cem`, `fig4_probe`, `table1_probes`), `paper/figures/ablations/` the appendix figures (`fig5_spectral_recovery`,
`fig6_stop_gradient`, `fig7_data_ablation`, `fig8_cost_ablation`, `fig9_norm_ablation`, and the centering / tempering ablation `figB_*` / `tableB_*`); each
folder's `vector/` carries the PDF and SVG. `scripts/reproduce/figures.sh` regenerates all of them.
