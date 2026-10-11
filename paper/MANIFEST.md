# Paper manifest

Every figure, table and in-text number of the paper, with the script that draws it, the data it reads and the
runs behind it. Anything in the repository that no row here reaches is not needed for the paper.

Run names: `<env>-<arm>-<seed>`; every seed's cells are in `results/` (`seed_aggregate.py` also accepts a second
tree through `SEED_RESULTS_DIR`). CEM cells live in `results/<run>-s<step>/O<offset>.txt` (scene and cube at O25, CALVIN at O50,
100 tasks). Held-out probes live in `results/paper/data/probe_heldout_*.csv`.

| item | script | output | data | runs |
|---|---|---|---|---|
| Fig. 2 (spectral alignment on one trajectory) | `scripts/plot_fig2_pipeline.py` | `figures/fig2_pipeline.png` | one scene trajectory of `scene_play.h5`, a checkpoint | `scene-mixlate-lag-1` |
| Fig. 3 (CEM success, 3 envs) | `scripts/plot_cem_3env.py` | `figures/fig3_cem.png` | `data/finals_seeds.csv` (`scripts/seed_aggregate.py`), `data/cem_3env_floor.csv` | LeWM, temporal, quasimetric x 3 seeds x 3 envs; random floors |
| Fig. 4 (object probe, 3 envs) | `scripts/plot_probe_3env.py` | `figures/fig4_probe.png` | `data/probe_heldout_unified_{cube,scene}.csv`, `data/probe_heldout_paper_calvin.csv` | the same runs, final checkpoint; `random-init` |
| Table 1 (kernel ablation, scene) | `scripts/make_table.py` | `figures/table1_probes.{md,tex}` | `data/table1_probe_exact.json` (scores, protocol, run/checkpoint provenance, reproduction command); probe rows sampled from the whole corpus, episodes held out from the readout only (`PROBING_PROTOCOL.md`) | run IDs, checkpoints and hashes in the JSON (checkpoint binaries not distributed); recipes `scene_{temporal,quasimetric,lewm}`, `scene_table1_clip_{state,proprio,depth}`, `scene_state_ablation_centered_tempered`, `scene_table1_full_{proprio,depth}` (`RUNS.tsv` role `table1`). Depth runs read `scene_play_mix_lateonset_depth_raw.h5` (`data/scene/render_scene_depth.py`, `data/scene/build_scene_raw_depth_h5.py`) |
| Fig. 5 (spectral recovery, App. B.1) | `scripts/plot_spectral_recovery.py` | `figures/ablations/fig5_spectral_recovery.png` | `battery/conversion.csv` (`scripts/spectral_battery.py`); depth runs read `scene_play_depth.h5` (`data/scene/render_scene_depth.py`, `data/scene/build_scene_depth_h5.py`) | fully specified and underspecified state / proprio / depth; untrained |
| Fig. 6 (stop-gradient, App. B.2) | `scripts/plot_anchor_tradeoff.py` | `figures/ablations/fig6_stop_gradient.png` | as Fig. 3 + Fig. 4 | + the `*-det-*` arms |
| Fig. 7 (failure-data ablation, App. B.3) | `scripts/plot_play_3env.py --metric both` | `figures/ablations/fig7_data_ablation.png` | cells + probe files above | `*-play-lag-1`, `*-lewm-play-1`, `calvin-play-lag-far-1`, `calvin-play-lag-far-sigreg-1` |
| Fig. 8 (CEM cost, App. B.4) | `scripts/plot_scene_ablations.py --which cost` | `figures/ablations/fig8_cost_ablation.png` | `data/scene_ablations_displayed.json` (three seeds x final three checkpoints) | Scene, LeWM vs SpecWM (temporal) |
| Fig. 9 (normalisation layer, App. B.5) | `scripts/plot_scene_ablations.py --which normalization` | `figures/ablations/fig9_norm_ablation.png` | `data/scene_ablations_displayed.json` (one seed) | Scene, LeWM vs SpecWM (temporal) |
| centering / tempering ablation (not in the current draft) | `scripts/kab_table.py`, `scripts/plot_kab.py`, `scripts/plot_kab_curves.py`, `scripts/kab_conversion.py plot-fade`, `scripts/kab_conversion.py plot` | `figures/ablations/tableB_centering_tempering.md`, `figures/ablations/figB_centering_tempering{,_curves,_conversion,_heatmap}.png` | `data/probe_heldout_unified_scene.csv`, `data/probe_heldout_paper_calvin.csv`, `battery/kab_conversion{,_calvin}.csv` (`scripts/kab_conversion.py compute --env scene|calvin`) | `scene-kab-full-state-buttons-pa05-rel-{,nocenter-,notemper-,raw-}s2`; CALVIN `calvin-kab-full-state-{both,raw,center,temper}-1` (running) |
| Table 2 (CEM configuration) | none (text) | | `config/eval/{scene,cube,calvin}.yaml`, `config/eval/solver/cem.yaml` | |
| Tables 3-5 (architecture, actions, training) | none (text) | | the recipes in `tests/paper_recipes.txt` | |
| "improves upon LeWM by an average of 21 percentage points" | `scripts/seed_aggregate.py` | | `data/finals_seeds.csv` | Fig. 3 runs |

Paper recipes (training): `tests/paper_recipes.txt` (one leaf per arm under `config/train/`, the stop-gradient
variants as the `loss.spectral.anchor.stop_gradient=true` override). Teachers: `{cube,scene,calvin}_teacher` (model
specs `config/teachers/scene_linear_fs5{,_calvin}.yaml`). Data builders: `data/{scene,cube,calvin}/` late-onset
mixtures, offset task sets, probe packs.
