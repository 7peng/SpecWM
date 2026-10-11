# SpecWM: spectral world models from pixels

A JEPA world model whose encoder is trained by a spectral loss: the Gram matrix of the embeddings
of a clip is regressed onto a target kernel over the clip's frames (a temporal kernel, or the
kernel of a learned quasimetric), and the predictor is anchored to the embeddings of the frames it
predicts. LeWM (next-embedding MSE + SIGReg) is the reference. Planning is sampling-based MPC
(CEM) on the goal-offset task sets of three environments: OGBench scene, OGBench cube and CALVIN.

Everything runs from this directory. `paper/MANIFEST.md` maps every figure, table and number of
the paper to the script that produces it, the data it reads and the runs behind it; `paper/RUNS.tsv`
lists every training run.

## Installation

```bash
uv sync --frozen
source .venv/bin/activate
cp .env.example .env            # then set the paths below
```

`pyproject.toml` pins `torch==2.12.0+cu126`; a driver older than CUDA 13 needs the matching wheel
index (`uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --reinstall`).
The CALVIN simulator needs its own environment (`CALVIN_VENV`, a Python with `calvin_env` and
pybullet); it is used only by the CALVIN planning evaluation, through a worker process.

Paths are read from `.env` (`utils.load_env`; exported variables win):

| variable | what |
|---|---|
| `STABLEWM_HOME` | datasets under `datasets/<name>.h5`, checkpoints under `checkpoints/lewm/<run_id>/` |
| `SCENE_STAGE_DIR`, `CUBE_STAGE_DIR`, `CALVIN_STAGE_DIR` | node-local copies of the training corpora (read at random offsets) |
| `CALVIN_ROOT`, `CALVIN_ENV_SRC`, `CALVIN_VENV` | the CALVIN dataset root, the `calvin_env` checkout and its Python |
| `OGBENCH_PACK_DIR`, `CALVIN_PROBE_PACK_SRC`, `CALVIN_PROBE_PACK_DIR` | probe packs (training-time diagnostics) |
| `SEED_RESULTS_DIR` | optional: a second results tree holding replicate seeds (`seed_aggregate.py`) |

## Data

| environment | corpus | built by |
|---|---|---|
| scene | `scene_play.h5` (1,000 episodes rendered from OGBench `scene-play-v0`), `scene_play_mix_lateonset.h5` (play + late-onset noised re-executions), `scene_play_mix_lateonset_depth_raw.h5` (the mixture plus full-resolution depth, Table 1 depth runs), `scene_play_depth.h5` (play plus gated, z-scored depth, Fig. 5 depth runs) | `data/scene/build_scene_noised.py`, `data/scene/build_scene_lateonset.py`, `data/scene/render_scene_depth.py` + `data/scene/build_scene_raw_depth_h5.py` / `data/scene/build_scene_depth_h5.py`; checksums in `data/scene/data_manifest.txt`, guarantees in `data/scene/REPRODUCIBILITY.md` |
| cube | `cube_play.h5`, `cube_play_mix_lateonset.h5` | `data/cube/build_cube_noised.py`, `data/cube/build_cube_lateonset.py` |
| CALVIN | `calvin_abc_d.h5` (task_ABC_D packed, blocks in canonical order), `calvin_abc_d_mix_lateonset.h5` | `data/calvin/download_calvin.sh`, `build_calvin_h5.py`, `build_calvin_noised.py`, `build_calvin_lateonset.py` (`data/calvin/README.md`) |
| task sets | `<env>_offset_tasks_O<k>.h5`: 100 start/goal pairs per offset | `data/<env>/make_<env>_offset_tasks.py` |
| probe packs | i.i.d. object placements in a fixed scene | `data/ogbench/make_ogbench_suite.py`, `data/calvin/make_calvin_suite.py`, `data/probe_suite/build_probe_pack.py` |

`scripts/stage_data.sh` and `scripts/stage_calvin.sh` copy a corpus to the node-local staging
directory the recipes read from.

## Training

One recipe per arm in `config/train/`, composed on the environment's shared file in
`config/train/env/`; two entry points.

| arm | scene | cube | CALVIN |
|---|---|---|---|
| LeWM | `scene_lewm` | `cube_lewm` | `calvin_lewm` |
| SpecWM, temporal kernel | `scene_temporal` | `cube_temporal` | `calvin_temporal` |
| SpecWM, quasimetric kernel | `scene_quasimetric` | `cube_quasimetric` | `calvin_quasimetric` |
| quasimetric teacher | `scene_teacher` | `cube_teacher` | `calvin_teacher` |
| play corpus only | `scene_{lewm,temporal}_play` | `cube_{lewm,temporal}_play` | `calvin_{lewm,temporal}_play`, `calvin_temporal_play_sigreg` |
| Table 1 kernel ablation (scene, late-onset mixture) | `scene_{temporal,quasimetric}`, `scene_table1_clip_{state,proprio,depth}`, `scene_state_ablation_centered_tempered`, `scene_table1_full_{proprio,depth}` | | |
| spectral recovery, Fig. 5 (scene, play corpus) | `scene_clip_{temporal,quasimetric,state,proprio,depth}`, `scene_full_{state,proprio,depth}` | | |
| centering / tempering ablation | `scene_state_ablation_{raw,centered,tempered,centered_tempered}` | | `calvin_state_ablation_{...}` |

```bash
MUJOCO_GL=egl python src/train_lewm.py --config-name=scene_lewm run_id=scene-lewm-1
MUJOCO_GL=egl python src/train_spec.py --config-name=scene_temporal run_id=scene-temporal-1
MUJOCO_GL=egl python src/train_spec.py --config-name=scene_temporal run_id=scene-temporal-sg-1 loss.spectral.anchor.stop_gradient=true
MUJOCO_GL=egl python src/train_spec.py --config-name=scene_temporal run_id=scene-temporal-2 seed=2
```

The stop-gradient variant of any SpecWM arm is the override above; replicate seeds pass `seed=`
(the validation-episode split is fixed by `split_seed`). A quasimetric arm reads its kernel from a
frozen teacher named in the recipe (`loss.spectral.teachers`), so the teacher recipe of that
environment runs first. `scripts/reproduce/train.sh <run_id> [gpu]` launches any row of
`paper/RUNS.tsv`. Checkpoints land under `$STABLEWM_HOME/checkpoints/lewm/<run_id>/` (weights per
epoch, step snapshots, the resolved `config.json`, `train_meta.json`); a run warm-resumes from its
latest snapshot.

The objective is `loss.spectral` in the recipe: `scope` (`clip`: one kernel per clip on within-clip
pairs; `batch`: the fully specified kernel over the whole batch, tempered), `factors` (the kernel as a
sum of factor kernels: `temporal`, `quasimetric`, `laplace` / `cos` / `indicator` on a raw state
column), `anchor` (the prediction anchor and its stop-gradient switch), `temper` (spectrum power,
retained modes, centering) and `queries` (`rows`: the encoder's rows; `predictions`: the one-step
predictions against the rows).

## Planning evaluation

```bash
scripts/eval_cem.py <cell> lewm/<run_id>/weights_epoch_N.pt --offsets 25      # scene, cube
scripts/eval_cem.py <cell> lewm/<run_id>/weights_epoch_N.pt --offsets 50      # CALVIN
scripts/eval_cem.py <env>-floor random --offsets 25
scripts/reproduce/score.sh <run_id> [gpu]                                    # every 4k snapshot + the final
```

Protocol `ICLR-scene-cem-v3` (`config/eval/<env>.yaml`, `config/eval/solver/cem.yaml`): 100 tasks per
offset, plan horizon `ceil(offset / frameskip)` blocks with receding horizon 3, env budget
`max(2 x offset, 60)` steps, CEM with 300 samples, 30 iterations, top 30, seed 42; the predictor is
rolled one block at a time from the observed frame history and the endpoint scored by
`1 - cos` to the encoded goal. Action statistics come from the checkpoint's training corpus. The
figure value of a run is the mean over its 20k, 24k and 28k snapshots, then over seeds.

## Probing

`results/paper/PROBING_PROTOCOL.md`: 5,000 frames from the run's own held-out validation
episodes, the final checkpoint, grouped ridge with cross-validated alpha, R² on held-out episodes,
a random-init encoder as the floor.

```bash
scripts/reproduce/probe.sh <run_id> [gpu]
python results/paper/scripts/probe_unified.py --protocol heldout --env scene --final --random --out results/paper/data/probe_heldout_unified_scene.csv
```

## Figures and tables

`scripts/reproduce/figures.sh` redraws every figure and table from the tracked CSVs into
`results/paper/figures/` (main text) and `figures/ablations/` (appendix), named by the paper's numbers.

| | Random | LeWM | SpecWM (temporal) | SpecWM (quasimetric) |
|---|---|---|---|---|
| cube, success at offset 25 | 13 | 52 ± 5 | 68 ± 3 | 71 ± 3 |
| scene, success at offset 25 | 21 | 49 ± 4 | 77 ± 3 | 74 ± 4 |
| CALVIN, success at offset 50 | 17 | 29 ± 3 | 51 ± 1 | 47 ± 2 |

## Layout

```
src/        train_spec.py (SpecWM and the teachers), train_lewm.py, train_common.py (data, optimiser,
            callbacks, resume), jepa.py, networks.py, losses.py (SIGReg, QRLHead), optim.py (MuonAdamW),
            throughput.py, utils.py, eval.py (the planner), parallel_env_pool.py, calvin_sim_env.py,
            quasimetric/ (the frozen teacher), diagnostics/ (training-time probes)
config/     train/ (recipes + env/), eval/ (scene, cube, calvin, solver/cem), teachers/ (teacher model specs)
data/       scene/, cube/, calvin/, ogbench/, probe_suite/
scripts/    eval_cem.py, write_config_json.py, export_step_weights.py, stage_*.sh, reproduce/
tests/      plain scripts (tests/run_all.sh <report>); regression_baseline.py fingerprints a recipe
results/    <cell>/ planning records, paper/ (data, scripts, figures/ with ablations/)
paper/      MANIFEST.md, RUNS.tsv
```

## Acknowledgements

The LeWM reference and the training and evaluation scaffolding build on
[LeWorldModel](https://le-wm.github.io/) ([paper](https://arxiv.org/pdf/2603.19312v1)) and its
libraries [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) and
[stable-pretraining](https://github.com/galilai-group/stable-pretraining).

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
