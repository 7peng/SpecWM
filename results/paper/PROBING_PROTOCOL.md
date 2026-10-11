# The paper-ready probing protocols

## Table 1: saved historical probe

Table 1 is generated from `data/table1_probe_exact.json`, the full-precision saved probe output used
for the submitted table. That single artifact also records the run and checkpoint-relative filename
for every row, the exact sampling and grouped-ridge protocol, hashes, and the reproduction command.
`scripts/make_table.py` reads it directly; it does not average training-time logs. No checkpoint
binary is distributed.

This historical Table 1 protocol sampled 5,000 rows from the complete Scene late-onset corpus with
seed 0. Episodes are held out from the probe readout in five grouped outer splits; the sampled pool
is **not** restricted to episodes held out from encoder training. Features and targets are centered
and scaled from each training fold only, with no per-episode centering. The JSON is authoritative for
the remaining details. This differs from the validation-episode protocol below. The paper's "held-out
OGBench Scene episodes" for Table 1 means held out from the probe readout, not from encoder training.

## Figures: validation-episode probe

Every probe number in the paper's probe figures (Fig. 4 object probe, Fig. 6 stop-gradient, Fig. 7 play data) comes from one
instrument, `results/paper/scripts/probe_unified.py --protocol heldout`. Nothing else is a paper
number: not the training-time probe the callback logs to wandb (`probe/emb/*`), not the placement suite,
not the per-run report tables built from training logs. Those differ from this protocol in ways that
move the numbers (see "What it is not"), and they are diagnostics.

## Definition

1. **Frames.** 5,000 rows drawn (seed 0, without replacement) from the episodes in the run's own
   corpus that the encoder NEVER trained on: the validation split of `train_common` (10% of episodes
   by a `torch.randperm` seeded with `split_seed: 3072`, identical for every run and seed). On the
   late-onset corpora that is 200 episodes per environment. `probe_val_episodes.val_episodes` re-derives
   the set; `probe_unified.py` reads pixels and raw state from the mixture file itself.
2. **Encoder.** The final checkpoint (`weights_epoch_<max>.pt`, the same file Fig. 3 plans with), loaded
   by `diagnostics.probe_encoders.build_encoder`: ViT CLS token through the projector, i.e. the planner's
   embedding, fp32, ImageNet normalisation, 224 px (CALVIN's 200 px frames resized bilinearly as in
   training). No pooling of patch tokens, no whitening, no augmentation.
3. **Targets.** Raw state columns, not centred, not per-episode adjusted: cube xyz and arm joints on
   cube; cube xyz, arm, drawer, window, buttons on scene; the three block xyz (roster-canonical: red,
   blue, pink), slider, drawer, switch, arm on CALVIN. "Object" is cube xyz on cube and scene and the
   median over the three blocks on CALVIN.
4. **Fit.** `diagnostics.probe_heads.grouped_fit`: ridge with intercept, features and targets
   standardised on the fit rows, alpha chosen by inner episode-grouped CV, five outer splits that hold
   whole episodes out of the fit (seeds 0-4), R² on the held-out episodes averaged over the target's
   coordinates and over the five splits.
5. **Floor.** A random-init ViT of the same width (192) run through steps 1-4 in the same pass
   (`--random`).
6. **Seeds.** Where several training seeds exist, the figure bar is the mean over seeds of the per-seed
   value and the whisker one sample standard deviation; single-seed bars carry no whisker.

Outputs: `data/probe_heldout_unified_<env>.csv` (one row per run, all targets, plus embedding
effective rank and mean cosine). Fig. 4 (`plot_probe_3env.py`) and Fig. 6 (`plot_anchor_tradeoff.py`) read those
files and nothing else.

## What it is not

- **Training-time probe** (`src/diagnostics/probe_callback.py`, wandb `probe/emb/*`, the per-run
  battery reports): 5,000 rows from the WHOLE corpus (about 90% training episodes), live weights every
  500 steps, the same grouped ridge. The readout never sees its test episodes but the encoder has, so an
  encoder carrying episode-specific content scores through it. Reads higher than the paper protocol for
  every arm, most for LeWM on scene (0.17-0.18 there against 0.07-0.12 here).
- **Placement suite** (`probe_pack_*`, `--protocol suite`): rendered frames with the arm frozen and
  objects re-placed independently, camera-frame labels, random rows within a cell. A controlled
  instrument with a high random floor; kept for the CALVIN single-block vs three-block binding test.
- **Any probe on training frames, any MLP head, any centred or excess-over-arm variant**: diagnostics
  in the ledger, never a figure number.

## Reproduce

```bash
python results/paper/scripts/probe_unified.py --protocol heldout --env scene --final --random \
    --out results/paper/data/probe_heldout_unified_scene.csv          # likewise cube, calvin
python results/paper/scripts/plot_probe_3env.py --out results/paper/figures/fig4_probe.png
python results/paper/scripts/plot_anchor_tradeoff.py --out results/paper/figures/ablations/fig6_stop_gradient.png
```

`--runs <ids>` restricts to given runs; a replicate trained elsewhere only needs its checkpoint
directory under `$STABLEWM_HOME/checkpoints/lewm/<run>/` (a symlink is fine).
