# CALVIN contrastive baselines

| Objective | O50 success (%, mean ± sample SD across three seeds) |
| --- | ---: |
| Temporal InfoNCE with within-clip negatives | 40.8 ± 2.7 |
| OGBench CRL with separate critic heads | 22.2 ± 2.0 |

Each seed score averages the 20k, 24k, and 28k checkpoints. Each checkpoint was evaluated on 100 CALVIN O50 tasks using mixture-based action statistics. The raw O50 outputs, checkpoint metadata, and summaries for every 4k checkpoint and the final model are in `results/<run>-<checkpoint>/`. The six final held-out probe rows are in `results/paper/data/probe_heldout_paper_calvin.csv`.

The six runs used SpecWM commit `55bb378` with the accompanying `source_changes.patch`. Each run trained for 33,375 batches with a 120,000-step learning-rate schedule and 6,000 warmup steps for the model optimizer. The first seed used the recipe default of 3072, followed by seeds 2 and 3. Each run directory here contains its training log, saved configuration, and training metadata.
