# Experiment result handoff — 2026-09-25

This directory is the machine-readable handoff for the runs completed and scored in the 2026-09-24/25 session.

- `run_summary.csv`: one row per scored student (13 rows).
- `cem_scores.csv`: every verified CEM checkpoint result (67 rows).
- `probe_scores.csv`: trained and matched-random rows for every student (26 rows).
- `support_and_excluded_runs.csv`: teachers, superseded runs, the canceled Cube LeWM run, and earlier handoff runs whose artifacts are absent from this checkout.
- `manifest.json`: counts, source locations, and audit provenance.

## Quick table

| run_id | family | treatment | cem_s4000 | cem_s8000 | cem_s12000 | cem_s16000 | cem_s20000 | cem_s24000 | cem_s28000 | cem_s32000 | cem_final | cem_best_pct | cem_best_checkpoint | probe_obj_r2 | matched_random_obj_r2 | probe_delta_r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calvin-lewm-mixlate-2 | LeWM | baseline |  |  |  |  | 24.0 | 27.0 | 30.0 |  | 35.0 | 35.0 | final | 0.3935221754894148 | -0.010189089449073737 | 0.4037112649384886 |
| calvin-lewm-mixlate-3 | LeWM | baseline |  |  |  |  | 33.0 | 32.0 | 30.0 |  | 31.0 | 33.0 | s20000 | 0.5189735229954747 | -0.010189089449073737 | 0.5291626124445484 |
| calvin-mixlate-lag-far-det-1 | temporal | far frames; stop-gradient |  |  |  |  | 40.00000000000001 | 41.0 | 42.0 |  | 39.0 | 42.0 | s28000 | 0.45496923639125014 | -0.029115289078025054 | 0.4840845254692752 |
| calvin-mixlate-lag-far-det-2 | temporal | far frames; stop-gradient |  |  |  |  | 43.0 | 54.0 | 45.0 |  | 35.0 | 54.0 | s24000 | 0.42594745139441875 | -0.029115289078025054 | 0.4550627404724438 |
| calvin-mixlate-lag-far-det-3 | temporal | far frames; stop-gradient |  |  |  |  | 46.0 | 39.0 | 43.0 |  | 45.0 | 46.0 | s20000 | 0.43945767054795687 | -0.029115289078025054 | 0.46857295962598194 |
| calvin-mixlate-far-2 | quasimetric | far frames; attached |  |  |  |  | 51.0 | 47.0 | 47.0 |  | 46.0 | 51.0 | s20000 | 0.04905709842999978 | -0.029115289078025054 | 0.07817238750802483 |
| calvin-mixlate-far-3 | quasimetric | far frames; attached |  |  |  |  | 46.0 | 42.0 | 47.0 |  | 48.0 | 48.0 | final | 0.05071926651158445 | -0.029115289078025054 | 0.0798345555896095 |
| calvin-mixlate-far-det-1 | quasimetric | far frames; stop-gradient |  |  |  |  | 45.0 | 54.0 | 50.0 |  | 52.0 | 54.0 | s24000 | 0.18584473048113187 | -0.029115289078025054 | 0.21496001955915692 |
| calvin-mixlate-far-det-2 | quasimetric | far frames; stop-gradient |  |  |  |  | 47.0 | 49.0 | 49.0 |  | 51.0 | 51.0 | final | 0.220802236231127 | -0.029115289078025054 | 0.24991752530915204 |
| calvin-mixlate-far-det-3 | quasimetric | far frames; stop-gradient |  |  |  |  | 49.0 | 53.0 | 53.0 |  | 55.0 | 55.0 | final | 0.1973654461108777 | -0.029115289078025054 | 0.22648073518890274 |
| scene-play-lag-1 | temporal | play-only | 59.0 | 51.0 | 49.0 | 51.0 | 48.0 | 50.0 | 46.0 | 44.0 | 53.0 | 59.0 | s4000 | 0.8813128374259765 | 0.09382604070502257 | 0.787486796720954 |
| scene-lewm-play-1 | LeWM | play-only | 26.0 | 39.0 | 44.0 | 40.0 | 43.0 | 46.0 | 38.0 | 46.0 | 51.0 | 51.0 | final | 0.20523512600059343 | 0.09382604070502257 | 0.11140908529557086 |
| cube-play-lag-1 | temporal | play-only | 48.0 | 49.0 | 43.0 | 46.0 | 47.0 | 52.0 | 45.0 | 44.0 | 48.0 | 52.0 | s24000 | 0.8650108929016838 | 0.6224567219554349 | 0.24255417094624887 |

## Protocol notes

- CEM uses the `cem-b20` protocol: 100 tasks in five batches of 20, delta mode. CALVIN uses goal offset 50; Scene/Cube use offset 25. Batch-20 RNG draws can differ from older unbatched results, so do not merge them silently.
- CALVIN checkpoints are final plus 20k/24k/28k. Play checkpoints are final plus every 4k from 4k through 32k.
- Probe `obj` is the median of red/blue/pink XYZ R2 for CALVIN and cube XYZ R2 for Scene/Cube. Each trained row has a random-init baseline evaluated on the identical sampled held-out data.
- Probe episodes reconstruct each resolved training split over drawable episodes. Different CALVIN clip reaches can yield different held-out episode sets.
- All 67 CEM cells, 13 probes, and six Modal checkpoint downloads passed `audit_v2.py`; see the manifest for the audit file.
- Rows marked artifact-absent in `support_and_excluded_runs.csv` came from the prior handoff, not from locally verified result files. They are included so agents do not mistake their absence for an intentional omission.
