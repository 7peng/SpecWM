# Reproducing the scene datasets

`data/scene/build_scene_noised.py` rebuilds `scene_play.h5`, `scene_play_noised.h5` and their
concatenation `scene_play_mix_noised.h5` from one pinned OGBench archive. This document records what
is guaranteed, what is not, and the measurements behind each claim.

```bash
data/scene/run_build_scene_noised.sh 36        # 36 workers over $NG GPUs; any worker count is equivalent
python data/scene/build_scene_noised.py spec   # print the frozen SPEC + its digest
python data/scene/build_scene_noised.py verify --out <file>.h5 [--deep]
```

## What the datasets are

`scene_play` replays OGBench `scene-play-v0` through the `swm` SceneEnv by teleporting to each
recorded state and rendering the front camera at 224x224. `scene_play_noised` instead *steps* from
each episode's recorded start state under `a_t = clip(expert_a_t + sigma * N(0,1), -1, 1)`, so the
trajectory diverges and the action is no longer `pi(state)`; `sigma` cycles over `{0, 0.3, 0.6}` by
episode index. Both are 1000 episodes x 1000 frames. The mix is the two concatenated, which is what
the training configs load.

## The guarantee

Two chains are recorded in `<file>.h5.manifest.json`, because they answer different questions.

| | covers | reproducible on another cluster? |
|---|---|---|
| `content_sha256_state` | `action`, `observation`, `qpos`, `qvel`, `button_states`, `step_idx` | **yes, exactly** |
| `content_sha256_full` | the above plus `pixels` | no — see below |
| `file_sha256` | the HDF5 file as shipped | yes, for a *downloaded* copy |

A rebuild must reproduce `content_sha256_state`. `verify` treats a state mismatch as a failure and a
pixels-only mismatch as an expected note. Because the artifact is distributed as a file rather than
re-rendered, pixel identity comes from checking `file_sha256` after download.

Digests are computed **per episode** and stored in the file (`episode_digest`,
`episode_state_digest`). If two builds disagree, the digest lists say which episodes diverged rather
than only that the files differ.

### The state chain is recomputed, not read back (fixed 2026-09-07)

`verify` used to read `episode_state_digest` out of the file and compare it to the manifest. Both
come from the same place — the per-worker block files, via `merge`/`mix`, which carry the digest
lists forward rather than re-deriving them — so the check compared the merge's bookkeeping against
itself and passed on a file whose columns were partly **unwritten**.

That is not hypothetical. `_copy_into`'s verbatim-chunk fast path walked the chunk grid along rows
at column offset 0 only, and h5py auto-chunks a narrow 2-d column by *both* axes: `qpos` (N, 25)
becomes (256, 7), so the chunks at column offsets 7/14/21 were never written and read back as fill,
i.e. **zero**. It fired for every block whose row count is a multiple of the 256-row chunk — which a
32-way split makes the first eight — so `scene_play_noised`, `cube_play` and `cube_play_noised`
shipped with `qpos`, `qvel` and `observation` identically zero past the arm for **episodes 0–255**,
and both mix files inherited it (`scene_play_mix_noised` 1000–1255; `cube_play_mix_noised` 0–255 and
1000–1255). Every `cube_xyz` / `drawer` / `window` probe read on those files scored a quarter of its
rows against a zero label; the frames were always correct.

Two changes, both in `build_scene_noised.py`:

- the fast path now requires the chunk to span the column's **full width** (`pixels`, chunked
  (50, 224, 224, 3), still takes it — it is the only column the optimisation was ever for);
- the default `verify` **recomputes the state chain from the columns** (`recompute_digests(...,
  states_only=True)`, seconds on a 50 GB file, since only `pixels` is expensive). `--deep` still adds
  the full chain including pixels.

Existing files are repaired in place by `data/scene/repair_chunk_hole.py`, which rewrites the
affected columns from the block files under `$STABLEWM_HOME/datasets/_{scene,cube}_build/` and gates
every episode on the digest the file already carries. Run `--check` first; it reports without
writing.

## Why pixels are not bit-reproducible

MuJoCo renders with 4x MSAA. There is no seed to set — the sample pattern is fixed — but the GPU
resolves the multisample buffer in a schedule-dependent order, and float addition is not associative,
so the last bit can land either way.

Measured on an L40S: replaying one episode twice differs on **up to 20 of 60 frames** by exactly
+/-1 LSB, while `qpos`/`qvel`/`action`/`observation` stay bit-identical.

A single frozen state re-rendered 250 times was bit-identical on both an idle and a fully saturated
GPU. **Repeating one frozen state is therefore not a valid determinism test here** — it passes while
the real workload still jitters. `preflight` and `selftest` both probe whole episodes for this
reason.

If exactly reproducible pixels are ever needed, set `offsamples: 0` with `supersample: 2` or `4` in
`SPEC`: that renders larger with the multisample path off and box-filters down in NumPy, where the
summation order is ours. Every episode-level trial of that configuration was bit-exact. The cost, for
all 2M frames:

| setting | RMSE vs the MSAA frames | GPU-hours |
|---|---|---|
| `offsamples: 4` (default, v1-identical) | 0 | 0.4 |
| `offsamples: 0, supersample: 1` (no AA) | 9.7 | 0.3 |
| `offsamples: 0, supersample: 2` | 5.9 | 4.9 |
| `offsamples: 0, supersample: 4` | 4.0 | 12.7 |

## Three bugs this rewrite fixes

These were found while auditing the v1 collectors (`scripts/collect_scene.py`,
`scripts/collect_scene_noised.py`) and are the reason a rebuild is not merely a re-run.

**1. An unseeded goal-ghost in every frame.** `ManipSpaceEnv.reset()` picks
`cur_task_id = np.random.randint(1, 6)` from the *global* `np.random`, which neither collector
seeded. With `mode='task'` and `visualize_info=True` that paints the task's goal cube into the scene
at alpha 0.2 — and `set_state` never touches `mocap_pos`, so whatever the draw picked persists for
the entire worker process. The ghost is drawn in the *same colour as the manipulated cube*
(`initialize_episode` copies the cube colour onto the target geoms), so every frame contains a second
cube-coloured object at a fixed image position.

Recovered from the shipped files by re-rendering stored states under each candidate task id:

- `scene_play.h5`: task 1 for the whole file — ghost on the table at `(0.35, 0.05, 0.02)`.
- `scene_play_noised.h5`: its 36 shards drew four different positions — 6 shards task 1, 6 task 2,
  11 task 3, 13 task 4/5 (4 and 5 share a goal position inside the drawer, where it is occluded).

So v1's noised half carries a nuisance variable that is constant within a 28-episode block and
changes between blocks. `SPEC` fixes this twice over: `task_id: 1` pins the draw (matching what v1's
expert half already had), and `visualize_info: False` keeps the overlay out of the render entirely.
Pinning alone would have left a second cube-coloured object at a fixed image position in every
frame — harmless for planning, but a direct confound for any analysis that asks whether the encoder
represents the cube. With the overlay off, task ids 1 and 4 render identically, so `reset()`'s draw
cannot reach the pixels by any path. Physics never saw the ghost: the state digests are unchanged.

**2. Contact forces leaking across episodes.** `set_state` ends at `mj_forward`, which does not
compute `cfrc_ext`; only `mj_rnePostConstraint` does, and that runs inside `step()`. The first
observation of an episode therefore carried the *previous* episode's contact forces through
`proprio/gripper_contact`. Measured: frame 0 of an episode following a grasp differed by up to 0.98.
This made an episode depend on what its worker rendered before it — a 3-worker and a 1-worker build
produced different `observation` columns. `reset_sim()` now recomputes it.

**3. Row order depended on the worker count.** v1 sharded strided (`e % W == k`) and merged shard by
shard, so the episode order in the file was a function of `W`, and `merge` silently skipped missing
shards. Workers now take contiguous blocks of the canonical order and `merge` refuses to run if a
block is absent.

## What is unchanged from v1

Verified by regenerating full 1000-frame episodes and diffing against the shipped files: for expert
episodes 0/500/999 and noised episodes 0/1/2/37 (all three sigma levels), `action`, `observation`,
`qpos`, `qvel`, `button_states` and `step_idx` are **bit-identical**. Same episode selection, same
sigma schedule, same per-episode noise stream (`default_rng(1000 + e)`), same physics. Rendering with
v1's settings reproduces the shipped pixels exactly (0 of 150528 subpixels differ).

The changes above therefore affect: the goal-ghost (gone from both halves, not merely pinned), the
first observation of episodes whose predecessor was in contact, and the row order. Removing the
overlay is the one change that moves pixels, so `GOLDEN`'s `pixel_mean` entries differ from v1's.

## Cross-machine notes

`preflight` compares this machine against `GOLDEN` — an exact hash of the simulation columns plus a
tolerance-checked pixel summary — and refuses to start a build on a machine that disagrees. Verified
identical across GPUs 0/1/2 of one node and across two different nodes (L40S, driver 595.71.05).
Different GPU vendors are untested; the OpenGL spec leaves the multisample resolve
implementation-defined, so treat a `preflight` pixel failure as "this machine cannot reproduce the
artifact — download it instead", not as a bug.

The manifest records a full environment fingerprint (mujoco, numpy, h5py, HDF5, GPU, driver) so a
mismatch can be attributed rather than guessed at.
