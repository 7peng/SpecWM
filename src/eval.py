"""Goal-offset CEM evaluation of a checkpoint: the planner rolls the predictor forward along each
candidate action plan from the observed frame history and scores the endpoint by cosine distance to
the encoded goal frame. Driven by scripts/eval_cem.py (the paper protocol); one-off:

    MUJOCO_GL=egl python src/eval.py --config-name=scene policy=lewm/<run>/weights_epoch_N.pt eval.dataset_name=...
"""
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

os.environ["MUJOCO_GL"] = "egl"

from utils import classify_columns, load_env

load_env()


@contextmanager
def _parallel_envpool_patch():
    """Swap swm's serial EnvPool for the process-parallel one (one env per process: MuJoCo EGL
    contexts sharing a process return each other's buffers above two envs), and re-observe each env
    after its episode is installed so the policy's first action sees the episode's own start frame."""
    import stable_worldmodel.world.world as _wmod
    from parallel_env_pool import ParallelEnvPool

    orig_pool, orig_apply = _wmod.EnvPool, _wmod._apply_callables

    def _apply_callables_and_refresh(env, callables, init_state):
        orig_apply(env, callables, init_state)
        refresh = getattr(env, "pool_refresh", None)
        assert refresh is not None, (
            "the episode was installed on an env this pool cannot re-observe, so the policy would "
            "act on the pre-episode state; run with the parallel pool")
        pool = object.__getattribute__(env, "_pool")
        idx = object.__getattribute__(env, "_idx")
        # the envs are refreshed together once the last episode of the reset is installed, so the
        # workers' rebuilds overlap; the queue lives on the pool because swm hands a fresh proxy per call
        queue = pool.__dict__.setdefault("_deferred_refresh", [])
        queue.append((idx, init_state.get("pixels", None)))
        if len(queue) < pool._n:
            return
        pending = list(queue)
        queue.clear()
        pool.refresh([i for i, _ in pending])
        for idx, ref in pending:
            if isinstance(ref, np.ndarray) and ref.ndim == 3:
                cur = np.asarray(pool._stacked_infos["pixels"][idx])
                cur = cur[-1] if cur.ndim == 4 else cur
                if cur.shape == ref.shape:
                    diff = float(np.abs(cur.astype(np.float32) - ref.astype(np.float32)).mean())
                    assert diff < 1.0, (
                        f"[start-frame] env {idx}: the rendered start differs from the stored start "
                        f"frame by {diff:.2f} mean-abs (uint8); the episode was not installed")
    _wmod.EnvPool = ParallelEnvPool
    _wmod._apply_callables = _apply_callables_and_refresh
    try:
        yield
    finally:
        # only the pool is restored: World reads it in __init__ alone, whereas `_apply_callables` is
        # looked up during evaluate(), after this context has exited
        _wmod.EnvPool = orig_pool


# stable_pretraining's ImageNet table, inlined (tests/test_eval_norm.py asserts they agree)
IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET_STATS),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        keys_to_load=cfg.dataset.get("keys_to_load"),
        cache_dir=dataset_path,
    )


class _FrameHistoryMixin:
    """Per-env ring buffers of the observed frames and the executed action blocks. Young episodes
    clamp to their oldest frame; an episode reset flushes both rings."""

    def _hist_init(self, n, history_frames, history_stride):
        self.hist_n = int(history_frames)
        self.hist_stride = int(history_stride)
        span = (self.hist_n - 1) * self.hist_stride + 1
        self._hist = [deque(maxlen=span) for _ in range(n)]
        self._ahist = [deque(maxlen=max(1, (self.hist_n - 1) * self.hist_stride)) for _ in range(n)]

    def _action_hist(self, n, adim):
        """(n, hist_n - 1, stride * adim): the executed action block from each history frame to the
        next, oldest first, in the model's normalised action space; zero blocks for clamped slots."""
        N, st = self.hist_n, self.hist_stride
        out = torch.zeros(n, N - 1, st * adim, dtype=torch.float32)
        for i in range(n):
            h = list(self._ahist[i]); L = len(h)
            for k in range(N - 1):
                lo = L - (N - 1 - k) * st            # absolute ring offset of frame k's block
                if lo < 0:
                    continue
                out[i, k] = torch.stack(h[lo:lo + st]).reshape(-1)
        return out

    def _hist_stack(self, cur, flush):
        """cur (n, C, H, W) -> (n, hist_n, C, H, W), oldest first."""
        n = cur.shape[0]
        for i in range(n):
            if flush is not None and bool(flush[i]):
                self._hist[i].clear()
            self._hist[i].append(cur[i])
        ctx = []
        for i in range(n):
            h = self._hist[i]
            idxs = [max(0, len(h) - 1 - k * self.hist_stride) for k in reversed(range(self.hist_n))]
            ctx.append(torch.stack([h[j] for j in idxs]))
        return torch.stack(ctx)


class FrameHistoryPolicy(_FrameHistoryMixin, swm.policy.WorldModelPolicy):
    """WorldModelPolicy with a multi-frame context: the last `history_frames` observed frames spaced
    `history_stride` env steps apart, plus the action blocks executed between them."""

    def __init__(self, *args, history_frames=3, history_stride=10, **kw):
        super().__init__(*args, **kw)
        self._hf, self._hs = int(history_frames), int(history_stride)
        self._hist = None

    def set_env(self, env):
        super().set_env(env)
        self._hist_init(getattr(env, "num_envs", 1), self._hf, self._hs)

    def _prepare_info(self, info_dict):
        out = super()._prepare_info(info_dict)
        flush = out.get("_needs_flush")
        px = out["pixels"]
        out["pixels"] = self._hist_stack(px[:, -1] if px.ndim == 5 else px, flush)
        n = out["pixels"].shape[0]
        for i in range(n):
            if flush is not None and bool(flush[i]):
                self._ahist[i].clear()
        out["action_hist"] = self._action_hist(n, int(self.env.single_action_space.shape[-1]))
        # consumed: swm only pops the flag from its copy, so on the world's dict it would stay
        # True from the first auto-reset on and flush the history every step
        if isinstance(info_dict, dict):
            info_dict.pop("_needs_flush", None)
        return out

    def get_action(self, info_dict, **kw):
        """swm's get_action, plus the emitted action remembered in the model's normalised space
        (swm inverse-transforms for the env right before returning, so it is re-applied)."""
        a = super().get_action(info_dict, **kw)
        an = self.process["action"].transform(a) if "action" in self.process else a
        an = torch.as_tensor(np.asarray(an), dtype=torch.float32).reshape(a.shape[0], -1)
        for i in range(an.shape[0]):
            self._ahist[i].append(an[i])
        return a


def _action_encoder_in_dim(enc):
    """Trained per-block action dim (frameskip * action_dim) of the action encoder."""
    pe = getattr(enc, "patch_embed", None)
    if pe is not None and hasattr(pe, "in_channels"):
        return pe.in_channels
    return None


def _history_windows(model):
    """Frames the predictor's context holds, read off the checkpoint's positional table."""
    p = getattr(model, "predictor", None)
    sp = getattr(p, "state_pos", None)
    if sp is None:
        sp = getattr(p, "pos_embedding", None)      # ARPredictor: (1, num_frames, D)
    return int(sp.shape[1]) if sp is not None else 3


class DeltaCosineCostModel:
    """Planner cost: the embedding at the end of each candidate plan, reached by autoregressive
    one-block predictions from the frame history, scored by 1 - cos to the goal embedding. An
    ARPredictor is rolled over its own context window, each context frame conditioned on the block
    executed from it; a HorizonPredictor is rolled at horizon 1 with each prediction re-normalised
    to encoder magnitude before it is fed back."""

    def __init__(self, model):
        self.model = model
        self.ar = type(model.predictor).__name__ == "ARPredictor"

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info_dict, action_candidates):
        m = self.model
        device = next(m.parameters()).device
        with torch.no_grad():
            for k in list(info_dict.keys()):
                if torch.is_tensor(info_dict[k]):
                    info_dict[k] = info_dict[k].to(device)
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in list(goal.keys()):
                if k.startswith("goal_"):
                    goal[k[len("goal_"):]] = goal.pop(k)
            goal.pop("action", None)
            goal_emb = m.encode(goal)["emb"][..., -1, :]
            ctx = {k: v[:, 0] for k, v in info_dict.items()
                   if torch.is_tensor(v) and k != "goal" and not k.startswith("goal_")}
            ctx.pop("action", None)
            ctx.pop("action_hist", None)          # consumed by the AR branch below, not the encoder
            ctx_emb = m.encode(ctx)["emb"]
        B, S, T = action_candidates.shape[:3]
        ce = ctx_emb.unsqueeze(1).expand(B, S, *ctx_emb.shape[1:]).reshape(B * S, *ctx_emb.shape[1:])
        act = action_candidates.to(device).reshape(B * S, T, -1)
        enc_in = _action_encoder_in_dim(m.action_encoder)
        assert enc_in is None or act.shape[-1] == enc_in, (
            f"plan_config.action_block yields a {act.shape[-1]}-d action block, but the model's "
            f"action_encoder expects {enc_in} (= trained frameskip * action_dim). Set "
            f"plan_config.action_block to this checkpoint's training frameskip.")
        if self.ar:
            Hh = ce.shape[1]
            ah = info_dict.get("action_hist")
            assert ah is not None, "the AR cost path needs FrameHistoryPolicy's action_hist"
            ah = ah[:, 0].to(device)                                   # (B, Hh-1, A)
            assert ah.shape[1] == Hh - 1 and ah.shape[-1] == act.shape[-1], (ah.shape, act.shape)
            ah = ah.unsqueeze(1).expand(B, S, Hh - 1, act.shape[-1]).reshape(B * S, Hh - 1, act.shape[-1])
            acts = torch.cat([ah, torch.nan_to_num(act, 0.0)], dim=1)  # acts[j] = block FROM frame j
            W = _history_windows(m)
            assert Hh <= W, f"{Hh}-frame history exceeds the ARPredictor's {W}-frame window"
            emb = ce
            for i in range(T):
                w = min(W, Hh + i)
                a_emb = m.action_encoder(acts[:, Hh + i - w: Hh + i])   # one token per window frame
                assert a_emb.shape[1] == w, "ARPredictor takes one action token per context frame"
                nxt = m.predict(emb[:, -w:], a_emb)[:, -1:]
                emb = torch.cat([emb, nxt], dim=1)
            pred = emb[:, -1]
        else:
            gap = m.action_encoder(torch.nan_to_num(act, 0.0))
            sfs = gap.shape[1] // T
            Hh = ce.shape[1]
            run = ce
            one = torch.ones(B * S, dtype=torch.long, device=device)
            m1 = torch.ones(B * S, 1, 1, sfs, dtype=torch.bool, device=device)
            for i in range(T):
                nxt = m.emb_norm(m.predict_horizon(run, gap[:, i * sfs:(i + 1) * sfs], one, m1))
                run = torch.cat([run, nxt.unsqueeze(1)], dim=1)[:, -Hh:]
            pred = run[:, -1]
        self.last_pred = pred.detach()
        pn = torch.nn.functional.normalize(pred, dim=-1)
        g = torch.nn.functional.normalize(goal_emb, dim=-1)
        return 1.0 - (pn.reshape(B, S, -1) * g.unsqueeze(1)).sum(-1)


def seed_everything(seed):
    """Seed the action sampler as well as the environments, so a re-run reproduces and two
    checkpoints face the same draws."""
    import random
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_eval_context(cfg):
    """The world, the pixel transforms, the action normaliser and the task set: everything a run
    needs that does not depend on the checkpoint."""
    with _parallel_envpool_patch():
        world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    # the task sets' action column is identically zero, which fits an identity scaler, so the file
    # whose action statistics z-score the planner's candidates is named explicitly (the training file)
    assert cfg.dataset.get("stats"), (
        "dataset.stats is unset. It names the file whose action statistics the planner's candidates "
        "are z-scored in, and it must be the checkpoint's training file (scripts/eval_cem.py takes it "
        "from train_meta.json).")
    stats_name = cfg.dataset.stats
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = (dataset if stats_name == cfg.eval.dataset_name
                     else get_dataset(cfg, stats_name))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    # the same ledger training is held to: every cached column gets a StandardScaler below
    _norm, _raw = classify_columns(cfg.dataset.keys_to_cache, [])
    assert not _raw, (
        f"eval caches {_raw}, which train/utils classify as never-z-scored, and eval has no raw path")

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        if col == "action":
            # a constant column fits scale_ = 1 (sklearn's rule), which would execute the planner's
            # z-scores as raw actions; a wholly constant file is refused, a constant channel is
            # clamped as training's normaliser clamps it (utils.get_column_normalizer)
            dead = np.flatnonzero(np.asarray(processor.var_) <= 1e-12).tolist()
            n_ch = int(np.asarray(processor.var_).shape[0])
            assert len(dead) < max(1, n_ch // 2), (
                f"dataset.stats={stats_name!r} has constant `action` columns in channels {dead}; "
                f"point dataset.stats at the checkpoint's training file, not at a task set.")
            if dead:
                processor.scale_ = np.asarray(processor.scale_, np.float64).copy()
                processor.scale_[dead] = 1e-6
                print(f"[stats] constant action channels {dead} clamped to std 1e-6", flush=True)
            print(f"[stats] action normaliser from {stats_name!r}: "
                  f"mean {np.round(processor.mean_, 4).tolist()} "
                  f"std {np.round(processor.scale_, 4).tolist()}", flush=True)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    return dict(world=world, transform=transform, process=process, dataset=dataset,
                ep_indices=ep_indices, col_name=col_name)


def build_eval_policy(cfg, ctx):
    """Load the checkpoint and wrap it in the planning policy (or the random floor). Returns
    (policy, model)."""
    policy_config = cfg.get("policy", None)
    if not policy_config:
        raise ValueError("cfg.policy must name a trained checkpoint "
                         "(e.g. lewm/<run>/weights_epoch_N.pt) or 'random'.")
    is_random = str(policy_config) == "random"
    model = None if is_random else swm.wm.utils.load_pretrained(cfg.policy)
    if model is not None:
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    transform, process = ctx["transform"], ctx["process"]

    if is_random:
        return swm.policy.RandomPolicy(seed=cfg.seed), model
    solver = hydra.utils.instantiate(cfg.solver, model=DeltaCosineCostModel(model))
    # the context is H frames spaced action_block apart, H read off the checkpoint
    policy = FrameHistoryPolicy(
        solver=solver, config=config, process=process, transform=transform,
        history_frames=_history_windows(model), history_stride=cfg.plan_config.action_block,
    )
    return policy, model


@hydra.main(version_base=None, config_path="../config/eval", config_name="scene")
def run(cfg: DictConfig):
    return evaluate_config(cfg)


def evaluate_config(cfg: DictConfig, ctx=None):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, \
        "Planning horizon must be smaller than or equal to eval_budget"

    # the goal-offset mode replays the last O+1 frames of a demo; 2x the budget lets a plan overshoot
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if ctx is None:
        ctx = build_eval_context(cfg)
    world, dataset = ctx["world"], ctx["dataset"]
    ep_indices, col_name = ctx["ep_indices"], ctx["col_name"]
    policy, model = build_eval_policy(cfg, ctx)

    results_path = (Path(cfg.results_dir) if cfg.get("results_dir")
                    else Path(swm.data.utils.get_cache_dir(), cfg.policy).parent)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    # eval.episode_batches: every valid start, sorted, scored in equal batches in this process (the
    # model and the env-worker pool stay up); the offset's number is the driver's mean over the batches
    nb = cfg.eval.get("episode_batches")
    n_pick = len(valid_indices) if nb is not None else min(cfg.eval.num_eval, len(valid_indices))
    if nb is None and n_pick < cfg.eval.num_eval:
        print(f"num_eval={cfg.eval.num_eval} > {len(valid_indices)} valid starts; using all {n_pick}")
    random_episode_indices = g.choice(len(valid_indices), size=n_pick, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    batch_slices = None
    if nb is not None:
        nb, ne = int(nb), int(cfg.eval.num_eval)
        assert n_pick == nb * ne, (
            f"eval.episode_batches={nb} x eval.num_eval={ne} must equal the {n_pick} valid "
            f"starts exactly; the batches partition the sorted set")
        batch_slices = [(b * ne, (b + 1) * ne) for b in range(nb)]
        print(f"[episode-batches] {n_pick} valid starts as {nb} in-process batches of {ne}",
              flush=True)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)
    # immediately before the rollout, so the action draws are the same on a re-run
    seed_everything(cfg.seed)

    # videos go in a per-run subdirectory: world.evaluate names them episode_<i>.mp4 off the episode
    # index alone, so concurrent runs sharing a results_dir would overwrite each other's
    video_path = results_path / Path(cfg.output.filename).stem
    if not cfg.get("no_video"):
        video_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if batch_slices is None:
        metrics = _evaluate_rows(world, dataset, cfg, eval_start_idx, eval_episodes,
                                 None if cfg.get("no_video") else video_path)
    else:
        stem, suffix = Path(cfg.output.filename).stem, (Path(cfg.output.filename).suffix or ".txt")
        for b, (lo, hi) in enumerate(batch_slices):
            t0 = time.time()
            seed_everything(cfg.seed)          # the action draws of a fresh process
            world.set_policy(policy)           # solver.configure + action buffer + frame rings anew
            if hasattr(policy, "_next_init"):
                policy._next_init = None
            vp = None if cfg.get("no_video") else video_path / f"b{b}"
            if vp is not None:
                vp.mkdir(parents=True, exist_ok=True)
            m = _evaluate_rows(world, dataset, cfg, eval_start_idx[lo:hi], eval_episodes[lo:hi], vp)
            print(f"[episode-batches] batch {b} rows {lo}:{hi} ({time.time() - t0:.0f}s): {m}",
                  flush=True)
            _write_results(results_path / f"{stem}_b{b}{suffix}", cfg, m, time.time() - t0)
        return
    end_time = time.time()
    print(metrics)
    _write_results(results_path / cfg.output.filename, cfg, metrics, end_time - start_time)
    return ctx, metrics, end_time - start_time


def _evaluate_rows(world, dataset, cfg, start_idx, episodes, video_path):
    return world.evaluate(
        dataset=dataset,
        start_steps=np.asarray(start_idx).tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=np.asarray(episodes).tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_path,
    )


def _write_results(path, cfg, metrics, seconds):
    """The results txt scripts/eval_cem.py parses (`'success_rate': x` in the metrics repr)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {seconds} seconds\n")


if __name__ == "__main__":
    run()
