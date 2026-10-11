#!/usr/bin/env python
"""The CEM planning protocol (ICLR-scene-cem-v3): one checkpoint, every requested goal offset, 100
tasks each, one eval.py process per offset.

    eval_cem.py ARM POLICY [--gpus 0,1,2,3] [--offsets 25,50,75,100] [--batches N] [EXTRA_HYDRA_OVERRIDES...]

ARM names results/<arm>/ (O<k>.txt, O<k>.log, run_meta.json, summary.csv). POLICY is a checkpoint
path relative to $STABLEWM_HOME, or 'random' for the floor. The eval config (scene, cube, calvin),
the task-set family and the action statistics come from the checkpoint's train_meta.json; the
action block is the trained frameskip, read from its config.json.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import h5py

ICLR = Path(__file__).resolve().parent.parent
REPO = ICLR
for _name in (".env", ".env.example"):
    if (REPO / _name).exists():
        for _line in (REPO / _name).read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"])
PYTHON = str(REPO / ".venv/bin/python")
OFFSETS = [25, 50, 75, 100]
# raw action width per env: OGBench's 5-d command, CALVIN's 7-d rel_actions; the trained frameskip
# is the checkpoint's action_encoder.input_dim divided by it
ACTION_DIM = {"scene": 5, "cube": 5, "calvin": 7}
TASKSETS = {"scene": "scene_offset_tasks", "cube": "cube_offset_tasks", "calvin": "calvin_offset_tasks"}
PROTOCOL = "ICLR-scene-cem-v3"  # v2: AR horizon-1 rollout; v3: goal marker off in the agent render


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def gpu_util(gpu):
    out = sh(["nvidia-smi", "--query-gpu=utilization.gpu",
              "--format=csv,noheader,nounits", "-i", str(gpu)])
    return int(out.stdout.strip())


def taskset(args, offset):
    return f"{TASKSETS[args.eval_config]}_O{offset}"


def n_tasks(args, offset):
    with h5py.File(STABLEWM_HOME / "datasets" / f"{taskset(args, offset)}.h5", "r") as f:
        return len(f["ep_len"])


def data_id(name):
    """(path, bytes, mtime) of a dataset for the run record: a name alone does not identify a
    file that has been rebuilt under it."""
    p = STABLEWM_HOME / "datasets" / f"{name}.h5"
    if not p.is_file():
        return {"name": name, "path": str(p), "missing": True}
    st = p.stat()
    return {"name": name, "path": str(p), "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()}


def ckpt_sha(policy):
    if policy == "random":
        return "random"
    for cand in [STABLEWM_HOME / policy, STABLEWM_HOME / "checkpoints" / policy]:
        if cand.is_file():
            return hashlib.sha256(cand.read_bytes()).hexdigest()[:16]
    sys.exit(f"checkpoint not found: {policy}")


def config_action_block(eval_config):
    """The eval config's own `plan_config.action_block`, read as text (no hydra composition)."""
    text = (ICLR / "config" / "eval" / f"{eval_config}.yaml").read_text()
    m = re.search(r"^\s*action_block:\s*(\d+)", text, re.M)
    assert m, f"config/eval/{eval_config}.yaml declares no plan_config.action_block"
    return int(m.group(1))


def action_block(policy, eval_config):
    """The trained frameskip from the checkpoint's config.json; the random floor takes the eval
    config's block (RandomPolicy samples the action space every step, so only the horizon bound
    depends on it)."""
    if policy == "random":
        return config_action_block(eval_config)
    for base in [STABLEWM_HOME / policy, STABLEWM_HOME / "checkpoints" / policy]:
        cj = base.parent / "config.json"
        if cj.is_file():
            in_dim = int(json.load(open(cj))["action_encoder"]["input_dim"])
            adim = ACTION_DIM[eval_config]
            assert in_dim % adim == 0, (
                f"{cj}: action_encoder.input_dim={in_dim} is not a multiple of the env's {adim}-d "
                f"action, so the trained frameskip cannot be recovered from it")
            return in_dim // adim
    sys.exit(f"no config.json found next to {policy}; run scripts/write_config_json.py for a step snapshot")


def train_meta(policy, dataset=None):
    """The checkpoint's train_meta.json (written by train_common): its `dataset` names the eval
    config and the action statistics. `--dataset` supplies it for a checkpoint without the file."""
    for base in [STABLEWM_HOME / policy, STABLEWM_HOME / "checkpoints" / policy]:
        p = base.parent / "train_meta.json"
        if p.is_file():
            m = json.load(open(p))
            if dataset:
                m["dataset"] = dataset
            return m
    if dataset:
        return {"dataset": dataset}
    sys.exit(f"no train_meta.json next to {policy}; pass --dataset <training corpus name>")


def missing_tasksets(args, offsets):
    return [str(STABLEWM_HOME / "datasets" / f"{taskset(args, o)}.h5") for o in offsets
            if not (STABLEWM_HOME / "datasets" / f"{taskset(args, o)}.h5").is_file()]


def overrides(args, offset):
    horizon = -(-offset // args.action_block)
    budget = max(2 * offset, 60)
    out_dir = ICLR / "results" / args.arm
    ov = [
        f"policy={args.policy}",
        f"plan_config.horizon={horizon}",
        f"plan_config.receding_horizon={min(3, horizon)}",   # cannot recede past the plan
        f"plan_config.action_block={args.action_block}",
        f"eval.dataset_name={taskset(args, offset)}",
        f"eval.num_eval={n_tasks(args, offset)}",
        f"eval.goal_offset_steps={offset}",
        f"eval.eval_budget={budget}",
        f"output.filename={out_dir}/O{offset}.txt",
        f"+results_dir={out_dir}",
    ]
    # the planner's candidates are z-scored with the TRAINING file's action statistics
    if args.meta.get("dataset"):
        ov.append(f"dataset.stats={args.meta['dataset']}")
    return ov + args.extra


def run_offset(args, offset, gpu):
    out_dir = ICLR / "results" / args.arm
    txt, log = out_dir / f"O{offset}.txt", out_dir / f"O{offset}.log"
    if txt.exists() and txt.stat().st_size > 0:
        print(f"O{offset}: exists, skip")
        return parse_sr(txt) if parse_sr(txt) is not None else parse_sr(log)
    env = os.environ | {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu),
                        "CUDA_VISIBLE_DEVICES": str(gpu), "STABLEWM_HOME": str(STABLEWM_HOME),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    n, nb = n_tasks(args, offset), args.batches
    print(f"O{offset} on GPU{gpu}: n={n}" + (f" in {nb} batches" if nb > 1 else ""))
    if nb == 1:
        cmd = [PYTHON, "src/eval.py", "--config-name", args.eval_config, *overrides(args, offset)]
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=ICLR, env=env, stdout=f, stderr=subprocess.STDOUT)
        sr = parse_sr(log)
    else:
        # equal batches partition the sorted valid starts inside one process (eval.episode_batches);
        # the offset's number is their mean, the per-batch records land in O<k>_b<b>.txt
        assert n % nb == 0, f"O{offset}: {n} windows do not split into {nb} equal batches"
        ne = n // nb
        edges = [b * ne for b in range(nb + 1)]
        base = [o for o in overrides(args, offset)
                if not o.startswith(("eval.num_eval=", "output.filename="))]
        cmd = [PYTHON, "src/eval.py", "--config-name", args.eval_config,
               *base, f"eval.num_eval={ne}", f"eval.episode_batches={nb}", f"output.filename={txt}"]
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=ICLR, env=env, stdout=f, stderr=subprocess.STDOUT)
        srs = [parse_sr(out_dir / f"O{offset}_b{b}.txt") for b in range(nb)]
        for b in range(nb):
            print(f"  O{offset} batch {b} rows {edges[b]}:{edges[b + 1]}: success_rate={srs[b]}", flush=True)
        sr = None if any(x is None for x in srs) else sum(x * (edges[i + 1] - edges[i]) for i, x in enumerate(srs)) / n
        with open(log, "a") as f:
            f.write(f"batches: {srs} edges {edges}\n" + (f"'success_rate': {sr}\n" if sr is not None else ""))
        if sr is not None:
            with open(txt, "w") as f:
                f.write(f"batches: {srs} edges {edges}\n'success_rate': {sr}\n")
    if sr is None:
        txt.unlink(missing_ok=True)
        print(f"O{offset} FAILED (see {log})", file=sys.stderr)
    else:
        print(f"O{offset} done: success_rate={sr}")
    return sr


def parse_sr(path):
    """Last `'success_rate': x` in a driver log or an eval results txt (both are append-mode)."""
    if not path.exists():
        return None
    hits = re.findall(r"'success_rate': ([0-9.]+)", path.read_text(errors="ignore"))
    return float(hits[-1]) if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm")
    p.add_argument("policy")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--offsets", default=",".join(map(str, OFFSETS)))
    p.add_argument("--eval-config", default=None, choices=sorted(TASKSETS),
                   help="default from train_meta.json's dataset: cube for cube_play*, calvin for calvin*, else scene")
    p.add_argument("--dataset", default=None,
                   help="training corpus name, for a checkpoint without train_meta.json")
    p.add_argument("--batches", type=int, default=1, help="equal in-process batches per offset")
    p.add_argument("extra", nargs="*", default=[])
    args = p.parse_args()
    args.meta = (train_meta(args.policy, dataset=args.dataset)
                 if args.policy != "random" else {})
    if args.eval_config is None:
        ds = str(args.meta.get("dataset", ""))
        args.eval_config = "cube" if ds.startswith("cube_play") else "calvin" if ds.startswith("calvin") else "scene"
    args.action_block = action_block(args.policy, args.eval_config)
    if "frameskip" in args.meta:
        assert args.action_block == int(args.meta["frameskip"]), (
            f"action_block={args.action_block} from config.json's action_encoder.input_dim / "
            f"{ACTION_DIM[args.eval_config]}, but train_meta.json says frameskip={args.meta['frameskip']}")
    print(f"{args.policy}: action_block={args.action_block} eval_config={args.eval_config} "
          f"train_meta={ {k: args.meta[k] for k in ('dataset', 'max_pred_horizon') if k in args.meta} }")

    gpus = [int(g) for g in args.gpus.split(",")]
    offsets = [int(o) for o in args.offsets.split(",")]
    if (miss := missing_tasksets(args, offsets)):
        sys.exit("task sets missing for the requested offsets:\n  " + "\n  ".join(miss)
                 + "\nBuild them (data/<env>/make_*_offset_tasks.py) or pass --offsets with the ones that exist.")
    for g in gpus:
        for _ in range(60):
            if (u := gpu_util(g)) < 20:
                break
            time.sleep(5)
        else:
            if os.environ.get("EVAL_CEM_ALLOW_BUSY"):
                print(f"GPU {g} busy ({u}%), continuing (EVAL_CEM_ALLOW_BUSY)", flush=True)
            else:
                sys.exit(f"GPU {g} busy ({u}%)")

    out_dir = ICLR / "results" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    git_sha = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    dirty = len(sh(["git", "status", "--porcelain"], cwd=REPO).stdout.splitlines())
    stats_name = args.meta.get("dataset")
    meta = {"arm": args.arm, "policy": args.policy,
            "action_block": args.action_block, "ckpt_sha256_16": ckpt_sha(args.policy),
            "git_sha": git_sha, "git_dirty": dirty, "date": datetime.now().astimezone().isoformat(),
            "extra": args.extra, "protocol": PROTOCOL, "eval_config": args.eval_config,
            "batches": args.batches,
            "stats_dataset": stats_name, "dataset_override": args.dataset,
            "stablewm_home": str(STABLEWM_HOME),
            "action_stats": data_id(stats_name) if stats_name else "eval config default",
            "task_sets": {o: data_id(taskset(args, o)) for o in offsets}}
    # a finished offset is never re-run, so a changed protocol or checkpoint must not be mixed into
    # an arm directory that already holds results
    prev_path = out_dir / "run_meta.json"
    if prev_path.is_file() and any((out_dir / f"O{o}.txt").exists() for o in offsets):
        prev = json.loads(prev_path.read_text())
        for k in ("protocol", "ckpt_sha256_16", "eval_config", "action_block"):
            if k in prev and prev[k] != meta[k]:
                sys.exit(
                    f"{out_dir} already holds results taken with {k}={prev[k]!r}, but this run is "
                    f"{k}={meta[k]!r}. Use a new arm name or remove {out_dir}.")
    prev_path.write_text(json.dumps(meta, indent=1))

    by_gpu = {g: [o for i, o in enumerate(offsets) if gpus[i % len(gpus)] == g] for g in gpus}

    def run_group(gpu):
        return {o: run_offset(args, o, gpu) for o in by_gpu[gpu]}

    results = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for part in pool.map(run_group, gpus):
            results |= part

    # the summary is the arm's record, not this invocation's: every offset with a result on disk is carried
    for o in OFFSETS:
        if o in results:
            continue
        sr = parse_sr(out_dir / f"O{o}.txt")
        if sr is not None:
            print(f"O{o}: carrying {sr} from a previous run of this arm")
            results[o] = sr
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset", "n", "success_rate"])
        for o in sorted(results):
            w.writerow([o, n_tasks(args, o), results[o] if results[o] is not None else "FAILED"])
        ok = [r for r in results.values() if r is not None]
        if ok:
            w.writerow(["macro", "", round(sum(ok) / len(ok), 4)])

    print(f"=== {args.arm} COMPLETE ===")
    print((out_dir / "summary.csv").read_text())


if __name__ == "__main__":
    main()
