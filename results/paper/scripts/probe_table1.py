#!/usr/bin/env python3
"""Reproduce the Table 1 grouped linear probes from the recorded checkpoints.

Checkpoint binaries are not shipped. Pass roots containing the run directories
named below. This script deliberately records only relative checkpoint names and
content hashes in its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from diagnostics.probe_heads import grouped_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_val_3env import features  # noqa: E402

N_FRAMES = 5000
SEED = 0
SAMPLE_ROWS_SHA256 = "296f43ca76b58f32ca1d4be538ee6068cf27e8536d714778ab0a2e24c3495132"
EPISODE_GROUPS_SHA256 = "75c0d2098167c9f1065909e96033f283af1d036ea7bed89ff85f82f4657cbbbd"
TARGETS = {
    "cube": ("qpos", slice(14, 17)),
    "arm": ("qpos", slice(0, 7)),
    "drawer": ("qpos", slice(23, 24)),
    "window": ("qpos", slice(24, 25)),
    "buttons": ("button_states", slice(None)),
}
RUNS = [
    ("Underspecified", "Temporal", "scene-mixlate-lag-2", "step_snaps/step_33000.pt", 33000),
    ("Underspecified", "Quasimetric", "scene-mixlate-2", "step_snaps/step_33000.pt", 33000),
    ("Underspecified", "State", "scene-kab-within-state-buttons-s2", "step_snaps/step_33000.pt", 33000),
    ("Underspecified", "Proprioception", "scene-kab-within-proprio-s2", "step_snaps/step_33000.pt", 33000),
    ("Underspecified", "Depth", "scene-kab-within-depth-s2", "step_snaps/step_33000.pt", 33000),
    ("Fully specified", "State", "scene-kab-full-state-buttons-pa05-rel-s2", "step_snaps/step_33000.pt", 33000),
    ("Fully specified", "Proprioception", "scene-kab-full-proprio-pa05-rel-s2", "step_snaps/step_33000.pt", 33000),
    ("Fully specified", "Depth", "scene-kab-full-depth-pa05-rel-s2", "step_snaps/step_33000.pt", 33000),
    ("Reference", "LeWM", "scene-lewm-mixlate-2", "weights_epoch_3.pt", 33375),
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def evaluate(name: str, checkpoint: str, pixels: np.ndarray, targets: dict, groups: np.ndarray) -> dict:
    embedding = features(checkpoint, pixels)
    scores = {}
    for target, values in targets.items():
        scores[target] = float(grouped_fit(
            embedding, values, groups, ridge_alpha="cv", n_splits=5, seed0=SEED
        )["r2"])
    scores["scene_average"] = float(np.mean(list(scores.values())))
    print(f"{name}: " + ", ".join(f"{key}={value:.6f}" for key, value in scores.items()), flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        rows = np.sort(np.random.default_rng(SEED).choice(len(handle["pixels"]), N_FRAMES, replace=False))
        pixels = handle["pixels"][rows]
        groups = handle["ep_idx"][rows].astype(np.int64)
        targets = {key: handle[column][rows][:, selection].astype(np.float64)
                   for key, (column, selection) in TARGETS.items()}
    if array_sha256(rows) != SAMPLE_ROWS_SHA256 or array_sha256(groups) != EPISODE_GROUPS_SHA256:
        raise RuntimeError("Dataset rows or episode grouping do not match the saved Table 1 sample")

    output = {"rows": []}
    for objective, kernel, run_id, relative, step in RUNS:
        checkpoint = args.checkpoint_root / run_id / relative
        output["rows"].append({
            "objective": objective,
            "kernel": kernel,
            "run_id": run_id,
            "checkpoint": relative,
            "checkpoint_step": step,
            "checkpoint_sha256": file_sha256(checkpoint),
            "r2": evaluate(kernel, str(checkpoint), pixels, targets, groups),
        })
    output["rows"].append({
        "objective": "Reference",
        "kernel": "Untrained encoder",
        "run_id": "random-init-seed-0",
        "checkpoint": None,
        "checkpoint_step": 0,
        "checkpoint_sha256": None,
        "r2": evaluate("Untrained encoder", "random-init", pixels, targets, groups),
    })
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
