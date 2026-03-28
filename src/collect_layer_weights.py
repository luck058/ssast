#!/usr/bin/env python3
"""
Batch-collect learned layer weights from all trained probing and weighted-finetuning
checkpoints and write a single long-format CSV.

Usage:
  python collect_layer_weights.py [-o layer_weights.csv] [--model-size tiny]

Run from src/. Edit the MODELS dict below to match your pretrain experiment names.

Output columns: model, task, protocol, layer_idx, layer_label, weight
"""

import argparse
import csv
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURE: map pretrain experiment directory name -> short label for CSV
# ---------------------------------------------------------------------------
MODELS = {
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb0.0-mpg0.0-mpc1.0":  "mpc",
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb0.0-mpg1.0-mpc0.0":  "mpg",
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb0.0-mpg10.0-mpc1.0": "mpg_mpc",    # MPG+MPC 10:1
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb1.0-mpg0.0-mpc0.0":  "clp",
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb1.0-mpg0.0-mpc1.0":  "clp_mpc",
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb1.0-mpg1.0-mpc0.0":  "clp_mpg",    # CLP+MPG 1:1
    "mask03-tiny-f128-t2-b32-lr1e-4-m250-e75-pretrain_mpmhb-librispeech360-mpmhb1.0-mpg1.0-mpc1.0":  "all",        # MPG+MPC+CLP 1:1:1
}
# ---------------------------------------------------------------------------

# Base path for finetune directories, relative to this script (i.e. src/)
FINETUNE_ROOT = os.path.join(os.path.dirname(__file__), "finetune")

# Task/protocol configurations.
# 'pattern' uses {ms} = model_size, {model} = pretrain dir name.
# 'folds' = True means ESC-50-style fold subdirectories (fold1..fold5).
TASKS = [
    {
        "task": "esc50",
        "protocol": "probe",
        "base_dir": os.path.join(FINETUNE_ROOT, "esc50"),
        "pattern": "test01-esc50-f10-t1-b48-lr1e-3-probe_cls-{ms}-{model}-1-noiseTrue",
        "folds": True,
    },
    {
        "task": "esc50",
        "protocol": "wft",
        "base_dir": os.path.join(FINETUNE_ROOT, "esc50"),
        "pattern": "test01-esc50-f10-t1-b48-lr1e-4-ft_avgtok_weighted-{ms}-{model}-1-noiseTrue",
        "folds": True,
    },
    {
        "task": "sc",
        "protocol": "probe",
        "base_dir": os.path.join(FINETUNE_ROOT, "speechcommands_v2_35"),
        "pattern": "test01-speechcommands-f128-t1-b128-lr1e-3-probe_cls-{ms}-{model}-1x-noiseTrue",
        "folds": False,
    },
    {
        "task": "sc",
        "protocol": "wft",
        "base_dir": os.path.join(FINETUNE_ROOT, "speechcommands_v2_35"),
        "pattern": "test01-speechcommands-f128-t1-b128-lr2.5e-4-ft_avgtok_weighted-{ms}-{model}-1x-noiseTrue",
        "folds": False,
    },
    {
        "task": "asr",
        "protocol": "probe",
        "base_dir": os.path.join(FINETUNE_ROOT, "librispeech"),
        "pattern": "probe-asr-librispeech-f10-t1-b24-lr1e-3-probe_asr-{ms}-{model}",
        "folds": False,
    },
    {
        "task": "asr",
        "protocol": "wft",
        "base_dir": os.path.join(FINETUNE_ROOT, "librispeech"),
        "pattern": "test01-librispeech-f10-t1-b24-lr1e-4-ft_asr_weighted-{ms}-{model}-1-noiseFalse",
        "folds": False,
    },
    {
        "task": "pr",
        "protocol": "probe",
        "base_dir": os.path.join(FINETUNE_ROOT, "librispeech"),
        "pattern": "probe-pr-librispeech-f10-t1-b24-lr1e-3-probe_pr-{ms}-{model}",
        "folds": False,
    },
    {
        "task": "pr",
        "protocol": "wft",
        "base_dir": os.path.join(FINETUNE_ROOT, "librispeech"),
        "pattern": "test01-librispeech-f10-t1-b24-lr1e-4-ft_pr_weighted-{ms}-{model}-1-noiseFalse",
        "folds": False,
    },
]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def load_layer_weights(pth_path):
    """Load raw layer_weights from a ProbingModel checkpoint. Returns numpy array."""
    import torch
    sd = torch.load(pth_path, map_location="cpu")
    key = "module.layer_weights" if "module.layer_weights" in sd else "layer_weights"
    if key not in sd:
        raise KeyError(f"layer_weights not found in {pth_path}")
    return sd[key].float().numpy()


def collect_weights(model_dir, task_cfg, model_size):
    """
    Returns softmax-normalised mean weights (numpy array) for one model/task/protocol,
    or None if no checkpoints were found.
    """
    exp_dir = task_cfg["pattern"].format(ms=model_size, model=model_dir)
    exp_path = os.path.join(task_cfg["base_dir"], "exp", exp_dir)

    if task_cfg["folds"]:
        pth_paths = []
        for fold in range(1, 6):
            p = os.path.join(exp_path, f"fold{fold}", "models", "best_audio_model.pth")
            if os.path.isfile(p):
                pth_paths.append(p)
        if not pth_paths:
            return None, exp_path
        all_softmax = [softmax(load_layer_weights(p)) for p in pth_paths]
    else:
        p = os.path.join(exp_path, "models", "best_audio_model.pth")
        if not os.path.isfile(p):
            return None, exp_path
        all_softmax = [softmax(load_layer_weights(p))]

    mean_w = np.mean(all_softmax, axis=0)
    mean_w /= mean_w.sum()
    return mean_w, exp_path


def main():
    parser = argparse.ArgumentParser(description="Batch-collect layer weights from probe/WFT checkpoints.")
    parser.add_argument("--output", "-o", default="layer_weights.csv", help="Output CSV path (default: layer_weights.csv)")
    parser.add_argument("--model-size", default="tiny", help="Model size string used in exp dir names (default: tiny)")
    args = parser.parse_args()

    if not MODELS:
        print("ERROR: MODELS dict is empty. Edit the MODELS dict at the top of this script.", file=sys.stderr)
        sys.exit(1)

    n_layers = 13
    layer_labels = ["patch_embed"] + [f"transformer_{i}" for i in range(1, n_layers)]

    rows = []
    missing = []

    for model_dir, label in MODELS.items():
        for cfg in TASKS:
            weights, exp_path = collect_weights(model_dir, cfg, args.model_size)
            if weights is None:
                missing.append(f"  MISSING [{label}] {cfg['task']}/{cfg['protocol']}: {exp_path}")
                continue
            for idx, (layer_label, w) in enumerate(zip(layer_labels, weights)):
                rows.append({
                    "model": label,
                    "task": cfg["task"],
                    "protocol": cfg["protocol"],
                    "layer_idx": idx,
                    "layer_label": layer_label,
                    "weight": round(float(w), 6),
                })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "protocol", "layer_idx", "layer_label", "weight"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(rows) // n_layers} combinations) to {args.output}")

    if missing:
        print(f"\n{len(missing)} missing checkpoint(s) skipped:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)


if __name__ == "__main__":
    main()
