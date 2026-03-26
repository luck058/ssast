#!/usr/bin/env python3
"""
Extract and display learned layer weights from trained ProbingModel checkpoints.

Each checkpoint's raw logits are softmax-normalised individually, then averaged
across checkpoints (useful for ESC-50 which has 5 folds).

Usage examples:
  # Single checkpoint (ASR, PR, KS2)
  python extract_layer_weights.py path/to/exp/models/best_audio_model.pth

  # Multiple folds (ESC-50) — weights are averaged across folds
  python extract_layer_weights.py path/to/exp/fold{1..5}/models/best_audio_model.pth

  # Save to CSV
  python extract_layer_weights.py path/to/exp/models/best_audio_model.pth -o weights.csv

Layer index 0 = patch embedding output; indices 1-12 = transformer block outputs.
"""

import argparse
import csv
import sys
import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def load_weights(path):
    try:
        import torch
        sd = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if 'layer_weights' not in sd:
        print(f"Error: 'layer_weights' key not found in {path}.", file=sys.stderr)
        print(f"Available keys: {[k for k in sd.keys() if 'weight' in k.lower()]}", file=sys.stderr)
        sys.exit(1)

    return sd['layer_weights'].float().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Extract learned probing layer weights from ProbingModel checkpoints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'checkpoints', nargs='+',
        help='Path(s) to best_audio_model.pth. Multiple paths are averaged (e.g. ESC-50 folds).',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Optional path to save results as CSV.',
    )
    args = parser.parse_args()

    # Softmax-normalise each checkpoint individually, then average
    all_softmax = []
    for path in args.checkpoints:
        raw = load_weights(path)
        all_softmax.append(softmax(raw))

    mean_weights = np.mean(all_softmax, axis=0)
    mean_weights /= mean_weights.sum()  # renormalise after averaging

    n_layers = len(mean_weights)
    labels = ['patch_embed'] + [f'transformer_{i}' for i in range(1, n_layers)]

    # Print table
    n = len(args.checkpoints)
    print(f"\nLayer weights — averaged over {n} checkpoint(s), softmax-normalised:")
    print(f"{'Idx':<6} {'Layer':<16} {'Weight':>10} {'Weight %':>10}")
    print('-' * 44)
    for i, (label, w) in enumerate(zip(labels, mean_weights)):
        print(f"{i:<6} {label:<16} {w:>10.4f} {w * 100:>9.2f}%")
    print('-' * 44)
    print(f"{'Total':<22} {mean_weights.sum():>10.4f} {mean_weights.sum() * 100:>9.2f}%")

    if args.output:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_idx', 'label', 'weight'])
            for i, (label, w) in enumerate(zip(labels, mean_weights)):
                writer.writerow([i, label, round(float(w), 6)])
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
