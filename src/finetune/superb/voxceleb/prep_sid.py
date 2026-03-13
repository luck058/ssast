# -*- coding: utf-8 -*-
# prep_sid.py -- VoxCeleb1 data preparation for Speaker Identification probing.
# Expected structure:
#   <vox1_root>/dev/wav/id<XXXXX>/<session>/<utterance>.wav
#   <vox1_root>/test/wav/id<XXXXX>/<session>/<utterance>.wav

import os
import json
import csv
import argparse
import random


def collect_utterances(split_dir, split_name):
    wav_dir = os.path.join(split_dir, 'wav')
    utterances = []
    if not os.path.isdir(wav_dir):
        print(f'Warning: {wav_dir} not found, skipping {split_name}')
        return utterances
    for speaker_id in sorted(os.listdir(wav_dir)):
        spk_path = os.path.join(wav_dir, speaker_id)
        if not os.path.isdir(spk_path):
            continue
        for session in sorted(os.listdir(spk_path)):
            sess_path = os.path.join(spk_path, session)
            if not os.path.isdir(sess_path):
                continue
            for fname in sorted(os.listdir(sess_path)):
                if fname.endswith('.wav') or fname.endswith('.m4a'):
                    wav_path = os.path.join(sess_path, fname)
                    utterances.append((wav_path, speaker_id))
    return utterances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vox1_root', type=str, required=True,
                        help='Root dir containing dev/ and test/ subdirectories')
    parser.add_argument('--val_fraction', type=float, default=0.1)
    args = parser.parse_args()

    dev_dir = os.path.join(args.vox1_root, 'dev')
    test_dir = os.path.join(args.vox1_root, 'test')

    dev_utts = collect_utterances(dev_dir, 'dev')
    test_utts = collect_utterances(test_dir, 'test')

    if not dev_utts:
        raise RuntimeError(f'No utterances found under {dev_dir}/wav/')

    all_speakers = sorted(set(spk for _, spk in dev_utts + test_utts))
    spk2idx = {spk: idx for idx, spk in enumerate(all_speakers)}
    print(f'Total speakers: {len(all_speakers)}')

    out_dir = './data/datafiles'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'speakers.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['speaker_id', 'index'])
        for spk, idx in spk2idx.items():
            writer.writerow([spk, idx])
    print(f'Saved speakers.csv with {len(all_speakers)} speakers')

    random.seed(42)
    random.shuffle(dev_utts)
    n_val = int(len(dev_utts) * args.val_fraction)
    val_utts = dev_utts[:n_val]
    train_utts = dev_utts[n_val:]

    n_class = len(all_speakers)

    def make_json(utts, name):
        json_out = []
        for wav_path, spk in utts:
            idx = spk2idx[spk]
            one_hot = [0] * n_class
            one_hot[idx] = 1
            json_out.append({'wav': wav_path, 'labels': ','.join(map(str, one_hot))})
        out_path = os.path.join(out_dir, f'voxceleb1_{name}_data.json')
        with open(out_path, 'w') as f:
            json.dump({'data': json_out}, f, indent=1)
        print(f'Saved {name}: {len(json_out)} utterances -> {out_path}')

    make_json(train_utts, 'train')
    make_json(val_utts, 'val')
    make_json(test_utts, 'test')
    print(f'n_class = {n_class} (use --n_class {n_class} in run_sid_probe.sh)')


if __name__ == '__main__':
    main()
