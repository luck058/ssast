# -*- coding: utf-8 -*-
# prep_pr.py — Phoneme Recognition data preparation for LibriSpeech 100h
# Converts text transcriptions to ARPAbet phoneme sequences using g2p-en.
# Produces JSON files in the same format as prep_ls.py, plus phonemes.json vocab.

import os
import glob
import json
import random
import argparse
import torchaudio

# ARPAbet phoneme set (39 phones) + stress markers stripped
ARPABET_PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

def build_vocab():
    """Build phoneme vocabulary: 0=blank, 1=space, 2..N=phones"""
    vocab = {"<blank>": 0, "<space>": 1}
    for i, ph in enumerate(ARPABET_PHONES):
        vocab[ph] = i + 2
    return vocab

def text_to_phonemes(text, g2p):
    """Convert text to a space-separated sequence of ARPAbet phones."""
    phones = g2p(text)
    # Strip stress digits (AH0 -> AH), filter non-phone tokens
    cleaned = []
    for ph in phones:
        if ph == " ":
            cleaned.append("<space>")
        else:
            ph_stripped = ph.rstrip("012")
            if ph_stripped in ARPABET_PHONES:
                cleaned.append(ph_stripped)
    return cleaned

def get_transcripts(data_root):
    samples = []
    print(f"Scanning {data_root}...")
    for speaker_id in os.listdir(data_root):
        speaker_path = os.path.join(data_root, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            trans_file = glob.glob(os.path.join(chapter_path, "*.trans.txt"))
            if not trans_file:
                continue
            trans_file = trans_file[0]
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    file_id = parts[0]
                    text = " ".join(parts[1:]).upper()
                    wav_path = os.path.join(chapter_path, file_id + ".flac")
                    if os.path.exists(wav_path):
                        wav, sr = torchaudio.load(wav_path)
                        duration = wav.shape[1] / sr
                        samples.append({"wav": wav_path, "text": text, "duration": duration})
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_duration", type=float, default=15.0)
    args = parser.parse_args()

    try:
        from g2p_en import G2p
        g2p = G2p()
        print("Using g2p-en for G2P conversion.")
    except ImportError:
        raise ImportError("Please install g2p-en: pip install g2p-en")

    base_path = args.dataset_path
    if not base_path.endswith("/"):
        base_path += "/"

    data_root = os.path.join(base_path, "train-clean-100")
    if not os.path.exists(data_root):
        data_root = os.path.join(base_path, "LibriSpeech", "train-clean-100")

    all_samples = get_transcripts(data_root)
    print(f"Total samples found: {len(all_samples)}")

    valid_samples = [s for s in all_samples if s["duration"] <= args.max_duration]
    print(f"Kept {len(valid_samples)} samples (dropped {len(all_samples)-len(valid_samples)} long ones)")

    # Build vocab
    vocab = build_vocab()
    if not os.path.exists("./data/datafiles"):
        os.makedirs("./data/datafiles")
    with open("./data/datafiles/phonemes.json", "w") as f:
        json.dump(vocab, f, indent=1)
    print(f"Phoneme vocab size: {len(vocab)}. Saved to phonemes.json")

    # Convert text to phoneme sequences
    print("Converting transcripts to phoneme sequences (this may take a few minutes)...")
    for s in valid_samples:
        phones = text_to_phonemes(s["text"], g2p)
        s["labels"] = " ".join(phones)

    # Split 80/10/10
    random.seed(42)
    random.shuffle(valid_samples)
    n = len(valid_samples)
    n_val = int(n * 0.1)
    n_eval = int(n * 0.1)
    splits = {
        "train": valid_samples[: -n_val - n_eval],
        "validation": valid_samples[-n_val - n_eval : -n_eval],
        "eval": valid_samples[-n_eval:],
    }

    for name, data in splits.items():
        json_out = [{"wav": d["wav"], "labels": d["labels"]} for d in data]
        out_path = f"./data/datafiles/librispeech_{name}_data_pr.json"
        with open(out_path, "w") as f:
            json.dump({"data": json_out}, f, indent=1)
        print(f"Saved {name}: {len(json_out)} samples -> {out_path}")

if __name__ == "__main__":
    main()
