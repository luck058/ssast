# -*- coding: utf-8 -*-
# prep_ls.py
import os
import glob
import json
import random
import argparse
import torchaudio

def get_transcripts(data_root):
    """
    Parses Librispeech directory structure to find .trans.txt files
    Returns a list of {"wav": path, "text": transcript, "duration": seconds}
    """
    samples = []
    # Librispeech structure: speaker_id/chapter_id/filename.flac
    # Transcripts are in: speaker_id/chapter_id/chapter_id.trans.txt
    
    print(f"Scanning {data_root}...")
    # Walk through speaker folders
    for speaker_id in os.listdir(data_root):
        speaker_path = os.path.join(data_root, speaker_id)
        if not os.path.isdir(speaker_path): continue
            
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path): continue
            
            # Find transcript file
            trans_file = glob.glob(os.path.join(chapter_path, "*.trans.txt"))
            if not trans_file: continue
            trans_file = trans_file[0]
            
            # Parse transcript
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    file_id = parts[0]
                    text = " ".join(parts[1:]).upper() # ASR is typically uppercase
                    
                    wav_path = os.path.join(chapter_path, file_id + ".flac")
                    if os.path.exists(wav_path):
                        # Get duration for filtering
                        wav, sr = torchaudio.load(wav_path)
                        duration = wav.shape[1] / sr
                        
                        samples.append({
                            "wav": wav_path,
                            "text": text,
                            "duration": duration
                        })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--max_duration', type=float, default=15.0, help="Drop audio longer than this (seconds)")
    args = parser.parse_args()

    base_path = args.dataset_path
    if not base_path.endswith('/'): base_path += '/'
    
    # 1. Collect Data
    data_root = os.path.join(base_path, 'train-clean-100')
    if not os.path.exists(data_root):
        # Fallback if user unzipped differently
        data_root = os.path.join(base_path, 'LibriSpeech', 'train-clean-100')
        
    all_samples = get_transcripts(data_root)
    print(f"Total samples found: {len(all_samples)}")
    
    # 2. Drop Long Data
    valid_samples = [s for s in all_samples if s['duration'] <= args.max_duration]
    dropped = len(all_samples) - len(valid_samples)
    print(f"Dropped {dropped} samples ({dropped/len(all_samples)*100:.2f}%) longer than {args.max_duration}s")
    
    # 3. Build Vocabulary (Character Level)
    # ASR requires: <pad>=0, <s>, </s>, <unk>, <blank> (for CTC)
    # Standard CTC Map: 0=Blank, 1..N=Chars
    chars = set()
    for s in valid_samples:
        chars.update(s['text'])
    
    sorted_chars = sorted(list(chars))
    # Create vocab dictionary: 0 is reserved for CTC blank in PyTorch
    # We will use 0=blank, 1=space, 2..N=A-Z
    vocab = {"<blank>": 0, "<space>": 1}
    idx = 2
    for c in sorted_chars:
        if c != " ": # Handle space separately
            vocab[c] = idx
            idx += 1
            
    # Save Vocab
    if not os.path.exists('./data/datafiles'): os.makedirs('./data/datafiles')
    with open('./data/datafiles/vocab.json', 'w') as f:
        json.dump(vocab, f, indent=1)
    print(f"Vocab size: {len(vocab)}. Saved to vocab.json")

    # 4. Split Data (80/10/10)
    random.seed(42)
    random.shuffle(valid_samples)
    n = len(valid_samples)
    n_val = int(n * 0.1)
    n_eval = int(n * 0.1)
    
    splits = {
        'train': valid_samples[:-n_val-n_eval],
        'validation': valid_samples[-n_val-n_eval:-n_eval],
        'eval': valid_samples[-n_eval:]
    }

    # 5. Save Datafiles
    for name, data in splits.items():
        # Format for dataloader
        json_out = []
        for d in data:
            json_out.append({
                "wav": d['wav'],
                "labels": d['text'] # Pass raw text, dataloader tokenizes
            })
            
        with open(f'./data/datafiles/librispeech_{name}_data.json', 'w') as f:
            json.dump({'data': json_out}, f, indent=1)
        print(f"Saved {name}: {len(json_out)} samples")

if __name__ == '__main__':
    main()