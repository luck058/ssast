# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        # Check if it's a simple vocab json (ASR) or a CSV (Classification)
        if label_csv.endswith('.json'):
            vocab = json.load(f)
            return vocab
            
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
            
        # Determine Task Type based on labels
        # ASR uses 'vocab.json', Classification uses 'class_labels_indices.csv'
        if label_csv and label_csv.endswith('.json'):
            print('ASR task detected. Using vocab from {:s}'.format(label_csv))
            self.task_type = "ASR"
            self.vocab = make_index_dict(label_csv)
            # Ensure space has a unique index
            max_idx = max(self.vocab.values()) if self.vocab else 0
            self.space_idx = self.vocab.get("<space>", max_idx + 2)
        else:
            print('Classification task detected. Using labels from {:s}'.format(label_csv))
            self.task_type = "CLS"
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(self.index_dict)
            print('number of classes is {:d}'.format(self.label_num))
        
        # Initialize Cluster ID Storage for MelHuBERT loss during pretraining
        self.cluster_ids = None 
        self.use_cluster_labels = False
        
    def generate_cluster_labels(self, audio_model, batch_size=64, num_workers=4):
        """
        Runs the entire dataset through the model to generate cluster IDs.
        Stores them in self.cluster_ids for retrieval during training.
        """
        print("Generating cluster labels for the entire dataset (Teacher Pass)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Temporarily disable mixup/noise/SpecAugment to get clean labels for the original audio
        original_mixup = self.mixup
        self.mixup = 0 
        self.noise = False 
        original_freqm = self.freqm
        original_timem = self.timem
        self.freqm = 0
        self.timem = 0
        
        # Create a sequential loader (no shuffle)
        temp_loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        all_ids = []
        
        # Handle DataParallel unwrapping
        if isinstance(audio_model, torch.nn.DataParallel):
            model = audio_model.module
        else:
            model = audio_model
            
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(temp_loader):
                fbank = batch[0]
                fbank = fbank.to(device)
                # Ensure input shape is [B, 1, F, T] for AST
                if fbank.dim() == 3:
                     fbank = fbank.unsqueeze(1).transpose(2, 3)
                
                # Get IDs from model (defined in ASTModel next)
                ids = model.get_cluster_labels(fbank)
                all_ids.append(ids)
                
                if i % 100 == 0:
                    print(f"Labeled batch {i}/{len(temp_loader)}")

        # Concatenate and Store: [Total_Samples, N_Patches]
        self.cluster_ids = torch.cat(all_ids, dim=0)
        self.use_cluster_labels = True
        
        # Restore original augmentation settings
        self.mixup = original_mixup
        self.noise = self.audio_conf.get('noise')
        self.freqm = original_freqm
        self.timem = original_timem
        
        print(f"Finished labeling. Stored IDs shape: {self.cluster_ids.shape}")

    def _wav2fbank(self, filename, filename2=None):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        mix_lambda = 0.0
        if filename2:
            waveform1 = waveform
            waveform2, _ = torchaudio.load(filename2)
            waveform2 = waveform2 - waveform2.mean()
            
            # Match lengths
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[:, 0:waveform1.shape[1]]
                
            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        # Normalize
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
            valid_length = n_frames
        elif p < 0:
            fbank = fbank[0:target_length, :]
            valid_length = target_length
        else:
            valid_length = n_frames

        return fbank, mix_lambda, valid_length

    def text_to_tensor(self, text):
        indices = []
        for c in text:
            if c == " ": indices.append(self.space_idx)
            elif c in self.vocab: indices.append(self.vocab[c])
        # Changed: Updated to modern tensor constructor
        return torch.tensor(indices, dtype=torch.long)
    
    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        
        if self.task_type == "ASR":
            # No Mixup for ASR
            # Changed: Fixed unpacking error (3 values returned)
            fbank, _, valid_len = self._wav2fbank(datum['wav'])
            
            # SpecAugment
            if self.freqm > 0:
                fbank = torchaudio.transforms.FrequencyMasking(self.freqm)(fbank.unsqueeze(0).transpose(1,2)).transpose(1,2).squeeze(0)
            if self.timem > 0:
                fbank = torchaudio.transforms.TimeMasking(self.timem)(fbank.unsqueeze(0).transpose(1,2)).transpose(1,2).squeeze(0)

            # Label Processing
            label_tensor = self.text_to_tensor(datum['labels'])
            
            # Pad Labels for Batching
            max_label_len = 400
            label_len = len(label_tensor)
            if label_len > max_label_len:
                label_tensor = label_tensor[:max_label_len]
                label_len = max_label_len
            padded_label = torch.full((max_label_len,), 0, dtype=torch.long)
            padded_label[:label_len] = label_tensor

            return fbank, padded_label, valid_len, label_len
        
        # Classification Task with optional Mixup
        else:
            # do mix-up for this sample (controlled by the given mixup rate)
            if random.random() < self.mixup:
                datum = self.data[index]
                # find another sample to mix, also do balance sampling
                # sample the other sample from the multinomial distribution, will make the performance worse
                # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
                # sample the other sample from the uniform distribution
                mix_sample_idx = random.randint(0, len(self.data)-1)
                mix_datum = self.data[mix_sample_idx]
                # get the mixed fbank
                # Changed: Fixed unpacking error (3 values returned)
                fbank, mix_lambda, _ = self._wav2fbank(datum['wav'], mix_datum['wav'])
                # initialize the label
                label_indices = np.zeros(self.label_num)
                # add sample 1 labels
                for label_str in datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] += mix_lambda
                # add sample 2 labels
                for label_str in mix_datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
                # Changed: Updated to modern tensor constructor
                label_indices = torch.tensor(label_indices, dtype=torch.float32)
            # if not do mixup
            else:
                datum = self.data[index]
                label_indices = np.zeros(self.label_num)
                # Changed: Fixed unpacking error (3 values returned)
                fbank, mix_lambda, _ = self._wav2fbank(datum['wav'])
                for label_str in datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] = 1.0

                # Changed: Updated to modern tensor constructor
                label_indices = torch.tensor(label_indices, dtype=torch.float32)

            # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            # this is just to satisfy new torchaudio version.
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            # squeeze back
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

            # normalize the input for both training and test
            # NOTE: normalization is already applied inside _wav2fbank(),
            # so we do NOT normalize again here.

            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                if not self.use_cluster_labels:
                    # Only apply roll when not using cluster labels to ensure consistency as cluster labels are precomputed
                    fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
                else:
                    print("Warning: Noise augmentation is enabled but cluster labels are being used. Disabled torch.roll but keeps additive noise.")
                
            # Get pre-computed cluster ID if available
            cluster_target = -1 # Default placeholder
            if self.use_cluster_labels and self.cluster_ids is not None:
                # We map the index directly. Note: if mixup is active, this ID corresponds 
                # to the primary sample. Standard HuBERT does not mixup targets.
                cluster_target = self.cluster_ids[index]
                if self.mixup > 0:
                    print("Warning: Mixup is enabled but cluster targets correspond only to primary samples.")

            # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            return fbank, label_indices, cluster_target

    def __len__(self):
        return len(self.data)