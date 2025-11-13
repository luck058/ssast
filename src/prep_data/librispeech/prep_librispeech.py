# -*- coding: utf-8 -*-
# @Time    : 7/11/21 6:55 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_librispeech.py

# prepare librispeech data for ssl pretraining

import os,torchaudio,pickle,json,time
import argparse

def walk(path, name, testset_perc=None, save_train_set=True):
    """Walk through the dataset directory and create a JSON file listing all audio files.

    Args:
        path (str): path to the dataset 
        name (str): path to save the json file
        testset_perc (float, optional): Percentage of data to use as test set. If None, saves all as training data. Defaults to None.
        save_train_set (bool, optional): Whether to save the training set. Only used if testset_perc is not None. Defaults to True.
    """
    sample_cnt = 0
    pathdata = os.walk(path)
    wav_list = []
    begin_time = time.time()
    for root, dirs, files in pathdata:
        for file in files:
            if file.endswith('.flac'):
                sample_cnt += 1

                cur_path = root + os.sep + file
                # give a dummy label of 'speech' ('/m/09x0r' in AudioSet label ontology) to all librispeech samples
                # the label is not used in the pretraining, it is just to make the dataloader.py satisfy.
                cur_dict = {"wav": cur_path, "labels": '/m/09x0r'}
                wav_list.append(cur_dict)

                if sample_cnt % 1000 == 0:
                    end_time = time.time()
                    print('find {:d}k .wav files, time eclipse: {:.1f} seconds.'.format(int(sample_cnt/1000), end_time-begin_time))
                    begin_time = end_time
                if sample_cnt % 1e4 == 0:
                    with open(name + '.json', 'w') as f:
                        json.dump({'data': wav_list}, f, indent=1)
                    print('file saved.')
                    
    # Split into train and test if requested
    if testset_perc is not None:
        # Remove last n% of data for test set
        split_idx = int(len(wav_list) * (1 - testset_perc))  # e.g., 90% train, 10% test
        
        if save_train_set:
            # Save training set
            train_list = wav_list[:split_idx]
            with open(name + '.json', 'w') as f:
                json.dump({'data': train_list}, f, indent=1)
            print(f'Training set saved with {len(train_list)} samples.')
        
        # Save test set
        test_list = wav_list[split_idx:]
        test_name = name + '_test'
        with open(test_name + '.json', 'w') as f:
            json.dump({'data': test_list}, f, indent=1)
        print(f'Test set saved with {len(test_list)} samples to {test_name}.json')
    # Don't split, just save all data
    else:
        with open(name + '.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)


# combine json files
def combine_json(file_list, name='librispeech_tr960'):
    wav_list = []
    for file in file_list:
        with open(file + '.json', 'r') as f:
            cur_json = json.load(f)
        wav_list = wav_list + cur_json['data']
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare LibriSpeech data for SSL pretraining')
    parser.add_argument('--testset_perc', type=float, default=None, 
                        help='Percentage of data to use as test set (e.g., 0.1 for 10%%)')
    args = parser.parse_args()

    # Train set
    librispeech360_path = '~/../../disk/scratch/s2283874/librispeech/train-clean-360'
    save_loc = '~/../../disk/scratch/s2283874/librispeech/librispeech_tr360_cut'
    walk(librispeech360_path, save_loc, testset_perc=None)
    
    # Evaluation set
    librispeech100_path = '~/../../disk/scratch/s2283874/librispeech/train-clean-100'
    save_loc = '~/../../disk/scratch/s2283874/librispeech/librispeech_tr100_cut'
    walk(librispeech100_path, save_loc, testset_perc=0.36, save_training=False)
    
    # librispeech100_path = '/data/sls/scratch/yuangong/l2speak/data/librispeech/LibriSpeech/train-clean-100/'
    # walk(librispeech100_path, 'librispeech_tr100_cut')

    # librispeech100_path = '/data/sls/scratch/yuangong/l2speak/data/librispeech/LibriSpeech/train-clean-360/'
    # walk(librispeech100_path, 'librispeech_tr360_cut')
    
    # librispeech100_path = '/data/sls/scratch/yuangong/l2speak/data/librispeech/LibriSpeech/train-other-500/'
    # walk(librispeech100_path, 'librispeech_tr500_cut')
    
    # combine_json(['librispeech_tr500_cut', 'librispeech_tr360_cut', 'librispeech_tr100_cut'], name='librispeech_tr960_cut')
