#!/bin/bash
#SBATCH --job-name="ssast-pretrain-maskpatch-tiny"
#SBATCH --partition=Teach-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
. /home/htang2/toolchain-20251006/toolchain.rc
source ~/ssast/venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir exp
mkdir slurm_log

# Unzip librispeech dataset to scratch disk if not already done
librispeech360_dir=~/../../disk/scratch/s2283874/librispeech/train-clean-360
librispeech100_dir=~/../../disk/scratch/s2283874/librispeech/train-clean-100
if [ ! -d "$librispeech360_dir" ] || [ ! -d "$librispeech100_dir" ]; then
    echo "LibriSpeech directory not found. Extracting archive..."
    tar -xzf ~/ssast/src/prep_data/librispeech/train-clean-360.tar.gz -C ~/../../disk/scratch/s2283874/librispeech/
    tar -xzf ~/ssast/src/prep_data/librispeech/train-clean-100.tar.gz -C ~/../../disk/scratch/s2283874/librispeech/
else
    echo "LibriSpeech directory already exists at $librispeech_dir. Skipping extraction."
fi

# Check if JSON file exists, if not, run prep_librispeech.py
'~/../../disk/scratch/s2283874/librispeech/train-clean-360'
json_file=~/../../disk/scratch/s2283874/librispeech/librispeech_tr360_cut.json
json_test_file=~/../../disk/scratch/s2283874/librispeech/librispeech_tr360_cut_test.json
if [ ! -f "$json_file" ] || [ ! -f "$json_test_file" ]; then
    echo "JSON file not found. Running prep_librispeech.py..."
    python ~/ssast/src/prep_data/librispeech/prep_librispeech.py
else
    echo "JSON file already exists at $json_file. Skipping preparation."
fi

task=pretrain_joint
mask_patch=400

# audioset and librispeech
# dataset=asli
dataset=librispeech360
tr_data=~/../../disk/scratch/s2283874/librispeech/librispeech_tr360_cut.json
te_data=~/../../disk/scratch/s2283874/librispeech/librispeech_tr100_cut_test.json
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128

model_size=tiny
# no patch split overlap
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
batch_size=120
lr=5e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=10
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}

CUDA_CACHE_DISABLE=1 python -W ignore ../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 800