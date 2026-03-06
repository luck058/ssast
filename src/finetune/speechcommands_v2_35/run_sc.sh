#!/bin/bash
#SBATCH --job-name="ssast-speechcommandsV2"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=damnii[07-12],landonia[01-08,21-25]
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
. /home/htang2/toolchain-20251006/toolchain.rc
source ../../../venvssast/bin/activate
export LD_LIBRARY_PATH=""
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp

AFS_ROOT="/home/s2283874/ssast/datasets/speech_commands_v0.02"
# Use SLURM_TMPDIR if available, otherwise a temp path
SCRATCH_ROOT="/disk/scratch/${USER}/speech_commands_v0.02"

echo "AFS Path: $AFS_ROOT"
echo "Scratch Path: $SCRATCH_ROOT"

# Prepare AFS Storage
mkdir -p $AFS_ROOT
if [ ! -f "${AFS_ROOT}/speech_commands_v0.02.tar.gz" ]; then
    echo "Downloading dataset to AFS..."
    wget 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz' -O "${AFS_ROOT}/speech_commands_v0.02.tar.gz"
else
    echo "Dataset found on AFS."
fi

# Copy and Extract to Scratch
echo "Extracting data to Scratch..."
mkdir -p $SCRATCH_ROOT
# Extract directly from AFS file to Scratch folder
tar -xzf "${AFS_ROOT}/speech_commands_v0.02.tar.gz" -C "$SCRATCH_ROOT"

# 4. Prepare Datafiles (Always run this to update paths in JSONs)
# Note: We removed the 'if exists' check because paths change per job
echo "Generating JSON files pointing to Scratch..."
# Clear old datafiles to ensure fresh generation
rm -rf ./data/datafiles 
python prep_sc.py --dataset_path $SCRATCH_ROOT


# if [ -e SSAST-Base-Frame-400.pth ]
# then
#     echo "pretrained model already downloaded."
# else
#     wget https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1 -O SSAST-Base-Frame-400.pth
# fi

pretrain_exp=./../../pretrain/exp/
# pretrain_model=mask01-tiny-f16-t16-b64-lr5e-4-m400-pretrain_mpmhb-librispeech360
pretrain_model="$1"
pretrain_path=./${pretrain_exp}/${pretrain_model}/models/best_audio_model.pth

dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

bal=none
lr=2.5e-5
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=128
fshape=128
tshape=2
fstride=128
tstride=1

task=ft_avgtok
model_size="${2:-tiny}"
head_lr=10

exp_dir=./exp/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-${pretrain_model}-${head_lr}x-noise${noise}

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc
