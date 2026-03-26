#!/bin/bash
#SBATCH --job-name="ssast-speechcommandsV2-blank"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=damnii[07-12],landonia[01-08,21-25]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=./slurm_log/log_%j.txt

set -x
. /home/htang2/toolchain-20251006/toolchain.rc
source ../../../venvssast/bin/activate
export LD_LIBRARY_PATH=""
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp
mkdir -p ./slurm_log

AFS_ROOT="/home/s2283874/ssast/datasets/speech_commands_v0.02"
SCRATCH_ROOT="/disk/scratch/${USER}/speech_commands_v0.02"

echo "AFS Path: $AFS_ROOT"
echo "Scratch Path: $SCRATCH_ROOT"

mkdir -p $AFS_ROOT
if [ ! -f "${AFS_ROOT}/speech_commands_v0.02.tar.gz" ]; then
    echo "Downloading dataset to AFS..."
    wget 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz' -O "${AFS_ROOT}/speech_commands_v0.02.tar.gz"
else
    echo "Dataset found on AFS."
fi

echo "Extracting data to Scratch..."
mkdir -p $SCRATCH_ROOT
tar -xzf "${AFS_ROOT}/speech_commands_v0.02.tar.gz" -C "$SCRATCH_ROOT"

echo "Generating JSON files pointing to Scratch..."
rm -rf ./data/datafiles
python prep_sc.py --dataset_path $SCRATCH_ROOT

dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

bal=none
lr=2.5e-4
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
model_size="${1:-tiny}"
head_lr=10

exp_dir=./exp/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-blank-${head_lr}x-noise${noise}

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--blank_model True \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc
