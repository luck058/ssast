#!/bin/bash
#SBATCH --job-name="ssast-librispeech100-blank"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=damnii[07-12],landonia[01-08,21-25]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=./slurm_log/log_%j.txt

set -x
. /home/htang2/toolchain-20251006/toolchain.rc
source ../../../venvssast/bin/activate
export LD_LIBRARY_PATH=""
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp
mkdir -p ./slurm_log

# --- CONFIGURATION ---
SOURCE_TAR=~/ssast/src/prep_data/librispeech/train-clean-100.tar.gz

SCRATCH_ROOT="/disk/scratch/${USER}/librispeech100"

echo "Source: $SOURCE_TAR"
echo "Scratch Target: $SCRATCH_ROOT"

# --- DATA PREPARATION ---

if [ -d "$SCRATCH_ROOT/LibriSpeech" ]; then
    echo "Data directory $SCRATCH_ROOT/LibriSpeech already exists. Skipping unzip."
else
    echo "Unzipping dataset to Scratch..."
    mkdir -p $SCRATCH_ROOT
    if [ -f "$SOURCE_TAR" ]; then
        tar -xzf "$SOURCE_TAR" -C "$SCRATCH_ROOT"
        echo "Unzip complete."
    else
        echo "Error: Source file $SOURCE_TAR not found!"
        exit 1
    fi
fi

echo "Generating JSON files and Label CSV..."
python prep_ls.py --dataset_path "$SCRATCH_ROOT/LibriSpeech"

# --- MODEL TRAINING ---

dataset=librispeech
dataset_mean=-4.2677393
dataset_std=4.5689974

target_length=1024
noise=False

tr_data=./data/datafiles/librispeech_train_data.json
val_data=./data/datafiles/librispeech_validation_data.json
eval_data=./data/datafiles/librispeech_eval_data.json
label_csv=./data/datafiles/vocab.json

bal=none
lr=1e-4
freqm=0
timem=30
mixup=0
epoch=50
batch_size=24
fshape=128
tshape=2
fstride=10
tstride=1

task=ft_asr
model_size="${1:-tiny}"
head_lr=1

n_class=$(python -c "import json; print(len(json.load(open('$label_csv'))))")
echo "Detected $n_class classes (vocab size)."

exp_dir=./exp/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-blank-${head_lr}-noise${noise}

echo "Starting Training..."

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule True \
--blank_model True \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False \
--loss CTC --metrics wer
