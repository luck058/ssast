#!/bin/bash
#SBATCH --job-name="ssast-sid-probe"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --output=./log_sid_probe_%j.txt

set -x
. /home/htang2/toolchain-20251006/toolchain.rc
source ../../../venvssast/bin/activate
export LD_LIBRARY_PATH=""
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp
mkdir -p ./data/datafiles

# --- DATA ---
ZIP_PATH="./datasets/voxceleb1.zip"
SCRATCH_ROOT="/disk/scratch/${USER}/data/voxceleb1"
mkdir -p ${SCRATCH_ROOT}

if [ ! -d "${SCRATCH_ROOT}/dev" ]; then
    echo "Unzipping VoxCeleb1 to ${SCRATCH_ROOT}..."
    unzip -q ${ZIP_PATH} -d ${SCRATCH_ROOT}
else
    echo "Data already exists at ${SCRATCH_ROOT}. Skipping unzip."
fi

echo "Generating VoxCeleb1 JSON files..."
python prep_sid.py --vox1_root ${SCRATCH_ROOT}

# --- TRAINING ---
pretrain_exp=./../../pretrain/exp/
pretrain_model="$1"
pretrain_path=./${pretrain_exp}/${pretrain_model}/models/best_audio_model.pth

if [ ! -f "$pretrain_path" ]; then
    echo "Warning: Pretrained model not found at $pretrain_path"
fi

dataset=voxceleb1
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=300
noise=False

tr_data=./data/datafiles/voxceleb1_train_data.json
val_data=./data/datafiles/voxceleb1_val_data.json
label_csv=./data/datafiles/speakers.csv

bal=none
lr=1e-3
freqm=0
timem=0
mixup=0
epoch=50
batch_size=32
fshape=128
tshape=2
fstride=10
tstride=1

task=probe_sid
model_size="${2:-tiny}"
n_class=1251

exp_dir=./exp/probe-sid-f${fstride}-t${tstride}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_model}

echo "Starting Probe SID Training..."

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \n--data-train ${tr_data} --data-val ${val_data} --exp-dir $exp_dir \n--label-csv ${label_csv} --n_class ${n_class} \n--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \n--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \n--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \n--model_size ${model_size} --adaptschedule True \n--pretrained_mdl_path ${pretrain_path} \n--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \n--num_mel_bins 128 --head_lr 1 --noise ${noise} \n--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False \n--loss CE --metrics acc
