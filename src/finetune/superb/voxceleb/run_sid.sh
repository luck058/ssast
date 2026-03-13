#!/bin/bash
#SBATCH --job-name="s3p-sid"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30000
#SBATCH --output=./log_sid_%j.txt

set -x
# --- 1. SETUP ENVIRONMENT ---
source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir -p exp

# --- 2. UNZIP DATA TO SCRATCH ---
# Assumes you zipped vox1_dev_wav and vox1_test_wav into one zip file
ZIP_PATH="../datasets/voxceleb1.zip" 
SCRATCH_ROOT="/disk/scratch/${USER}/data/voxceleb1"
mkdir -p ${SCRATCH_ROOT}

# S3PRL usually expects the root containing 'dev' and 'test' folders
DATA_DIR="${SCRATCH_ROOT}"

if [ ! -d "${DATA_DIR}/dev" ]; then
    echo "Unzipping VoxCeleb1 to ${SCRATCH_ROOT}..."
    unzip -q ${ZIP_PATH} -d ${SCRATCH_ROOT}
else
    echo "Data directory already exists at ${DATA_DIR}. Skipping unzip."
fi

# --- 3. RUN TRAINING ---
mdl=ssast_frame_base_10s
lr=1e-4

expname=sid_${mdl}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

python3 run_downstream.py --expdir $expdir \
    -m train \
    -u $mdl \
    -d voxceleb1 \
    -c config_sid.yaml \
    -s hidden_states \
    -o "config.optimizer.lr=${lr},config.downstream_expert.datarc.root='${DATA_DIR}'" \
    -f