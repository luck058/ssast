#!/bin/bash
#SBATCH --job-name=esc50_result
#SBATCH --output=esc50_result_%j.log
#SBATCH --error=esc50_result_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB

set -x
. /home/htang2/toolchain-20251006/toolchain.rc
source ~/ssast/venvssast/bin/activate

EXP_PATH=${1:-.}

python src/finetune/esc50/get_esc_result.py --exp_path "$EXP_PATH"
