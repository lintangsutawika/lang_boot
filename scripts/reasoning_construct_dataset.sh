#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

# LANG="zho"
# LANG_CODE="zh-cn"
# LANG="jpn"
# LANG_CODE="ja"
# LANG="ind"
# LANG_CODE="id"

TASK=$1
LANG=$2
LANG_CODE=$3
DATA_PATH=$4
MAX_SAMPLES=$5

python lang_boot/lang_boot/construct.py \
    --translate_path ${DATA_PATH}/raw_traces/translated:${TASK}:${LANG}/ \
    --generate_path ${DATA_PATH}/raw_traces/generated:${TASK}:${LANG}/ \
    --output_path ${DATA_PATH}/prep_traces/translated:${TASK}:${LANG}:${MAX_SAMPLES}/ \
    --max_samples ${MAX_SAMPLES} \
    --lang_code ${LANG_CODE}
