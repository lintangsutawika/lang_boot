#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

# open_r1_math_220k
# gsm8k

MODEL_PATH=$1
TASK=$2
N_SAMPLES=$3
N_TOKENS=$4
SAVE_PATH=$5
PORT=$6
MASTER_PORT=$7

python -m lang_boot.main \
    --task_name ${TASK} \
    --model_name ${MODEL_PATH} \
    --output_path ${SAVE_PATH}/response/ \
    --save_model_path ${SAVE_PATH} \
    --task_path lang_boot/tasks/ \
    --lang eng \
    --api_base http://localhost:${PORT}/v1/ \
    --sample_args "temperature=1.0,top_p=0.9" \
    --n_samples $N_SAMPLES \
    --max_iteration 1 \
    --budget $N_TOKENS \
    --serve \
    --sft \
    --master_port ${MASTER_PORT} \
    --no_sampling \
    --training_data_path ${SAVE_PATH}/response/gsm8k:eng:ori/
