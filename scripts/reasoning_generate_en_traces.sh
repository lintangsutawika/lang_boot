#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# Example usage:
# PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
# sbatch lang_boot/scripts/reasoning_generate_en_traces.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     gsm8k_train ${PORT}

. ./lang_boot/config/.env

MODEL=$1
TASK=$2
PORT="${3:-8000}"
OTHER_ARGS=$4
PP_SIZE="${5:-1}"
TP_SIZE="${6:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')

MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task "${TASK}t//en_measure" \
    --include_path lang_boot/tasks/ \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK:en:generated:traces \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS
pkill vllm
sleep 2m
