#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:2
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# Example usage:
# bash lang_boot/scripts/reasoning_traces_translate.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     gsm8k_train \
#     ind \
#     data/ \
#     8723 "" 1 2

. ./lang_boot/config/.env

MODEL=$1
TASK=$2
LANG=$3
DATA_PATH=$4
PORT="${5:-8000}"
OTHER_ARGS=$6
PP_SIZE="${7:-1}"
TP_SIZE="${8:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DATA_DIR="${DATA_PATH}/${MODEL_ALIAS}/raw_traces/generated:traces:${TASK}:eng/output.jsonl"

MAX_TOKEN=8192
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task local_json_taskt//${LANG}_translate \
    --include_path lang_boot/tasks/ \
    --data_kwargs "{'data_files': '${DATA_DIR}'}" \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name translated:traces:$TASK:$LANG \
    --sample_args n=50,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS

pkill vllm
sleep 2m
