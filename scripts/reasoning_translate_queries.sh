#!/bin/bash
#SBATCH --job-name=langb
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

# Example usage:
# for LANG in de fr es ru th te bn sw ja zh id
# for LANG in de es ja id
# do
# PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
# sbatch lang_boot/scripts/reasoning_translate_queries.sh Qwen/Qwen2.5-7B-Instruct \
#     math_train \
#     ${LANG} \
#     ${PORT}
# done

. ./lang_boot/config/.sft_env

MODEL=$1
TASK=$2
LANG=$3
PORT="${4:-8000}"
OTHER_ARGS=$5
PP_SIZE="${6:-1}"
TP_SIZE="${7:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')

MAX_TOKEN=2048
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task ${TASK}_problemt//${LANG}_translate \
    --include_path lang_boot/tasks/ \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK:$LANG:translated:queries \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS

pkill vllm
sleep 2m
