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
# sbatch lang_boot/scripts/reasoning_translate_traces.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     math_train \
#     ${LANG} \
#     data/ \
#     ${PORT}
# done

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
DATA_DIR="${DATA_PATH}/${MODEL_ALIAS}/raw_traces/${TASK}:en:generated:traces/output.jsonl"

MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task json_highest_log_${TASK}t//${LANG}_translate \
    --include_path lang_boot/tasks/ \
    --data_kwargs "{'data_files': '${DATA_DIR}'}" \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK:$LANG:translated:traces \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS

pkill vllm
sleep 2m
