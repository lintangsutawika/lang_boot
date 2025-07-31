#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

while getopts ":s:m:l:r:o:p:t:e:" opt; do
  case ${opt} in
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    # \? ) echo "Usage: cmd [-p] [-m] [-l] [-o] [-pp] [-tp]";;
  esac
done

PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

TASK_LIST=(
    mgsm_$LANGUAGE
)

PROMPT_LANG_LIST=(
    ${LANGUAGE}_measure
)

MAX_TOKEN=2048
vllm serve ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

for TASK in ${TASK_LIST[@]}
do
    for PROMPT in ${PROMPT_LANG_LIST[@]}
    do
        # for N in {1..10}
        for N in 1
        do
            yeval \
                --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
                --sample_args n=16,temperature=1.0,logprobs=True \
                --task "${TASK}t//${PROMPT}" \
                --include_path lang_boot/tasks/ \
                --api_base "http://localhost:${PORT}/v1" \
                --run_name $MODEL:$TASK \
                --trust_remote_code \
                --output_path ./eval_scores/ $OTHER_ARGS
        done
    done
done
pkill vllm
sleep 2m

