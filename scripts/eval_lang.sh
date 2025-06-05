#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

MODEL=$1
LANGUAGE=$2
PORT="${3:-8000}"
OTHER_ARGS=$4
PP_SIZE="${5:-1}"
# Use PP_SIZE 2 for >32B Models
TP_SIZE="${6:-1}"
# Use TP_SIZE 4 for >32B Models

TASK_LIST=(
    mgsm_eng
    mgsm_$LANGUAGE
    # sea_eval_cross_mmlu_$LANGUAGE
    # sea_eval_cross_logiqa_$LANGUAGE
    # tydiqa_goldp_$LANGUAGE
)

PROMPT_LANG_LIST=(
    eng_reason
    eng_reason_in_${LANGUAGE}
    ${LANGUAGE}_reason
)

PROMPT_VARIATION_LIST=(
    A
    B
    C
)

PROMPT_ANSWER_LIST=(
    box
)

PROMPT_PLACEMENT_LIST=(
    "before"
    "after"
)

MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

for TASK in ${TASK_LIST[@]}
do
    for P_LANG in ${PROMPT_LANG_LIST[@]}
    do
        for P_VAR in ${PROMPT_VARIATION_LIST[@]}
        do
            for P_ANS in ${PROMPT_ANSWER_LIST[@]}
            do
                for P_PLC in ${PROMPT_PLACEMENT_LIST[@]}
                do
                    PROMPT=${P_LANG}_${P_VAR}_${P_ANS}_${P_PLC}
                    # for N in {1..10}
                    for N in 1
                    do
                    yeval \
                        --model $MODEL \
                        --sample_args "temperature=0.6,top_p=0.9" \
                        --task "${TASK}t//${PROMPT}" \
                        --include_path lang_boot/tasks/ \
                        --api_base "http://localhost:${PORT}/v1" \
                        --run_name $MODEL:$TASK:$P_LANG:$P_VAR:$P_ANS:$P_PLC:$N \
                        --trust_remote_code \
                        --output_path ./eval_scores/ $OTHER_ARGS
                    done
                done
            done
        done
    done
done
pkill vllm
sleep 2m
