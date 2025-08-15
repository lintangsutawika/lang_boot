#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=babel-9-3,babel-4-25,babel-14-29,babel-12-9,babel-13-1,babel-7-1

. ./lang_boot/config/.env

# babel-6-5
# sbatch lang_boot/scripts/train_grpo_baseline_gcs_new.sh \
#   -m Qwen/Qwen2.5-7B-Instruct \
#   -l ja \
#   -t gsm8k_train \
#   -d data/ \
#   -s /scratch/lsutawik/checkpoints/ \
#   -f compute_score_no_lang_no_penalty

while getopts ":m:l:n:t:d:s:f:r:v:g:e:j:p:w:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FUNCTION_NAME=$OPTARG;;
    r ) RUN_LABEL=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
    e ) SOURCE_TYPE=$OPTARG;;
    j ) USE_JUDGE=$OPTARG;;
    p ) USE_PRIVILEGED=$OPTARG;;
    w ) USE_REWARD_FN=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
# Get number of GPUs available
USE_JUDGE="${USE_JUDGE:-False}"
USE_PRIVILEGED="${USE_PRIVILEGED:-False}"
USE_REWARD_FN="${USE_REWARD_FN:-False}"
NUM_GPUS=$(nvidia-smi -L | wc -l)
USE_GCS="${USE_GCS:-False}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
RUN_NUMBER="${RUN_NUMBER:-0}"
FUNCTION_NAME="${FUNCTION_NAME:-compute_score}"
MAX_QUERY_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
RUN_LABEL="${RUN_LABEL:-grpo}"
SOURCE_TYPE="${SOURCE_TYPE:-generated}"
RUN_NAME=${RUN_LABEL}:${MODEL_ALIAS}:${TASK}:${LANGUAGE}:${RUN_NUMBER}
FULL_DATA_PATH=${DATA_PATH}${MODEL_ALIAS}/prep_traces/${TASK}:${LANGUAGE}:${SOURCE_TYPE}/
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
LOGPROB_BS=32
PPO_BS=16

echo $RUN_NAME

python -m lang_boot.main_grpo \
    +trainer.lang_code=${LANGUAGE} \
    +trainer.task=${TASK} \
    +trainer.use_gcs=${USE_GCS} \
    +trainer.gcs_project=${GCS_PROJECT} \
    +trainer.gcs_token=${GCS_TOKEN} \
    +trainer.gcs_path=${GCS_PATH}${RUN_NAME} \
    +trainer.use_judge=${USE_JUDGE} \
    +trainer.use_reward_fn=${USE_REWARD_FN} \
    +trainer.use_privileged=${USE_PRIVILEGED} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${FULL_DATA_PATH}/train.parquet \
    data.val_files=${FULL_DATA_PATH}/test.parquet \
    data.prompt_key=input \
    data.train_batch_size=64 \
    data.max_prompt_length=${MAX_QUERY_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_BS} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    +actor_rollout_ref.rollout.compare=${N_ROLLOUTS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lbr-lang_boot' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=255 \
    trainer.default_local_dir=${FULL_SAVE_PATH}/checkpoints/ \
    trainer.validation_data_dir=${FULL_SAVE_PATH}/evaluations/ \
    custom_reward_function.path=lang_boot/lang_boot/reward_functions/reward_fn.py \
    custom_reward_function.name=${FUNCTION_NAME}
