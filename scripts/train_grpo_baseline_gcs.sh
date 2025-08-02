#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --exclude=babel-9-3,babel-4-25,babel-14-29,babel-12-9

. ./lang_boot/config/.env

# babel-6-5
# sbatch lang_boot/scripts/train_grpo_baseline.sh \
#   -m Qwen/Qwen2.5-7B-Instruct \
#   -l id \
#   -t gsm8k_train \
#   -d data/ \
#   -s /project/flame/lsutawik/03-lang-rl/checkpoint/

while getopts ":m:l:n:t:d:s:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
# Get number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
N_ROLLOUTS="${N_ROLLOUTS:-16}"

RUN_NAME=grpo-baseline:${MODEL_ALIAS}:${TASK}:${LANGUAGE}
FULL_DATA_PATH=${DATA_PATH}${MODEL_ALIAS}/prep_traces/${TASK}:${LANGUAGE}:generated:-1/
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
LOGPROB_BS=16
PPO_BS=16

python -m lang_boot.main_grpo \
    +trainer.privileged=False \
    +trainer.use_gcs=True \
    +trainer.gcs_project=${GCS_PROJECT} \
    +trainer.gcs_token=${GCS_TOKEN} \
    +trainer.gcs_path=${GCS_PATH}${RUN_NAME} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${FULL_DATA_PATH}/train.parquet \
    data.val_files=${FULL_DATA_PATH}/valid.parquet \
    data.prompt_key=input \
    data.reward_fn_key=input_selected \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_BS} \
    actor_rollout_ref.actor.use_kl_loss=True \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lang_boot' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir=${FULL_SAVE_PATH} \
    custom_reward_function.path=lang_boot/lang_boot/reward_functions/reward_fn.py
