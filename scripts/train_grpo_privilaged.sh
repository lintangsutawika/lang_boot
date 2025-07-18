#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

MODEL_PATH=$1
N_ROLLOUTS=$2
DATA_PATH=$3
SAVE_MODEL_PATH=$4

python -m lang_boot.train_grpo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_PATH}/train.parquet \
    data.val_files=${DATA_PATH}/valid.parquet \
    data.prompt_key=input \
    data.reward_fn_key=input_selected \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    +actor_rollout_ref.rollout.compare=4 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lang_boot' \
    trainer.experiment_name='grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.test_freq=25 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=${SAVE_MODEL_PATH} \
    custom_reward_function.path=lang_boot/lang_boot/reward_functions/reward_fn.py

    # trainer.save_freq=20 \
    # reward_model.model.use_remove_padding=True \



    # reward_model.enable=True \
    # reward_model.model.path=${MODEL_PATH} \
    # reward_model.strategy=fsdp \
    # reward_model.model.fsdp_config.param_offload=True \
    # reward_model.micro_batch_size_per_gpu=${N_ROLLOUTS} \
    # actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    # actor_rollout_ref.model.lora_rank=64 \
    # actor_rollout_ref.model.lora_alpha=32 \