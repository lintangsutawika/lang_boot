#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

MODEL=$1
DATA_PATH=$2
SAVE_MODEL_PATH=$3
EPOCH=("${4:-5}")

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo $DATA_PATH
torchrun \
    --nproc-per-node=$NUM_GPUS \
    --rdzv-endpoint=0.0.0.0:29392 \
    -m verl.trainer.fsdp_sft_trainer \
        optim.lr=1e-5 \
        data.train_files=${DATA_PATH}/train.parquet \
        data.val_files=${DATA_PATH}/valid.parquet \
        data.multiturn.messages_key=messages \
        data.multiturn.enable=True \
        +data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.max_length=2048 \
        data.train_batch_size=16 \
        data.micro_batch_size_per_gpu=4 \
        model.partial_pretrain=${MODEL} \
        model.fsdp_config.model_dtype=bf16 \
        model.fsdp_config.cpu_offload=True \
        model.fsdp_config.offload_params=True \
        model.enable_gradient_checkpointing=True \
        model.use_liger=True \
        model.strategy=fsdp \
        trainer.default_local_dir=${SAVE_MODEL_PATH} \
        trainer.project_name=lang_boot \
        trainer.experiment_name=sft \
        trainer.total_epochs=${EPOCH} \
        trainer.save_freq=20 \
        trainer.total_training_steps=1000 \
        trainer.logger="['console']"

        # trainer.logger="['console', 'wandb']"
        # data.prompt_key=extra_info \
        # data.response_key=extra_info \
        # data.prompt_dict_keys=['question'] \
        # +data.response_dict_keys=['answer'] \

# sbatch lang_boot/scripts/train_sft.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     data/Qwen-Qwen2.5-7B-Instruct/prep_traces/translated:gsm8k_train:ind:1000/ \
#     /data/user_data/lsutawik/01-solution-path-routing/qwen2.5-7b:gsm8k_train:ind:1000:5/ \
#     5

# sbatch lang_boot/scripts/train_sft.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     data/Qwen-Qwen2.5-7B-Instruct/prep_traces/translated:gsm8k_train:ind:5000/ \
#     /data/user_data/lsutawik/01-solution-path-routing/qwen2.5-7b:gsm8k_train:ind:5000:1/ \
#     1

# for LANG in ind jpn zho
# for LANG in ind
# do
#     for ACC in False True
#     do
#         for USE_LANG in False True
#         do
#             DATA_PATH="gsm8k_train:${LANG}:1000:logprob-True:acc-${ACC}:lang-${USE_LANG}"
#             sbatch lang_boot/scripts/train_sft.sh \
#                 Qwen/Qwen2.5-7B-Instruct \
#                 data/Qwen-Qwen2.5-7B-Instruct/prep_traces/translated:${DATA_PATH}/ \
#                 /data/user_data/lsutawik/05-lang-rl/checkpoints/qwen2.5-7b:${DATA_PATH}/ \
#                 5
#         done
#     done
# done

# DATA_PATH="gsm8k_train:ind:1000"
# bash lang_boot/scripts/train_sft.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     data/Qwen-Qwen2.5-7B-Instruct/prep_traces/translated:${DATA_PATH}/ \
#     /data/user_data/lsutawik/05-lang-rl/checkpoints/qwen2.5-7b:${DATA_PATH}/ \
#     5

# DATA_PATH="gsm8k_train:ind:1000:logprob-True:acc-False:lang-False"
# bash lang_boot/scripts/train_sft.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     data/Qwen-Qwen2.5-7B-Instruct/prep_traces/translated:${DATA_PATH}/ \
#     /data/user_data/lsutawik/05-lang-rl/checkpoints/qwen2.5-7b:${DATA_PATH}/ \
#     5
