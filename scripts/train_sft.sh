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

. ./lang_boot/config/.sft_env
# . ./lang_boot/config/.env

# for LANG in id de es ja
# do
#     for SOURCE in generated translated
#     do
    # sbatch lang_boot/scripts/train_sft.sh \
    #     -s ${SOURCE} \
    #     -m Qwen/Qwen2.5-7B-Instruct \
    #     -l ${LANG} \
    #     -t gsm8k_train \
    #     -d data/Qwen-Qwen2.5-7B-Instruct/prep_traces/ \
    #     -x /scratch/lsutawik/checkpoints/ \
    #     -y /data/user_data/lsutawik/05-lang-rl/checkpoints/
#     done
# done

while getopts ":s:m:l:t:d:x:y:" opt; do
  case ${opt} in
    s ) SOURCE=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    x ) TMP_SAVE_PATH=$OPTARG;;
    y ) END_SAVE_PATH=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
RUN_NAME=${SOURCE}:${MODEL_ALIAS}:${TASK}:${LANGUAGE}
TRAIN_DATA_PATH=${DATA_PATH}${TASK}:${LANGUAGE}:${SOURCE}:-1/
TMP_PATH=${TMP_SAVE_PATH}${RUN_NAME}
rm -rf TMP_PATH

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo $DATA_PATH
torchrun \
    --nproc-per-node=$NUM_GPUS \
    --rdzv-endpoint=0.0.0.0:29392 \
    -m verl.trainer.fsdp_sft_trainer \
        optim.lr=1e-5 \
        data.train_files=${TRAIN_DATA_PATH}train.parquet \
        data.val_files=${TRAIN_DATA_PATH}valid.parquet \
        data.prompt_key=input_selected \
        data.response_key=output_selected \
        +data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.max_length=2048 \
        data.train_batch_size=32 \
        data.micro_batch_size_per_gpu=4 \
        model.partial_pretrain=${MODEL} \
        model.fsdp_config.model_dtype=bf16 \
        model.fsdp_config.cpu_offload=True \
        model.fsdp_config.offload_params=True \
        model.enable_gradient_checkpointing=True \
        model.use_liger=True \
        model.strategy=fsdp \
        trainer.default_local_dir=${TMP_PATH} \
        trainer.project_name=lang_boot \
        trainer.experiment_name=sft \
        trainer.save_freq=50 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=500 \
        trainer.logger="['console']"

        # trainer.logger="['console', 'wandb']"
        # data.prompt_key=extra_info \
        # data.response_key=extra_info \
        # data.prompt_dict_keys=['question'] \
        # +data.response_dict_keys=['answer'] \

for STEP in 500 450 400 350 300 250 200 150 100 50
do
    MODEL_STEP=global_step_${STEP}
    PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
    bash lang_boot/scripts/eval_mgsm.sh \
        -s ${TMP_SAVE_PATH} \
        -m ${RUN_NAME}/${MODEL_STEP} \
        -l ${LANGUAGE} \
        -r ${PORT} 
done

mv ${TMP_PATH} ${END_SAVE_PATH}
