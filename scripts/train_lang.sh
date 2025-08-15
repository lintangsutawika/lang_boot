while getopts ":m:l:t:s:v:g:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
  esac
done

# RUN_NUMBER=0
# MODEL=Qwen/Qwen2.5-7B-Instruct
# LANGUAGE=id
# TASK=gsm8k_train
# SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
# USE_GCS=False

# r_rand
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_rand -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_rand

# r_acc
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc

# r_acc-a-r_lang_fn
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-a-r_lang_fn -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_lang_fn

# r_acc-m-r_lang_fn
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-m-r_lang_fn -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_mult_lang_fn

# r_acc-r_lang_lm
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_lang_lm -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p False -j True -w True

# r_privileged
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score -p True -j True

# r_acc-r_privileged
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_privileged -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p True -j True -w True

