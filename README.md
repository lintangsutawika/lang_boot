# Lang Boot

Bootstrapping language model capability in non-English languages

## Constructing Dataset

1.  Generate traces
2.  Choose based on reward
2b. Translate
3.  Train on traces

## Training

### SFT

```
python -m lang_boot.main \
    --task_name open_r1_math_220k \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_path output/ \
    --task_path lang_boot/tasks/ \
    --lang ind \
    --api_base http://localhost:9100/v1/ \
    --max_iteration 4 \
    --sample_args "temperature=1.0,top_p=0.9,n=2"  \
    --save_model_path model_ckpt/ \
    --n_samples 4000 \
    --serve
```
### GRPO

```
CHECKPOINT_PATH=/data/user_data/lsutawik/05-lang-rl/checkpoints/
bash lang_boot/scripts/train_grpo_privilaged.sh \
	${MODEL_PATH} \
	${N_ROLLOUTS}$ \
	${DATA_PATH}$ \
	${SAVE_MODEL_PATH}$
```
