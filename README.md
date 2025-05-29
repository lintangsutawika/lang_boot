# Lang Boot

Bootstrapping language model capability in non-English languages


## Training

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