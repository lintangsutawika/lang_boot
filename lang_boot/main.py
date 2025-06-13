import os
import sys
import asyncio
import subprocess
import argparse

from functools import partial

from yeval.utils import import_modules
from yeval.task import TASK_LIST, YevalTask
from yeval.evaluation import EvaluateSystem
from yeval.model import Server, get_host_and_port
from yeval.utils import simple_parse_args_string

from .construct import construct_preference

def main(args):

    if args.task_path is not None:
        import_modules(args.task_path)

    # Check if the model checkpoint exists and not empty
    print(f"Checking current iteration")
    for iteration in range(args.max_iterations):
        if args.grpo:
            train = "grpo"
        else:
            train = "dpo" if args.dpo else "sft"
        model_checkpoint = os.path.join(args.save_model_path, f"{train}_{iteration}_model")
        if not os.path.exists(model_checkpoint) or not os.listdir(model_checkpoint):
            print(f"Continuing from iteration {iteration}")
            current_iteration = iteration
            break

    for iteration in range(args.max_iterations):

        print(f"Running iteration {iteration} of {args.max_iterations-1} on budget {args.budget} tokens")
        if iteration < current_iteration:
            continue

        if iteration == 0:
            model_name = args.model_name
        else:
            train = "dpo" if args.dpo else "sft"
            model_name = f"{args.save_model_path}/{train}_{iteration-1}_model/"

        # Deploy VLLM Here
        if args.serve:
            host, port = get_host_and_port(args.api_base)
            model_server = Server(
                model_name=model_name,
                host=host, port=port, backend=args.backend,
                pp_size=args.pp_size, tp_size=args.tp_size,
                max_model_len=args.max_model_len
                )
            process = model_server.start()

            evaluator = EvaluateSystem(
                model=model_name,
                api_base=args.api_base,
                api_key=args.api_key,
                output_path=args.output_path,
                max_rps=args.max_rps,
                )

        prompt_list = [
            # "eng_reason_A_box_after",
            # "eng_reason_in_{}_A_box_after",
            "{}_reason_A_box_after"
        ]

        for idx, prompt in enumerate(prompt_list):
            if args.no_sampling:
                break
            prompt_modifier = prompt.format(args.lang)
            print(f"Running task: {args.task_name} with prompt: {prompt_modifier}")
            base_run_name = f"{iteration}:{args.lang}:{args.task_name}:{prompt_modifier}"

            # for query in ["reasoning", "translate", "default"]:
            for query in ["translate", "reasoning"]:
                task_run_name = base_run_name+f":{query}"

                if (query == "translate"):
                    if args.no_translate:
                        continue
                    translate_file = f"{args.task_name}:{args.lang}:translate"
                    try:
                        if translate_file in os.listdir(args.output_path):
                            continue
                    except FileNotFoundError:
                        task_run_name = translate_file

                    sub_task_name = "{}_translate".format(args.lang)
                    task_kwargs = {
                        "subtask_list": [
                            partial(
                                TASK_LIST[sub_task_name],
                                name=sub_task_name,
                            )
                        ]
                    }

                    sampling_args = None
                    n_samples = None

                    task_object = TASK_LIST[args.task_name](
                        evaluation=TASK_LIST[sub_task_name].evaluation,
                        sample_agg_fn=TASK_LIST[sub_task_name].sample_agg_fn,
                        **task_kwargs,
                        )
                else:

                    task_kwargs = {
                        "subtask_list": [
                            partial(
                                TASK_LIST[f"budget={args.budget}"],
                                name=f"budget={args.budget}",
                                user_message=TASK_LIST[prompt_modifier].user_message,
                            ),
                        ]
                    }
                    sampling_args = simple_parse_args_string(args.sample_args) if args.sample_args else None
                    n_samples = args.n_samples
                    def input_text(x):
                        candidate = x["answer"]
                        score = x["lang"]
                        return candidate[score.index(max(score))]

                    def shuffle(dataset, seed=0):
                        shuffled_dataset = dataset.shuffle(seed=seed)
                        return shuffled_dataset.flatten_indices()

                    shuffle_fn = partial(shuffle, seed=1000+iteration)

                    if args.no_translate:
                        task_object = TASK_LIST[args.task_name](
                            preprocessing=shuffle_fn,
                            user_message=TASK_LIST[prompt_modifier].user_message,
                            **task_kwargs,
                        )
                    else:
                        task_object = TASK_LIST[args.task_name](
                            preprocessing=shuffle_fn,
                            user_message=TASK_LIST[prompt_modifier].user_message,
                            data_path="json",
                            data_name=None,
                            test_split="train",
                            input_text=input_text,
                            output_text="ground_truth",
                            data_kwargs={
                                "data_files": {
                                    os.path.join(
                                        args.output_path,
                                        translate_file,
                                        "output.jsonl"
                                    )
                                }
                            },
                            **task_kwargs
                            )

                asyncio.run(
                    evaluator.run(
                        task_object,
                        run_name=task_run_name,
                        n_samples=n_samples,
                        sampling_args=sampling_args,
                    )
                )

        if args.serve:
            model_server.stop(process)

        for training in ["sft", "dpo", "grpo"]:

            training_specific = [
                "--learning_rate", "1e-5",
                "--max_samples", str(args.n_samples),
                "--max_epochs", "5",
            ]

            if (training == "sft") and args.sft:
                save_path = os.path.join(args.save_model_path, f"sft_{iteration}_model/")
                cli_command = "openrlhf.cli.train_sft"
                data_key = [
                    "--input_key", "question",
                    "--output_key", "response_i",
                ]

            elif (training == "dpo") and args.dpo:
                save_path = os.path.join(args.save_model_path, f"dpo_{iteration}_model/")
                cli_command = "openrlhf.cli.train_dpo"
                data_key = [
                    "--prompt_key", "question",
                    "--chosen_key", "response_i",
                    "--rejected_key", "response_j",
                ]

                training_specific += [
                    "--ref_offload",
                    "--beta", "0.1",
                ]
            elif (training == "grpo") and args.grpo:
                save_path = os.path.join(args.save_model_path, f"grpo_{iteration}_model/")
                cli_command = "openrlhf.cli.train_ppo_ray"
                data_key = [
                    "--input_key", "question",
                    "--label_key", "response_i",
                ]

                training_specific = [
                    "--actor_num_gpus_per_node", "1",
                    "--vllm_num_engines", "1",
                    "--colocate_all_models",
                    "--actor_learning_rate", "1e-5",
                    "--advantage_estimator", "group_norm",
                    "--use_kl_loss",
                    "--init_kl_coef", "0",
                    "--normalize_reward",
                    "--n_samples_per_prompt", "16",
                    "--prompt_max_len", "1024", 
                    "--generate_max_len", "1024", 
                    "--micro_rollout_batch_size", "32", 
                    "--rollout_batch_size", "1024", 
                    "--remote_rm_url", "lang_boot/lang_boot/reward_func.py",
                    "--max_samples", str(10000),
                    "--prompt_split", "train",
                    "--eval_split", "test",
                ]
            else:
                continue

            print(f"Running training: {training} for iteration {iteration}")

            if args.training_data_path:
                training_data_path = args.training_data_path
            else:
                training_data_path = os.path.join(
                    args.output_path,
                    f"{iteration}:{training}:{args.lang}:{args.task_name}"
                    )

                # Construct Data
                construct_preference(
                    iteration, args.lang, args.task_name,
                    args.output_path,
                    training_data_path,
                    sft=True if training != "dpo" else False,
                    )

            if training == "grpo":
                data_key += [
                    "--prompt_data", f"json@{training_data_path}",
                ]
            else:
                data_key += [
                    "--dataset", f"json@{training_data_path}",
                ]

            ckpt_path = os.path.join(f"{args.save_model_path}", "ckpt/")

            training_command = [
                "deepspeed", 
                "--master_port", f"{args.master_port}",
                "--module", f"{cli_command}",
                "--save_path", save_path,
                "--ckpt_path", ckpt_path,
                "--save_steps", "62",
                "--logging_steps", "1",
                "--eval_steps", "-1",
                "--train_batch_size", "16",
                "--micro_train_batch_size", "8",
                "--lr_warmup_ratio", "0.05",
                "--pretrain", model_name,
                "--save_hf_ckpt",
                "--bf16",
                "--max_len", "4096",
                "--zero_stage", "3",
                "--l2", "1e-4",
                "--apply_chat_template",
                "--input_template", "None",
                "--flash_attn", "--gradient_checkpointing",
                "--adam_offload",
                # "--use_liger_kernel",
                "--packing_samples",
            ] + data_key + training_specific

            try:
                process = subprocess.Popen(training_command)
                exit_code = process.wait()
            except Exception as e:
                print(f"Error during training: {e}")
                if "process" in locals():
                    process.terminate()
                    process.wait()
                sys.exit()

            model_name = save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute DPO training pipeline.")
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--serve", action="store_true", help="Whether to serve the model")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1/", help="API base URL for the model server")
    parser.add_argument("--api_key", type=str, default="None", help="API key for the model server")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend for the model server")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--no_translate", action="store_true", help="Whether to serve the model")
    parser.add_argument("--budget", type=int, default=512, help="Sampling budget in tokens")
    # Sampling parameters
    parser.add_argument("--lang", type=str, required=True, help="Path to the model")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4000, help="Number of samples")
    parser.add_argument("--sample_args", type=str, default=None, help="Sampling arguments")
    parser.add_argument("--save_model_path", type=str, required=True, help="Output path for results")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
    parser.add_argument("--no_sampling", action="store_true", help="Disable sampling")
    parser.add_argument("--training_data_path", type=str, default=None, help="Set training path manually")

    # Training parameters
    parser.add_argument("--sft", action="store_true", help="Enable SFT training")
    parser.add_argument("--dpo", action="store_true", help="Enable DPO training")
    parser.add_argument("--grpo", action="store_true", help="Enable GRPO training")

    parser.add_argument("--task_path", type=str, default=None, help="Path to the task modules")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--max_rps", type=int, default=512, help="Maximum requests per second")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum number of iterations for training")

    args = parser.parse_args()
    main(args)
