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

reasoning_prompt_list = [
    "_".join([prompt, order]) for prompt in [
    # "eng_reason_A_box",
    # "eng_reason_in_{}_A_box",
    "{}_reason_A_box"
    ] for order in [
        # "before",
        "after",
        ]
]

def main(args):

    if args.task_path is not None:
        import_modules(args.task_path)

    # Check if the model checkpoint exists and not empty
    print(f"Checking current iteration")
    for iteration in range(args.max_iterations):
        model_checkpoint = os.path.join(args.save_model_path, f"{iteration}_model")
        if not os.path.exists(model_checkpoint) or not os.listdir(model_checkpoint):
            print(f"Continuing from iteration {iteration}")
            current_iteration = iteration
            break

    for iteration in range(args.max_iterations):

        print(f"Running iteration {iteration} of {args.max_iterations}")
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

        # if args.sft:
        #     # Skip the first prompt for SFT
        #     prompt_list = reasoning_prompt_list[1:]
        # else:
        #     # Use all prompts for DPO
        #     prompt_list = reasoning_prompt_list

        for idx, prompt in enumerate(reasoning_prompt_list):
            prompt_modifier = prompt.format(args.lang)
            print(f"Running task: {args.task_name} with prompt: {prompt_modifier}")
            base_run_name = f"{iteration}:{args.lang}:{args.task_name}:{prompt_modifier}"

            for query in ["reasoning", "translate", "default"]:
                task_run_name = base_run_name+f":{query}"

                if query == "reasoning":

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

                    def shuffle(dataset, seed=0):
                        shuffled_dataset = dataset.shuffle(seed=seed)
                        return shuffled_dataset.flatten_indices()

                    shuffle_fn = partial(shuffle, seed=1000+iteration)
                    task_object = TASK_LIST[args.task_name](
                        preprocessing=shuffle_fn,
                        user_message=TASK_LIST[prompt_modifier].user_message,
                        **task_kwargs,
                        )
                else:

                    # Only need to run translate once
                    if idx > 0:
                        break

                    if query == "translate":
                        task_name = "{}_translate".format(args.lang)
                        source = "reasoning"
                    else:
                        if not args.dpo:
                            break
                        task_name = "default"
                        source = "translate"

                    task_kwargs = {
                        "data_kwargs": {
                            "data_files": {
                                os.path.join(
                                    args.output_path,
                                    base_run_name+f":{source}", # We need the queries from reasoning
                                    "output.jsonl"
                                )
                            }
                        }
                    }
                    sampling_args = None
                    task_object = TASK_LIST[task_name](
                        **task_kwargs
                        )

                asyncio.run(
                    evaluator.run(
                        task_object,
                        run_name=task_run_name,
                        n_samples=args.n_samples,
                        sampling_args=sampling_args,
                    )
                )

        if args.serve:
            model_server.stop(process)

        for training in ["sft", "dpo"]:
            if (training == "sft") and args.sft:
                save_path = os.path.join(args.save_model_path, f"sft_{iteration}_model/")
                cli_command = "openrlhf.cli.train_sft"
                data_key = [
                    "--input_key", "question",
                    "--output_key", "response_i",
                ]

                training_specific = []

            elif (training == "dpo") and args.dpo:
                save_path = os.path.join(args.save_model_path, f"dpo_{iteration}_model/")
                cli_command = "openrlhf.cli.train_dpo"
                data_key = [
                    "--prompt_key", "question",
                    "--chosen_key", "response_i",
                    "--rejected_key", "response_j",
                ]

                training_specific = [
                    "--ref_offload",
                    "--beta", "0.1",
                ]
            else:
                continue

            print(f"Running training: {training} for iteration {iteration}")
            training_data_path = os.path.join(
                args.output_path,
                f"{iteration}:{training}:{args.lang}:{args.task_name}"
                )

            # Construct Data
            construct_preference(
                iteration, args.lang, args.task_name,
                args.output_path,
                training_data_path,
                sft=True if training == "sft" else False,
                )

            ckpt_path = os.path.join(f"{args.save_model_path}", "ckpt/")

            training_command = [
                "deepspeed", "--module", f"{cli_command}",
                "--save_path", save_path,
                "--ckpt_path", ckpt_path,
                "--save_steps", "-1",
                "--logging_steps", "1",
                "--eval_steps", "-1",
                "--train_batch_size", "16",
                "--micro_train_batch_size", "4",
                "--lr_warmup_ratio", "0.05",
                "--pretrain", model_name,
                "--save_hf_ckpt",
                "--bf16",
                "--max_samples", str(1000),
                "--max_epochs", "1",
                "--max_len", "2048",
                "--zero_stage", "3",
                "--learning_rate", "1e-5",
                "--l2", "1e-4",
                "--dataset", f"json@{training_data_path}",
                "--apply_chat_template",
                "--input_template", "None",
                "--flash_attn", "--gradient_checkpointing",
                "--adam_offload", "--use_liger_kernel", "--packing_samples"
            ] + data_key + training_specific

            try:
                process = subprocess.Popen(training_command)
                exit_code = process.wait()
            except Exception as e:
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
    parser.add_argument("--translate", action="store_true", help="Whether to serve the model")
    parser.add_argument("--budget", type=int, default=512, help="Sampling budget in tokens")
    # Sampling parameters
    parser.add_argument("--lang", type=str, required=True, help="Path to the model")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4000, help="Number of samples")
    parser.add_argument("--sample_args", type=str, default=None, help="Sampling arguments")
    parser.add_argument("--save_model_path", type=str, required=True, help="Output path for results")

    # Training parameters
    parser.add_argument("--sft", action="store_true", help="Enable SFT training")
    parser.add_argument("--dpo", action="store_true", help="Enable DPO training")

    parser.add_argument("--task_path", type=str, default=None, help="Path to the task modules")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--max_rps", type=int, default=512, help="Maximum requests per second")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum number of iterations for training")

    args = parser.parse_args()
    main(args)
