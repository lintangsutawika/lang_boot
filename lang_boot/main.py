import os
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
    "eng_reason_A_box",
    "eng_reason_in_{}_A_box",
    "{}_reason_A_box"
    ] for order in [
        "before",
        "after",
        ]
]

translate_prompt_list = [
    "{}_translate",
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

        if iteration < current_iteration:
            continue

        if iteration == 0:
            model_name = args.model_name
        else:
            model_name = f"{args.save_model_path}/{iteration}_model/"

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

        for idx, prompt in enumerate(reasoning_prompt_list):
            prompt_modifier = prompt.format(args.lang)
            print(f"Running task: {args.task_name} with prompt: {prompt_modifier}")
            base_run_name = f"{iteration}:{args.lang}:{args.task_name}:{prompt_modifier}"

            for query in ["reasoning", "translate", "default"]:
                task_run_name = base_run_name+f":{query}"

                if query == "reasoning":
                    task_kwargs = {
                        "subtask_list": [partial(TASK_LIST[prompt_modifier], name=prompt_modifier)]
                    }
                    sampling_args = simple_parse_args_string(args.sample_args) if args.sample_args else None

                    def shuffle(dataset, seed=0):
                        shuffled_dataset = dataset.shuffle(seed=seed)
                        return shuffled_dataset.flatten_indices()

                    shuffle_fn = partial(shuffle, seed=1000+iteration)
                    task_object = TASK_LIST[args.task_name](
                        preprocessing=shuffle_fn,
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

        preference_data_path = os.path.join(
            args.output_path,
            f"{iteration}:preference_data:{args.lang}:{args.task_name}"
            )

        # Construct Data
        construct_preference(
            iteration, args.lang, args.task_name, args.output_path,
            preference_data_path,
            )

        # DPO Training
        training_command = [
            "deepspeed", "--module", "openrlhf.cli.train_dpo",
            "--save_path", f"{args.save_model_path}/{iteration}_model/",
            "--ckpt_path", f"{args.save_model_path}/ckpt/",
            "--save_steps", "-1",
            "--logging_steps", "1",
            "--eval_steps", "-1",
            "--train_batch_size", "64",
            "--micro_train_batch_size", "4",
            "--pretrain", model_name,
            "--save_hf_ckpt",
            "--bf16",
            "--max_samples", "640",
            "--max_epochs", "1",
            "--max_len", "2048",
            "--zero_stage", "3",
            "--ref_offload",
            "--learning_rate", "3e-7",
            "--l2", "0.05",
            "--beta", "0.05",
            "--dataset", f"json@{preference_data_path}",
            "--apply_chat_template",
            "--prompt_key", "question",
            "--chosen_key", "response_i",
            "--rejected_key", "response_j",
            "--flash_attn",
            "--gradient_checkpointing",
            "--adam_offload", "--use_liger_kernel", "--packing_samples"
        ]
        subprocess.run(training_command, check=True)
        model_name = f"{args.save_model_path}/{iteration}_model/"

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

    # Sampling parameters
    parser.add_argument("--lang", type=str, required=True, help="Path to the model")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4000, help="Number of samples")
    parser.add_argument("--sample_args", type=str, default=None, help="Sampling arguments")
    parser.add_argument("--save_model_path", type=str, required=True, help="Output path for results")

    parser.add_argument("--task_path", type=str, default=None, help="Path to the task modules")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--max_rps", type=int, default=512, help="Maximum requests per second")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum number of iterations for training")

    args = parser.parse_args()
    main(args)
