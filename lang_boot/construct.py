import os
import re
import string
import random
import argparse
import jsonlines
import multiprocessing

import pandas as pd
import numpy as np

from tqdm import tqdm
from functools import partial
from datasets import load_dataset
# from nltk.tokenize import word_tokenize

from yeval.task import TASK_LIST
from yeval.utils import import_modules

path = os.path.dirname(__file__)
import_modules(os.path.join(path, "../tasks/"))

from langdetect import detect_langs

# TODO Choose reward models
# df.apply(lambda row: row.step[0]['completion'][row['logprob'].index(max(row['logprob']))], axis=1)

def from_key(x, key):
    return x[key] if key in x else None

def from_completions(x):
    return x["step"][0]["completion"]

def get_lang_score(prediction, lang="id"):

    lang_prob = 0.0
    try:
        langs = detect_langs(prediction)
        for l in langs:
            if l.lang == lang:
                lang_prob = l.prob
    except Exception as e:
        lang_prob = 0.0

    return prediction, lang_prob

def select_best_candidate(row, col_name="input_candidates", use_logprob=True, use_accuracy=False, use_lang=False):
    
    candidates_dict = {'candidate': row[col_name]}
    
    # Sort by selected criteria (descending order)
    sort_columns = []
    if use_logprob:
        sort_columns.append('logprob')
        candidates_dict["logprob"] = row["logprob"]
    else:
        candidates_dict["logprob"] = [1.0] * len(row[col_name])

    if use_lang:
        sort_columns.append('lang')
        candidates_dict["lang"] = row["lang"]
    else:
        candidates_dict["lang"] = [1.0] * len(row[col_name])

    if use_accuracy:
        sort_columns.append('accuracy')
        candidates_dict["accuracy"] = row["accuracy"]
    else:
        candidates_dict["accuracy"] = [1.0] * len(row[col_name])
    
    candidates_df = pd.DataFrame(candidates_dict)

    candidates_df["score"] = (-1/candidates_df["logprob"]) * candidates_df["lang"] * candidates_df["accuracy"]

    candidates_df = candidates_df.sort_values(by="score", ascending=False)

    if len(candidates_df[candidates_df["score"] > 0]) == 0:
        return "None"
    
    return candidates_df.iloc[0]['candidate']


# system_message = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
system_message = [{"role": "system", "content": "Think about it step by step and give your answer at the end in \\boxed{}."}]

def construct_dataframe(
    response,
    task,
    lang,
    data_path,
    max_samples=-1, 
    use_accuracy=False, use_lang=False,
    use_en=False,
    # test_data_path="juletxara/mgsm",
    # test_data_name=None,
    ):

    df = pd.DataFrame()

    if (lang == "en") or use_en:
        response = "generated"
        query_source = "generated:traces"
    else:
        query_source = "translated:queries"

    prompt_functions = {
        f"{lang}_reason": TASK_LIST[f"{lang}_reason"].user_message,
        f"en_reason": TASK_LIST[f"en_reason"].user_message,
    }

    query_lang = "en" if use_en else lang
    query_path = os.path.join(data_path, f"raw_traces/{task}:{query_lang}:{query_source}/")
    response_path = os.path.join(data_path, f"raw_traces/{task}:{lang}:{response}:traces/")
    eng_response_path = os.path.join(data_path, f"raw_traces/{task}:en:generated:traces/")

    eng_response_df = pd.read_json(os.path.join(eng_response_path, "output.jsonl"), lines=True)
    eng_response_df['output_candidates'] = eng_response_df.apply(lambda row: from_key(row, "answer"), axis=1)
    df['eng_output_selected'] = eng_response_df.apply(
        lambda row: select_best_candidate(
            row, 
            col_name="output_candidates",
            use_logprob=True,
            use_accuracy=getattr(args, 'use_accuracy', True),
            use_lang=getattr(args, 'use_lang', False),
        ), 
        axis=1
    )

    query_df = pd.read_json(os.path.join(query_path, "output.jsonl"), lines=True)
    if (lang == "en") or use_en:
        def _get_query(row):
            for context_dict in row["step"][0]['full_input']:
                if context_dict['role'] == 'user':
                    return context_dict['content']
                
            return ""
        
        df['input_selected'] = query_df.apply(lambda row: _get_query(row), axis=1)
    else:
        query_df['input_candidates'] = query_df.apply(lambda row: from_key(row, "answer"), axis=1)
        df['input_selected'] = query_df.apply(
            lambda row: select_best_candidate(
                row, 
                col_name="input_candidates",
                use_logprob=True,
                use_accuracy=False,
                use_lang=getattr(args, 'use_lang', False),
            ), 
            axis=1
        )

    if lang == "en":
        df['output_selected'] = df['eng_output_selected']

        df['reward_model'] = eng_response_df.apply(
            lambda row: {
                "ground_truth": str(row["ground_truth"])
            },
            axis=1
        )

    else:
        response_df = pd.read_json(os.path.join(response_path, "output.jsonl"), lines=True)
        # response_df['output_candidates'] = response_df.apply(lambda row: from_completions(row), axis=1)
        response_df['output_candidates'] = response_df.apply(lambda row: from_key(row, "answer"), axis=1)
        df['output_selected'] = response_df.apply(
            lambda row: select_best_candidate(
                row, 
                col_name="output_candidates",
                use_logprob=True,
                use_accuracy=getattr(args, 'use_accuracy', False),
                use_lang=getattr(args, 'use_lang', False),
            ), 
            axis=1
        )

        df['reward_model'] = response_df.apply(
            lambda row: {
                "ground_truth": str(row["ground_truth"])
            },
            axis=1
        )

    # Post Processing
    # prompt_fn = prompt_functions[f"{lang}_reason"]
    if response == "translated":
        df = df[(df["input_selected"] != "None") & (df["output_selected"] != "None")]
    else:
        df = df[(df["eng_output_selected"] != "None")]

    dataset_df = pd.DataFrame()
    prompt_name = f"{lang}_reason"
    prompt_fn = prompt_functions[prompt_name]
    prompt_df = df.copy()
    prompt_df["input_selected"] = prompt_df.apply(
        lambda row: prompt_fn(row["input_selected"]),
        axis=1
    )
    prompt_df['input'] = prompt_df.apply(lambda row: system_message + [{"role": "user", "content": row["input_selected"]}], axis=1)
    prompt_df['output'] = prompt_df.apply(lambda row: [{"role": "assistant", "content": row["output_selected"]}], axis=1)
    prompt_df['messages'] = prompt_df.apply(lambda row: row['input'] + row['output'], axis=1)
    prompt_df['data_source'] = "train_dataset"
    prompt_df['raw_prompt'] = prompt_df.apply(
        lambda row: system_message + [
            {"role": "user", "content": row["input_selected"]},
            ],
        axis=1
    )

    prompt_df["extra_info"] = prompt_df.apply(
        lambda row: {
            "task": f"train_dataset/{prompt_name}",
            "ground_truth": str(row["reward_model"]["ground_truth"]),
            "use_lang": use_lang,
            "use_accuracy": use_accuracy,
            "lang": lang,
        }, 
        axis=1
    )

    dataset_df = pd.concat([dataset_df, prompt_df], ignore_index=True)

    # df = df[['input', 'output']]
    # df = df[['messages']]
    # Remove unsuccessful responses
    dataset_df.sample(frac=1).reset_index(drop=True)
    task_size = len(dataset_df)
    train_size = int(task_size * 0.8)
    valid_size = int(task_size * 0.1)
    test_size = int(task_size * 0.1)

    train_df = dataset_df.iloc[:train_size]
    valid_df = dataset_df.iloc[train_size:train_size + valid_size]
    test_df = dataset_df.iloc[-test_size-1:]

    print(train_df.head())

    for task_name in ["math500", "mgsm", "global_mmlu", "belebele", "mt_math100"]:
        if task_name == "math500":
            full_task_name = task_name
        else:
            full_task_name = f'{task_name}_{lang}'

        eval_dataset = TASK_LIST[full_task_name]()
        x, y = zip(*[eval_dataset.dataset.__getitem__(idx) for idx in range(eval_dataset.dataset.__len__())])

        for prompt_name, prompt_fn in prompt_functions.items():
            print(f"{task_name}/{prompt_name}")
            task_lang = prompt_name.split("_")[0]
            if not prompt_name.startswith("en_"):
                _, *prompt_name = prompt_name.split("_")
                prompt_name = "_".join(["x"]+prompt_name)

            eval_df = pd.DataFrame()
            eval_df['input_selected'] = [prompt_fn(_x) for _x in x]
            eval_df['output_selected'] = [str(_y) for _y in y]
            eval_df['input'] = eval_df.apply(lambda row: system_message + [{"role": "user", "content": row["input_selected"]}], axis=1)
            eval_df['output'] = eval_df.apply(lambda row: [{"role": "assistant", "content": ""}], axis=1)
            eval_df['messages'] = eval_df.apply(lambda row: [{"role": "assistant", "content": ""}], axis=1)
            eval_df['raw_prompt'] = eval_df.apply(lambda row: [{"role": "assistant", "content": ""}], axis=1)
            eval_df['data_source'] = f"{task_name}/{prompt_name}"
            eval_df["extra_info"] = eval_df.apply(
                lambda row: {
                    "task": f"{task_name}/{prompt_name}",
                    "ground_truth": str(row["output_selected"]),
                    "lang": task_lang,
                }, 
                axis=1
            )
            eval_df['reward_model'] = eval_df.apply(
                lambda row: {
                    "ground_truth": str(row["output_selected"]),
                },
                axis=1
            )

            eval_df = eval_df[['input', 'output', 'input_selected', 'output_selected', 'messages', 'data_source', 'raw_prompt', 'reward_model', 'extra_info']]
            test_df = pd.concat([test_df, eval_df], ignore_index=True)

    if use_en:
        output_path = os.path.join(data_path, f"prep_traces/{task}:{lang}:en_generated/")
    else:
        output_path = os.path.join(data_path, f"prep_traces/{task}:{lang}:{response}/")
    os.makedirs(output_path, exist_ok=True)
    train_df.to_parquet(os.path.join(output_path, "train.parquet"))
    valid_df.to_parquet(os.path.join(output_path, "valid.parquet"))
    test_df.to_parquet(os.path.join(output_path, "test.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--lang', type=str, default=None, help='Language')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--use_en', action='store_true', default=False)
    parser.add_argument('--use_accuracy', action='store_true', default=False)
    parser.add_argument('--use_lang', action='store_true', default=False)
    args = parser.parse_args()
    construct_dataframe(
        args.response,
        args.task,
        args.lang,
        use_en=args.use_en,
        data_path=args.data_path,
        max_samples=args.max_samples,
        use_accuracy=args.use_accuracy,
        use_lang=args.use_lang,
    )
