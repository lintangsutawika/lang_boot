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

from langdetect import detect_langs

# TODO Choose reward models
# df.apply(lambda row: row.step[0]['completion'][row['logprob'].index(max(row['logprob']))], axis=1)

lang_map = {
    "eng": "en",
    "ind": "id",
    "jpn": "ja",
    "zho": "zh",
}

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
    
    return candidates_df.iloc[0]['candidate']

# def rank_response(input_text, response, score, lang):

#     response_i_list = []
#     response_j_list = []

#     response_dict = {
#         "answers": [],
#         "lang": [],
#         "score": [],
#         "overlap": [],
#     }

#     score_fn = partial(get_lang_score, lang=lang)
#     results = map(score_fn, response)
#     for s, (prediction, lang_prob) in zip(score, list(results)):
#         response_dict["answers"].append(prediction)
#         response_dict["lang"].append(lang_prob)
#         response_dict["score"].append(s)

#     all_responses = pd.DataFrame(response_dict)
#     all_responses = all_responses.sort_values(by=['score', 'lang', 'overlap'], ascending=False)
#     all_responses = all_responses.reset_index(drop=True)

#     preferred = all_responses[(all_responses['lang'] > 0.5) & (all_responses['score'] == 1)]
#     dispreferred = all_responses[(all_responses['lang'] <= 0.5)]
#     dispreferred.loc[:, "lang"] = 1 - dispreferred.loc[:, "lang"]
#     dispreferred = dispreferred.sort_values(by=['score', 'lang', 'logprob'], ascending=False)

#     return preferred, dispreferred

system_message = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]

def construct_dataframe(
    translate_path, generate_path, output_path,
    max_samples=-1, keep_keys=None, use_accuracy=False, use_lang=False, lang_code="id",
    ):

    collected_responses = {}

    translate_df = pd.read_json(os.path.join(translate_path, "output.jsonl"), lines=True)
    generate_df = pd.read_json(os.path.join(generate_path, "output.jsonl"), lines=True)

    df = pd.DataFrame()
    translate_df['input_candidates'] = translate_df.apply(lambda row: from_key(row, "answer"), axis=1)
    # df['input_selected'] = translate_df.apply(lambda row: row["input_candidates"][row["logprob"].index(max(row["logprob"]))], axis=1)
    df['input_selected'] = translate_df.apply(
        lambda row: select_best_candidate(
            row, 
            col_name="input_candidates",
            use_logprob=True,
            use_accuracy=False,
            use_lang=getattr(args, 'use_lang', False),
        ), 
        axis=1
    )

    # generate_df['output_candidates'] = generate_df.apply(lambda row: from_completions(row), axis=1)
    generate_df['output_candidates'] = generate_df.apply(lambda row: from_key(row, "answer"), axis=1)
    # df['output_selected'] = generate_df.apply(lambda row: row["output_candidates"][row["logprob"].index(max(row["logprob"]))], axis=1)
    df['output_selected'] = generate_df.apply(
        lambda row: select_best_candidate(
            row, 
            col_name="output_candidates",
            use_logprob=True,
            use_accuracy=getattr(args, 'use_accuracy', False),
            use_lang=getattr(args, 'use_lang', False),
        ), 
        axis=1
    )

    # Post Processing
    df['input'] = df.apply(lambda row: [{"role": "user", "content": row["input_selected"]}], axis=1)
    df['output'] = df.apply(lambda row: [{"role": "assistant", "content": row["output_selected"]}], axis=1)
    df['messages'] = df.apply(lambda row: system_message + row['input'] + row['output'], axis=1)

    df['raw_prompt'] = df.apply(
        lambda row: [
            {"role": "system", "content": row["input_selected"]},
            {"role": "user", "content": row["input_selected"]},
            ],
        axis=1
    )

    df['reward_model'] = generate_df.apply(
        lambda row: {
            "ground_truth": row["ground_truth"]
        },
        axis=1
    )

    df["extra_info"] = df.apply(
        lambda row: {
            "ground_truth": row["reward_model"]["ground_truth"],
            "use_lang": use_lang,
            "use_accuracy": use_accuracy,
            "lang": lang_code,
        }, 
        axis=1
    )

    # df = df[['input', 'output']]
    # df = df[['messages']]

    df.sample(frac=1).reset_index(drop=True)
    task_size = len(df)
    train_size = int(task_size * 0.8)
    valid_size = int(task_size * 0.1)
    test_size = int(task_size * 0.1)
    if max_samples == -1:
        train_df = df.iloc[:train_size]
    else:
        train_df = df.iloc[:min(train_size, max_samples)]
    valid_df = df.iloc[train_size:train_size + valid_size]
    test_df = df.iloc[-test_size-1:]

    print(train_df.head())

    os.makedirs(output_path, exist_ok=True)
    train_df.to_parquet(os.path.join(output_path, "train.parquet"))
    valid_df.to_parquet(os.path.join(output_path, "valid.parquet"))
    test_df.to_parquet(os.path.join(output_path, "test.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate_path', type=str)
    parser.add_argument('--generate_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--use_accuracy', action='store_true', default=False)
    parser.add_argument('--use_lang', action='store_true', default=False)
    parser.add_argument('--lang_code', type=str, default='id', help='Language code for the responses (default: id for Indonesian)')
    args = parser.parse_args()
    construct_dataframe(
        args.translate_path,
        args.generate_path,
        args.output_path,
        max_samples=args.max_samples,
        use_accuracy=args.use_accuracy,
        use_lang=args.use_lang,
        lang_code=args.lang_code,
    )
