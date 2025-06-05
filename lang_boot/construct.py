import os
import re
import string
import random
import argparse
import jsonlines
import multiprocessing

import pandas as pd

from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from langdetect import detect_langs

lang_map = {
    "eng": "en",
    "ind": "id",
    "jpn": "ja",
    "zho": "zh",
}

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

def process_query(query, lang, as_string=False):

    query_list = []

    response_dict = {
        "query": [],
        "lang": [],
    }

    score_fn = partial(get_lang_score, lang=lang)
    results = map(score_fn, query)
    for q, lang_prob in list(results):
        response_dict["query"].append(q)
        response_dict["lang"].append(lang_prob)

    all_responses = pd.DataFrame(response_dict)
    all_responses = all_responses.sort_values(by=['lang'], ascending=False)
    all_responses = all_responses.reset_index(drop=True)

    prompt = all_responses.iloc[0]["query"]
    if as_string:
        return prompt

    return [{"role": "user", "content": prompt}]

def calculate_overlap(row):
    input_words = set(word for word in np.unique(row["input"].split()).tolist() if word not in stop_words)
    output_words = set(word for word in np.unique(row["output"].split()).tolist() if word not in stop_words)
    overlap = len(input_words & output_words) / len(input_words | output_words)
    return overlap

def n_gram_overlap(input_text, output_text, n=2):
    input_words = word_tokenize(input_text)
    output_words = word_tokenize(output_text)
    
    input_ngrams = set(tuple(input_words[i:i+n]) for i in range(len(input_words)-n+1))
    output_ngrams = set(tuple(output_words[i:i+n]) for i in range(len(output_words)-n+1))
    
    overlap = len(input_ngrams & output_ngrams) / len(input_ngrams | output_ngrams)
    return overlap

# , as_string=False, sft=False
def rank_response(input_text, response, score, lang):

    response_i_list = []
    response_j_list = []

    response_dict = {
        "answers": [],
        "lang": [],
        "score": [],
        "overlap": [],
    }

    score_fn = partial(get_lang_score, lang=lang)
    results = map(score_fn, response)
    for s, (prediction, lang_prob) in zip(score, list(results)):
        response_dict["answers"].append(prediction)
        response_dict["lang"].append(lang_prob)
        response_dict["score"].append(s)
        response_dict["overlap"].append(n_gram_overlap(input_text, prediction))

    all_responses = pd.DataFrame(response_dict)
    all_responses = all_responses.sort_values(by=['score', 'lang', 'overlap'], ascending=False)
    all_responses = all_responses.reset_index(drop=True)

    preferred = all_responses[(all_responses['lang'] > 0.5) & (all_responses['score'] == 1)]
    dispreferred = all_responses[(all_responses['lang'] <= 0.5)]
    dispreferred.loc[:, "lang"] = 1 - dispreferred.loc[:, "lang"]
    dispreferred = dispreferred.sort_values(by=['score', 'lang'], ascending=False)

    return preferred, dispreferred

    # for idx in range(len(all_responses)):
    #     preferred = all_responses.iloc[idx]
    #     if (preferred['lang'] < 0.1) or (preferred['score'] == 0):
    #         break

    #     dispreferred = all_responses.iloc[idx+1:]
    #     if len(dispreferred) == 0:
    #         break

    #     response_1, lang_1, score_1 = preferred
    #     for idx_2, (response_2, lang_2, score_2) in dispreferred.iterrows():
    #         if (lang_2 > 0.50) and (score_2 == 1):
    #             continue
    #         if as_string:
    #             response_i_list.append(response_1)
    #             response_j_list.append(response_2)
    #         else:
    #             response_i_list.append([{"role": "assistant", "content":response_1}])
    #             response_j_list.append([{"role": "assistant", "content":response_2}])

    # preferred = all_responses[(all_responses['lang'] > 0.5) & (all_responses['score'] == 1)]
    # if len(preferred) == 0:
    #     return None

    # dispreferred = all_responses[~all_responses.isin(preferred)].dropna()
    # for idx_1, (response_1, lang_1, score_1) in preferred.iterrows():
    #     for idx_2, (response_2, lang_2, score_2) in dispreferred.iterrows():
    #         # data['question'].append(input_text)
    #         question_list.append([{"role":"user", "content":input_text}])
    #         # data['response_i'].append(response_1)
    #         response_i_list.append([{"role": "assistant", "content":response_1}])
    #         # data['response_j'].append(response_2)
    #         response_j_list.append([{"role": "assistant", "content":response_2}])

    return response_i_list, response_j_list

def input_text(x):
    text = "\n"+x["step"][0]["full_input"][0]["content"]
    text = re.sub("(\n((?!\n).)*?boxed{}.)", "", text)
    return text.strip()

def construct_preference(iteration, lang, task, input_path, output_path, as_string=False, sft=False):

    lang_id = lang_map[lang]
    prefix = f"{iteration}:{lang}:{task}"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    preference_data = []

    sample_list = [_dir for _dir in os.listdir(input_path) if _dir.startswith(prefix)]

    collected_responses = {}
    for sample_dir in sample_list:

        sample_lines = load_dataset("json", data_files=os.path.join(input_path, sample_dir, "output.jsonl"))['train']
        for sample in sample_lines:
            idx = sample['idx']
            if idx not in collected_responses:
                collected_responses[idx] = {
                    "query": [],
                    "ground_truth": None,
                    "prompted_score": [],
                    "default_score": [],
                    "prompted_response": [],
                    "default_response": [],
                }

            if sample_dir.endswith("translate"):
                sampled_text = sample["answer"]
                collected_responses[idx]["query"] = sampled_text
                collected_responses[idx]["ground_truth"] = sample["ground_truth"]
            else:
                if sample_dir.endswith("reasoning"):
                    response_key, score_key = "prompted_response", "prompted_score"
                elif sample_dir.endswith("default"):
                    response_key, score_key = "default_response", "default_score"

                collected_responses[idx]["query"] = input_text(sample)
                collected_responses[idx]["ground_truth"] = sample["ground_truth"]
                for step in sample["step"]:
                    sampled_text = step["completion"]
                    sampled_acc = step["eval"]["acc"]
                    collected_responses[idx][response_key].extend(sampled_text)
                    collected_responses[idx][score_key].extend(sampled_acc)

    _rank_response = partial(rank_response, lang=lang_id)
    result_list = []
    for idx in tqdm(range(len(collected_responses))):
        # Skip if no responses are correct
        if sum(collected_responses[idx]["prompted_score"]) == 0:
            continue

        # query = process_query(collected_responses[idx]["query"], lang=lang_id)
        query_text = collected_responses[idx]["query"]
        query = [{"role": "user", "content": query_text}]
        ground_truth = collected_responses[idx]["ground_truth"]
        responses = collected_responses[idx]["prompted_response"]+collected_responses[idx]["default_response"]
        scores = collected_responses[idx]["prompted_score"]+collected_responses[idx]["default_score"]
        preferred, dispreferred = _rank_response(
            input_text=query_text,
            response=responses,
            score=scores,
        )

        if sft == False and len(dispreferred) == 0:
            print(f"Skipping idx {idx} as no dispreferred responses found")
            continue

        max_samples = 1
        preferred = preferred.iloc[:max_samples]
        dispreferred = dispreferred.iloc[:max_samples]

        line = {"ground_truth": ground_truth, "question": query}

        if sft:
            for idx_0 in range(len(preferred)):
                response_1, *_ = preferred.iloc[idx_0]

                if not as_string:
                    response_1 = [{"role": "assistant", "content": response_1}]
                preference_data.append({**line, "response_i": response_1})
        else:
            for idx_0 in range(len(preferred)):
                for idx_1 in range(len(dispreferred)):

                    response_1, *_ = preferred.iloc[idx_0]
                    response_2, *_ = dispreferred.iloc[idx_1]

                    if not as_string:
                        response_1 = [{"role": "assistant", "content":response_1}]
                        response_2 = [{"role": "assistant", "content":response_2}]

                    preference_data.append({**line, "response_i": response_1, "response_j": response_2})

    random.shuffle(preference_data)
    train_split = int(len(preference_data) * 0.9)

    train_data = preference_data[:train_split]
    test_data = preference_data[train_split:]

    with jsonlines.open(os.path.join(output_path, "train.jsonl"), mode='w') as writer:
        writer.write_all(train_data)

    with jsonlines.open(os.path.join(output_path, "test.jsonl"), mode='w') as writer:
        writer.write_all(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--task', type=str)

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--as_string', action='store_true', default=False)
    parser.add_argument('--sft', action='store_true', default=False)
    args = parser.parse_args()
    construct_preference(
        args.iteration,
        args.lang,
        args.task,
        args.input_path,
        args.output_path,
        as_string=args.as_string,
        sft=args.sft,
    )