import pandas as pd
import os
import re
import random
from tqdm import tqdm
from datasets import load_dataset
import argparse
import multiprocessing
import jsonlines
from functools import partial
from yeval.response.math_responses import get_boxed_answer
from math_verify import parse, verify

from langdetect import detect_langs

lang_map = {
    "ind": "id",
    "jpn": "ja",
    "zho": "zh",
}

def get_score(prediction, ground_truth, lang="id", score=None):
    lang_prob = 0.0
    try:
        if score is None:
            score = 0
            pr = parse(get_boxed_answer(prediction))
            gt = parse(ground_truth)
            score = int(verify(gt, pr))
        # print("pr", pr, "gt", gt, "score", score)
        langs = detect_langs(prediction)
        for l in langs:
            if l.lang == lang:
                lang_prob = l.prob
    except Exception as e:
        print(f"Error: {e}")

    return prediction, lang_prob, score

def process_response(ground_truth, response, lang, as_string=False):

    response_i_list = []
    response_j_list = []

    response_dict = {
        "answers": [],
        "lang": [],
        "score": [],
    }

    score_fn = partial(get_score, ground_truth=ground_truth, lang=lang)
    results = map(score_fn, response)
    for prediction, lang_prob, score in list(results):
        response_dict["answers"].append(prediction)
        response_dict["lang"].append(lang_prob)
        response_dict["score"].append(score)

    all_responses = pd.DataFrame(response_dict)
    all_responses = all_responses.sort_values(by=['score', 'lang'], ascending=False)
    all_responses = all_responses.reset_index(drop=True)
    # all_responses = all_responses[all_responses["score"] == 1]
    # return all_responses

    for idx in range(len(all_responses)):
        preferred = all_responses.iloc[idx]
        if (preferred['lang'] < 0.1) or (preferred['score'] == 0):
            break

        dispreferred = all_responses.iloc[idx+1:]
        if len(dispreferred) == 0:
            break

        response_1, lang_1, score_1 = preferred
        for idx_2, (response_2, lang_2, score_2) in dispreferred.iterrows():
            if (lang_2 > 0.50) and (score_2 == 1):
                continue
            if as_string:
                response_i_list.append(response_1)
                response_j_list.append(response_2)
            else:
                response_i_list.append([{"role": "assistant", "content":response_1}])
                response_j_list.append([{"role": "assistant", "content":response_2}])

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

def construct_preference(iteration, lang, task, input_path, output_path, as_string=False):

    prefix = f"{iteration}:{lang}:{task}"

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    preference_data = {
        # 'task':[],
        # 'solution':[],
        'question':[],
        'response_i':[],
        'response_j':[],
        'ground_truth':[]
        }

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
                    "response_reason": [],
                    "response_default": [],
                }

            if sample_dir.endswith("translate"):
                sampled_text = sample["answer"]
                collected_responses[idx]["query"].extend(sampled_text)
                collected_responses[idx]["ground_truth"] = sample["ground_truth"]
            elif sample_dir.endswith("reasoning"):
                sampled_text = sample["step"][0]["completion"]
                collected_responses[idx]["response_reason"].extend(sampled_text)
            elif sample_dir.endswith("default"):
                sampled_text = sample["step"][0]["completion"]
                collected_responses[idx]["response_default"].extend(sampled_text)

    lang_id = lang_map[lang]
    _process_response = partial(process_response, lang=lang_id, as_string=as_string)
    result_list = []
    for _, response in tqdm(collected_responses.items(), total=len(collected_responses)):
        result_list.append(
            _process_response(
                response["ground_truth"],
                response["response_reason"]+response["response_default"]
                )
        )

    result_list = [result for result in result_list if len(result[0]) != 0]
    for idx in range(len(result_list)):
        response_i_list, response_j_list = result_list[idx]
        assert len(response_i_list) == len(response_j_list), "Response lists must be of the same length"
        query = collected_responses[idx]["query"]
        ground_truth = collected_responses[idx]["ground_truth"]
        
        # data['task'].extend([task] * len(response_i_list))
        preference_data['question'].extend([query]* len(response_i_list))
        preference_data['ground_truth'].extend([ground_truth] * len(response_i_list))
        preference_data['response_i'].extend(response_i_list)
        preference_data['response_j'].extend(response_j_list)

    with jsonlines.open(os.path.join(output_path, "train.jsonl"), mode='w') as writer:
        for q, i, j, g in zip(preference_data['question'], preference_data['response_i'], preference_data['response_j'], preference_data['ground_truth']):
            line = {"ground_truth": g, "question": q, "response_i": i, "response_j": j}
            writer.write(line)

    test_samples = 128
    with jsonlines.open(os.path.join(output_path, "test.jsonl"), mode='w') as writer:
        for idx, (q, i, j, g) in enumerate(zip(preference_data['question'], preference_data['response_i'], preference_data['response_j'], preference_data['ground_truth'])):

            if idx == test_samples:
                break

            line = {"ground_truth": g, "question": q, "response_i": i, "response_j": j}
            writer.write(line)

    #df = pd.DataFrame(data)
    #df = df.sample(frac=1).reset_index(drop=True)
    #df.to_csv(os.path.join(args.output_path, "data.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--task', type=str)

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--as_string', action='store_true', default=False)
    args = parser.parse_args()
    construct_preference(
        args.iteration,
        args.lang,
        args.task,
        args.input_path,
        args.output_path,
        as_string=args.as_string
    )