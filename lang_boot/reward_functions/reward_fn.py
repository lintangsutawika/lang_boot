import re

from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer
from lang_boot.utils import math_eval_with_postprocessing, get_lang_score

from lang_boot.reward_functions.repetition_penalty import repetition_penalty

def compute_score(data_source, solution_str, ground_truth, extra_info=None, use_lang=True, use_penalty=True):

    try:
        gold = get_boxed_answer(ground_truth)
        if gold == "None":
            gold = ground_truth
    except:
        gold = ground_truth

    try:
        ans = get_boxed_answer(solution_str)
    except:
        ans = solution_str

    ans_score = 0.0
    if extra_info["task"] == "mmlu-global":
        gold_letter, gold_answer = gold.split(":::")
        if ans.lower() == gold_letter.lower():
            ans_score = 1.0
        elif ans.lower() == gold_answer.lower():
            ans_score = 1.0
        elif ans.lower()[:1] == gold_letter.lower():
            ans_score = 1.0
    else:
        ans_score = math_eval(ans, gold)

    lang = extra_info["lang"]
    if use_lang and (lang != "en"):
        _, lang_score = get_lang_score(solution_str, lang=lang)
        reward = ans_score * lang_score
    else:
        reward = ans_score

    if use_penalty and (lang != "en"):
        penalty = 0
        N_gram_sizes = [2, 3, 4, 5]
        for N in N_gram_sizes:
            penalty += repetition_penalty(solution_str, window_size=N)
        penalty /= len(N_gram_sizes)
        reward -= penalty

    return {
        "score": reward,
        "task_score": ans_score,
        "lang_score": lang_score if use_lang else 1.0,
        "repetition_penalty": penalty if use_penalty else 0.0,
    }
