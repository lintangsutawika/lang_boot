import re

from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer
from lang_boot.utils import math_eval_with_postprocessing, get_lang_score

from lang_boot.reward_functions.repetition_penalty import repetition_penalty

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    try:
        gold = get_boxed_answer(ground_truth)
    except:
        gold = ground_truth

    try:
        ans = get_boxed_answer(solution_str)
    except:
        ans = solution_str

    # ans_score = math_eval_with_postprocessing(solution_str, gold)
    ans_score = math_eval(ans, gold)
    _, lang_score = get_lang_score(solution_str, lang=extra_info["lang"])

    reward = ans_score * lang_score
    penalty = 0
    N_gram_sizes = [2, 3, 4, 5]
    for N in N_gram_sizes:
        penalty += repetition_penalty(solution_str, window_size=N)
    penalty /= len(N_gram_sizes)

    reward -= penalty
    return reward