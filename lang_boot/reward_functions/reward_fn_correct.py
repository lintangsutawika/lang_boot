import re

from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    try:
        gold = get_boxed_answer(ground_truth)
    except:
        gold = ground_truth

    try:
        ans = get_boxed_answer(solution_str)
    except:
        ans = solution_str

    reward = math_eval(ans, gold)
    return reward